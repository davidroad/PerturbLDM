#!/usr/bin/env python3
"""Shared, train-only preprocessing and metric definitions for MLP and RF."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


RUN = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@dataclass
class TrainingCache:
    root: Path
    metadata: pd.DataFrame
    responses: np.ndarray
    controls: np.ndarray
    control_metadata: pd.DataFrame
    control_index: np.ndarray
    genes: np.ndarray
    train_mask: np.ndarray
    valid_mask: np.ndarray


def load_training_cache(root: Path) -> TrainingCache:
    root = root.resolve()
    manifest = json.loads((root / "cache_manifest.json").read_text())
    require(manifest["status"] == "CACHE_OK", "cache manifest did not pass")
    require(manifest["test_response_accessed"] is False, "model-search cache accessed test response")

    metadata = pd.read_csv(root / "train/metadata.csv")
    response = np.load(root / "train/response_means.npy", mmap_mode="r")
    counts = np.load(root / "train/response_means_counts.npy")
    control_metadata = pd.read_csv(root / "control/metadata.csv")
    controls = np.load(root / "control/control_means.npy", mmap_mode="r")
    control_counts = np.load(root / "control/control_means_counts.npy")
    genes = pd.read_csv(root / "genes.csv")["gene"].astype(str).to_numpy()

    require(response.shape == (len(metadata), len(genes)), "response shape differs from metadata/genes")
    require(controls.shape == (len(control_metadata), len(genes)), "control shape differs from metadata/genes")
    require(np.all(counts == 500), "response condition cell counts are not all 500")
    require(np.all(control_counts == 500), "control cell-line counts are not all 500")
    require(metadata["condition_id"].is_unique, "condition IDs are duplicated")
    require(control_metadata["normalized_cell_line"].is_unique, "control cell lines are duplicated")
    require(len(genes) == 13_784 and len(set(genes)) == len(genes), "gene identity contract failed")
    require(np.isfinite(response).all() and np.isfinite(controls).all(), "cache contains non-finite values")

    control_lookup = {
        value: int(index)
        for index, value in enumerate(control_metadata["normalized_cell_line"].astype(str))
    }
    normalized_cell = metadata["cell_line"].astype(str).str.replace("CVCL_", "CVCL-", regex=False)
    missing_controls = set(normalized_cell).difference(control_lookup)
    require(not missing_controls, f"matched controls missing for {len(missing_controls)} cell lines")
    control_index = np.asarray([control_lookup[value] for value in normalized_cell], dtype=np.int32)
    train_mask = metadata["search_split"].eq("train").to_numpy()
    valid_mask = metadata["search_split"].eq("valid").to_numpy()
    require(train_mask.any() and valid_mask.any(), "cache must contain train and validation conditions")
    require(not np.any(train_mask & valid_mask), "train/validation overlap")

    return TrainingCache(
        root=root,
        metadata=metadata,
        responses=response,
        controls=controls,
        control_metadata=control_metadata,
        control_index=control_index,
        genes=genes,
        train_mask=train_mask,
        valid_mask=valid_mask,
    )


def build_delta_cache(cache: TrainingCache) -> Path:
    output_dir = cache.root / "model_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "delta_expression.npy"
    if output.is_file():
        delta = np.load(output, mmap_mode="r")
        require(delta.shape == cache.responses.shape, "existing delta cache shape changed")
        require(np.isfinite(delta).all(), "existing delta cache contains non-finite values")
        return output
    partial = output_dir / ".delta_expression.partial.npy"
    delta = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float32, shape=cache.responses.shape
    )
    for start in range(0, len(cache.metadata), 256):
        stop = min(len(cache.metadata), start + 256)
        delta[start:stop] = (
            np.asarray(cache.responses[start:stop], dtype=np.float32)
            - np.asarray(cache.controls[cache.control_index[start:stop]], dtype=np.float32)
        )
    delta.flush()
    require(np.isfinite(delta).all(), "new delta cache contains non-finite values")
    del delta
    partial.replace(output)
    return output


def fit_train_only_condition_encoder(cache: TrainingCache) -> tuple[np.ndarray, dict]:
    train = cache.metadata.loc[cache.train_mask]
    drug_vocab = sorted(train["drug"].astype(str).unique())
    cell_vocab = sorted(train["cell_line"].astype(str).unique())
    require(set(cache.metadata.loc[cache.valid_mask, "drug"].astype(str)).issubset(drug_vocab), "validation contains unseen drug")
    require(set(cache.metadata.loc[cache.valid_mask, "cell_line"].astype(str)).issubset(cell_vocab), "validation contains unseen cell line")

    drug_encoder = OneHotEncoder(
        categories=[drug_vocab], handle_unknown="error", sparse_output=False, dtype=np.float32
    )
    cell_encoder = OneHotEncoder(
        categories=[cell_vocab], handle_unknown="error", sparse_output=False, dtype=np.float32
    )
    drug_encoder.fit(train[["drug"]].astype(str))
    cell_encoder.fit(train[["cell_line"]].astype(str))
    drug = drug_encoder.transform(cache.metadata[["drug"]].astype(str))
    cell = cell_encoder.transform(cache.metadata[["cell_line"]].astype(str))
    train_dose = cache.metadata.loc[cache.train_mask, "dose"].to_numpy(dtype=np.float64)
    dose_mean = float(train_dose.mean())
    dose_sd = float(train_dose.std())
    require(dose_sd > 0, "training dose has zero variance")
    dose = ((cache.metadata["dose"].to_numpy(dtype=np.float32) - dose_mean) / dose_sd).reshape(-1, 1)
    encoded = np.hstack([drug, cell, dose]).astype(np.float32, copy=False)
    contract = {
        "fit_scope": "internal training conditions only",
        "drug_vocabulary": drug_vocab,
        "cell_line_vocabulary": cell_vocab,
        "drug_feature_count": len(drug_vocab),
        "cell_line_feature_count": len(cell_vocab),
        "dose_mean": dose_mean,
        "dose_standard_deviation": dose_sd,
        "dose_transform": "(dose - train_mean) / train_standard_deviation",
        "validation_unknown_drugs": 0,
        "validation_unknown_cell_lines": 0,
    }
    return encoded, contract


def build_mlp_design(cache: TrainingCache, condition_features: np.ndarray) -> Path:
    output = cache.root / "model_inputs/mlp_design.npy"
    expected_shape = (len(cache.metadata), condition_features.shape[1] + len(cache.genes))
    if output.is_file():
        design = np.load(output, mmap_mode="r")
        require(design.shape == expected_shape, "existing MLP design shape changed")
        return output
    partial = output.parent / ".mlp_design.partial.npy"
    design = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=expected_shape)
    condition_dim = condition_features.shape[1]
    for start in range(0, len(cache.metadata), 256):
        stop = min(len(cache.metadata), start + 256)
        design[start:stop, :condition_dim] = condition_features[start:stop]
        design[start:stop, condition_dim:] = cache.controls[cache.control_index[start:stop]]
    design.flush()
    require(np.isfinite(design).all(), "MLP design contains non-finite values")
    del design
    partial.replace(output)
    return output


def select_train_only_control_features(cache: TrainingCache, feature_count: int) -> tuple[np.ndarray, np.ndarray]:
    train_control_index = cache.control_index[cache.train_mask]
    weights = np.bincount(train_control_index, minlength=len(cache.control_metadata)).astype(np.float64)
    require(int(weights.sum()) == int(cache.train_mask.sum()), "control feature weights differ from training count")
    profiles = np.asarray(cache.controls, dtype=np.float64)
    weighted_mean = np.average(profiles, axis=0, weights=weights)
    variance = np.average((profiles - weighted_mean) ** 2, axis=0, weights=weights)
    require(feature_count <= len(variance), "requested too many control features")
    order = np.lexsort((np.arange(len(variance)), -variance))
    selected = order[:feature_count].astype(np.int32)
    require(len(np.unique(selected)) == feature_count, "selected control features are duplicated")
    return selected, variance[selected]


def build_rf_design(
    cache: TrainingCache,
    condition_features: np.ndarray,
    selected_genes: np.ndarray,
) -> Path:
    output = cache.root / "model_inputs/rf_design.npy"
    expected_shape = (len(cache.metadata), condition_features.shape[1] + len(selected_genes))
    if output.is_file():
        design = np.load(output, mmap_mode="r")
        require(design.shape == expected_shape, "existing RF design shape changed")
        return output
    partial = output.parent / ".rf_design.partial.npy"
    design = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=expected_shape)
    condition_dim = condition_features.shape[1]
    for start in range(0, len(cache.metadata), 512):
        stop = min(len(cache.metadata), start + 512)
        design[start:stop, :condition_dim] = condition_features[start:stop]
        design[start:stop, condition_dim:] = cache.controls[
            cache.control_index[start:stop]
        ][:, selected_genes]
    design.flush()
    require(np.isfinite(design).all(), "RF design contains non-finite values")
    del design
    partial.replace(output)
    return output


def rowwise_pearson(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    observed_centered = observed - observed.mean(axis=1, keepdims=True)
    predicted_centered = predicted - predicted.mean(axis=1, keepdims=True)
    numerator = np.sum(observed_centered * predicted_centered, axis=1)
    denominator = np.sqrt(
        np.sum(observed_centered**2, axis=1) * np.sum(predicted_centered**2, axis=1)
    )
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def regression_metrics(
    observed_delta: np.ndarray,
    predicted_delta: np.ndarray,
    matched_control: np.ndarray,
) -> dict:
    observed_delta = np.asarray(observed_delta, dtype=np.float32)
    predicted_delta = np.asarray(predicted_delta, dtype=np.float32)
    matched_control = np.asarray(matched_control, dtype=np.float32)
    require(observed_delta.shape == predicted_delta.shape == matched_control.shape, "metric arrays differ in shape")
    require(np.isfinite(observed_delta).all() and np.isfinite(predicted_delta).all(), "metric arrays contain non-finite values")
    difference = predicted_delta.astype(np.float64) - observed_delta.astype(np.float64)
    pooled_mse = float(np.mean(difference**2))
    effect_pearson = rowwise_pearson(observed_delta, predicted_delta)
    observed_absolute = observed_delta + matched_control
    predicted_absolute = predicted_delta + matched_control
    absolute_pearson = rowwise_pearson(observed_absolute, predicted_absolute)
    residual_ss = np.sum(
        (predicted_absolute.astype(np.float64) - observed_absolute.astype(np.float64)) ** 2,
        axis=1,
    )
    total_ss = np.sum(
        (observed_absolute.astype(np.float64) - observed_absolute.mean(axis=1, keepdims=True)) ** 2,
        axis=1,
    )
    absolute_r2 = np.divide(
        residual_ss,
        total_ss,
        out=np.full_like(residual_ss, np.nan),
        where=total_ss > 0,
    )
    absolute_r2 = 1.0 - absolute_r2
    return {
        "condition_count": int(len(observed_delta)),
        "gene_count": int(observed_delta.shape[1]),
        "delta_mse": pooled_mse,
        "median_condition_delta_pearson": float(np.nanmedian(effect_pearson)),
        "mean_condition_delta_pearson": float(np.nanmean(effect_pearson)),
        "mean_condition_absolute_r2": float(np.nanmean(absolute_r2)),
        "median_condition_absolute_r2": float(np.nanmedian(absolute_r2)),
        "mean_condition_absolute_pearson": float(np.nanmean(absolute_pearson)),
        "median_condition_absolute_pearson": float(np.nanmedian(absolute_pearson)),
    }
