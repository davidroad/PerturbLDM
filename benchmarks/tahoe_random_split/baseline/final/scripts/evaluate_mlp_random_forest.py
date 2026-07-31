#!/usr/bin/env python3
"""Gated OOD inference and condition-level evaluation for the final MLP and RF.

The program deliberately opens the OOD cache only after both model selections
have passed a frozen-selection and artifact-integrity gate.  The scientific
unit is one drug-dose-cell-line condition.  Both models predict expression
change relative to the same matched cell-line control; absolute expression is
recovered by adding that control before expression-profile metrics are
calculated.

Formal mode requires exactly 13,942 OOD conditions and 13,784 genes.  A
strictly /tmp-scoped synthetic mode exists only for executable testing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy.stats import rankdata
from torch import nn


RUN = Path(__file__).resolve().parents[1]
FORMAL_OOD_CONDITIONS = 13_942
FORMAL_GENE_COUNT = 13_784
METRIC_COLUMNS = [
    "MSE",
    "MAE",
    "R2",
    "Pearson_r",
    "Spearman_r",
    "Chatterjee",
    "effect_MSE",
    "effect_MAE",
    "effect_Pearson_r",
    "effect_Spearman_r",
    "effect_Chatterjee",
    "observed_effect_rms",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_strings(values: list[str] | np.ndarray | pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def set_hash(values: list[str] | np.ndarray | pd.Series) -> str:
    return hash_strings(sorted(str(value) for value in values))


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv_atomic(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def file_record(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "sha256": sha256(path),
    }


def resolve_recorded_path(value: object, expected: Path, label: str) -> Path:
    recorded = Path(str(value)).resolve()
    expected = expected.resolve()
    require(recorded == expected, f"{label} path differs from the formal artifact path")
    require(expected.is_file(), f"{label} is missing: {expected}")
    return expected


def read_json(path: Path, label: str) -> dict:
    require(path.is_file(), f"{label} is missing: {path}")
    payload = json.loads(path.read_text())
    require(isinstance(payload, dict), f"{label} is not a JSON object")
    return payload


def normalize_cell_line(value: object) -> str:
    return str(value).strip().replace("CVCL_", "CVCL-")


class MLP(nn.Module):
    """Architecture frozen by run_mlp_search.py."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden: list[int], dropout: float
    ):
        super().__init__()
        require(hidden == [512, 256], "MLP hidden layers differ from [512, 256]")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.BatchNorm1d(hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def load_torch_checkpoint(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    require(isinstance(payload, dict), "MLP checkpoint is not a dictionary")
    return payload


def verify_hash(path: Path, expected: object, label: str) -> None:
    require(str(expected) == sha256(path), f"{label} SHA256 mismatch")


def validate_frozen_selections(run_root: Path, synthetic: bool) -> dict:
    """Validate both selections before any OOD-cache path is opened."""

    contract_path = run_root / "config/hyperparameter_search.json"
    contract = read_json(contract_path, "search contract")
    contract_hash = sha256(contract_path)
    if not synthetic:
        require(
            contract.get("contract_version") == "2026-07-27.2",
            "unexpected formal search-contract version",
        )
        require(
            int(contract["split"]["external_ood_conditions"])
            == FORMAL_OOD_CONDITIONS,
            "search contract does not specify 13,942 OOD conditions",
        )

    selections: dict[str, dict] = {}
    for method in ("mlp", "rf"):
        selection_path = run_root / f"results/{method}/selection.json"
        selection = read_json(selection_path, f"{method.upper()} selection")
        require(
            selection.get("status") == "SELECTION_FROZEN",
            f"{method.upper()} selection is not SELECTION_FROZEN",
        )
        require(
            selection.get("test_response_accessed") is False,
            f"{method.upper()} selection reports test-response access",
        )
        require(
            bool(selection.get("smoke_test")) is synthetic,
            f"{method.upper()} selection mode differs from requested mode",
        )
        resolve_recorded_path(
            selection["contract"], contract_path, f"{method.upper()} contract"
        )
        require(
            selection.get("contract_sha256") == contract_hash,
            f"{method.upper()} contract hash mismatch",
        )
        selections[method] = selection

    mlp_checks_path = run_root / "checks/mlp/final_checks.json"
    mlp_checks = read_json(mlp_checks_path, "MLP final checks")
    require(
        mlp_checks.get("status") == "MLP_TWO_STAGE_SEARCH_OK"
        and mlp_checks.get("test_response_accessed") is False,
        "MLP final checks did not pass without test access",
    )
    rf_checks_path = run_root / "checks/rf/final_checks.json"
    rf_checks = read_json(rf_checks_path, "RF final checks")
    require(
        rf_checks.get("status") == "RF_TWO_STAGE_SEARCH_OK"
        and rf_checks.get("test_response_accessed") is False,
        "RF final checks did not pass without test access",
    )

    training_cache_manifest_path = run_root / "cache/train_control_search/cache_manifest.json"
    training_manifest = read_json(
        training_cache_manifest_path, "training/control cache manifest"
    )
    require(
        training_manifest.get("status") == "CACHE_OK"
        and training_manifest.get("test_response_accessed") is False,
        "training/control cache is not a test-free CACHE_OK artifact",
    )
    require(
        selections["mlp"].get("cache_manifest_sha256")
        == sha256(training_cache_manifest_path),
        "MLP selection does not match the training/control cache manifest",
    )

    mlp_encoder_path = run_root / "provenance/mlp/condition_encoder.json"
    rf_encoder_path = run_root / "provenance/rf/condition_encoder.json"
    mlp_encoder = read_json(mlp_encoder_path, "MLP condition encoder")
    rf_encoder = read_json(rf_encoder_path, "RF condition encoder")
    require(mlp_encoder == rf_encoder, "MLP and RF condition encoders differ")
    require(
        selections["mlp"].get("condition_encoder_sha256")
        == sha256(mlp_encoder_path),
        "MLP condition-encoder hash mismatch",
    )
    require(
        mlp_encoder.get("fit_scope") == "internal training conditions only",
        "condition encoder was not fitted only on internal training",
    )

    training_genes_path = run_root / "cache/train_control_search/genes.csv"
    training_genes = pd.read_csv(training_genes_path)["gene"].astype(str)
    require(training_genes.is_unique, "training gene names are duplicated")
    if not synthetic:
        require(
            len(training_genes) == FORMAL_GENE_COUNT,
            "formal training gene count differs from 13,784",
        )

    mlp_checkpoint_path = resolve_recorded_path(
        selections["mlp"]["selected_checkpoint"],
        run_root / "models/mlp/selected_best_validation_checkpoint.pt",
        "MLP selected checkpoint",
    )
    verify_hash(
        mlp_checkpoint_path,
        selections["mlp"]["selected_checkpoint_sha256"],
        "MLP selected checkpoint",
    )
    mlp_frozen_path = resolve_recorded_path(
        selections["mlp"]["frozen_hyperparameters"],
        run_root / "results/mlp/frozen_hyperparameters.json",
        "MLP frozen hyperparameters",
    )
    verify_hash(
        mlp_frozen_path,
        selections["mlp"]["frozen_hyperparameters_sha256"],
        "MLP frozen hyperparameters",
    )
    mlp_frozen = read_json(mlp_frozen_path, "MLP frozen hyperparameters")
    require(
        mlp_frozen.get("status") == "HYPERPARAMETERS_FROZEN"
        and mlp_frozen.get("test_response_accessed") is False,
        "MLP hyperparameters are not frozen without test access",
    )
    require(
        float(mlp_frozen["dropout"])
        == float(selections["mlp"]["selected"]["dropout"]),
        "MLP selected dropout differs from frozen hyperparameters",
    )
    require(
        float(mlp_frozen["learning_rate"])
        == float(selections["mlp"]["selected"]["learning_rate"]),
        "MLP selected learning rate differs from frozen hyperparameters",
    )
    mlp_final_result_path = resolve_recorded_path(
        selections["mlp"]["final_result"],
        run_root / "results/mlp/final/result.json",
        "MLP final result",
    )
    verify_hash(
        mlp_final_result_path,
        selections["mlp"]["final_result_sha256"],
        "MLP final result",
    )
    mlp_final_result = read_json(mlp_final_result_path, "MLP final result")
    require(mlp_final_result.get("status") == "STAGE_OK", "MLP final stage did not pass")
    checkpoint = load_torch_checkpoint(mlp_checkpoint_path)
    require(
        checkpoint.get("stage_name") == "final_refit_all_formal_train",
        "MLP checkpoint is not the all-formal-training refit",
    )
    require(
        checkpoint.get("stage_contract_sha256")
        == mlp_final_result.get("stage_contract_sha256"),
        "MLP checkpoint and final-result stage contracts differ",
    )
    require(
        checkpoint.get("hidden_layers") == [512, 256],
        "MLP checkpoint architecture differs from the frozen design",
    )
    require(
        float(checkpoint["dropout"]) == float(mlp_frozen["dropout"]),
        "MLP checkpoint dropout differs from the frozen setting",
    )
    require(
        int(checkpoint["output_dim"]) == len(training_genes),
        "MLP output dimension differs from the training gene list",
    )
    condition_dim = (
        int(mlp_encoder["drug_feature_count"])
        + int(mlp_encoder["cell_line_feature_count"])
        + 1
    )
    require(
        int(checkpoint["input_dim"]) == condition_dim + len(training_genes),
        "MLP input dimension differs from condition plus matched-control features",
    )

    rf_model_path = resolve_recorded_path(
        selections["rf"]["final_model"],
        run_root / "models/rf/final_random_forest.joblib",
        "RF final model",
    )
    verify_hash(
        rf_model_path, selections["rf"]["final_model_sha256"], "RF final model"
    )
    rf_frozen_path = resolve_recorded_path(
        selections["rf"]["frozen_hyperparameters"],
        run_root / "results/rf/frozen_hyperparameters.json",
        "RF frozen hyperparameters",
    )
    verify_hash(
        rf_frozen_path,
        selections["rf"]["frozen_hyperparameters_sha256"],
        "RF frozen hyperparameters",
    )
    rf_frozen = read_json(rf_frozen_path, "RF frozen hyperparameters")
    require(
        rf_frozen.get("status") == "HYPERPARAMETERS_FROZEN"
        and rf_frozen.get("test_response_accessed") is False,
        "RF hyperparameters are not frozen without test access",
    )
    require(
        int(rf_frozen["selected_min_samples_leaf"])
        == int(selections["rf"]["selected_min_samples_leaf"]),
        "RF selected leaf size differs from frozen hyperparameters",
    )
    rf_metrics_path = resolve_recorded_path(
        selections["rf"]["formal_validation_metrics"],
        run_root / "results/rf/formal_validation_metrics.json",
        "RF formal-validation metrics",
    )
    verify_hash(
        rf_metrics_path,
        selections["rf"]["formal_validation_metrics_sha256"],
        "RF formal-validation metrics",
    )
    rf_artifact_path = run_root / "models/rf/final_random_forest_artifact.json"
    rf_artifact = read_json(rf_artifact_path, "RF model artifact")
    require(
        rf_artifact.get("status") == "FINAL_MODEL_SAVED"
        and rf_artifact.get("test_response_accessed") is False,
        "RF model artifact is not a test-free final model",
    )
    require(
        rf_artifact.get("model_sha256") == sha256(rf_model_path),
        "RF artifact model hash mismatch",
    )
    require(
        int(rf_artifact["model_parameters"]["min_samples_leaf"])
        == int(selections["rf"]["selected_min_samples_leaf"]),
        "RF artifact parameters differ from the frozen selection",
    )

    rf_input_hashes_path = run_root / "provenance/rf/input_hashes_and_stats.json"
    rf_input_hashes = read_json(rf_input_hashes_path, "RF input-hash record")
    require(
        rf_input_hashes["condition_encoder"]["sha256"] == sha256(rf_encoder_path),
        "RF condition-encoder hash mismatch",
    )
    require(
        rf_input_hashes["genes"]["sha256"] == sha256(training_genes_path),
        "RF training-gene hash mismatch",
    )
    require(
        rf_input_hashes["cache_manifest"]["sha256"]
        == sha256(training_cache_manifest_path),
        "RF training/control cache-manifest hash mismatch",
    )

    rf_features_path = run_root / "provenance/rf/selected_train_only_control_features.csv"
    rf_features = pd.read_csv(rf_features_path)
    required_feature_columns = {
        "selection_rank_one_based",
        "gene_index_zero_based",
        "gene",
        "weighted_formal_train_control_variance",
    }
    require(
        set(rf_features.columns) == required_feature_columns,
        "RF selected-feature table has unexpected columns",
    )
    expected_feature_count = (
        int(contract["random_forest"]["matched_control_feature_count"])
        if not synthetic
        else len(rf_features)
    )
    require(
        len(rf_features) == expected_feature_count,
        "RF selected-control feature count differs from the contract",
    )
    feature_indices = rf_features["gene_index_zero_based"].to_numpy(dtype=np.int64)
    require(
        len(np.unique(feature_indices)) == len(feature_indices)
        and np.all(feature_indices >= 0)
        and np.all(feature_indices < len(training_genes)),
        "RF selected-control gene indices are invalid or duplicated",
    )
    require(
        rf_features["selection_rank_one_based"].to_numpy(dtype=np.int64).tolist()
        == list(range(1, len(rf_features) + 1)),
        "RF selected-control feature ranks are not contiguous",
    )
    require(
        training_genes.iloc[feature_indices].tolist()
        == rf_features["gene"].astype(str).tolist(),
        "RF selected-control gene names do not match training gene order",
    )

    gate = {
        "status": "FROZEN_SELECTION_GATE_OK",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "formal_mode": not synthetic,
        "ood_cache_opened": False,
        "contract": file_record(contract_path),
        "training_cache_manifest": file_record(training_cache_manifest_path),
        "training_genes": file_record(training_genes_path),
        "mlp_selection": file_record(run_root / "results/mlp/selection.json"),
        "mlp_final_checks": file_record(mlp_checks_path),
        "mlp_checkpoint": file_record(mlp_checkpoint_path),
        "mlp_encoder": file_record(mlp_encoder_path),
        "rf_selection": file_record(run_root / "results/rf/selection.json"),
        "rf_final_checks": file_record(rf_checks_path),
        "rf_model": file_record(rf_model_path),
        "rf_encoder": file_record(rf_encoder_path),
        "rf_selected_control_features": file_record(rf_features_path),
        "test_response_accessed_in_training_or_selection": False,
    }
    return {
        "gate": gate,
        "contract": contract,
        "contract_path": contract_path,
        "training_genes": training_genes.to_numpy(),
        "training_genes_path": training_genes_path,
        "encoder": mlp_encoder,
        "mlp_selection": selections["mlp"],
        "mlp_checkpoint": checkpoint,
        "mlp_checkpoint_path": mlp_checkpoint_path,
        "rf_selection": selections["rf"],
        "rf_model_path": rf_model_path,
        "rf_features": rf_features,
        "rf_feature_indices": feature_indices,
    }


def resolve_split_manifest(
    run_root: Path, contract: dict, override: Path | None
) -> Path:
    if override is not None:
        return override.resolve()
    value = Path(str(contract["split"]["manifest"]))
    if value.is_absolute():
        return value.resolve()
    return (run_root / value).resolve()


def load_ood_cache(
    cache_root: Path,
    training_genes: np.ndarray,
    split_manifest_path: Path,
    synthetic: bool,
) -> dict:
    """Open and validate the OOD cache. Called only after the frozen gate."""

    cache_root = cache_root.resolve()
    manifest_path = cache_root / "cache_manifest.json"
    manifest = read_json(manifest_path, "OOD cache manifest")
    require(
        manifest.get("status") == "CACHE_OK"
        and manifest.get("test_response_accessed") is True,
        "OOD cache manifest must be CACHE_OK with explicit test-response access",
    )
    require(
        "test" in {
            str(report.get("source"))
            for report in manifest.get("datasets", [])
            if isinstance(report, dict)
        },
        "OOD cache manifest does not record a test-response dataset",
    )
    metadata_path = cache_root / "test/metadata.csv"
    response_path = cache_root / "test/response_means.npy"
    response_counts_path = cache_root / "test/response_means_counts.npy"
    control_metadata_path = cache_root / "control/metadata.csv"
    control_path = cache_root / "control/control_means.npy"
    control_counts_path = cache_root / "control/control_means_counts.npy"
    genes_path = cache_root / "genes.csv"
    for path in (
        metadata_path,
        response_path,
        response_counts_path,
        control_metadata_path,
        control_path,
        control_counts_path,
        genes_path,
    ):
        require(path.is_file(), f"OOD cache artifact is missing: {path}")

    metadata = pd.read_csv(metadata_path)
    required_metadata = {
        "condition_id",
        "cell_line",
        "drug",
        "dose",
        "search_split",
    }
    require(
        required_metadata.issubset(metadata.columns),
        "OOD metadata lacks required condition fields",
    )
    expected_conditions = len(metadata) if synthetic else FORMAL_OOD_CONDITIONS
    require(
        len(metadata) == expected_conditions,
        f"OOD condition count differs from {expected_conditions}",
    )
    require(metadata["condition_id"].is_unique, "OOD condition IDs are duplicated")
    require(
        metadata["search_split"].astype(str).eq("ood").all(),
        "OOD metadata contains a non-OOD split",
    )
    require(
        metadata["condition_id"].astype(str).eq(
            metadata["condition_id"].astype(str).str.strip()
        ).all(),
        "OOD condition IDs contain leading/trailing whitespace",
    )

    split_manifest = pd.read_csv(split_manifest_path, dtype=str)
    require(
        {"condition_id", "cpa_split"}.issubset(split_manifest.columns),
        "split manifest lacks condition_id/cpa_split",
    )
    require(
        split_manifest["condition_id"].is_unique,
        "split-manifest condition IDs are duplicated",
    )
    expected_ids = split_manifest.loc[
        split_manifest["cpa_split"].eq("ood"), "condition_id"
    ].astype(str)
    require(
        len(expected_ids) == expected_conditions,
        "split-manifest OOD count differs from cache metadata",
    )
    metadata_ids = metadata["condition_id"].astype(str)
    require(
        set(metadata_ids) == set(expected_ids),
        "OOD cache and split manifest do not contain the same condition IDs",
    )

    genes = pd.read_csv(genes_path)["gene"].astype(str).to_numpy()
    require(
        np.array_equal(genes, training_genes),
        "OOD gene names/order differ from the training gene list",
    )
    require(len(set(genes)) == len(genes), "OOD gene names are duplicated")
    if not synthetic:
        require(len(genes) == FORMAL_GENE_COUNT, "OOD gene count differs from 13,784")

    response = np.load(response_path, mmap_mode="r")
    response_counts = np.load(response_counts_path)
    control_metadata = pd.read_csv(control_metadata_path)
    controls = np.load(control_path, mmap_mode="r")
    control_counts = np.load(control_counts_path)
    require(
        response.shape == (expected_conditions, len(genes)),
        "OOD response shape differs from metadata/genes",
    )
    require(
        controls.shape == (len(control_metadata), len(genes)),
        "control shape differs from metadata/genes",
    )
    expected_cells_per_group = 3 if synthetic else 500
    require(
        np.all(response_counts == expected_cells_per_group),
        "OOD response group counts differ from the required count",
    )
    require(
        np.all(control_counts == expected_cells_per_group),
        "OOD control group counts differ from the required count",
    )
    require(
        control_metadata["normalized_cell_line"].astype(str).is_unique,
        "control normalized cell lines are duplicated",
    )
    require(
        np.isfinite(response).all() and np.isfinite(controls).all(),
        "OOD cache contains non-finite expression values",
    )
    control_lookup = {
        value: int(index)
        for index, value in enumerate(
            control_metadata["normalized_cell_line"].astype(str)
        )
    }
    normalized_cells = metadata["cell_line"].map(normalize_cell_line)
    missing = set(normalized_cells).difference(control_lookup)
    require(not missing, f"matched controls are missing for {len(missing)} cell lines")
    control_indices = np.asarray(
        [control_lookup[value] for value in normalized_cells], dtype=np.int32
    )

    return {
        "root": cache_root,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "metadata": metadata,
        "metadata_path": metadata_path,
        "response": response,
        "response_path": response_path,
        "response_counts_path": response_counts_path,
        "controls": controls,
        "control_path": control_path,
        "control_metadata_path": control_metadata_path,
        "control_counts_path": control_counts_path,
        "control_indices": control_indices,
        "genes": genes,
        "genes_path": genes_path,
        "split_manifest_path": split_manifest_path,
        "condition_order_sha256": hash_strings(metadata_ids),
        "condition_set_sha256": set_hash(metadata_ids),
    }


def encode_conditions(metadata: pd.DataFrame, encoder: dict) -> np.ndarray:
    drug_vocab = [str(value) for value in encoder["drug_vocabulary"]]
    cell_vocab = [str(value) for value in encoder["cell_line_vocabulary"]]
    require(
        len(drug_vocab) == int(encoder["drug_feature_count"]),
        "drug vocabulary length differs from encoder feature count",
    )
    require(
        len(cell_vocab) == int(encoder["cell_line_feature_count"]),
        "cell-line vocabulary length differs from encoder feature count",
    )
    require(
        len(set(drug_vocab)) == len(drug_vocab)
        and len(set(cell_vocab)) == len(cell_vocab),
        "condition-encoder vocabularies contain duplicates",
    )
    drug_lookup = {value: index for index, value in enumerate(drug_vocab)}
    cell_lookup = {value: index for index, value in enumerate(cell_vocab)}
    drugs = metadata["drug"].astype(str).tolist()
    cells = metadata["cell_line"].astype(str).tolist()
    unknown_drugs = sorted(set(drugs).difference(drug_lookup))
    unknown_cells = sorted(set(cells).difference(cell_lookup))
    require(not unknown_drugs, f"OOD encoder has {len(unknown_drugs)} unknown drugs")
    require(
        not unknown_cells, f"OOD encoder has {len(unknown_cells)} unknown cell lines"
    )
    dose_sd = float(encoder["dose_standard_deviation"])
    require(math.isfinite(dose_sd) and dose_sd > 0, "encoder dose SD is invalid")
    output = np.zeros(
        (len(metadata), len(drug_vocab) + len(cell_vocab) + 1), dtype=np.float32
    )
    rows = np.arange(len(metadata))
    output[rows, [drug_lookup[value] for value in drugs]] = 1.0
    output[
        rows, len(drug_vocab) + np.asarray([cell_lookup[value] for value in cells])
    ] = 1.0
    output[:, -1] = (
        metadata["dose"].to_numpy(dtype=np.float64) - float(encoder["dose_mean"])
    ) / dose_sd
    require(np.isfinite(output).all(), "encoded OOD conditions are non-finite")
    return output


def create_memmap(path: Path, shape: tuple[int, int]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name("." + path.name + ".partial")
    if partial.exists():
        partial.unlink()
    return np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float32, shape=shape
    )


def publish_memmap(path: Path, array: np.memmap) -> None:
    array.flush()
    require(np.isfinite(array).all(), f"non-finite values in output array: {path.name}")
    del array
    partial = path.with_name("." + path.name + ".partial")
    partial.replace(path)


def materialize_reference_arrays(cache: dict, array_root: Path, batch_size: int) -> dict:
    shape = cache["response"].shape
    paths = {
        "observed_expression": array_root / "observed_expression.npy",
        "matched_control_expression": array_root / "matched_control_expression.npy",
        "observed_effect": array_root / "observed_effect.npy",
    }
    arrays = {name: create_memmap(path, shape) for name, path in paths.items()}
    for start in range(0, shape[0], batch_size):
        stop = min(shape[0], start + batch_size)
        observed = np.asarray(cache["response"][start:stop], dtype=np.float32)
        control = np.asarray(
            cache["controls"][cache["control_indices"][start:stop]], dtype=np.float32
        )
        arrays["observed_expression"][start:stop] = observed
        arrays["matched_control_expression"][start:stop] = control
        arrays["observed_effect"][start:stop] = observed - control
    for name, path in paths.items():
        publish_memmap(path, arrays[name])
    return paths


def predict_mlp(
    artifacts: dict,
    condition_features: np.ndarray,
    controls: np.ndarray,
    control_indices: np.ndarray,
    paths: dict,
    device: torch.device,
    batch_size: int,
) -> None:
    checkpoint = artifacts["mlp_checkpoint"]
    model = MLP(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        hidden=[int(value) for value in checkpoint["hidden_layers"]],
        dropout=float(checkpoint["dropout"]),
    )
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    require(not missing and not unexpected, "MLP state-dictionary load was not strict")
    model.to(device)
    model.eval()
    shape = (len(condition_features), int(checkpoint["output_dim"]))
    effect = create_memmap(paths["effect"], shape)
    expression = create_memmap(paths["expression"], shape)
    with torch.no_grad():
        for start in range(0, shape[0], batch_size):
            stop = min(shape[0], start + batch_size)
            matched = np.asarray(
                controls[control_indices[start:stop]], dtype=np.float32
            )
            design = np.concatenate(
                [condition_features[start:stop], matched], axis=1
            ).astype(np.float32, copy=False)
            require(
                design.shape[1] == int(checkpoint["input_dim"]),
                "MLP OOD design dimension mismatch",
            )
            prediction = (
                model(torch.from_numpy(design).to(device))
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            effect[start:stop] = prediction
            expression[start:stop] = prediction + matched
    publish_memmap(paths["effect"], effect)
    publish_memmap(paths["expression"], expression)


def predict_rf(
    artifacts: dict,
    condition_features: np.ndarray,
    controls: np.ndarray,
    control_indices: np.ndarray,
    paths: dict,
    batch_size: int,
) -> None:
    model = joblib.load(artifacts["rf_model_path"])
    require(
        hasattr(model, "predict") and hasattr(model, "n_outputs_"),
        "RF artifact is not a fitted multi-output regressor",
    )
    require(
        int(model.n_outputs_) == len(artifacts["training_genes"]),
        "RF output dimension differs from the gene list",
    )
    feature_indices = artifacts["rf_feature_indices"]
    expected_input_dim = condition_features.shape[1] + len(feature_indices)
    require(
        int(model.n_features_in_) == expected_input_dim,
        "RF input dimension differs from condition plus selected-control features",
    )
    shape = (len(condition_features), len(artifacts["training_genes"]))
    effect = create_memmap(paths["effect"], shape)
    expression = create_memmap(paths["expression"], shape)
    for start in range(0, shape[0], batch_size):
        stop = min(shape[0], start + batch_size)
        matched = np.asarray(controls[control_indices[start:stop]], dtype=np.float32)
        design = np.concatenate(
            [condition_features[start:stop], matched[:, feature_indices]], axis=1
        ).astype(np.float32, copy=False)
        prediction = np.asarray(model.predict(design), dtype=np.float32)
        require(
            prediction.shape == (stop - start, shape[1]),
            "RF OOD prediction shape mismatch",
        )
        effect[start:stop] = prediction
        expression[start:stop] = prediction + matched
    publish_memmap(paths["effect"], effect)
    publish_memmap(paths["expression"], expression)


def rowwise_pearson(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    observed_centered = observed - observed.mean(axis=1, keepdims=True)
    predicted_centered = predicted - predicted.mean(axis=1, keepdims=True)
    numerator = np.sum(observed_centered * predicted_centered, axis=1)
    denominator = np.sqrt(
        np.sum(observed_centered**2, axis=1)
        * np.sum(predicted_centered**2, axis=1)
    )
    values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    values[~np.isfinite(values)] = 0.0
    return values


def rowwise_spearman(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    observed_rank = rankdata(observed, method="average", axis=1)
    predicted_rank = rankdata(predicted, method="average", axis=1)
    return rowwise_pearson(observed_rank, predicted_rank)


def chatterjee_correlation(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Match the benchmark's asymmetric Xi(observed, predicted) formula."""

    n = len(observed)
    if n <= 1:
        return 0.0
    order = np.argsort(observed)
    ranks = rankdata(predicted[order], method="ordinal")
    value = 1.0 - (3.0 * np.abs(np.diff(ranks)).sum()) / (n**2 - 1)
    return float(value) if np.isfinite(value) else 0.0


def metric_block(observed: np.ndarray, predicted: np.ndarray) -> dict[str, np.ndarray]:
    observed64 = np.asarray(observed, dtype=np.float64)
    predicted64 = np.asarray(predicted, dtype=np.float64)
    difference = predicted64 - observed64
    squared = difference**2
    mse = squared.mean(axis=1)
    mae = np.abs(difference).mean(axis=1)
    residual_ss = squared.sum(axis=1)
    centered = observed64 - observed64.mean(axis=1, keepdims=True)
    total_ss = (centered**2).sum(axis=1)
    r2 = np.divide(
        residual_ss,
        total_ss,
        out=np.zeros_like(residual_ss),
        where=total_ss > 0,
    )
    r2 = 1.0 - r2
    constant = total_ss <= 0
    r2[constant] = np.where(residual_ss[constant] <= 0, 1.0, 0.0)
    pearson = rowwise_pearson(observed64, predicted64)
    spearman = rowwise_spearman(observed64, predicted64)
    chatterjee = np.asarray(
        [
            chatterjee_correlation(observed64[index], predicted64[index])
            for index in range(len(observed64))
        ],
        dtype=np.float64,
    )
    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
        "chatterjee": chatterjee,
    }


def condition_metrics(
    metadata: pd.DataFrame,
    observed_expression_path: Path,
    observed_effect_path: Path,
    predicted_expression_path: Path,
    predicted_effect_path: Path,
    method: str,
    batch_size: int,
) -> pd.DataFrame:
    observed_expression = np.load(observed_expression_path, mmap_mode="r")
    observed_effect = np.load(observed_effect_path, mmap_mode="r")
    predicted_expression = np.load(predicted_expression_path, mmap_mode="r")
    predicted_effect = np.load(predicted_effect_path, mmap_mode="r")
    require(
        observed_expression.shape
        == observed_effect.shape
        == predicted_expression.shape
        == predicted_effect.shape,
        f"{method} metric arrays have different shapes",
    )
    rows: list[pd.DataFrame] = []
    for start in range(0, len(metadata), batch_size):
        stop = min(len(metadata), start + batch_size)
        observed_abs = np.asarray(observed_expression[start:stop], dtype=np.float32)
        predicted_abs = np.asarray(predicted_expression[start:stop], dtype=np.float32)
        observed_delta = np.asarray(observed_effect[start:stop], dtype=np.float32)
        predicted_delta = np.asarray(predicted_effect[start:stop], dtype=np.float32)
        absolute = metric_block(observed_abs, predicted_abs)
        effect = metric_block(observed_delta, predicted_delta)
        block = metadata.iloc[start:stop][
            ["condition_id", "drug", "dose", "cell_line", "search_split"]
        ].reset_index(drop=True)
        block.insert(0, "condition_row_zero_based", np.arange(start, stop))
        block["method"] = method
        block["n_genes"] = observed_abs.shape[1]
        block["MSE"] = absolute["mse"]
        block["MAE"] = absolute["mae"]
        block["R2"] = absolute["r2"]
        block["Pearson_r"] = absolute["pearson"]
        block["Spearman_r"] = absolute["spearman"]
        block["Chatterjee"] = absolute["chatterjee"]
        block["effect_MSE"] = effect["mse"]
        block["effect_MAE"] = effect["mae"]
        block["effect_Pearson_r"] = effect["pearson"]
        block["effect_Spearman_r"] = effect["spearman"]
        block["effect_Chatterjee"] = effect["chatterjee"]
        block["observed_effect_rms"] = np.sqrt(np.mean(observed_delta**2, axis=1))
        rows.append(block)
    table = pd.concat(rows, ignore_index=True)
    require(
        len(table) == len(metadata)
        and table["condition_id"].astype(str).tolist()
        == metadata["condition_id"].astype(str).tolist(),
        f"{method} metric rows lost condition order",
    )
    require(
        np.isfinite(table[METRIC_COLUMNS].to_numpy(dtype=np.float64)).all(),
        f"{method} condition metrics contain non-finite values",
    )
    return table


def metric_summary(table: pd.DataFrame) -> dict:
    summary: dict[str, dict] = {}
    for metric in METRIC_COLUMNS:
        values = table[metric].to_numpy(dtype=np.float64)
        summary[metric] = {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=1)),
            "minimum": float(np.min(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75)),
            "q95": float(np.quantile(values, 0.95)),
            "maximum": float(np.max(values)),
        }
    return summary


def deterministic_audit_indices(condition_ids: pd.Series, count: int = 10) -> np.ndarray:
    scores = np.asarray(
        [
            int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)
            for value in condition_ids.astype(str)
        ],
        dtype=np.uint64,
    )
    return np.sort(np.argsort(scores, kind="stable")[: min(count, len(scores))])


def audit_metrics(
    combined: pd.DataFrame,
    metadata: pd.DataFrame,
    array_paths: dict,
    tolerance: float = 2e-10,
) -> pd.DataFrame:
    indices = deterministic_audit_indices(metadata["condition_id"])
    records: list[dict] = []
    observed_abs = np.load(array_paths["observed_expression"], mmap_mode="r")
    observed_effect = np.load(array_paths["observed_effect"], mmap_mode="r")
    for method in ("MLP", "RF"):
        predicted_abs = np.load(
            array_paths[f"{method.lower()}_predicted_expression"], mmap_mode="r"
        )
        predicted_effect = np.load(
            array_paths[f"{method.lower()}_predicted_effect"], mmap_mode="r"
        )
        for index in indices:
            absolute = metric_block(
                np.asarray(observed_abs[index : index + 1]),
                np.asarray(predicted_abs[index : index + 1]),
            )
            effect = metric_block(
                np.asarray(observed_effect[index : index + 1]),
                np.asarray(predicted_effect[index : index + 1]),
            )
            row = combined.loc[
                combined["method"].eq(method)
                & combined["condition_row_zero_based"].eq(index)
            ]
            require(len(row) == 1, "audit condition lookup is not one-to-one")
            row = row.iloc[0]
            recomputed = {
                "MSE": absolute["mse"][0],
                "MAE": absolute["mae"][0],
                "R2": absolute["r2"][0],
                "Pearson_r": absolute["pearson"][0],
                "Spearman_r": absolute["spearman"][0],
                "Chatterjee": absolute["chatterjee"][0],
                "effect_MSE": effect["mse"][0],
                "effect_MAE": effect["mae"][0],
                "effect_Pearson_r": effect["pearson"][0],
                "effect_Spearman_r": effect["spearman"][0],
                "effect_Chatterjee": effect["chatterjee"][0],
                "observed_effect_rms": float(
                    np.sqrt(np.mean(np.asarray(observed_effect[index]) ** 2))
                ),
            }
            maximum_difference = max(
                abs(float(row[key]) - float(value))
                for key, value in recomputed.items()
            )
            require(
                maximum_difference <= tolerance,
                f"{method} metric audit failed for row {index}: {maximum_difference}",
            )
            records.append(
                {
                    "method": method,
                    "condition_row_zero_based": int(index),
                    "condition_id": metadata.iloc[index]["condition_id"],
                    "maximum_absolute_metric_difference": maximum_difference,
                    "status": "AUDIT_OK",
                }
            )
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN)
    parser.add_argument(
        "--ood-cache-root", type=Path, default=RUN / "cache/ood_test_control"
    )
    parser.add_argument(
        "--output-root", type=Path, default=RUN / "results/ood_evaluation_mlp_rf"
    )
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mlp-batch-size", type=int, default=64)
    parser.add_argument("--rf-batch-size", type=int, default=16)
    parser.add_argument("--metric-batch-size", type=int, default=16)
    parser.add_argument(
        "--synthetic-test-mode",
        action="store_true",
        help="Relax formal dimensions only for a fully /tmp-scoped synthetic test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_root = args.output_root.resolve()
    ood_cache_root = args.ood_cache_root.resolve()
    require(
        args.mlp_batch_size > 0
        and args.rf_batch_size > 0
        and args.metric_batch_size > 0,
        "batch sizes must be positive",
    )
    if args.synthetic_test_mode:
        temporary_root = Path("/tmp").resolve()
        require(
            run_root.is_relative_to(temporary_root)
            and output_root.is_relative_to(temporary_root)
            and ood_cache_root.is_relative_to(temporary_root),
            "synthetic mode is restricted to /tmp and cannot relax formal artifacts",
        )
    require(
        not (output_root / "checks/final_checks.json").exists(),
        "a completed OOD evaluation already exists; use a new versioned output root",
    )
    # CRITICAL: this gate completes before any output is published or any
    # file below ood_cache_root is read.
    artifacts = validate_frozen_selections(run_root, args.synthetic_test_mode)
    output_root.mkdir(parents=True, exist_ok=True)
    gate_path = output_root / "checks/frozen_selection_gate.json"
    write_json_atomic(gate_path, artifacts["gate"])

    split_manifest_path = resolve_split_manifest(
        run_root, artifacts["contract"], args.split_manifest
    )
    cache = load_ood_cache(
        ood_cache_root,
        artifacts["training_genes"],
        split_manifest_path,
        args.synthetic_test_mode,
    )
    artifacts["gate"]["ood_cache_opened"] = True
    artifacts["gate"]["ood_cache_manifest"] = file_record(cache["manifest_path"])
    artifacts["gate"]["ood_cache_condition_count"] = len(cache["metadata"])
    write_json_atomic(gate_path, artifacts["gate"])

    condition_features = encode_conditions(cache["metadata"], artifacts["encoder"])
    condition_order_path = output_root / "condition_order.csv"
    condition_order = cache["metadata"][
        ["condition_id", "drug", "dose", "cell_line", "search_split"]
    ].copy()
    condition_order.insert(0, "condition_row_zero_based", np.arange(len(condition_order)))
    write_csv_atomic(condition_order_path, condition_order)
    genes_output_path = output_root / "genes.csv"
    write_csv_atomic(
        genes_output_path,
        pd.DataFrame(
            {
                "gene_index_zero_based": np.arange(len(cache["genes"])),
                "gene": cache["genes"],
            }
        ),
    )

    array_root = output_root / "arrays"
    array_paths = materialize_reference_arrays(
        cache, array_root, max(args.mlp_batch_size, args.rf_batch_size)
    )
    array_paths.update(
        {
            "mlp_predicted_effect": array_root / "mlp_predicted_effect.npy",
            "mlp_predicted_expression": array_root
            / "mlp_predicted_expression.npy",
            "rf_predicted_effect": array_root / "rf_predicted_effect.npy",
            "rf_predicted_expression": array_root / "rf_predicted_expression.npy",
        }
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    predict_mlp(
        artifacts,
        condition_features,
        cache["controls"],
        cache["control_indices"],
        {
            "effect": array_paths["mlp_predicted_effect"],
            "expression": array_paths["mlp_predicted_expression"],
        },
        device,
        args.mlp_batch_size,
    )
    predict_rf(
        artifacts,
        condition_features,
        cache["controls"],
        cache["control_indices"],
        {
            "effect": array_paths["rf_predicted_effect"],
            "expression": array_paths["rf_predicted_expression"],
        },
        args.rf_batch_size,
    )

    metrics_root = output_root / "condition_metrics"
    mlp_metrics = condition_metrics(
        cache["metadata"],
        array_paths["observed_expression"],
        array_paths["observed_effect"],
        array_paths["mlp_predicted_expression"],
        array_paths["mlp_predicted_effect"],
        "MLP",
        args.metric_batch_size,
    )
    rf_metrics = condition_metrics(
        cache["metadata"],
        array_paths["observed_expression"],
        array_paths["observed_effect"],
        array_paths["rf_predicted_expression"],
        array_paths["rf_predicted_effect"],
        "RF",
        args.metric_batch_size,
    )
    mlp_metrics_path = metrics_root / "mlp_ood_condition_metrics.csv"
    rf_metrics_path = metrics_root / "rf_ood_condition_metrics.csv"
    combined_path = metrics_root / "mlp_rf_ood_condition_metrics.csv"
    write_csv_atomic(mlp_metrics_path, mlp_metrics)
    write_csv_atomic(rf_metrics_path, rf_metrics)
    combined = pd.concat([mlp_metrics, rf_metrics], ignore_index=True)
    require(
        combined.groupby("method")["condition_id"].nunique().to_dict()
        == {"MLP": len(cache["metadata"]), "RF": len(cache["metadata"])},
        "combined metrics do not contain one row per OOD condition and method",
    )
    write_csv_atomic(combined_path, combined)

    summary_path = output_root / "summary/metric_summary.json"
    write_json_atomic(
        summary_path,
        {
            "status": "OOD_METRIC_SUMMARY_OK",
            "unit_of_analysis": "drug-dose-cell-line condition",
            "absolute_expression_claim": (
                "predicted treated-state expression compared with measured "
                "treated-condition mean across genes within each condition"
            ),
            "effect_claim": (
                "predicted matched-control-relative effect compared with measured "
                "treated-minus-the-same-matched-control effect across genes"
            ),
            "r2_axis": "per condition across genes",
            "condition_count_per_method": len(cache["metadata"]),
            "gene_count": len(cache["genes"]),
            "MLP": metric_summary(mlp_metrics),
            "RF": metric_summary(rf_metrics),
        },
    )
    audit = audit_metrics(combined, cache["metadata"], array_paths)
    audit_path = output_root / "checks/deterministic_metric_recomputation.csv"
    write_csv_atomic(audit_path, audit)

    output_records = {
        "condition_order": file_record(condition_order_path),
        "genes": file_record(genes_output_path),
        "mlp_condition_metrics": file_record(mlp_metrics_path),
        "rf_condition_metrics": file_record(rf_metrics_path),
        "combined_condition_metrics": file_record(combined_path),
        "metric_summary": file_record(summary_path),
        "metric_recomputation_audit": file_record(audit_path),
        "arrays": {
            key: file_record(path) for key, path in sorted(array_paths.items())
        },
    }
    final_checks = {
        "status": "MLP_RF_OOD_EVALUATION_OK",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "formal_mode": not args.synthetic_test_mode,
        "unit_of_analysis": "drug-dose-cell-line condition",
        "condition_count": len(cache["metadata"]),
        "gene_count": len(cache["genes"]),
        "method_count": 2,
        "rows_per_method": {
            method: int(count)
            for method, count in combined["method"].value_counts().sort_index().items()
        },
        "condition_ids_unique": bool(cache["metadata"]["condition_id"].is_unique),
        "condition_order_sha256": cache["condition_order_sha256"],
        "condition_set_sha256": cache["condition_set_sha256"],
        "split_manifest_condition_set_equal": True,
        "gene_order_identical_to_training": True,
        "matched_control_alignment": True,
        "prediction_target": (
            "matched-control-relative delta expression, with absolute expression "
            "recovered by addition of the same matched control"
        ),
        "deterministic_metric_rows_recomputed": len(audit),
        "maximum_metric_audit_difference": float(
            audit["maximum_absolute_metric_difference"].max()
        ),
        "selection_gate": file_record(gate_path),
        "inputs": {
            "ood_cache_manifest": file_record(cache["manifest_path"]),
            "ood_metadata": file_record(cache["metadata_path"]),
            "ood_response_means": file_record(cache["response_path"]),
            "ood_response_counts": file_record(cache["response_counts_path"]),
            "control_means": file_record(cache["control_path"]),
            "control_metadata": file_record(cache["control_metadata_path"]),
            "control_counts": file_record(cache["control_counts_path"]),
            "ood_genes": file_record(cache["genes_path"]),
            "split_manifest": file_record(cache["split_manifest_path"]),
            "mlp_checkpoint": file_record(artifacts["mlp_checkpoint_path"]),
            "rf_model": file_record(artifacts["rf_model_path"]),
        },
        "outputs": output_records,
        "software": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "joblib": joblib.__version__,
            "device": str(device),
        },
        "resource": {
            "peak_process_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
            "logical_cpu_count": os.cpu_count(),
        },
    }
    final_path = output_root / "checks/final_checks.json"
    write_json_atomic(final_path, final_checks)
    print(json.dumps(final_checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
