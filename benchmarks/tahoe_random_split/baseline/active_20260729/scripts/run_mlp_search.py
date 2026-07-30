#!/usr/bin/env python3
"""Two-stage validation-based MLP baseline search without OOD/test access.

Six fixed learning-rate/dropout configurations are tuned on a shared
4,800/1,200 development split drawn only from the 29,277 formal training
conditions. The selected hyperparameters are then refitted once on all 29,277
training conditions; the unchanged 3,252 formal validation conditions select
the final checkpoint by strict minimum matched-control-relative delta MSE.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import random
import resource
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from baseline_common import (
    RUN,
    build_delta_cache,
    build_mlp_design,
    fit_train_only_condition_encoder,
    load_training_cache,
    regression_metrics,
    require,
    sha256,
    write_json_atomic,
)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        require(len(hidden) == 2, "MLP must have exactly two hidden layers")
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predict(model: nn.Module, values: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    output = np.empty((len(values), model.network[-1].out_features), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            stop = min(len(values), start + batch_size)
            batch = torch.from_numpy(values[start:stop]).to(device, non_blocking=True)
            output[start:stop] = model(batch).detach().cpu().numpy().astype(np.float32)
    return output


def checkpoint_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_csv_atomic(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def freeze_json(path: Path, payload: dict) -> None:
    if path.is_file():
        require(json.loads(path.read_text()) == payload, f"frozen JSON changed: {path}")
    else:
        write_json_atomic(path, payload)


def freeze_csv(path: Path, table: pd.DataFrame) -> None:
    if path.is_file():
        prior = pd.read_csv(path, keep_default_na=False)
        current = table.reset_index(drop=True)
        require(list(prior.columns) == list(current.columns), f"frozen CSV columns changed: {path}")
        require(len(prior) == len(current), f"frozen CSV row count changed: {path}")
        for column in current.columns:
            if pd.api.types.is_numeric_dtype(current[column]):
                require(
                    np.allclose(
                        prior[column].to_numpy(dtype=np.float64),
                        current[column].to_numpy(dtype=np.float64),
                        rtol=1e-12,
                        atol=1e-15,
                    ),
                    f"frozen CSV numeric content changed in {column}: {path}",
                )
            else:
                require(
                    prior[column].astype(str).tolist()
                    == current[column].astype(str).tolist(),
                    f"frozen CSV text content changed in {column}: {path}",
                )
    else:
        write_csv_atomic(path, table)


def file_stat(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def resource_snapshot() -> dict:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "process_user_seconds": float(usage.ru_utime),
        "process_system_seconds": float(usage.ru_stime),
        "process_peak_rss_kib": int(usage.ru_maxrss),
    }


def hash_strings(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def config_name(learning_rate: float, dropout: float) -> str:
    return f"lr_{learning_rate:.0e}__dropout_{dropout}".replace("-", "m").replace(".", "p")


def safe_batch_size(row_count: int, requested: int) -> int:
    require(row_count >= 2, "MLP stage requires at least two training rows")
    batch_size = min(row_count, requested)
    while row_count % batch_size == 1 and batch_size > 2:
        batch_size -= 1
    require(batch_size >= 2, "batch size is too small for BatchNorm")
    return batch_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=RUN)
    parser.add_argument("--contract", type=Path, default=RUN / "config/search_contract.json")
    parser.add_argument(
        "--development-split",
        type=Path,
        default=RUN / "config/shared_development_split.csv",
    )
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def resolve_development_split(
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    development_path: Path,
    seed: int,
    smoke_test: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    condition_ids = metadata["condition_id"].astype(str)
    train_ids = condition_ids.iloc[train_indices]
    require(train_ids.is_unique, "formal training condition IDs are duplicated")
    if smoke_test:
        require(len(train_indices) >= 5, "smoke cache needs at least five train rows")
        shuffled = np.random.default_rng(seed).permutation(train_indices)
        n_valid = max(1, int(round(0.2 * len(train_indices))))
        tuning_valid = np.sort(shuffled[:n_valid])
        tuning_train = np.sort(shuffled[n_valid:])
        resolved = pd.DataFrame(
            {
                "condition_id": np.concatenate(
                    [condition_ids.iloc[tuning_train], condition_ids.iloc[tuning_valid]]
                ),
                "development_split": ["tuning_train"] * len(tuning_train)
                + ["tuning_validation"] * len(tuning_valid),
            }
        )
        source = {
            "mode": "deterministic smoke-only 80/20 split of cache train rows",
            "seed": seed,
            "requested_path_not_read": str(development_path.resolve()),
        }
    else:
        development_path = development_path.resolve()
        require(development_path.is_file(), f"shared development split missing: {development_path}")
        resolved = pd.read_csv(development_path, keep_default_na=False)
        require(
            list(resolved.columns) == ["condition_id", "development_split"],
            "development split must have exactly condition_id and development_split columns",
        )
        resolved = resolved.astype(str)
        require(resolved["condition_id"].is_unique, "development condition IDs are duplicated")
        require(
            set(resolved["development_split"]) == {"tuning_train", "tuning_validation"},
            "development labels must be tuning_train and tuning_validation",
        )
        counts = resolved["development_split"].value_counts()
        require(
            int(counts["tuning_train"]) == 4_800 and int(counts["tuning_validation"]) == 1_200,
            "formal development split must contain 4800/1200 rows",
        )
        lookup = {value: int(index) for index, value in zip(train_indices, train_ids)}
        missing = set(resolved["condition_id"]).difference(lookup)
        require(not missing, f"development split has {len(missing)} IDs outside formal train")
        tuning_train = np.asarray(
            [lookup[value] for value in resolved.loc[resolved["development_split"].eq("tuning_train"), "condition_id"]],
            dtype=np.int64,
        )
        tuning_valid = np.asarray(
            [lookup[value] for value in resolved.loc[resolved["development_split"].eq("tuning_validation"), "condition_id"]],
            dtype=np.int64,
        )
        source = {"mode": "authoritative shared development split", **file_stat(development_path), "sha256": sha256(development_path)}
    require(set(tuning_train).isdisjoint(tuning_valid), "tuning train/validation overlap")
    require(
        set(tuning_train).union(tuning_valid).issubset(set(train_indices)),
        "development split is not contained in formal train",
    )
    source.update(
        {
            "tuning_train_conditions": int(len(tuning_train)),
            "tuning_validation_conditions": int(len(tuning_valid)),
            "tuning_train_ids_sha256": hash_strings(condition_ids.iloc[tuning_train]),
            "tuning_validation_ids_sha256": hash_strings(condition_ids.iloc[tuning_valid]),
        }
    )
    return tuning_train, tuning_valid, resolved, source


def run_training_stage(
    *,
    stage_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    valid_controls: np.ndarray,
    train_condition_ids: list[str],
    valid_condition_ids: list[str],
    learning_rate: float,
    dropout: float,
    hidden: list[int],
    weight_decay: float,
    requested_batch_size: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    seed: int,
    device: torch.device,
    result_dir: Path,
    model_dir: Path,
    common_contract: dict,
    smoke_test: bool,
) -> dict:
    """Train/resume one stage and verify its exact strict-minimum checkpoint."""
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    batch_size = safe_batch_size(len(x_train), requested_batch_size)
    stage_contract = {
        **common_contract,
        "stage_name": stage_name,
        "learning_rate": float(learning_rate),
        "dropout": float(dropout),
        "hidden_layers": hidden,
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "max_epochs": int(max_epochs),
        "early_stopping_patience": int(patience),
        "early_stopping_min_delta": float(min_delta),
        "checkpoint_rule": "strict minimum validation delta MSE",
        "patience_rule": "min_delta only resets patience; it never changes checkpoint selection",
        "training_conditions": int(len(x_train)),
        "validation_conditions": int(len(x_valid)),
        "training_ids_sha256": hash_strings(train_condition_ids),
        "validation_ids_sha256": hash_strings(valid_condition_ids),
        "training_validation_overlap": int(len(set(train_condition_ids).intersection(valid_condition_ids))),
        "smoke_test": bool(smoke_test),
    }
    require(stage_contract["training_validation_overlap"] == 0, f"row overlap: {stage_name}")
    stage_contract_path = result_dir / "stage_contract.json"
    freeze_json(stage_contract_path, stage_contract)
    stage_contract_hash = sha256(stage_contract_path)
    history_path = result_dir / "history.csv"
    best_path = model_dir / "best_validation_checkpoint.pt"
    last_path = model_dir / "last_epoch_resume_checkpoint.pt"
    metrics_path = result_dir / "best_validation_metrics.json"
    completed_path = result_dir / "result.json"

    if completed_path.is_file():
        completed = json.loads(completed_path.read_text())
        require(completed["status"] == "STAGE_OK", f"incomplete prior stage: {stage_name}")
        require(completed["stage_contract_sha256"] == stage_contract_hash, f"stage contract changed: {stage_name}")
        row = completed["selection_row"]
        require(best_path.is_file() and history_path.is_file() and metrics_path.is_file(), f"stage artifact missing: {stage_name}")
        require(row["checkpoint_sha256"] == sha256(best_path), f"checkpoint hash changed: {stage_name}")
        require(row["history_sha256"] == sha256(history_path), f"history hash changed: {stage_name}")
        require(row["validation_metrics_sha256"] == sha256(metrics_path), f"metric hash changed: {stage_name}")
        return row

    set_seed(seed)
    model = MLP(x_train.shape[1], y_train.shape[1], hidden, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss(reduction="mean")
    history: list[dict] = []
    best_mse = float("inf")
    patience_reference = float("inf")
    best_epoch = -1
    wait_count = 0
    start_epoch = 0
    elapsed_before_resume = 0.0
    if last_path.is_file():
        resume = torch.load(last_path, map_location=device)
        require(resume["stage_contract_sha256"] == stage_contract_hash, f"resume contract changed: {stage_name}")
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        history = resume["history"]
        best_mse = float(resume["best_validation_delta_mse"])
        patience_reference = float(resume["patience_reference"])
        best_epoch = int(resume["best_epoch_zero_based"])
        wait_count = int(resume["wait_count"])
        start_epoch = int(resume["epoch_zero_based"]) + 1
        elapsed_before_resume = float(resume["elapsed_seconds"])
        torch.set_rng_state(resume["torch_rng_state"].cpu())
        if torch.cuda.is_available() and resume["cuda_rng_state_all"] is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in resume["cuda_rng_state_all"]]
            )
        require(len(history) == start_epoch, f"resume history mismatch: {stage_name}")
        write_csv_atomic(history_path, pd.DataFrame(history))

    started = time.monotonic()
    for epoch in range(start_epoch, max_epochs):
        if wait_count >= patience:
            break
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
            drop_last=False,
        )
        model.train()
        squared_error_sum = 0.0
        value_count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            batch_prediction = model(batch_x)
            loss = criterion(batch_prediction, batch_y)
            loss.backward()
            optimizer.step()
            squared_error_sum += float(loss.detach().cpu()) * batch_y.numel()
            value_count += int(batch_y.numel())
        train_mse = squared_error_sum / value_count
        validation_prediction = predict(model, x_valid, device, batch_size)
        validation_metrics = regression_metrics(y_valid, validation_prediction, valid_controls)
        validation_mse = float(validation_metrics["delta_mse"])
        require(np.isfinite(validation_mse), f"non-finite validation MSE: {stage_name}")
        elapsed = elapsed_before_resume + time.monotonic() - started
        history.append(
            {
                "epoch_zero_based": int(epoch),
                "epoch_one_based": int(epoch + 1),
                "training_delta_mse": float(train_mse),
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
                "elapsed_seconds": float(elapsed),
            }
        )
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_epoch = epoch
            checkpoint_atomic(
                best_path,
                {
                    "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "input_dim": int(x_train.shape[1]),
                    "output_dim": int(y_train.shape[1]),
                    "hidden_layers": hidden,
                    "dropout": float(dropout),
                    "learning_rate": float(learning_rate),
                    "weight_decay": float(weight_decay),
                    "seed": int(seed),
                    "stage_name": stage_name,
                    "stage_contract_sha256": stage_contract_hash,
                    "best_epoch_zero_based": int(best_epoch),
                    "best_validation_delta_mse": float(best_mse),
                },
            )
        if validation_mse < patience_reference - min_delta:
            patience_reference = validation_mse
            wait_count = 0
        else:
            wait_count += 1
        checkpoint_atomic(
            last_path,
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch_zero_based": int(epoch),
                "history": history,
                "best_validation_delta_mse": float(best_mse),
                "patience_reference": float(patience_reference),
                "best_epoch_zero_based": int(best_epoch),
                "wait_count": int(wait_count),
                "elapsed_seconds": float(elapsed),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "stage_contract_sha256": stage_contract_hash,
            },
        )
        write_csv_atomic(history_path, pd.DataFrame(history))
        del loader, validation_prediction

    require(best_path.is_file() and best_epoch >= 0, f"best checkpoint missing: {stage_name}")
    checkpoint = torch.load(best_path, map_location=device)
    require(checkpoint["stage_contract_sha256"] == stage_contract_hash, f"best checkpoint contract changed: {stage_name}")
    model.load_state_dict(checkpoint["model_state_dict"])
    exact_prediction = predict(model, x_valid, device, batch_size)
    exact_metrics = regression_metrics(y_valid, exact_prediction, valid_controls)
    require(
        np.isclose(exact_metrics["delta_mse"], best_mse, rtol=1e-7, atol=1e-10),
        f"reloaded MSE differs from strict minimum: {stage_name}",
    )
    history_table = pd.DataFrame(history)
    require(best_epoch == int(history_table["validation_delta_mse"].astype(float).idxmin()), f"best epoch is not argmin: {stage_name}")
    write_json_atomic(
        metrics_path,
        {
            "status": "RELOADED_CHECKPOINT_OK",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "best_epoch_zero_based": int(best_epoch),
            "metrics": exact_metrics,
            "checkpoint": str(best_path),
            "checkpoint_sha256": sha256(best_path),
            "test_response_accessed": False,
            "smoke_test": bool(smoke_test),
        },
    )
    row = {
        "stage": stage_name,
        "learning_rate": float(learning_rate),
        "dropout": float(dropout),
        "best_epoch_zero_based": int(best_epoch),
        "completed_epochs": int(len(history)),
        "stopped_early": bool(len(history) < max_epochs),
        "validation_delta_mse": float(exact_metrics["delta_mse"]),
        "validation_median_condition_delta_pearson": float(exact_metrics["median_condition_delta_pearson"]),
        "validation_mean_condition_absolute_r2": float(exact_metrics["mean_condition_absolute_r2"]),
        "validation_mean_condition_absolute_pearson": float(exact_metrics["mean_condition_absolute_pearson"]),
        "checkpoint": str(best_path),
        "checkpoint_sha256": sha256(best_path),
        "history": str(history_path),
        "history_sha256": sha256(history_path),
        "validation_metrics": str(metrics_path),
        "validation_metrics_sha256": sha256(metrics_path),
        "stage_contract_sha256": stage_contract_hash,
    }
    write_json_atomic(
        completed_path,
        {
            "status": "STAGE_OK",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "smoke_test": bool(smoke_test),
            "stage_contract_sha256": stage_contract_hash,
            "selection_row": row,
            "resource_final": resource_snapshot(),
        },
    )
    del model, optimizer, exact_prediction
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def main() -> None:
    args = parse_args()
    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text())
    mlp_contract = contract["mlp"]
    seed = int(contract["seed"])
    cache = load_training_cache(args.cache_root)
    require(
        set(cache.metadata["search_split"].astype(str).unique()) == {"train", "valid"},
        "MLP cache contains a split other than internal train/validation",
    )
    train_indices = np.flatnonzero(cache.train_mask)
    formal_valid_indices = np.flatnonzero(cache.valid_mask)
    require(set(train_indices).isdisjoint(formal_valid_indices), "formal train/validation overlap")
    if not args.smoke_test:
        require(
            len(train_indices) == int(contract["split"]["internal_train_conditions"]) == 29_277,
            "formal train count differs from 29277",
        )
        require(
            len(formal_valid_indices) == int(contract["split"]["validation_conditions"]) == 3_252,
            "formal validation count differs from 3252",
        )
    tuning_train_indices, tuning_valid_indices, development_split, split_source = resolve_development_split(
        cache.metadata, train_indices, args.development_split, seed, args.smoke_test
    )
    if not args.smoke_test:
        require(
            len(tuning_train_indices) == 4_800 and len(tuning_valid_indices) == 1_200,
            "formal tuning split differs from 4800/1200",
        )

    delta_path = build_delta_cache(cache)
    condition_features, encoder_contract = fit_train_only_condition_encoder(cache)
    design_path = build_mlp_design(cache, condition_features)
    delta = np.load(delta_path, mmap_mode="r")
    design = np.load(design_path, mmap_mode="r")
    require(delta.shape == cache.responses.shape, "delta target shape mismatch")
    require(design.shape[0] == len(cache.metadata), "MLP design row count mismatch")
    require(design.shape[1] == condition_features.shape[1] + len(cache.genes), "MLP design feature count mismatch")

    output_root = args.output_root.resolve()
    result_root = output_root / "results/mlp"
    model_root = output_root / "models/mlp"
    provenance_root = output_root / "provenance/mlp"
    check_root = output_root / "checks/mlp"
    for path in (result_root, model_root, provenance_root, check_root):
        path.mkdir(parents=True, exist_ok=True)
    development_copy = provenance_root / "shared_development_split_resolved.csv"
    freeze_csv(development_copy, development_split)
    encoder_path = provenance_root / "condition_encoder.json"
    freeze_json(encoder_path, encoder_contract)
    input_hashes = {
        "contract": {**file_stat(contract_path), "sha256": sha256(contract_path)},
        "cache_manifest": {**file_stat(cache.root / "cache_manifest.json"), "sha256": sha256(cache.root / "cache_manifest.json")},
        "training_metadata": {**file_stat(cache.root / "train/metadata.csv"), "sha256": sha256(cache.root / "train/metadata.csv")},
        "control_metadata": {**file_stat(cache.root / "control/metadata.csv"), "sha256": sha256(cache.root / "control/metadata.csv")},
        "genes": {**file_stat(cache.root / "genes.csv"), "sha256": sha256(cache.root / "genes.csv")},
        "development_split_resolved": {**file_stat(development_copy), "sha256": sha256(development_copy)},
        "development_split_source": split_source,
        "condition_encoder": {**file_stat(encoder_path), "sha256": sha256(encoder_path)},
        "delta_cache": file_stat(delta_path),
        "mlp_design": file_stat(design_path),
        "script": {**file_stat(Path(__file__)), "sha256": sha256(Path(__file__))},
    }
    freeze_json(provenance_root / "input_hashes_and_stats.json", input_hashes)
    freeze_json(
        provenance_root / "software_environment.json",
        {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "requested_device": args.device,
            "logical_cpu_count": os.cpu_count(),
            "random_seed": seed,
        },
    )
    condition_ids = cache.metadata["condition_id"].astype(str)
    alignment = {
        "status": "ALIGNMENT_OK",
        "unit_of_analysis": contract["unit_of_analysis"],
        "target": contract["target"],
        "hyperparameter_selection_metric": "tuning-validation matched-control-relative delta-expression MSE",
        "final_checkpoint_selection_metric": "formal-validation matched-control-relative delta-expression MSE",
        "formal_train_conditions": int(len(train_indices)),
        "formal_validation_conditions": int(len(formal_valid_indices)),
        "tuning_train_conditions": int(len(tuning_train_indices)),
        "tuning_validation_conditions": int(len(tuning_valid_indices)),
        "tuning_is_subset_of_formal_train": True,
        "tuning_train_validation_overlap": 0,
        "formal_train_validation_overlap": 0,
        "condition_ids_unique": bool(cache.metadata["condition_id"].is_unique),
        "gene_count": int(len(cache.genes)),
        "gene_names_unique": bool(len(set(cache.genes)) == len(cache.genes)),
        "matched_control_rows_aligned": True,
        "condition_encoder_fit_scope": encoder_contract["fit_scope"],
        "validation_unknown_drugs": encoder_contract["validation_unknown_drugs"],
        "validation_unknown_cell_lines": encoder_contract["validation_unknown_cell_lines"],
        "formal_train_ids_sha256": hash_strings(condition_ids.iloc[train_indices]),
        "formal_validation_ids_sha256": hash_strings(condition_ids.iloc[formal_valid_indices]),
        "test_response_accessed": False,
        "smoke_test": bool(args.smoke_test),
    }
    freeze_json(check_root / "alignment_checks.json", alignment)

    hidden = [int(value) for value in mlp_contract["hidden_layers"]]
    learning_rates = [float(value) for value in mlp_contract["learning_rates"]]
    dropout_rates = [float(value) for value in mlp_contract["dropout_rates"]]
    require(hidden == [512, 256], "MLP architecture differs from fixed contract")
    require(
        learning_rates == [0.0001, 0.0005, 0.001] and dropout_rates == [0.0, 0.2],
        "MLP six-configuration grid differs from fixed contract",
    )
    max_epochs = 3 if args.smoke_test else int(mlp_contract["max_epochs"])
    patience = 2 if args.smoke_test else int(mlp_contract["early_stopping_patience"])
    require(args.smoke_test or patience == 10, "formal MLP patience must be 10")
    batch_size = int(mlp_contract["batch_size"])
    min_delta = float(mlp_contract["early_stopping_min_delta"])
    weight_decay = float(mlp_contract["weight_decay"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    common_contract = {
        "contract_sha256": sha256(contract_path),
        "cache_manifest_sha256": sha256(cache.root / "cache_manifest.json"),
        "training_metadata_sha256": sha256(cache.root / "train/metadata.csv"),
        "genes_sha256": sha256(cache.root / "genes.csv"),
        "development_split_resolved_sha256": sha256(development_copy),
        "condition_encoder_sha256": sha256(encoder_path),
        "script_sha256": sha256(Path(__file__)),
        "input_dim": int(design.shape[1]),
        "output_dim": int(delta.shape[1]),
        "target": contract["target"],
        "test_response_accessed": False,
    }

    # Grid search materializes only the shared development rows.
    x_tuning_train = np.asarray(design[tuning_train_indices], dtype=np.float32)
    y_tuning_train = np.asarray(delta[tuning_train_indices], dtype=np.float32)
    x_tuning_valid = np.asarray(design[tuning_valid_indices], dtype=np.float32)
    y_tuning_valid = np.asarray(delta[tuning_valid_indices], dtype=np.float32)
    tuning_valid_controls = np.asarray(cache.controls[cache.control_index[tuning_valid_indices]], dtype=np.float32)
    require(x_tuning_train.shape[0] == y_tuning_train.shape[0] == len(tuning_train_indices), "tuning train alignment failed")
    require(x_tuning_valid.shape[0] == y_tuning_valid.shape[0] == len(tuning_valid_indices), "tuning validation alignment failed")
    require(y_tuning_train.shape[1] == y_tuning_valid.shape[1] == len(cache.genes), "tuning gene alignment failed")
    require(
        np.isfinite(x_tuning_train).all() and np.isfinite(y_tuning_train).all()
        and np.isfinite(x_tuning_valid).all() and np.isfinite(y_tuning_valid).all()
        and np.isfinite(tuning_valid_controls).all(),
        "tuning arrays contain non-finite values",
    )
    search_rows = []
    for learning_rate, dropout in itertools.product(learning_rates, dropout_rates):
        name = config_name(learning_rate, dropout)
        row = run_training_stage(
            stage_name=f"tuning__{name}",
            x_train=x_tuning_train,
            y_train=y_tuning_train,
            x_valid=x_tuning_valid,
            y_valid=y_tuning_valid,
            valid_controls=tuning_valid_controls,
            train_condition_ids=condition_ids.iloc[tuning_train_indices].tolist(),
            valid_condition_ids=condition_ids.iloc[tuning_valid_indices].tolist(),
            learning_rate=learning_rate,
            dropout=dropout,
            hidden=hidden,
            weight_decay=weight_decay,
            requested_batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            seed=seed,
            device=device,
            result_dir=result_root / "tuning" / name,
            model_dir=model_root / "tuning" / name,
            common_contract=common_contract,
            smoke_test=args.smoke_test,
        )
        search_rows.append({"config": name, **row})
    search_columns = [
        "config",
        "learning_rate",
        "dropout",
        "best_epoch_zero_based",
        "completed_epochs",
        "stopped_early",
        "validation_delta_mse",
        "validation_median_condition_delta_pearson",
        "validation_mean_condition_absolute_r2",
        "validation_mean_condition_absolute_pearson",
        "checkpoint",
        "checkpoint_sha256",
        "history",
        "history_sha256",
        "validation_metrics",
        "validation_metrics_sha256",
        "stage",
        "stage_contract_sha256",
    ]
    search = (
        pd.DataFrame(search_rows)[search_columns]
        .sort_values(["validation_delta_mse", "config"], kind="stable")
        .reset_index(drop=True)
    )
    require(len(search) == 6 and search["config"].is_unique, "expected six unique MLP configurations")
    search_path = result_root / "tuning_search_summary.csv"
    freeze_csv(search_path, search)
    best = search.iloc[0].to_dict()
    require(best["config"] == search.loc[search["validation_delta_mse"].idxmin(), "config"], "tuning choice is not strict argmin")
    frozen = {
        "status": "HYPERPARAMETERS_FROZEN",
        "selection_stage": "six-configuration 4800/1200 development search" if not args.smoke_test else "six-configuration deterministic smoke 80/20 development search",
        "selection_metric": "tuning-validation delta-expression MSE",
        "selection_mode": "strict minimum",
        "selected_config": best["config"],
        "learning_rate": float(best["learning_rate"]),
        "dropout": float(best["dropout"]),
        "tuning_best_epoch_zero_based": int(best["best_epoch_zero_based"]),
        "tuning_validation_delta_mse": float(best["validation_delta_mse"]),
        "tuning_checkpoint": best["checkpoint"],
        "tuning_checkpoint_sha256": best["checkpoint_sha256"],
        "configuration_count": int(len(search)),
        "search_summary": str(search_path),
        "search_summary_sha256": sha256(search_path),
        "test_response_accessed": False,
        "smoke_test": bool(args.smoke_test),
    }
    frozen_path = result_root / "frozen_hyperparameters.json"
    freeze_json(frozen_path, frozen)
    del x_tuning_train, y_tuning_train, x_tuning_valid, y_tuning_valid, tuning_valid_controls
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # One final refit uses all formal train rows and the unchanged formal validation.
    x_final_train = np.asarray(design[train_indices], dtype=np.float32)
    y_final_train = np.asarray(delta[train_indices], dtype=np.float32)
    x_formal_valid = np.asarray(design[formal_valid_indices], dtype=np.float32)
    y_formal_valid = np.asarray(delta[formal_valid_indices], dtype=np.float32)
    formal_valid_controls = np.asarray(cache.controls[cache.control_index[formal_valid_indices]], dtype=np.float32)
    require(x_final_train.shape[0] == y_final_train.shape[0] == len(train_indices), "final train alignment failed")
    require(x_formal_valid.shape[0] == y_formal_valid.shape[0] == len(formal_valid_indices), "formal validation alignment failed")
    require(
        np.isfinite(x_final_train).all() and np.isfinite(y_final_train).all()
        and np.isfinite(x_formal_valid).all() and np.isfinite(y_formal_valid).all()
        and np.isfinite(formal_valid_controls).all(),
        "final arrays contain non-finite values",
    )
    final_row = run_training_stage(
        stage_name="final_refit_all_formal_train",
        x_train=x_final_train,
        y_train=y_final_train,
        x_valid=x_formal_valid,
        y_valid=y_formal_valid,
        valid_controls=formal_valid_controls,
        train_condition_ids=condition_ids.iloc[train_indices].tolist(),
        valid_condition_ids=condition_ids.iloc[formal_valid_indices].tolist(),
        learning_rate=float(frozen["learning_rate"]),
        dropout=float(frozen["dropout"]),
        hidden=hidden,
        weight_decay=weight_decay,
        requested_batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        seed=seed,
        device=device,
        result_dir=result_root / "final",
        model_dir=model_root / "final",
        common_contract={
            **common_contract,
            "frozen_hyperparameters_sha256": sha256(frozen_path),
            "final_fit_scope": "all formal internal-training conditions",
            "final_validation_scope": "unchanged formal validation conditions",
        },
        smoke_test=args.smoke_test,
    )
    final_source = Path(final_row["checkpoint"])
    require(final_source.is_file(), "verified final-refit checkpoint is missing")
    selected_checkpoint = model_root / "selected_best_validation_checkpoint.pt"
    copy_atomic(final_source, selected_checkpoint)
    selected_hash = sha256(selected_checkpoint)
    require(selected_hash == final_row["checkpoint_sha256"], "selected checkpoint differs from verified final refit")

    # selection.json is deliberately impossible to publish before the final
    # model has been fitted, reloaded and re-evaluated above.
    final_result_path = result_root / "final/result.json"
    require(final_result_path.is_file(), "final result missing before selection publication")
    selection = {
        "status": "SELECTION_FROZEN",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "two-stage development search followed by one full refit",
        "hyperparameter_selection_metric": "tuning-validation delta-expression MSE",
        "final_checkpoint_selection_metric": "formal-validation delta-expression MSE",
        "selection_mode": "strict minimum at both stages",
        "selected": {
            "config": frozen["selected_config"],
            "learning_rate": frozen["learning_rate"],
            "dropout": frozen["dropout"],
            "tuning_validation_delta_mse": frozen["tuning_validation_delta_mse"],
            "final_best_epoch_zero_based": final_row["best_epoch_zero_based"],
            "final_validation_delta_mse": final_row["validation_delta_mse"],
            "final_validation_median_condition_delta_pearson": final_row["validation_median_condition_delta_pearson"],
            "final_validation_mean_condition_absolute_r2": final_row["validation_mean_condition_absolute_r2"],
            "final_validation_mean_condition_absolute_pearson": final_row["validation_mean_condition_absolute_pearson"],
        },
        "selected_checkpoint": str(selected_checkpoint),
        "selected_checkpoint_sha256": selected_hash,
        "frozen_hyperparameters": str(frozen_path),
        "frozen_hyperparameters_sha256": sha256(frozen_path),
        "tuning_search_summary": str(search_path),
        "tuning_search_summary_sha256": sha256(search_path),
        "final_result": str(final_result_path),
        "final_result_sha256": sha256(final_result_path),
        "tuning_train_conditions": int(len(tuning_train_indices)),
        "tuning_validation_conditions": int(len(tuning_valid_indices)),
        "train_conditions": int(len(train_indices)),
        "validation_conditions": int(len(formal_valid_indices)),
        "train_validation_overlap": 0,
        "test_response_accessed": False,
        "smoke_test": bool(args.smoke_test),
        "contract": str(contract_path),
        "contract_sha256": sha256(contract_path),
        "cache_manifest_sha256": sha256(cache.root / "cache_manifest.json"),
        "development_split_resolved_sha256": sha256(development_copy),
        "condition_encoder_sha256": sha256(encoder_path),
        "resource_final": resource_snapshot(),
    }
    selection_path = result_root / "selection.json"
    if selection_path.is_file():
        previous = json.loads(selection_path.read_text())
        require(previous["status"] == "SELECTION_FROZEN", "prior MLP selection is not frozen")
        require(previous["smoke_test"] is bool(args.smoke_test), "prior MLP selection mode differs")
        require(
            previous["selected"]["config"] == selection["selected"]["config"]
            and previous["selected_checkpoint_sha256"] == selected_hash
            and previous["tuning_search_summary_sha256"] == selection["tuning_search_summary_sha256"],
            "recomputed MLP selection differs from frozen selection",
        )
        selection = previous
    else:
        write_json_atomic(selection_path, selection)
    checks = {
        "status": "MLP_TWO_STAGE_SEARCH_OK",
        "configuration_count": int(len(search)),
        "configuration_names": sorted(search["config"].tolist()),
        "selected_config": selection["selected"]["config"],
        "tuning_selection_is_argmin": bool(selection["selected"]["config"] == search.iloc[0]["config"]),
        "final_checkpoint_present": bool(selected_checkpoint.is_file()),
        "final_checkpoint_matches_verified_refit": bool(sha256(selected_checkpoint) == final_row["checkpoint_sha256"]),
        "tuning_is_subset_of_formal_train": True,
        "formal_train_validation_overlap": 0,
        "test_response_accessed": False,
        "selection_written_after_final_result": bool(final_result_path.is_file()),
        "smoke_test": bool(args.smoke_test),
    }
    require(checks["tuning_selection_is_argmin"], "MLP tuning selection is not argmin")
    require(checks["final_checkpoint_present"] and checks["final_checkpoint_matches_verified_refit"], "final checkpoint verification failed")
    require(checks["selection_written_after_final_result"], "selection preceded final model")
    write_json_atomic(check_root / "final_checks.json", checks)
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
