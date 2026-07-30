#!/usr/bin/env python3
"""Isolated full-train MLP dropout=0.3 extension without OOD/test access.

The frozen formal dropout=0 result and completed 0/0.1/0.2 sensitivity table are read only. Dropout 0.3 is
trained on the same 29,277 conditions and checked every epoch on the same
3,252-condition validation set. The imported formal trainer provides
restartable checkpoints and strict-minimum validation-MSE checkpoint reload.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline_common import RUN, load_training_cache, require, sha256, write_json_atomic
from run_mlp_search import (
    hash_strings,
    run_training_stage,
    write_csv_atomic,
)


LEARNING_RATE = 1e-4
DROPOUTS = (0.3,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RUN / "sensitivity/mlp_full_train_dropout_0p3_extension_20260727",
    )
    parser.add_argument(
        "--contract", type=Path, default=RUN / "config/search_contract.json"
    )
    parser.add_argument("--formal-root", type=Path, default=RUN)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def row_for_comparison(label: str, row: dict, validation_hash: str) -> dict:
    return {
        "candidate": label,
        "learning_rate": float(row["learning_rate"]),
        "dropout": float(row["dropout"]),
        "training_conditions": int(29_277),
        "validation_conditions": int(3_252),
        "validation_ids_sha256": validation_hash,
        "best_epoch_zero_based": int(row["best_epoch_zero_based"]),
        "completed_epochs": int(row["completed_epochs"]),
        "stopped_early": bool(row["stopped_early"]),
        "validation_delta_mse": float(row["validation_delta_mse"]),
        "validation_median_condition_delta_pearson": float(
            row["validation_median_condition_delta_pearson"]
        ),
        "validation_mean_condition_absolute_r2": float(
            row["validation_mean_condition_absolute_r2"]
        ),
        "validation_mean_condition_absolute_pearson": float(
            row["validation_mean_condition_absolute_pearson"]
        ),
        "checkpoint": str(row["checkpoint"]),
        "checkpoint_sha256": str(row["checkpoint_sha256"]),
        "history": str(row["history"]),
        "history_sha256": str(row["history_sha256"]),
        "validation_metrics": str(row["validation_metrics"]),
        "validation_metrics_sha256": str(row["validation_metrics_sha256"]),
        "stage_contract_sha256": str(row["stage_contract_sha256"]),
    }


def load_formal_dropout_zero(
    formal_root: Path,
    contract_hash: str,
    train_hash: str,
    validation_hash: str,
) -> tuple[dict, dict, str, str]:
    selection_path = formal_root / "results/mlp/selection.json"
    result_path = formal_root / "results/mlp/final/result.json"
    stage_path = formal_root / "results/mlp/final/stage_contract.json"
    selection = json.loads(selection_path.read_text())
    result = json.loads(result_path.read_text())
    stage = json.loads(stage_path.read_text())
    require(selection["status"] == "SELECTION_FROZEN", "formal MLP is not frozen")
    require(selection["test_response_accessed"] is False, "formal MLP accessed test")
    require(selection["contract_sha256"] == contract_hash, "formal contract changed")
    require(float(selection["selected"]["learning_rate"]) == LEARNING_RATE, "formal LR differs")
    require(float(selection["selected"]["dropout"]) == 0.0, "formal dropout differs")
    require(result["status"] == "STAGE_OK", "formal dropout=0 stage did not pass")
    row = result["selection_row"]
    require(float(row["learning_rate"]) == LEARNING_RATE, "formal row LR differs")
    require(float(row["dropout"]) == 0.0, "formal row dropout differs")
    require(stage["training_conditions"] == 29_277, "formal train count differs")
    require(stage["validation_conditions"] == 3_252, "formal validation count differs")
    require(stage["training_ids_sha256"] == train_hash, "formal train IDs differ")
    require(stage["validation_ids_sha256"] == validation_hash, "formal validation IDs differ")
    require(stage["checkpoint_rule"] == "strict minimum validation delta MSE", "formal checkpoint rule differs")
    history = pd.read_csv(row["history"])
    require(
        int(history["validation_delta_mse"].astype(float).idxmin())
        == int(row["best_epoch_zero_based"]),
        "formal dropout=0 checkpoint is not the strict argmin",
    )
    return row, stage, sha256(selection_path), sha256(
        formal_root / "models/mlp/selected_best_validation_checkpoint.pt"
    )


def main() -> None:
    args = parse_args()
    cache_root = args.cache_root.resolve()
    output_root = args.output_root.resolve()
    formal_root = args.formal_root.resolve()
    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text())
    mlp = contract["mlp"]
    seed = int(contract["seed"])
    require([int(x) for x in mlp["hidden_layers"]] == [512, 256], "architecture changed")
    require(int(mlp["batch_size"]) == 1024, "batch size changed")
    require(int(mlp["max_epochs"]) == 100, "max epochs changed")
    require(int(mlp["early_stopping_patience"]) == 10, "patience changed")
    require(float(mlp["early_stopping_min_delta"]) == 1e-5, "min_delta changed")
    require(float(mlp["weight_decay"]) == 0.0, "weight decay changed")

    cache = load_training_cache(cache_root)
    require(
        set(cache.metadata["search_split"].astype(str).unique()) == {"train", "valid"},
        "cache contains a split other than train/valid",
    )
    train_indices = np.flatnonzero(cache.train_mask)
    validation_indices = np.flatnonzero(cache.valid_mask)
    if args.smoke_test:
        require(len(train_indices) >= 4 and len(validation_indices) >= 2, "smoke cache too small")
    else:
        require(len(train_indices) == 29_277, "train count differs from 29277")
        require(len(validation_indices) == 3_252, "validation count differs from 3252")
    require(set(train_indices).isdisjoint(validation_indices), "train/validation overlap")
    condition_ids = cache.metadata["condition_id"].astype(str)
    train_hash = hash_strings(condition_ids.iloc[train_indices])
    validation_hash = hash_strings(condition_ids.iloc[validation_indices])

    delta_path = cache_root / "model_inputs/delta_expression.npy"
    design_path = cache_root / "model_inputs/mlp_design.npy"
    require(delta_path.is_file() and design_path.is_file(), "precomputed MLP inputs missing")
    delta = np.load(delta_path, mmap_mode="r")
    design = np.load(design_path, mmap_mode="r")
    require(delta.shape == cache.responses.shape, "delta shape differs")
    require(design.shape[0] == len(cache.metadata), "design row count differs")
    if not args.smoke_test:
        require(design.shape[1] == 14_211, "formal MLP input dimension differs")

    formal_row = None
    formal_stage = None
    formal_selection_hash = None
    formal_checkpoint_hash = None
    if not args.smoke_test:
        (
            formal_row,
            formal_stage,
            formal_selection_hash,
            formal_checkpoint_hash,
        ) = load_formal_dropout_zero(
            formal_root,
            sha256(contract_path),
            train_hash,
            validation_hash,
        )

    x_train = np.asarray(design[train_indices], dtype=np.float32)
    y_train = np.asarray(delta[train_indices], dtype=np.float32)
    x_valid = np.asarray(design[validation_indices], dtype=np.float32)
    y_valid = np.asarray(delta[validation_indices], dtype=np.float32)
    valid_controls = np.asarray(
        cache.controls[cache.control_index[validation_indices]], dtype=np.float32
    )
    require(x_train.shape[0] == y_train.shape[0] == len(train_indices), "train alignment failed")
    require(x_valid.shape[0] == y_valid.shape[0] == len(validation_indices), "validation alignment failed")
    require(y_train.shape[1] == y_valid.shape[1] == len(cache.genes), "gene alignment failed")

    output_root.mkdir(parents=True, exist_ok=True)
    max_epochs = 2 if args.smoke_test else 100
    patience = 2 if args.smoke_test else 10
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    common = {
        "analysis": "isolated full-train dropout sensitivity",
        "contract_sha256": sha256(contract_path),
        "cache_manifest_sha256": sha256(cache_root / "cache_manifest.json"),
        "script_sha256": sha256(Path(__file__).resolve()),
        "training_implementation_sha256": sha256(
            Path(run_training_stage.__code__.co_filename).resolve()
        ),
        "input_dim": int(design.shape[1]),
        "output_dim": int(delta.shape[1]),
        "target": contract["target"],
        "fit_scope": "all formal internal-training conditions",
        "validation_scope": "unchanged formal validation conditions",
        "test_response_accessed": False,
    }
    rows = []
    for dropout in DROPOUTS:
        label = f"dropout_{dropout:.1f}".replace(".", "p")
        candidate_root = output_root / label
        row = run_training_stage(
            stage_name=f"full_train_dropout_sensitivity__{label}",
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            valid_controls=valid_controls,
            train_condition_ids=condition_ids.iloc[train_indices].tolist(),
            valid_condition_ids=condition_ids.iloc[validation_indices].tolist(),
            learning_rate=LEARNING_RATE,
            dropout=dropout,
            hidden=[512, 256],
            weight_decay=0.0,
            requested_batch_size=1024,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=1e-5,
            seed=seed,
            device=device,
            result_dir=candidate_root,
            model_dir=candidate_root / "model",
            common_contract=common,
            smoke_test=args.smoke_test,
        )
        stage = json.loads((candidate_root / "stage_contract.json").read_text())
        require(stage["training_ids_sha256"] == train_hash, f"{label} train IDs differ")
        require(stage["validation_ids_sha256"] == validation_hash, f"{label} validation IDs differ")
        require(stage["checkpoint_rule"] == "strict minimum validation delta MSE", f"{label} rule differs")
        if not args.smoke_test:
            for key in (
                "batch_size",
                "early_stopping_min_delta",
                "early_stopping_patience",
                "hidden_layers",
                "input_dim",
                "learning_rate",
                "max_epochs",
                "output_dim",
                "training_conditions",
                "training_ids_sha256",
                "validation_conditions",
                "validation_ids_sha256",
                "weight_decay",
            ):
                require(stage[key] == formal_stage[key], f"{label} differs in {key}")
        rows.append(row_for_comparison(label, row, validation_hash))

    if args.smoke_test:
        table = pd.DataFrame(rows).sort_values("validation_delta_mse").reset_index(drop=True)
        write_csv_atomic(output_root / "smoke_comparison.csv", table)
        write_json_atomic(
            output_root / "smoke_checks.json",
            {
                "status": "SMOKE_OK",
                "candidate_count": 1,
                "strict_minimum_checkpoints_reloaded": True,
                "test_response_accessed": False,
            },
        )
        print(json.dumps(json.loads((output_root / "smoke_checks.json").read_text()), indent=2))
        return

    require(formal_row is not None, "formal dropout=0 result was not loaded")
    previous_path = formal_root / "sensitivity/mlp_full_train_dropout_search_20260727/full_train_dropout_comparison.csv"
    previous_summary_path = formal_root / "sensitivity/mlp_full_train_dropout_search_20260727/full_train_dropout_comparison.json"
    require(previous_path.is_file() and previous_summary_path.is_file(), "completed dropout 0/0.1/0.2 comparison is missing")
    previous_summary = json.loads(previous_summary_path.read_text())
    require(previous_summary["status"] == "FULL_TRAIN_DROPOUT_SENSITIVITY_OK", "previous sensitivity did not pass")
    require(previous_summary["test_response_accessed"] is False, "previous sensitivity accessed test")
    previous = pd.read_csv(previous_path)
    require(len(previous) == 3, "previous comparison must contain three candidates")
    require(set(previous["dropout"].astype(float)) == {0.0, 0.1, 0.2}, "previous dropout set changed")
    require(previous["validation_ids_sha256"].eq(validation_hash).all(), "previous validation IDs changed")
    require(len(rows) == 1 and float(rows[0]["dropout"]) == 0.3, "extension must contain only dropout 0.3")
    table = pd.concat([previous, pd.DataFrame(rows)], ignore_index=True).sort_values(
        ["validation_delta_mse", "dropout"], kind="stable"
    ).reset_index(drop=True)
    require(len(table) == 4, "comparison must contain dropout 0, 0.1, 0.2 and 0.3")
    minimum = float(table["validation_delta_mse"].min())
    require(int(table["validation_delta_mse"].eq(minimum).sum()) == 1, "strict argmin is tied")
    best = table.iloc[0]
    comparison_path = output_root / "full_train_dropout_0p0_0p1_0p2_0p3_comparison.csv"
    write_csv_atomic(comparison_path, table)
    require(
        sha256(formal_root / "results/mlp/selection.json") == formal_selection_hash,
        "formal selection changed",
    )
    require(
        sha256(formal_root / "models/mlp/selected_best_validation_checkpoint.pt")
        == formal_checkpoint_hash,
        "formal selected checkpoint changed",
    )
    summary = {
        "status": "FULL_TRAIN_DROPOUT_0P3_EXTENSION_OK",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "selection_metric": "3252-condition formal-validation delta-expression MSE",
        "selection_mode": "strict argmin across dropout 0, 0.1, 0.2 and 0.3",
        "strict_argmin_dropout": float(best["dropout"]),
        "strict_argmin_validation_delta_mse": float(best["validation_delta_mse"]),
        "comparison": str(comparison_path),
        "comparison_sha256": sha256(comparison_path),
        "formal_selection_modified": False,
        "test_response_accessed": False,
    }
    write_json_atomic(output_root / "full_train_dropout_0p3_extension.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
