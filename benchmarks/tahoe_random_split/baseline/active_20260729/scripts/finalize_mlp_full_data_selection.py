#!/usr/bin/env python3
"""Freeze the authoritative full-data MLP model-selection record.

This audit does not train a model or access the external OOD response data. It
combines the completed learning-rate and dropout comparisons, verifies that
both strict validation argmins identify the same existing checkpoint, and
writes a versioned selection record without overwriting the earlier
development-search record.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
LR_TABLE = (
    RUN
    / "sensitivity/mlp_full_data_learning_rate_stage_20260727"
    / "full_data_learning_rate_comparison.csv"
)
DROPOUT_TABLE = (
    RUN
    / "sensitivity/mlp_full_train_dropout_0p3_extension_20260727"
    / "full_train_dropout_0p0_0p1_0p2_0p3_comparison.csv"
)
UPSTREAM_SELECTION = RUN / "results/mlp/selection.json"
FINAL_HISTORY = RUN / "results/mlp/final/history.csv"
FINAL_CHECKPOINT = RUN / "models/mlp/final/best_validation_checkpoint.pt"
SELECTED_CHECKPOINT = RUN / "models/mlp/selected_best_validation_checkpoint.pt"
OUTPUT = RUN / "results/mlp/full_data_validated_selection_20260727.json"
AUDIT = RUN / "provenance/mlp/full_data_model_selection_audit_20260727.md"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def strict_argmin(table: pd.DataFrame, label: str) -> pd.Series:
    require(not table.empty, f"{label} table is empty")
    values = table["validation_delta_mse"].astype(float)
    require(np.isfinite(values).all(), f"{label} has non-finite validation MSE")
    minimum = float(values.min())
    require(int(values.eq(minimum).sum()) == 1, f"{label} strict argmin is tied")
    return table.loc[values.idxmin()]


def main() -> None:
    for path in (
        LR_TABLE,
        DROPOUT_TABLE,
        UPSTREAM_SELECTION,
        FINAL_HISTORY,
        FINAL_CHECKPOINT,
        SELECTED_CHECKPOINT,
    ):
        require(path.is_file(), f"required artifact missing: {path}")

    learning_rates = pd.read_csv(LR_TABLE)
    dropouts = pd.read_csv(DROPOUT_TABLE)
    require(len(learning_rates) == 3, "learning-rate table must contain three candidates")
    require(
        set(learning_rates["learning_rate"].astype(float)) == {1e-4, 5e-4, 1e-3},
        "learning-rate candidate set changed",
    )
    require(learning_rates["dropout"].astype(float).eq(0.0).all(), "LR stage changed dropout")
    require(len(dropouts) == 4, "dropout table must contain four candidates")
    require(
        set(dropouts["dropout"].astype(float)) == {0.0, 0.1, 0.2, 0.3},
        "dropout candidate set changed",
    )
    require(dropouts["learning_rate"].astype(float).eq(1e-4).all(), "dropout stage changed LR")

    for label, table in (("learning-rate", learning_rates), ("dropout", dropouts)):
        require(table["training_conditions"].astype(int).eq(29_277).all(), f"{label} train count changed")
        require(table["validation_conditions"].astype(int).eq(3_252).all(), f"{label} validation count changed")
        require(table["validation_ids_sha256"].astype(str).nunique() == 1, f"{label} validation membership changed")
    require(
        learning_rates["validation_ids_sha256"].iloc[0]
        == dropouts["validation_ids_sha256"].iloc[0],
        "LR and dropout stages used different validation memberships",
    )

    best_lr = strict_argmin(learning_rates, "learning-rate")
    best_dropout = strict_argmin(dropouts, "dropout")
    require(float(best_lr["learning_rate"]) == 1e-4, "unexpected best learning rate")
    require(float(best_lr["dropout"]) == 0.0, "unexpected LR-stage dropout")
    require(float(best_dropout["learning_rate"]) == 1e-4, "unexpected dropout-stage learning rate")
    require(float(best_dropout["dropout"]) == 0.0, "unexpected best dropout")
    require(
        str(best_lr["checkpoint_sha256"]) == str(best_dropout["checkpoint_sha256"]),
        "two full-data stages identify different checkpoints",
    )

    upstream = json.loads(UPSTREAM_SELECTION.read_text())
    require(upstream["status"] == "SELECTION_FROZEN", "upstream selection is not frozen")
    require(upstream["test_response_accessed"] is False, "upstream selection accessed test response")
    require(float(upstream["selected"]["learning_rate"]) == 1e-4, "upstream LR differs")
    require(float(upstream["selected"]["dropout"]) == 0.0, "upstream dropout differs")
    checkpoint_hash = sha256(FINAL_CHECKPOINT)
    require(checkpoint_hash == sha256(SELECTED_CHECKPOINT), "selected and final checkpoints differ")
    require(checkpoint_hash == str(best_lr["checkpoint_sha256"]), "checkpoint differs from full-data argmin")
    require(checkpoint_hash == upstream["selected_checkpoint_sha256"], "checkpoint differs from upstream selection")

    history = pd.read_csv(FINAL_HISTORY)
    require(len(history) == 100, "final MLP history must contain 100 completed epochs")
    require(
        history["epoch_zero_based"].astype(int).tolist() == list(range(100)),
        "final MLP history epochs are not contiguous",
    )
    require(history["validation_condition_count"].astype(int).eq(3_252).all(), "history validation count changed")
    history_best = history.loc[history["validation_delta_mse"].astype(float).idxmin()]
    require(int(history_best["epoch_zero_based"]) == 98, "strict history argmin epoch changed")
    require(
        np.isclose(
            float(history_best["validation_delta_mse"]),
            float(best_lr["validation_delta_mse"]),
            rtol=1e-12,
            atol=1e-15,
        ),
        "history and comparison validation MSE differ",
    )

    record = {
        "status": "FULL_DATA_MLP_SELECTION_FROZEN",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "authoritative_for": "MLP hyperparameters and checkpoint used in subsequent OOD evaluation",
        "unit_of_analysis": "drug-dose-cell-line condition",
        "target": "treated-condition mean minus matched cell-line control mean",
        "training_conditions": 29_277,
        "validation_conditions": 3_252,
        "validation_ids_sha256": str(best_lr["validation_ids_sha256"]),
        "selection_metric": "validation delta-expression MSE pooled across condition-gene pairs",
        "search": {
            "mode": "compact sequential full-data validation search",
            "learning_rates_at_dropout_0": [1e-4, 5e-4, 1e-3],
            "dropouts_at_selected_learning_rate": [0.0, 0.1, 0.2, 0.3],
            "learning_rate_comparison": str(LR_TABLE),
            "learning_rate_comparison_sha256": sha256(LR_TABLE),
            "dropout_comparison": str(DROPOUT_TABLE),
            "dropout_comparison_sha256": sha256(DROPOUT_TABLE),
        },
        "selected": {
            "architecture": [14_211, 512, 256, 13_784],
            "learning_rate": 1e-4,
            "dropout": 0.0,
            "optimizer": "Adam",
            "batch_size": 1_024,
            "best_epoch_zero_based": 98,
            "completed_epochs": 100,
            "training_delta_mse_at_best_epoch": float(history_best["training_delta_mse"]),
            "validation_delta_mse": float(best_lr["validation_delta_mse"]),
            "validation_median_condition_delta_pearson": float(
                best_lr["validation_median_condition_delta_pearson"]
            ),
            "validation_mean_condition_absolute_r2": float(
                best_lr["validation_mean_condition_absolute_r2"]
            ),
            "validation_mean_condition_absolute_pearson": float(
                best_lr["validation_mean_condition_absolute_pearson"]
            ),
        },
        "checkpoint": str(FINAL_CHECKPOINT),
        "checkpoint_sha256": checkpoint_hash,
        "history": str(FINAL_HISTORY),
        "history_sha256": sha256(FINAL_HISTORY),
        "upstream_development_selection": str(UPSTREAM_SELECTION),
        "upstream_development_selection_sha256": sha256(UPSTREAM_SELECTION),
        "test_response_accessed": False,
        "ood_results_used_for_selection": False,
    }
    write_json_atomic(OUTPUT, record)
    require(json.loads(OUTPUT.read_text()) == record, "written selection record changed")

    audit = f"""# Full-data MLP model-selection audit

Status: **PASS**

- Analysis unit: drug-dose-cell-line condition.
- Formal internal training set: 29,277 conditions.
- Fixed validation set: 3,252 conditions; membership SHA256 `{record['validation_ids_sha256']}`.
- Selection metric: pooled validation delta-expression MSE.
- Learning-rate candidates at dropout 0: 0.0001, 0.0005 and 0.001.
- Dropout candidates at learning rate 0.0001: 0, 0.1, 0.2 and 0.3.
- Strict joint optimum: learning rate 0.0001, dropout 0.
- Architecture: 14,211 -> 512 -> 256 -> 13,784.
- Best epoch: 99 (one-based; zero-based index 98).
- Validation delta-expression MSE: {record['selected']['validation_delta_mse']:.12g}.
- Validation median condition-level delta Pearson: {record['selected']['validation_median_condition_delta_pearson']:.12g}.
- Checkpoint SHA256: `{checkpoint_hash}`.
- External OOD responses were not used for model selection.

The earlier development-search record remains preserved as upstream provenance.
This versioned record is authoritative for the full-data validation confirmation
and the checkpoint to be used in subsequent OOD evaluation.
"""
    write_text_atomic(AUDIT, audit)
    print(json.dumps({"status": "PASS", "selection": str(OUTPUT), "audit": str(AUDIT)}, indent=2))


if __name__ == "__main__":
    main()
