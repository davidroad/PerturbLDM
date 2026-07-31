#!/usr/bin/env python3
"""Compute train-only simple baselines for the Tahoe benchmark.

The baselines are intentionally simple and interpretable controls:

- MatchedCtrl: matched control mean expression for the same cell line.
- CellLineMean: train-condition mean expression for the same cell line.
- DrugDoseMean: train-condition mean expression for the same drug-dose pair.
- AdditiveMean: CellLineMean + DrugDoseMean - GlobalMean.

All marginal means are estimated from the training split only. Test expression
means are used only as ground truth during evaluation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import rankdata


METHODS = ("MatchedCtrl", "CellLineMean", "DrugDoseMean", "AdditiveMean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute train-only simple baseline predictions and condition-level "
            "expression/effect metrics for the Tahoe cell-line/drug/dose split."
        )
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Processed Tahoe split folder containing collection/ and processed/.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("benchmark_simple_baselines"),
        help="Directory for derived benchmark metrics.",
    )
    parser.add_argument(
        "--condition_col",
        default="CondID",
        help="Condition identifier column in train_metadf.csv and test_metadf.csv.",
    )
    parser.add_argument(
        "--control_cell_col",
        default="cell_name",
        help="Cell-line column in control_metadf.csv.",
    )
    parser.add_argument(
        "--save_prediction_means",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save per-condition prediction-mean matrices as .npy files.",
    )
    parser.add_argument(
        "--metrics_batch_size",
        type=int,
        default=256,
        help="Number of test conditions per vectorized metric batch.",
    )
    parser.add_argument(
        "--read_backed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read h5ad files in backed read-only mode to reduce memory use.",
    )
    return parser.parse_args()


def parse_condition(condition: str) -> tuple[str, str, str]:
    parts = str(condition).rsplit("___", 2)
    if len(parts) != 3:
        raise ValueError(f"Cannot parse condition key: {condition!r}")
    return parts[0], parts[1], parts[2]


def dense_rows(matrix, indices: np.ndarray) -> np.ndarray:
    subset = matrix[indices]
    if hasattr(subset, "toarray"):
        subset = subset.toarray()
    return np.asarray(subset)


def mean_rows(matrix, indices: np.ndarray) -> np.ndarray:
    return np.asarray(dense_rows(matrix, indices).mean(axis=0)).ravel().astype(np.float32)


def validate_collection(data_root: Path) -> dict[str, Path]:
    collection = data_root / "collection"
    paths = {
        "train_adata": collection / "train_adata.h5ad",
        "train_meta": collection / "train_metadf.csv",
        "test_adata": collection / "test_adata.h5ad",
        "test_meta": collection / "test_metadf.csv",
        "control_adata": collection / "control_adata.h5ad",
        "control_meta": collection / "control_metadf.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required Tahoe processed-data files:\n" + "\n".join(missing))
    return paths


def grouped_indices(labels: Iterable[str]) -> list[tuple[str, np.ndarray]]:
    labels_array = np.asarray(list(labels), dtype=object)
    return [
        (str(label), np.flatnonzero(labels_array == label))
        for label in pd.unique(labels_array)
    ]


def aggregate_train_marginals(
    train_adata,
    train_meta: pd.DataFrame,
    condition_col: str,
) -> dict[str, object]:
    if condition_col not in train_meta.columns:
        raise KeyError(f"{condition_col!r} is missing from train metadata")
    if train_adata.n_obs != len(train_meta):
        raise ValueError(
            f"train_adata.n_obs ({train_adata.n_obs}) does not match "
            f"train metadata rows ({len(train_meta)})"
        )

    n_genes = train_adata.n_vars
    global_sum = np.zeros(n_genes, dtype=np.float64)
    global_count = 0
    cell_sum: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_genes, dtype=np.float64))
    drugdose_sum: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_genes, dtype=np.float64))
    cell_count: dict[str, int] = defaultdict(int)
    drugdose_count: dict[str, int] = defaultdict(int)

    for condition, idx in grouped_indices(train_meta[condition_col].astype(str)):
        drug, dose, cell = parse_condition(condition)
        drug_dose = f"{drug}___{dose}"
        condition_mean = mean_rows(train_adata.X, idx).astype(np.float64, copy=False)

        global_sum += condition_mean
        global_count += 1
        cell_sum[cell] += condition_mean
        cell_count[cell] += 1
        drugdose_sum[drug_dose] += condition_mean
        drugdose_count[drug_dose] += 1

    if global_count == 0:
        raise ValueError("No training conditions were found")

    return {
        "global_mean": (global_sum / global_count).astype(np.float32),
        "cell_mean": {
            key: (value / cell_count[key]).astype(np.float32)
            for key, value in cell_sum.items()
        },
        "drugdose_mean": {
            key: (value / drugdose_count[key]).astype(np.float32)
            for key, value in drugdose_sum.items()
        },
        "cell_count": dict(cell_count),
        "drugdose_count": dict(drugdose_count),
        "global_count": global_count,
    }


def compute_control_means(
    control_adata,
    control_meta: pd.DataFrame,
    cell_col: str,
) -> dict[str, object]:
    if cell_col not in control_meta.columns:
        raise KeyError(f"{cell_col!r} is missing from control metadata")
    if control_adata.n_obs != len(control_meta):
        raise ValueError(
            f"control_adata.n_obs ({control_adata.n_obs}) does not match "
            f"control metadata rows ({len(control_meta)})"
        )

    control_means: dict[str, np.ndarray] = {}
    labels = control_meta[cell_col].astype(str)
    for cell, idx in grouped_indices(labels):
        control_means[cell] = mean_rows(control_adata.X, idx)

    if not control_means:
        raise ValueError("No control cell-line means were found")

    control_global = np.vstack(list(control_means.values())).mean(axis=0).astype(np.float32)
    return {
        "control_mean": control_means,
        "control_global_mean": control_global,
        "control_cell_count": labels.value_counts().to_dict(),
    }


def row_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = y_true.astype(np.float64, copy=False)
    y_pred = y_pred.astype(np.float64, copy=False)
    centered_true = y_true - y_true.mean(axis=1, keepdims=True)
    centered_pred = y_pred - y_pred.mean(axis=1, keepdims=True)
    denom = np.sqrt((centered_true * centered_true).sum(axis=1) * (centered_pred * centered_pred).sum(axis=1))
    out = np.full(y_true.shape[0], np.nan, dtype=np.float64)
    ok = denom > 0
    out[ok] = (centered_true[ok] * centered_pred[ok]).sum(axis=1) / denom[ok]
    return out


def row_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true_rank = rankdata(y_true, axis=1, method="average")
    pred_rank = rankdata(y_pred, axis=1, method="average")
    return row_pearson(true_rank, pred_rank)


def row_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = y_true.astype(np.float64, copy=False)
    y_pred = y_pred.astype(np.float64, copy=False)
    ss_res = ((y_true - y_pred) ** 2).sum(axis=1)
    centered = y_true - y_true.mean(axis=1, keepdims=True)
    ss_tot = (centered * centered).sum(axis=1)
    out = np.full(y_true.shape[0], np.nan, dtype=np.float64)
    ok = ss_tot > 0
    out[ok] = 1.0 - ss_res[ok] / ss_tot[ok]
    return out


def row_chatterjee(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    order = np.argsort(y_true, axis=1)
    pred_sorted = np.take_along_axis(y_pred, order, axis=1)
    pred_ranks = rankdata(pred_sorted, axis=1, method="average")
    diff_sum = np.abs(np.diff(pred_ranks, axis=1)).sum(axis=1)
    n_genes = y_true.shape[1]
    if n_genes < 2:
        return np.full(y_true.shape[0], np.nan, dtype=np.float64)
    return 1.0 - (3.0 * diff_sum) / (n_genes**2 - 1)


def compute_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
    diff = y_true.astype(np.float64, copy=False) - y_pred.astype(np.float64, copy=False)
    return {
        "MSE": np.mean(diff * diff, axis=1),
        "MAE": np.mean(np.abs(diff), axis=1),
        "R2": row_r2(y_true, y_pred),
        "Pearson_r": row_pearson(y_true, y_pred),
        "Spearman_r": row_spearman(y_true, y_pred),
        "Chatterjee": row_chatterjee(y_true, y_pred),
    }


def get_predictions(
    drug_dose: str,
    cell: str,
    aggregates: dict[str, object],
    controls: dict[str, object],
    fallback_counts: dict[str, int],
) -> dict[str, np.ndarray]:
    global_mean: np.ndarray = aggregates["global_mean"]  # type: ignore[assignment]
    cell_mean_map: dict[str, np.ndarray] = aggregates["cell_mean"]  # type: ignore[assignment]
    drugdose_mean_map: dict[str, np.ndarray] = aggregates["drugdose_mean"]  # type: ignore[assignment]
    control_mean_map: dict[str, np.ndarray] = controls["control_mean"]  # type: ignore[assignment]
    control_global: np.ndarray = controls["control_global_mean"]  # type: ignore[assignment]

    ctrl = control_mean_map.get(cell)
    if ctrl is None:
        ctrl = control_global
        fallback_counts["MatchedCtrl"] += 1

    cell_mean = cell_mean_map.get(cell)
    if cell_mean is None:
        cell_mean = global_mean
        fallback_counts["CellLineMean"] += 1

    drugdose_mean = drugdose_mean_map.get(drug_dose)
    if drugdose_mean is None:
        drugdose_mean = global_mean
        fallback_counts["DrugDoseMean"] += 1

    if cell not in cell_mean_map or drug_dose not in drugdose_mean_map:
        fallback_counts["AdditiveMean"] += 1

    return {
        "MatchedCtrl": ctrl,
        "CellLineMean": cell_mean,
        "DrugDoseMean": drugdose_mean,
        "AdditiveMean": (cell_mean + drugdose_mean - global_mean).astype(np.float32),
    }


def evaluate_test_conditions(
    test_adata,
    test_meta: pd.DataFrame,
    condition_col: str,
    aggregates: dict[str, object],
    controls: dict[str, object],
    output_dir: Path,
    save_prediction_means: bool,
    metrics_batch_size: int,
) -> tuple[pd.DataFrame, dict[str, int], list[str]]:
    if condition_col not in test_meta.columns:
        raise KeyError(f"{condition_col!r} is missing from test metadata")
    if test_adata.n_obs != len(test_meta):
        raise ValueError(
            f"test_adata.n_obs ({test_adata.n_obs}) does not match "
            f"test metadata rows ({len(test_meta)})"
        )
    if metrics_batch_size < 1:
        raise ValueError("--metrics_batch_size must be positive")

    condition_groups = grouped_indices(test_meta[condition_col].astype(str))
    n_conditions = len(condition_groups)
    n_genes = test_adata.n_vars
    fallback_counts = {method: 0 for method in METHODS}
    records: list[dict[str, object]] = []
    condition_order: list[str] = []

    prediction_writers = None
    if save_prediction_means:
        prediction_writers = {
            method: np.lib.format.open_memmap(
                output_dir / f"{method}_pred_expr_mean.npy",
                mode="w+",
                dtype=np.float32,
                shape=(n_conditions, n_genes),
            )
            for method in METHODS
        }

    for start in range(0, n_conditions, metrics_batch_size):
        end = min(start + metrics_batch_size, n_conditions)
        batch_groups = condition_groups[start:end]
        batch_meta: list[dict[str, object]] = []
        gt_rows: list[np.ndarray] = []
        ctrl_rows: list[np.ndarray] = []
        pred_rows = {method: [] for method in METHODS}

        for i, (condition, idx) in enumerate(batch_groups, start=start):
            drug, dose, cell = parse_condition(condition)
            drug_dose = f"{drug}___{dose}"
            gt_expr = mean_rows(test_adata.X, idx)
            predictions = get_predictions(drug_dose, cell, aggregates, controls, fallback_counts)
            ctrl_mean = predictions["MatchedCtrl"]

            batch_meta.append(
                {
                    "condition_id": condition,
                    "drug": drug,
                    "dose": dose,
                    "cell_name": cell,
                    "drug_dose": drug_dose,
                    "n_test_cells": int(len(idx)),
                }
            )
            gt_rows.append(gt_expr)
            ctrl_rows.append(ctrl_mean)
            condition_order.append(condition)

            for method, pred_expr in predictions.items():
                pred_expr = pred_expr.astype(np.float32, copy=False)
                pred_rows[method].append(pred_expr)
                if prediction_writers is not None:
                    prediction_writers[method][i] = pred_expr

        gt_expr_batch = np.vstack(gt_rows)
        ctrl_batch = np.vstack(ctrl_rows)
        gt_effect_batch = gt_expr_batch - ctrl_batch

        for method in METHODS:
            pred_expr_batch = np.vstack(pred_rows[method])
            pred_effect_batch = pred_expr_batch - ctrl_batch
            expr_metrics = compute_matrix_metrics(gt_expr_batch, pred_expr_batch)
            effect_metrics = compute_matrix_metrics(gt_effect_batch, pred_effect_batch)

            for row_idx, base in enumerate(batch_meta):
                row = dict(base)
                row["method"] = method
                for metric_name, values in expr_metrics.items():
                    row[f"expr_{metric_name}"] = values[row_idx]
                for metric_name, values in effect_metrics.items():
                    row[f"effect_{metric_name}"] = values[row_idx]
                records.append(row)

        print(f"evaluated {end}/{n_conditions} test conditions", flush=True)

    if prediction_writers is not None:
        for writer in prediction_writers.values():
            writer.flush()
        with open(output_dir / "condition_order.json", "w") as fh:
            json.dump(condition_order, fh, indent=2)

    return pd.DataFrame.from_records(records), fallback_counts, condition_order


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in metrics.columns
        if col.startswith("expr_") or col.startswith("effect_")
    ]
    pieces = []
    grouped = metrics.groupby("method", observed=True)
    for method, group in grouped:
        row: dict[str, object] = {
            "method": method,
            "n_conditions": int(group["condition_id"].nunique()),
        }
        for col in numeric_cols:
            row[f"mean_{col}"] = group[col].mean()
            row[f"median_{col}"] = group[col].median()
        pieces.append(row)
    return pd.DataFrame(pieces).sort_values("median_expr_R2", ascending=False)


def main() -> None:
    args = parse_args()
    paths = validate_collection(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Tahoe simple baseline benchmark", flush=True)
    print(f"data_root={args.data_root}", flush=True)
    print(f"output_dir={args.output_dir}", flush=True)
    print("marginal baselines are estimated from the training split only", flush=True)

    read_kwargs = {"backed": "r"} if args.read_backed else {}
    train_adata = sc.read_h5ad(paths["train_adata"], **read_kwargs)
    test_adata = sc.read_h5ad(paths["test_adata"], **read_kwargs)
    control_adata = sc.read_h5ad(paths["control_adata"], **read_kwargs)
    train_meta = pd.read_csv(paths["train_meta"])
    test_meta = pd.read_csv(paths["test_meta"])
    control_meta = pd.read_csv(paths["control_meta"])

    if train_adata.n_vars != test_adata.n_vars or train_adata.n_vars != control_adata.n_vars:
        raise ValueError(
            "Gene dimensions differ across train/test/control adata files: "
            f"train={train_adata.n_vars}, test={test_adata.n_vars}, control={control_adata.n_vars}"
        )

    print(
        "loaded data: "
        f"train_cells={train_adata.n_obs}, test_cells={test_adata.n_obs}, "
        f"control_cells={control_adata.n_obs}, genes={train_adata.n_vars}",
        flush=True,
    )

    aggregates = aggregate_train_marginals(train_adata, train_meta, args.condition_col)
    controls = compute_control_means(control_adata, control_meta, args.control_cell_col)
    print(
        "train marginals: "
        f"conditions={aggregates['global_count']}, "
        f"cell_lines={len(aggregates['cell_mean'])}, "
        f"drug_doses={len(aggregates['drugdose_mean'])}",
        flush=True,
    )

    metrics, fallback_counts, condition_order = evaluate_test_conditions(
        test_adata=test_adata,
        test_meta=test_meta,
        condition_col=args.condition_col,
        aggregates=aggregates,
        controls=controls,
        output_dir=args.output_dir,
        save_prediction_means=args.save_prediction_means,
        metrics_batch_size=args.metrics_batch_size,
    )
    summary = summarize_metrics(metrics)

    metrics_path = args.output_dir / "simple_baseline_condition_metrics.csv"
    summary_path = args.output_dir / "simple_baseline_summary.csv"
    coverage_path = args.output_dir / "simple_baseline_coverage.json"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)

    coverage = {
        "data_root": str(args.data_root),
        "n_genes": int(train_adata.n_vars),
        "n_train_cells": int(train_adata.n_obs),
        "n_test_cells": int(test_adata.n_obs),
        "n_control_cells": int(control_adata.n_obs),
        "n_train_conditions": int(aggregates["global_count"]),
        "n_test_conditions": int(len(condition_order)),
        "n_train_cell_lines": int(len(aggregates["cell_mean"])),
        "n_train_drug_doses": int(len(aggregates["drugdose_mean"])),
        "n_control_cell_lines": int(len(controls["control_mean"])),
        "fallback_counts": fallback_counts,
        "baseline_definitions": {
            "MatchedCtrl": "matched control mean expression for the same cell line",
            "CellLineMean": "mean of training perturbation-condition means from the same cell line",
            "DrugDoseMean": "mean of training perturbation-condition means from the same drug-dose pair",
            "AdditiveMean": "CellLineMean + DrugDoseMean - GlobalMean, using training split marginals",
        },
        "no_leakage_rule": "Test expression means are used only as ground truth during evaluation.",
    }
    with open(coverage_path, "w") as fh:
        json.dump(coverage, fh, indent=2)

    print(f"saved condition metrics: {metrics_path}", flush=True)
    print(f"saved summary metrics: {summary_path}", flush=True)
    print(f"saved coverage note: {coverage_path}", flush=True)
    print(summary[["method", "n_conditions", "median_expr_R2", "median_effect_Pearson_r"]], flush=True)


if __name__ == "__main__":
    main()
