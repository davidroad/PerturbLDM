#!/usr/bin/env python3
"""Recompute a deterministic subset of CPA condition-level expression metrics."""

from __future__ import annotations

import argparse
import json
import math
import resource
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RUN = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = RUN.parents[1]
DEFAULT_TEST = BENCHMARK_ROOT / "external_inputs/tahoe/test_adata_processed.h5ad"
DEFAULT_INFERENCE = RUN / "results/random_inference_full_gauss"
DEFAULT_MANIFEST = BENCHMARK_ROOT / "manifests/cpa_condition_assignments_seed42.csv"
DEFAULT_OUTPUT = RUN / "results/audit_cpa_inference_metric_semantics.json"
CELL_LINES = ("CVCL-1715", "CVCL-1716")
METRICS = ("mse", "mae", "r2_score", "pearson_r", "spearman_r", "chatterjee_r")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def chatterjee_corr(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    order = np.argsort(x)
    ranks = rankdata(y[order], method="ordinal")
    return float(1 - (3 * np.abs(np.diff(ranks)).sum()) / (n**2 - 1))


def dense_float32(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def normalized_cell_line(value: str) -> str:
    return str(value).replace("CVCL_", "CVCL-")


def normalize_drug_series(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.replace("DMSO_TF", "DMSO-TF", regex=False)
        .str.strip()
    )


def dose_string_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(".", "-", regex=False)


def metric_values(mean_real: np.ndarray, mean_pred: np.ndarray) -> dict[str, float]:
    pearson = pearsonr(mean_real, mean_pred)
    spearman = spearmanr(mean_real, mean_pred)
    return {
        "mse": float(mean_squared_error(mean_real, mean_pred)),
        "mae": float(mean_absolute_error(mean_real, mean_pred)),
        "r2_score": float(r2_score(mean_real, mean_pred)),
        "pearson_r": float(pearson.statistic if hasattr(pearson, "statistic") else pearson[0]),
        "spearman_r": float(
            spearman.statistic if hasattr(spearman, "statistic") else spearman[0]
        ),
        "chatterjee_r": chatterjee_corr(mean_real, mean_pred),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-h5ad", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--inference-root", type=Path, default=DEFAULT_INFERENCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    before = args.test_h5ad.stat()
    merged = json.loads(
        (args.inference_root / "merged_metrics_by_condition.json").read_text()
    )["metrics_by_condition"]
    manifest = pd.read_csv(args.manifest, dtype=str)
    ood_conditions = set(
        manifest.loc[manifest["cpa_split"].eq("ood"), "condition_id"].astype(str)
    )

    test = sc.read_h5ad(args.test_h5ad, backed="r")
    test_obs = test.obs
    condition_reports: list[dict] = []
    maximum_absolute_difference = {metric: 0.0 for metric in METRICS}
    feature_count: int | None = None

    try:
        for cell_line in CELL_LINES:
            inference_file = (
                args.inference_root
                / "by_cell_line"
                / f"cpa_inference_{cell_line.replace('-', '_')}.h5ad"
            )
            require(inference_file.is_file(), f"missing inference file: {inference_file}")
            prediction = sc.read_h5ad(inference_file, backed="r")
            try:
                require(
                    prediction.var_names.equals(test.var_names),
                    f"gene order differs for {cell_line}",
                )
                if feature_count is None:
                    feature_count = int(prediction.n_vars)
                require(
                    int(prediction.n_vars) == feature_count,
                    f"feature count changed for {cell_line}",
                )

                pred_obs = prediction.obs.copy()
                require("condition" in pred_obs, f"condition column missing for {cell_line}")
                pred_matrix = dense_float32(prediction.obsm["CPA_pred"])
                require(
                    pred_matrix.shape == (prediction.n_obs, prediction.n_vars),
                    f"prediction matrix shape mismatch for {cell_line}",
                )

                raw_candidates = [cell_line, cell_line.replace("-", "_")]
                global_cell_rows = np.flatnonzero(
                    test_obs["cell_line"].isin(raw_candidates).to_numpy()
                )
                require(len(global_cell_rows) > 0, f"no measured rows for {cell_line}")
                local_test_obs = test_obs.iloc[global_cell_rows].copy()
                local_test_obs["_cell_line"] = [
                    normalized_cell_line(value)
                    for value in local_test_obs["cell_line"].astype(str)
                ]
                local_test_obs["_drug"] = normalize_drug_series(local_test_obs["drug"])
                if "dose_str" in local_test_obs:
                    local_test_obs["_dose_str"] = local_test_obs["dose_str"].astype(str)
                else:
                    local_test_obs["_dose_str"] = dose_string_series(
                        local_test_obs["dose"]
                    )

                for condition in sorted(pred_obs["condition"].astype(str).unique()):
                    pred_mask = pred_obs["condition"].astype(str).eq(condition).to_numpy()
                    require(pred_mask.any(), f"empty prediction condition: {condition}")
                    first = pred_obs.loc[pred_mask].iloc[0]
                    drug = str(first["drug"]).replace("DMSO_TF", "DMSO-TF").strip()
                    dose_str = str(first["dose_str"])
                    real_local_mask = (
                        local_test_obs["_cell_line"].eq(cell_line)
                        & local_test_obs["_drug"].eq(drug)
                        & local_test_obs["_dose_str"].eq(dose_str)
                    ).to_numpy()
                    real_rows = global_cell_rows[real_local_mask]
                    require(len(real_rows) > 0, f"no measured cells for {condition}")

                    real_matrix = dense_float32(test.X[real_rows, :])
                    mean_real = np.mean(real_matrix, axis=0)
                    mean_pred = np.mean(pred_matrix[pred_mask, :], axis=0)
                    recomputed = metric_values(mean_real, mean_pred)
                    require(condition in merged, f"condition absent from merged JSON: {condition}")
                    require(condition in ood_conditions, f"condition is not OOD: {condition}")
                    reported = merged[condition]

                    metric_differences = {}
                    for metric in METRICS:
                        observed_value = float(reported[metric])
                        recalculated_value = float(recomputed[metric])
                        difference = abs(observed_value - recalculated_value)
                        metric_differences[metric] = difference
                        maximum_absolute_difference[metric] = max(
                            maximum_absolute_difference[metric], difference
                        )
                        require(
                            math.isclose(
                                observed_value,
                                recalculated_value,
                                rel_tol=1e-6,
                                abs_tol=5e-7,
                            ),
                            f"{condition} {metric} mismatch: "
                            f"reported={observed_value} recomputed={recalculated_value}",
                        )

                    require(
                        int(reported["n_inference_cells"]) == int(pred_mask.sum()),
                        f"prediction-cell count mismatch for {condition}",
                    )
                    require(
                        int(reported["n_real_cells"]) == len(real_rows),
                        f"measured-cell count mismatch for {condition}",
                    )
                    condition_reports.append(
                        {
                            "condition": condition,
                            "cell_line": cell_line,
                            "drug": drug,
                            "dose_str": dose_str,
                            "n_prediction_cells": int(pred_mask.sum()),
                            "n_measured_cells": int(len(real_rows)),
                            "metric_absolute_differences": metric_differences,
                        }
                    )
            finally:
                prediction.file.close()
    finally:
        test.file.close()

    after = args.test_h5ad.stat()
    require(before.st_size == after.st_size, "raw test file size changed")
    require(before.st_mtime_ns == after.st_mtime_ns, "raw test file mtime changed")
    require(len(condition_reports) == 9, f"expected 9 audit conditions, found {len(condition_reports)}")

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "AUDIT_OK",
        "scientific_contract": {
            "unit_of_analysis": "one external OOD drug-dose-cell-line condition",
            "ground_truth": "mean expression across measured perturbed test cells",
            "prediction": "mean CPA_pred across matched-cell-line control-derived prediction cells",
            "feature_alignment": "exact ordered gene identity",
            "metric_axis": "per condition across all genes",
            "metric_family": "absolute condition-level expression reconstruction",
            "effect_fidelity_guardrail": (
                "This audit does not reinterpret absolute-expression metrics as "
                "matched-control perturbation-effect fidelity."
            ),
        },
        "inputs": {
            "test_h5ad": str(args.test_h5ad),
            "test_h5ad_mode": "backed read-only",
            "inference_root": str(args.inference_root),
            "ood_manifest": str(args.manifest),
        },
        "conditions_recomputed": len(condition_reports),
        "cell_lines": list(CELL_LINES),
        "feature_count": feature_count,
        "maximum_absolute_metric_difference": maximum_absolute_difference,
        "conditions": condition_reports,
        "raw_data_guardrail": {
            "size_bytes_before_after_equal": True,
            "mtime_ns_before_after_equal": True,
        },
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        f"AUDIT_OK conditions={len(condition_reports)} "
        f"features={feature_count} output={args.output}"
    )


if __name__ == "__main__":
    main()
