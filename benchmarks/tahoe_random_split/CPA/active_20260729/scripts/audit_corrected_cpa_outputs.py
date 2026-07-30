#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RUN = Path(__file__).resolve().parents[1]
EXPECTED_CONDITIONS = 13_942
EXPECTED_CELL_LINES = 47


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def external_conditions() -> set[str]:
    manifest = pd.read_csv(RUN / "split_manifest/cpa_condition_assignments_seed42.csv", dtype=str)
    values = set(manifest.loc[manifest["cpa_split"].eq("ood"), "condition_id"])
    require(len(values) == EXPECTED_CONDITIONS, f"OOD manifest count={len(values)}")
    return values


def audit_split(report: dict) -> None:
    data = json.loads((RUN / "split_manifest/cpa_split_audit_seed42.json").read_text())
    require(data["internal_train_condition_count"] == 29_277, "train condition count mismatch")
    require(data["validation_condition_count"] == 3_252, "validation condition count mismatch")
    require(data["external_ood_condition_count"] == EXPECTED_CONDITIONS, "OOD count mismatch")
    require(data["train_validation_condition_overlap"] == 0, "train/validation overlap")
    require(data["official_train_external_ood_condition_overlap"] == 0, "train/OOD overlap")
    report["split"] = data


def audit_training(report: dict) -> None:
    model = RUN / "random_gauss_result/models/cpa_global_model_gauss.pth"
    for name in ["model.pt", "CPA_info.json", "history.csv"]:
        require((model / name).is_file(), f"missing model artifact: {name}")
    history = pd.read_csv(model / "history.csv")
    train = history.loc[history["mode"].eq("train")]
    valid = history.loc[history["mode"].eq("valid")]
    train_epochs = train["epoch"].astype(int).tolist()
    valid_epochs = valid["epoch"].astype(int).tolist()
    require(train_epochs == valid_epochs, "train and validation epoch coverage differs")
    require(1 <= len(train_epochs) <= 20, "completed epoch count is outside 1..20")
    require(train_epochs == list(range(len(train_epochs))), "epochs are not contiguous from 0")
    selection = json.loads(
        (
            RUN
            / "random_gauss_result/checkpoints_40cpu_20epochs/best/selection.json"
        ).read_text()
    )
    require(selection["completed_epochs"] == len(train_epochs), "selection epoch count mismatch")
    require(selection["max_epochs"] == 20, "max_epochs changed")
    require(selection["early_stopping_enabled"] is True, "early stopping is disabled")
    require(selection["early_stopping_patience"] == 4, "early-stopping patience changed")
    require(
        selection["stopped_early"] is (len(train_epochs) < 20),
        "early-stopping status is inconsistent",
    )
    require((model / "model.pt").stat().st_size > 100_000_000, "model file is unexpectedly small")
    report["training"] = {
        "train_epochs": train_epochs,
        "validation_epochs_zero_based": valid_epochs,
        "completed_epochs": len(train_epochs),
        "max_epochs": 20,
        "stopped_early": selection["stopped_early"],
        "model_bytes": (model / "model.pt").stat().st_size,
        "final_train_recon": float(train.iloc[-1]["recon_loss"]),
        "final_validation_recon": float(valid.iloc[-1]["recon_loss"]),
    }


def audit_inference(report: dict) -> None:
    root = RUN / "results/random_inference_full_gauss"
    data = json.loads((root / "merged_metrics_by_condition.json").read_text())
    metrics = data["metrics_by_condition"]
    require(data["total_conditions"] == EXPECTED_CONDITIONS, "inference summary count mismatch")
    require(len(metrics) == EXPECTED_CONDITIONS, "inference metric count mismatch")
    require(set(metrics) == external_conditions(), "inference condition set differs from OOD manifest")
    files = sorted((root / "by_cell_line").glob("cpa_inference_*.h5ad"))
    require(len(files) == EXPECTED_CELL_LINES, f"inference H5AD count={len(files)}")
    required = {"mse", "mae", "r2_score", "pearson_r", "spearman_r", "chatterjee_r"}
    require(all(required.issubset(v) for v in metrics.values()), "inference metrics incomplete")
    report["inference"] = {"conditions": len(metrics), "cell_line_files": len(files)}


def audit_distribution(report: dict) -> None:
    path = RUN / "results/random_distribution_similarity_gauss/global_condition_metrics.csv"
    data = pd.read_csv(path, low_memory=False)
    require(len(data) == EXPECTED_CONDITIONS, f"distribution row count={len(data)}")
    require(data["condition"].nunique() == EXPECTED_CONDITIONS, "distribution condition duplicates")
    require(set(data["condition"]) == external_conditions(), "distribution condition set differs from OOD manifest")
    require(data["status"].eq("success").all(), "distribution contains failed conditions")
    metrics = ["MMD_RBF", "E_distance", "Wasserstein_Sliced", "Wasserstein_OT"]
    for metric in metrics:
        values = pd.to_numeric(data[metric], errors="coerce")
        require(values.notna().all(), f"missing {metric}")
        require(values.map(math.isfinite).all(), f"non-finite {metric}")
    require(data["Wasserstein_OT_type"].eq("ot_emd").all(), "OT metric is not exact EMD")
    report["distribution"] = {
        "conditions": len(data),
        "metric_means": {m: float(pd.to_numeric(data[m]).mean()) for m in metrics},
    }


def unchanged_non_target(before: pd.DataFrame, after: pd.DataFrame, mask: pd.Series, label: str) -> None:
    pd.testing.assert_frame_equal(
        before.loc[~mask].reset_index(drop=True),
        after.loc[~mask].reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=0,
        atol=0,
        obj=f"{label} non-CPA rows",
    )


def audit_final(report: dict) -> None:
    base = RUN / "results/pipeline_results_gauss_random"
    pred_before = pd.read_csv(base / "random_merged_condition_metrics_exact_6methods_RUN.csv", low_memory=False)
    pred_after = pd.read_csv(base / "random_merged_condition_metrics_exact_6methods_RUN_cpa_updated.csv", low_memory=False)
    require(list(pred_before.columns) == list(pred_after.columns), "prediction columns changed")
    pred_mask = pred_before["method"].eq("CPA")
    require(int(pred_mask.sum()) == EXPECTED_CONDITIONS, "prediction CPA row count mismatch")
    unchanged_non_target(pred_before, pred_after, pred_mask, "prediction")

    dist_before = pd.read_csv(base / "merged_metrics_distribution_random_diffusion_cpa_chemcpa.csv", low_memory=False)
    dist_after = pd.read_csv(base / "merged_metrics_distribution_random_diffusion_cpa_chemcpa_cpa_updated.csv", low_memory=False)
    require(list(dist_before.columns) == list(dist_after.columns), "distribution columns changed")
    dist_mask = dist_before["method"].astype(str).str.lower().eq("cpa")
    require(int(dist_mask.sum()) == EXPECTED_CONDITIONS, "distribution CPA row count mismatch")
    unchanged_non_target(dist_before, dist_after, dist_mask, "distribution")
    report["final_tables"] = {
        "prediction_rows": len(pred_after),
        "prediction_cpa_rows": int(pred_mask.sum()),
        "distribution_rows": len(dist_after),
        "distribution_cpa_rows": int(dist_mask.sum()),
        "non_cpa_rows_unchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["training", "inference", "distribution", "final"], required=True)
    args = parser.parse_args()
    report = {"stage": args.stage, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    audit_split(report)
    audit_training(report)
    if args.stage in {"inference", "distribution", "final"}:
        audit_inference(report)
    if args.stage in {"distribution", "final"}:
        audit_distribution(report)
    if args.stage == "final":
        audit_final(report)
    out = RUN / f"results/audit_{args.stage}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"AUDIT_OK stage={args.stage} report={out}")


if __name__ == "__main__":
    main()
