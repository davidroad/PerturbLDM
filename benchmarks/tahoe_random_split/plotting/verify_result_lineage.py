#!/usr/bin/env python3
"""Verify RF/MLP and CPA condition metrics against the released Fig. 2b table."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["MSE", "MAE", "R2", "Pearson_r", "Spearman_r"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learned-table", type=Path, required=True)
    parser.add_argument("--baseline-raw", type=Path, required=True)
    parser.add_argument("--cpa-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalize_cell_line(value: object) -> str:
    text = str(value).strip()
    return text.replace("_", "-", 1) if text.startswith("CVCL_") else text


def normalize_dose(value: object) -> str:
    text = str(value).strip().replace("-", ".")
    try:
        return f"{float(text):.12g}"
    except ValueError:
        return text


def normalize_drug(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def add_key(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_key"] = [
        "\t".join([
            normalize_cell_line(cell),
            normalize_drug(drug),
            normalize_dose(dose),
        ])
        for cell, drug, dose in zip(result["cellname"], result["drug"], result["dose"])
    ]
    return result


def compare(source: pd.DataFrame, released: pd.DataFrame,
            source_method: str, released_method: str) -> dict[str, object]:
    left = source.rename(columns={source_method: "_method"})
    right = released.rename(columns={released_method: "_method"})
    merged = left.merge(
        right, on=["_method", "_key"], suffixes=("_source", "_released"),
        validate="one_to_one",
    )
    maximum = {
        metric: float(np.max(np.abs(
            merged[f"{metric}_source"].to_numpy()
            - merged[f"{metric}_released"].to_numpy()
        )))
        for metric in METRICS
    }
    passed = (
        len(source) == len(released) == len(merged)
        and max(maximum.values()) <= 5.1e-9
    )
    return {
        "source_rows": int(len(source)),
        "released_rows": int(len(released)),
        "matched_rows": int(len(merged)),
        "maximum_absolute_difference": maximum,
        "status": "pass" if passed else "fail",
    }


def load_baseline(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    split = raw["CondID"].str.split("___", n=2, expand=True)
    if split.shape[1] != 3:
        raise ValueError("Baseline CondID is not drug___dose___cell-line")
    raw[["drug", "dose", "cellname"]] = split
    raw["method"] = raw["Model"].replace({"TrivialZero": "TrivalZero"})
    return add_key(raw)


def load_cpa(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        payload = json.load(handle)
    rows = []
    for condition, metrics in payload["metrics_by_condition"].items():
        left, dose = condition.rsplit("_", 1)
        cell_line, drug = left.split("_", 1)
        rows.append({
            "cellname": cell_line,
            "drug": drug,
            "dose": dose,
            "method": "CPA",
            "MSE": metrics["mse"],
            "MAE": metrics["mae"],
            "R2": metrics["r2_score"],
            "Pearson_r": metrics["pearson_r"],
            "Spearman_r": metrics["spearman_r"],
        })
    return add_key(pd.DataFrame(rows))


def main() -> None:
    args = parse_args()
    learned = add_key(pd.read_csv(args.learned_table, low_memory=False))

    baseline_raw = load_baseline(args.baseline_raw)
    baseline_methods = {"MLP", "RF", "TrivalZero"}
    baseline_released = learned.loc[learned["method_raw"].isin(baseline_methods)].copy()
    baseline = compare(
        baseline_raw.loc[baseline_raw["method"].isin(baseline_methods)],
        baseline_released,
        "method",
        "method_raw",
    )

    cpa_raw = load_cpa(args.cpa_json)
    cpa_released = learned.loc[learned["method_raw"].eq("CPA")].copy()
    cpa = compare(cpa_raw, cpa_released, "method", "method_raw")
    report = {
        "baseline": baseline,
        "cpa": cpa,
        "status": "pass" if baseline["status"] == cpa["status"] == "pass" else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
