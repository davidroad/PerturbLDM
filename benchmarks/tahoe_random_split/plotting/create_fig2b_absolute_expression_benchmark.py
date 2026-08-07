#!/usr/bin/env python3
"""Recreate Fig. 2b from released condition-level derived metrics.

This standalone wrapper recreates the panel from paired release tables.
It does not train or run any model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "PerturbLDM", "AdditiveMean", "CellLineMean", "MatchedCtrl",
    "MLP", "RF", "CPA", "chemCPA",
]
METHOD_LABELS = {
    "PerturbLDM": "PerturbLDM", "AdditiveMean": "Additive",
    "CellLineMean": "Cell-line", "MatchedCtrl": "Matched ctrl",
    "MLP": "MLP", "RF": "RF", "CPA": "CPA", "chemCPA": "chemCPA",
}
METHOD_CLASS = {
    "PerturbLDM": "PerturbLDM", "AdditiveMean": "Simple marginal",
    "CellLineMean": "Simple marginal", "MatchedCtrl": "Matched control",
    "MLP": "Learned baseline", "RF": "Learned baseline",
    "CPA": "Learned baseline", "chemCPA": "Learned baseline",
}
CLASS_COLORS = {
    "PerturbLDM": "#1f8a5b", "Simple marginal": "#c49a4a",
    "Matched control": "#9aa5b1", "Learned baseline": "#8aa7c4",
}
METRICS = [
    ("Pearson_r", "Pearson", (0.91, 1.002)),
    ("Spearman_r", "Spearman", (0.87, 1.002)),
    ("R2", r"$R^2$", (0.78, 1.002)),
    ("MAE", "MAE", (0.015, 0.075)),
]
EXPECTED_CONDITIONS = 13_942
EDGE = "#27313c"
GRID = "#dfe5ec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learned-table", type=Path, required=True)
    parser.add_argument("--simple-table", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--expected-summary", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def load_metrics(learned_path: Path, simple_path: Path) -> pd.DataFrame:
    learned = pd.read_csv(learned_path, low_memory=False)
    simple = pd.read_csv(simple_path, low_memory=False)
    required = {"method_canonical", *(metric for metric, _, _ in METRICS)}
    for label, table in [("learned", learned), ("simple", simple)]:
        missing = sorted(required.difference(table.columns))
        if missing:
            raise ValueError(f"{label} table is missing columns: {missing}")

    metrics = pd.concat([learned, simple], ignore_index=True, sort=False)
    metrics = metrics[metrics["method_canonical"].isin(METHOD_ORDER)].copy()
    counts = metrics.groupby("method_canonical", observed=True).size().reindex(METHOD_ORDER)
    if counts.isna().any():
        raise ValueError(f"Missing methods: {counts[counts.isna()].index.tolist()}")
    bad = counts[counts.ne(EXPECTED_CONDITIONS)]
    if not bad.empty:
        raise ValueError(f"Expected {EXPECTED_CONDITIONS:,} rows per method: {bad.to_dict()}")
    for metric, _, _ in METRICS:
        values = pd.to_numeric(metrics[metric], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Missing or non-numeric values in {metric}")
        metrics[metric] = values
    return metrics


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        subset = metrics.loc[metrics["method_canonical"].eq(method)]
        for metric, label, _ in METRICS:
            values = subset[metric].to_numpy()
            rows.append({
                "method_canonical": method,
                "method_label": METHOD_LABELS[method],
                "method_class": METHOD_CLASS[method],
                "metric": metric,
                "metric_label": label,
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "q05": float(np.quantile(values, 0.05)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "q95": float(np.quantile(values, 0.95)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            })
    return pd.DataFrame(rows)


def verify_summary(actual: pd.DataFrame, expected_path: Path) -> dict[str, object]:
    expected = pd.read_csv(expected_path)
    merged = actual.merge(
        expected, on=["method_canonical", "metric"], how="outer",
        suffixes=("_actual", "_expected"), indicator=True, validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Expected summary has different method-metric keys")
    if not (merged["n_actual"] == merged["n_expected"]).all():
        raise ValueError("Expected summary has different condition counts")
    numeric = ["mean", "median", "q05", "q25", "q75", "q95", "min", "max"]
    maximum = {
        column: float(np.max(np.abs(
            merged[f"{column}_actual"].to_numpy()
            - merged[f"{column}_expected"].to_numpy()
        )))
        for column in numeric
    }
    if max(maximum.values()) > 1e-12:
        raise ValueError(f"Expected summary mismatch: {maximum}")
    return {"rows": int(len(actual)), "maximum_absolute_difference": maximum, "status": "pass"}


def render(metrics: pd.DataFrame, outdir: Path) -> list[str]:
    for metric, label, ylim in METRICS:
        method_values = [
            metrics.loc[metrics["method_canonical"].eq(method), metric].to_numpy()
            for method in METHOD_ORDER
        ]
        visible_low = min(float(np.quantile(values, 0.05)) for values in method_values)
        visible_high = max(float(np.quantile(values, 0.95)) for values in method_values)
        if visible_low < ylim[0] or visible_high > ylim[1]:
            raise ValueError(
                f"{label} 5--95% whiskers ({visible_low:.6g}, {visible_high:.6g}) "
                f"fall outside the configured y-axis limits {ylim}"
            )

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.edgecolor": "#343b45", "axes.linewidth": 0.8,
    })
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.55), dpi=300)
    for index, (axis, (metric, label, ylim)) in enumerate(zip(axes.ravel(), METRICS)):
        data = [metrics.loc[metrics["method_canonical"].eq(method), metric].to_numpy()
                for method in METHOD_ORDER]
        boxes = axis.boxplot(
            data, patch_artist=True, widths=0.55, whis=(5, 95), showfliers=False,
            medianprops={"color": "#111827", "linewidth": 1.15},
            whiskerprops={"color": EDGE, "linewidth": 0.72},
            capprops={"color": EDGE, "linewidth": 0.72},
            boxprops={"edgecolor": EDGE, "linewidth": 0.78},
        )
        for patch, method in zip(boxes["boxes"], METHOD_ORDER):
            patch.set_facecolor(CLASS_COLORS[METHOD_CLASS[method]])
            patch.set_alpha(0.95 if method == "PerturbLDM" else 0.78)
        axis.set_title(label, fontsize=9.4, pad=5)
        axis.set_ylim(*ylim)
        axis.set_xlim(0.35, len(METHOD_ORDER) + 0.65)
        axis.set_xticks(range(1, len(METHOD_ORDER) + 1))
        if index >= 2:
            axis.set_xticklabels(
                [METHOD_LABELS[method] for method in METHOD_ORDER], rotation=30,
                ha="right", va="top", rotation_mode="anchor", fontsize=7.1,
            )
            axis.tick_params(axis="x", pad=4, length=3, width=0.7)
            for tick, method in zip(axis.get_xticklabels(), METHOD_ORDER):
                tick.set_x(tick.get_position()[0] + 0.16)
                if method == "PerturbLDM":
                    tick.set_color(CLASS_COLORS["PerturbLDM"])
                    tick.set_fontweight("bold")
        else:
            axis.set_xticklabels([])
            axis.tick_params(axis="x", length=0)
        for x_value in [1.5, 3.5, 4.5]:
            axis.axvline(x_value, color="#edf0f3", linewidth=0.7, zorder=0)
        axis.tick_params(axis="y", labelsize=7.2, length=3, width=0.7)
        axis.grid(axis="y", color=GRID, linestyle=(0, (3, 3)), linewidth=0.55)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    handles = [Patch(facecolor=CLASS_COLORS[name], edgecolor=EDGE, label=name)
               for name in ["PerturbLDM", "Simple marginal", "Matched control", "Learned baseline"]]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=6.8,
        handlelength=1.1,
        handletextpad=0.45,
        columnspacing=1.35,
    )
    fig.subplots_adjust(left=0.068, right=0.99, top=0.965, bottom=0.255,
                        wspace=0.20, hspace=0.28)
    outputs = []
    for extension, options in [("png", {"dpi": 300}), ("pdf", {}), ("svg", {})]:
        output = outdir / f"Fig2B_absolute_expression_benchmark.{extension}"
        fig.savefig(output, bbox_inches="tight", pad_inches=0.03, **options)
        outputs.append(str(output))
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(args.learned_table, args.simple_table)
    summary = summarize(metrics)
    summary_path = args.outdir / "Fig2B_absolute_expression_benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)
    report = {
        "condition_count_per_method": EXPECTED_CONDITIONS,
        "methods": METHOD_ORDER,
        "summary": str(summary_path),
    }
    if args.expected_summary:
        report["expected_summary_verification"] = verify_summary(summary, args.expected_summary)
    if not args.summary_only:
        report["figures"] = render(metrics, args.outdir)
    report["status"] = "pass"
    report_path = args.outdir / "Fig2B_absolute_expression_benchmark_validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
