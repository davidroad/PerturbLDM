#!/usr/bin/env python3
"""Recreate the FAO--OXPHOS comparison shown in manuscript Fig. 5h."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


CELL_TYPES = [
    ("B cells", "B cells"),
    ("CD8 T cells", "CD8 T"),
    ("FCGR3A+ Monocytes", "FCGR3A⁺ mono."),
]
METHODS = ["Measured", "PerturbLDM", "scGen"]
COLORS = {"Measured": "#2B2B2B", "PerturbLDM": "#0F4D92", "scGen": "#C44E52"}
MARKERS = {"Measured": "D", "PerturbLDM": "o", "scGen": "^"}
METHOD_Y_OFFSETS = {"Measured": 0.0, "PerturbLDM": 0.20, "scGen": -0.20}
RATIO_OFFSETS = {"Effect error": -0.11, "Distribution W1": 0.11}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def save_all(fig, base: Path) -> None:
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".png"), dpi=350, bbox_inches="tight", pad_inches=0.02)


def main() -> None:
    args = parse_args()
    scores = pd.read_csv(args.source_dir / "pbmc_fao_oxphos_pathway_scores.csv")
    ratios = pd.read_csv(
        args.source_dir / "pbmc_fao_oxphos_composite_fidelity_ratios.csv"
    ).set_index("cell_type")
    cell_order = [cell_type for cell_type, _ in CELL_TYPES]

    def program_wide(program: str) -> pd.DataFrame:
        return (
            scores.loc[scores["program"].eq(program)]
            .pivot(index="cell_type", columns="profile", values="matched_control_effect")
            .loc[cell_order]
        )

    composite = (program_wide("FAO") + program_wide("OXPHOS")) / 2
    if not (
        ratios["composite_effect_error_ratio"].lt(1).all()
        and ratios["composite_score_wasserstein_ratio"].lt(1).all()
    ):
        raise ValueError("One or more manuscript Fig. 5h ratios are not below 1")

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 5.7,
            "axes.titlesize": 6.0,
            "axes.labelsize": 5.7,
            "xtick.labelsize": 5.2,
            "ytick.labelsize": 5.5,
            "legend.fontsize": 4.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
        }
    )
    fig, axes = plt.subplots(
        2, 1, figsize=(2.55, 2.05), gridspec_kw={"height_ratios": [1.0, 1.0]}
    )
    y_positions = [2, 1, 0]

    top = axes[0]
    for y, (cell_type, _) in zip(y_positions, CELL_TYPES):
        measured = composite.loc[cell_type, "Measured"]
        for method in ["PerturbLDM", "scGen"]:
            predicted = composite.loc[cell_type, method]
            top.plot(
                [measured, predicted],
                [y, y + METHOD_Y_OFFSETS[method]],
                color=COLORS[method],
                alpha=0.35,
                linewidth=0.55,
                zorder=1,
            )
        for method in METHODS:
            top.scatter(
                composite.loc[cell_type, method],
                y + METHOD_Y_OFFSETS[method],
                s=19,
                marker=MARKERS[method],
                color=COLORS[method],
                edgecolor="white" if method != "Measured" else COLORS[method],
                linewidth=0.4,
                zorder=3,
            )
    top.axvline(0, color="#888888", linestyle=(0, (2, 2)), linewidth=0.45)
    top.set_xlim(-0.05, 0.012)
    top.set_xticks([-0.04, -0.02, 0, 0.01], ["−.04", "−.02", "0", ".01"])
    top.set_yticks(y_positions, [label for _, label in CELL_TYPES])
    top.set_ylim(-0.50, 2.50)
    top.set_xlabel("FAO–OXPHOS effect vs matched control", labelpad=2)
    top.grid(axis="x", color="#E7E7E7", linewidth=0.4)
    top.tick_params(axis="y", length=0)

    method_handles = [
        mpl.lines.Line2D(
            [], [], marker=MARKERS[method], linestyle="none",
            markerfacecolor=COLORS[method],
            markeredgecolor="white" if method != "Measured" else COLORS[method],
            markeredgewidth=0.3, markersize=3.4,
            label="GT" if method == "Measured" else method,
        )
        for method in METHODS
    ]
    top.legend(
        handles=method_handles, loc="lower center", bbox_to_anchor=(0.50, 1.02),
        ncol=3, frameon=False, handletextpad=0.15, columnspacing=0.25,
    )

    bottom = axes[1]
    metrics = [
        ("Effect error", "composite_effect_error_ratio", "o"),
        ("Distribution W1", "composite_score_wasserstein_ratio", "s"),
    ]
    for y, (cell_type, _) in zip(y_positions, CELL_TYPES):
        for label, column, marker in metrics:
            ratio = ratios.loc[cell_type, column]
            marker_y = y + RATIO_OFFSETS[label]
            bottom.plot([ratio, 1], [marker_y, y], color="#C7C7C7", linewidth=0.55)
            bottom.scatter(
                ratio, marker_y, s=18, marker=marker, color=COLORS["PerturbLDM"],
                edgecolor="white", linewidth=0.4, zorder=3,
            )
        bottom.scatter(
            1, y, s=19, marker="^", color=COLORS["scGen"], edgecolor="white",
            linewidth=0.4, zorder=4, clip_on=False,
        )
    bottom.axvline(1, color=COLORS["scGen"], linestyle=(0, (2, 2)), linewidth=0.45, alpha=0.55)
    bottom.set_xlim(0.25, 1.08)
    bottom.set_xticks([0.4, 0.6, 0.8, 1.0], [".4", ".6", ".8", "1.0"])
    bottom.set_yticks(y_positions, [label for _, label in CELL_TYPES])
    bottom.set_ylim(-0.45, 2.45)
    bottom.set_xlabel("PerturbLDM/scGen error ratio (lower is better)", labelpad=2)
    bottom.grid(axis="x", color="#E7E7E7", linewidth=0.45)
    bottom.tick_params(axis="y", length=0)
    ratio_handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="none", color=COLORS["PerturbLDM"], label="Effect error"),
        mpl.lines.Line2D([], [], marker="s", linestyle="none", color=COLORS["PerturbLDM"], label="Distribution W1"),
        mpl.lines.Line2D([], [], marker="^", linestyle="none", color=COLORS["scGen"], label="scGen reference (=1)"),
    ]
    bottom.legend(
        handles=ratio_handles, loc="lower center", bbox_to_anchor=(0.50, 1.02),
        ncol=3, frameon=False, handletextpad=0.15, columnspacing=0.35,
    )

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_linewidth(0.5)
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.12, top=0.94, hspace=0.92)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "figure5h_fao_oxphos"
    save_all(fig, output)
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
