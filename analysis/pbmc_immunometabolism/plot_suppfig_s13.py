#!/usr/bin/env python3
"""Recreate Supplementary Fig. S13a and S13b from released PBMC source tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CELL_TYPES = [
    ("B cells", "B cells"),
    ("CD8 T cells", "CD8 T"),
    ("FCGR3A+ Monocytes", "FCGR3A⁺ mono."),
]
PROFILES = ["Measured", "PerturbLDM", "scGen"]
PROGRAMS = ["FAO", "OXPHOS"]
COLORS = {"Measured": "#2B2B2B", "PerturbLDM": "#0F4D92", "scGen": "#C44E52"}
MARKERS = {"Measured": "D", "PerturbLDM": "o", "scGen": "^"}
OFFSETS = {"Measured": 0.0, "PerturbLDM": 0.12, "scGen": -0.12}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coordinates = pd.read_csv(args.source_dir / "pbmc_oxphos_umap_coordinates.csv")
    scores = pd.read_csv(args.source_dir / "pbmc_fao_oxphos_pathway_scores.csv")

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.5,
            "svg.fonttype": "path",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.2, 4.7))
    grid = fig.add_gridspec(
        2, 7, width_ratios=[1, 1, 1, 1, 1, 1, 0.10],
        height_ratios=[1.05, 0.95], hspace=0.48, wspace=0.48,
    )
    umap_axes = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]), fig.add_subplot(grid[0, 4:6])]
    color_axis = fig.add_subplot(grid[0, 6])
    vmin, vmax = np.quantile(coordinates["OXPHOS_score"], [0.02, 0.98])
    points = None
    for axis, profile in zip(umap_axes, PROFILES):
        subset = coordinates.loc[coordinates["profile"].eq(profile)]
        points = axis.scatter(
            subset["UMAP1"], subset["UMAP2"], c=subset["OXPHOS_score"],
            cmap="viridis", vmin=vmin, vmax=vmax, s=3.0, alpha=0.78,
            linewidth=0, rasterized=True,
        )
        axis.set_title(profile, fontweight="normal")
        for cell_type, short_label in [
            ("B cells", "B"), ("CD8 T cells", "CD8 T"),
            ("FCGR3A+ Monocytes", "FCGR3A⁺ mono."),
        ]:
            lineage = subset.loc[subset["cell_type"].eq(cell_type)]
            label = axis.text(
                lineage["UMAP1"].median(), lineage["UMAP2"].median(), short_label,
                fontsize=5.7, ha="center", va="center", color="#202020", zorder=5,
            )
            label.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    colorbar = fig.colorbar(points, cax=color_axis)
    colorbar.set_label("OXPHOS score")
    colorbar.outline.set_linewidth(0.4)

    score_axes = [fig.add_subplot(grid[1, 0:3]), fig.add_subplot(grid[1, 3:6])]
    limits = {"FAO": (-0.016, 0.013), "OXPHOS": (-0.078, 0.016)}
    y_positions = [2, 1, 0]
    for axis, program in zip(score_axes, PROGRAMS):
        subset = scores.loc[scores["program"].eq(program)].pivot(
            index="cell_type", columns="profile", values="matched_control_effect"
        )
        for y, (cell_type, _) in zip(y_positions, CELL_TYPES):
            measured = subset.loc[cell_type, "Measured"]
            for method in ["PerturbLDM", "scGen"]:
                predicted = subset.loc[cell_type, method]
                axis.plot(
                    [measured, predicted], [y, y + OFFSETS[method]],
                    color=COLORS[method], alpha=0.35, linewidth=0.7, zorder=1,
                )
            for profile in PROFILES:
                axis.scatter(
                    subset.loc[cell_type, profile], y + OFFSETS[profile], s=22,
                    marker=MARKERS[profile], color=COLORS[profile],
                    edgecolor="white" if profile != "Measured" else COLORS[profile],
                    linewidth=0.5, zorder=3,
                )
        axis.axvline(0, color="#777777", linestyle=(0, (2, 2)), linewidth=0.5)
        axis.set_xlim(*limits[program])
        axis.set_title(program, fontweight="normal")
        axis.set_xlabel("Effect vs matched control")
        axis.set_yticks(y_positions)
        axis.set_yticklabels([label for _, label in CELL_TYPES] if axis is score_axes[0] else ["", "", ""])
        axis.grid(axis="x", color="#E7E7E7", linewidth=0.5)
        axis.tick_params(axis="y", length=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_linewidth(0.5)

    handles = [
        mpl.lines.Line2D(
            [], [], marker=MARKERS[profile], linestyle="none",
            markerfacecolor=COLORS[profile],
            markeredgecolor="white" if profile != "Measured" else COLORS[profile],
            markeredgewidth=0.4, markersize=4.2,
            label="GT" if profile == "Measured" else profile,
        )
        for profile in PROFILES
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.005),
        ncol=3, frameon=False, handletextpad=0.25, columnspacing=0.8,
    )
    fig.text(0.015, 0.965, "a", fontsize=8.8)
    fig.text(0.015, 0.47, "b", fontsize=8.8)
    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.94)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "supplementary_figure_s13"
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
