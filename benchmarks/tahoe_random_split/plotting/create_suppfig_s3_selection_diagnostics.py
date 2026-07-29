#!/usr/bin/env python3
"""Plot model-specific selection diagnostics for Tahoe learned comparators."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import ScalarFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_SOURCE = RELEASE_ROOT / "figshare" / "inputs" / "plotting_inputs" / "tahoe" / "suppfig_s3_selection_diagnostics"
DEFAULT_OUTPUT = RELEASE_ROOT / "figshare" / "derived" / "validation" / "suppfig_s3_selection_diagnostics"
SOURCE = DEFAULT_SOURCE
ASSEMBLY = DEFAULT_OUTPUT
PANEL_FONT = "DejaVu Sans"
LIBERATION_FONT = Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf")
if LIBERATION_FONT.is_file():
    font_manager.fontManager.addfont(str(LIBERATION_FONT))
    PANEL_FONT = "Liberation Sans"

COLORS = {
    "training": "#2E7D61",
    "validation": "#B85C38",
    "selection": "#7A5195",
    "neutral": "#5F6B76",
    "grid": "#DCE3E8",
}


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5, linestyle=(0, (2, 2)))
    ax.tick_params(length=2.6, width=0.65)


def panel_label(ax, label, x=-0.15, y=1.08):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=20,
        fontfamily=PANEL_FONT,
        fontweight="normal",
        va="top",
    )


def load_chemcpa_reconstruction():
    rows = pd.read_csv(SOURCE / "chemcpa_training_diagnostics_source.csv")
    selected = rows[
        (rows["model"] == "chemCPA")
        & (rows["metric"] == "reconstruction_mse")
        & (rows["split"].isin(["train", "validation"]))
    ].copy()
    selected["epoch"] = selected["epoch"].astype(int)
    selected["value"] = selected["value"].astype(float)
    return selected.sort_values(["split", "epoch"])


def plot_mlp(ax):
    history = pd.read_csv(SOURCE / "mlp_final_training_history.csv")
    best_index = history["validation_delta_mse"].idxmin()
    best_epoch = int(history.loc[best_index, "epoch_one_based"])
    best_value = float(history.loc[best_index, "validation_delta_mse"])

    ax.plot(
        history["epoch_one_based"],
        history["training_delta_mse"],
        color=COLORS["training"],
        linewidth=1.35,
        label="Training",
    )
    ax.plot(
        history["epoch_one_based"],
        history["validation_delta_mse"],
        color=COLORS["validation"],
        linewidth=1.0,
        alpha=0.9,
        label="Validation",
    )
    ax.scatter(
        [best_epoch],
        [best_value],
        s=44,
        facecolors="white",
        edgecolors=COLORS["selection"],
        linewidths=1.4,
        zorder=5,
    )
    ax.axvline(best_epoch, color=COLORS["selection"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_xlim(0, 101)
    ax.set_xticks([1, 25, 50, 75, 100])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\Delta$-expression MSE")
    ax.set_title("MLP", loc="left", fontweight="normal", pad=5)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    style_axis(ax)
    panel_label(ax, "e")
    return {
        "method": "MLP",
        "selection_unit": "epoch",
        "selected": best_epoch,
        "training_metric": "delta-expression MSE",
        "training_value": float(history.loc[best_index, "training_delta_mse"]),
        "validation_metric": "delta-expression MSE",
        "validation_value": best_value,
        "selection_rule": "minimum validation delta-expression MSE",
    }


def plot_rf(ax):
    search = pd.read_csv(SOURCE / "rf_validation_search.csv").sort_values(
        "min_samples_leaf"
    )
    best_index = search["formal_validation_delta_mse"].idxmin()
    best_leaf = int(search.loc[best_index, "min_samples_leaf"])
    best_value = float(search.loc[best_index, "formal_validation_delta_mse"])

    ax.plot(
        search["min_samples_leaf"],
        search["formal_validation_delta_mse"],
        color=COLORS["neutral"],
        linewidth=1.0,
        zorder=2,
    )
    ax.scatter(
        search["min_samples_leaf"],
        search["formal_validation_delta_mse"],
        s=38,
        color=COLORS["neutral"],
        zorder=3,
    )
    ax.scatter(
        [best_leaf],
        [best_value],
        s=66,
        facecolors="white",
        edgecolors=COLORS["selection"],
        linewidths=1.5,
        zorder=4,
    )
    for _, row in search.iterrows():
        ax.annotate(
            f"{row['formal_validation_delta_mse']:.9f}",
            (row["min_samples_leaf"], row["formal_validation_delta_mse"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
        )
    span = float(
        search["formal_validation_delta_mse"].max()
        - search["formal_validation_delta_mse"].min()
    )
    pad = max(span * 0.45, 0.00000004)
    ax.set_ylim(
        search["formal_validation_delta_mse"].min() - pad,
        search["formal_validation_delta_mse"].max() + pad * 1.7,
    )
    ax.set_xticks(search["min_samples_leaf"].astype(int))
    ax.set_xlabel("Minimum samples per leaf")
    ax.set_ylabel(r"Validation $\Delta$-expression MSE")
    ax.set_title("Random forest", loc="left", fontweight="normal", pad=5)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, -3))
    formatter.set_useOffset(False)
    ax.yaxis.set_major_formatter(formatter)
    style_axis(ax)
    panel_label(ax, "f")
    return {
        "method": "Random forest",
        "selection_unit": "minimum samples per leaf",
        "selected": best_leaf,
        "training_metric": "",
        "training_value": "",
        "validation_metric": "delta-expression MSE",
        "validation_value": best_value,
        "selection_rule": "minimum validation delta-expression MSE",
    }


def plot_cpa(parent_spec, fig):
    inner = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=parent_spec, height_ratios=[1.05, 0.95], hspace=0.16
    )
    ax_loss = fig.add_subplot(inner[0])
    ax_metric = fig.add_subplot(inner[1], sharex=ax_loss)

    loss = pd.read_csv(SOURCE / "cpa_training_reconstruction_history.csv")
    metric = pd.read_csv(SOURCE / "cpa_validation_selection_metric.csv")
    best_index = metric["cpa_metric"].idxmax()
    best_epoch = int(metric.loc[best_index, "epoch_one_based"])
    best_metric = float(metric.loc[best_index, "cpa_metric"])

    ax_loss.plot(
        loss["epoch"],
        loss["train_recon_loss"],
        color=COLORS["training"],
        marker="o",
        markersize=3.0,
        linewidth=1.2,
        label="Training",
    )
    ax_loss.plot(
        loss["epoch"],
        loss["validation_recon_loss"],
        color=COLORS["validation"],
        marker="s",
        markersize=3.0,
        linewidth=1.2,
        label="Validation",
    )
    ax_loss.axvline(best_epoch, color=COLORS["selection"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax_loss.set_ylabel("Reconstruction loss")
    ax_loss.set_title("CPA", loc="left", fontweight="normal", pad=5)
    ax_loss.legend(frameon=False, loc="upper right", ncol=2)
    ax_loss.tick_params(labelbottom=False)
    style_axis(ax_loss)
    panel_label(ax_loss, "g")

    ax_metric.plot(
        metric["epoch_one_based"],
        metric["cpa_metric"],
        color=COLORS["selection"],
        marker="o",
        markersize=3.2,
        linewidth=1.25,
    )
    ax_metric.scatter(
        [best_epoch],
        [best_metric],
        s=48,
        facecolors="white",
        edgecolors=COLORS["selection"],
        linewidths=1.4,
        zorder=4,
    )
    ax_metric.axvline(best_epoch, color=COLORS["selection"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax_metric.set_xlim(0.7, 8.3)
    ax_metric.set_xticks(range(1, 9))
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("Validation CPA metric")
    style_axis(ax_metric)

    chosen_loss = loss.loc[loss["epoch"] == best_epoch].iloc[0]
    return {
        "method": "CPA",
        "selection_unit": "epoch",
        "selected": best_epoch,
        "training_metric": "reconstruction loss",
        "training_value": float(chosen_loss["train_recon_loss"]),
        "validation_metric": "CPA metric",
        "validation_value": best_metric,
        "selection_rule": "maximum validation CPA metric",
    }


def plot_chemcpa(ax):
    history = load_chemcpa_reconstruction()
    training = history[history["split"] == "train"]
    validation = history[history["split"] == "validation"]
    best_index = validation["value"].idxmin()
    best_epoch = int(validation.loc[best_index, "epoch"])
    best_value = float(validation.loc[best_index, "value"])

    ax.plot(
        training["epoch"],
        training["value"],
        color=COLORS["training"],
        marker="o",
        markersize=4.0,
        linewidth=1.35,
        label="Training",
    )
    ax.plot(
        validation["epoch"],
        validation["value"],
        color=COLORS["validation"],
        marker="s",
        markersize=4.0,
        linewidth=1.35,
        label="Validation",
    )
    ax.scatter(
        [best_epoch],
        [best_value],
        s=58,
        facecolors="white",
        edgecolors=COLORS["selection"],
        linewidths=1.4,
        zorder=5,
    )
    ax.axvline(best_epoch, color=COLORS["selection"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_xlim(0.7, 5.3)
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("chemCPA", loc="left", fontweight="normal", pad=5)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    style_axis(ax)
    panel_label(ax, "h")

    training_value = float(
        training.loc[training["epoch"] == best_epoch, "value"].iloc[0]
    )
    return {
        "method": "chemCPA",
        "selection_unit": "epoch",
        "selected": best_epoch,
        "training_metric": "reconstruction MSE",
        "training_value": training_value,
        "validation_metric": "reconstruction MSE",
        "validation_value": best_value,
        "selection_rule": "minimum validation reconstruction MSE",
    }


def save_summary(rows):
    path = ASSEMBLY / "baseline_selection_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Recreate Supplementary Fig. S3e-h from compact validation histories."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()



def main():
    global SOURCE, ASSEMBLY
    args = parse_args()
    SOURCE = args.source_dir.resolve()
    ASSEMBLY = args.output_dir.resolve()
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.labelsize": 9.2,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    ASSEMBLY.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.15, 6.6))
    outer = GridSpec(
        2,
        2,
        figure=fig,
        left=0.10,
        right=0.97,
        bottom=0.09,
        top=0.96,
        hspace=0.34,
        wspace=0.30,
    )

    rows = []
    rows.append(plot_mlp(fig.add_subplot(outer[0, 0])))
    rows.append(plot_rf(fig.add_subplot(outer[0, 1])))
    rows.append(plot_cpa(outer[1, 0], fig))
    rows.append(plot_chemcpa(fig.add_subplot(outer[1, 1])))
    save_summary(rows)

    output = ASSEMBLY / "supple3_baseline_selection_diagnostics"
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
