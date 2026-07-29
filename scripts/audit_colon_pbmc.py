#!/usr/bin/env python3
"""Audit task-specific fetal-colon and PBMC comparisons without pseudoreplication."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


RELEASE_ROOT = Path(__file__).resolve().parents[2]
OUT = RELEASE_ROOT / "figshare/derived/audits"
SEED = 20260717
N_BOOTSTRAP = 2000
BOOTSTRAP_CHUNK = 128

COLON_FEATURES = (
    RELEASE_ROOT / "figshare/inputs/plotting_inputs/colon/all_800_hvg_gene_performance_candidates.csv"
)
COLON_UMAP = (
    RELEASE_ROOT
    / "figshare/inputs/plotting_inputs/colon/source_bundle_20260426/"
    "development_umap_rerun/umap_coordinates.csv"
)
PBMC_COUNTS = RELEASE_ROOT / "figshare/inputs/plotting_inputs/pbmc/pbmc_evaluation_cell_counts.csv"
PBMC_EXPRESSION = (
    RELEASE_ROOT / "figshare/inputs/plotting_inputs/pbmc/pbmc_expression_metrics_by_cell_type.csv"
)
PBMC_DEG = (
    RELEASE_ROOT / "figshare/inputs/plotting_inputs/pbmc/pbmc_deg_metrics_by_cell_type.csv"
)


def bootstrap_median_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    estimates = np.empty(N_BOOTSTRAP, dtype=float)
    for start in range(0, N_BOOTSTRAP, BOOTSTRAP_CHUNK):
        stop = min(start + BOOTSTRAP_CHUNK, N_BOOTSTRAP)
        indices = rng.integers(
            0, values.size, size=(stop - start, values.size), dtype=np.int32
        )
        estimates[start:stop] = np.median(values[indices], axis=1)
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def audit_colon() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    features = pd.read_csv(COLON_FEATURES)
    if len(features) != 800 or features["gene"].nunique() != 800:
        raise ValueError(
            f"Expected 800 unique colon genes, found {len(features)} rows and "
            f"{features['gene'].nunique()} genes"
        )

    metric_definitions = [
        (
            "pseudo-bulk absolute mean-expression error",
            "pert_mean_error",
            "squidiff_mean_error",
            "improvement_mean_error_positive_better",
        ),
        (
            "single-gene Wasserstein distance",
            "pert_wasserstein",
            "squidiff_wasserstein",
            "improvement_wasserstein_positive_better",
        ),
        (
            "PCW-stratified mean-expression MAE",
            "pert_pcw_mean_mae",
            "squidiff_pcw_mean_mae",
            "pcw_mean_mae_improvement_positive_better",
        ),
    ]
    records: list[dict[str, object]] = []
    for index, (label, perturb_column, squidiff_column, improvement_column) in enumerate(
        metric_definitions
    ):
        improvement = features[improvement_column].to_numpy(dtype=float)
        finite = np.isfinite(improvement)
        improvement = improvement[finite]
        ci_low, ci_high = bootstrap_median_ci(improvement, SEED + index)
        records.append(
            {
                "metric": label,
                "analysis_unit": "gene feature in the prespecified 800-HVG space",
                "n_genes": int(improvement.size),
                "perturbldm_median": float(
                    np.nanmedian(features.loc[finite, perturb_column])
                ),
                "squidiff_median": float(
                    np.nanmedian(features.loc[finite, squidiff_column])
                ),
                "median_reduction_perturbldm_vs_squidiff": float(
                    np.median(improvement)
                ),
                "median_reduction_ci95_low": ci_low,
                "median_reduction_ci95_high": ci_high,
                "perturbldm_lower_error_fraction": float(np.mean(improvement > 0)),
                "bootstrap_replicates": N_BOOTSTRAP,
                "bootstrap_seed": SEED,
                "inference_boundary": (
                    "feature-level uncertainty over the selected 800 genes; "
                    "not biological-replicate inference"
                ),
                "p_value": "not reported",
            }
        )

    tolerance = 0.1
    tolerance_summary = pd.DataFrame(
        [
            {
                "method": "PerturbLDM",
                "absolute_error_tolerance": tolerance,
                "n_genes_within_tolerance": int(
                    (features["pert_mean_error"] <= tolerance).sum()
                ),
                "n_genes": len(features),
                "fraction_within_tolerance": float(
                    (features["pert_mean_error"] <= tolerance).mean()
                ),
            },
            {
                "method": "Squidiff",
                "absolute_error_tolerance": tolerance,
                "n_genes_within_tolerance": int(
                    (features["squidiff_mean_error"] <= tolerance).sum()
                ),
                "n_genes": len(features),
                "fraction_within_tolerance": float(
                    (features["squidiff_mean_error"] <= tolerance).mean()
                ),
            },
        ]
    )

    coordinates = pd.read_csv(COLON_UMAP)
    held_out = coordinates.loc[
        coordinates["domain"].eq("Ground truth") & coordinates["split"].eq("test")
    ].copy()
    stage_counts = (
        held_out.groupby("pcw", as_index=False)
        .size()
        .rename(columns={"size": "n_held_out_cells"})
        .sort_values("pcw")
    )
    if int(stage_counts["n_held_out_cells"].sum()) != 1529:
        raise ValueError(
            "Colon held-out count does not reproduce 1,529: "
            f"{stage_counts.to_dict('records')}"
        )

    metadata = {
        "feature_source": str(COLON_FEATURES.relative_to(RELEASE_ROOT)),
        "cell_count_source": str(COLON_UMAP.relative_to(RELEASE_ROOT)),
        "held_out_cell_total": 1529,
        "held_out_stage_counts": stage_counts.to_dict("records"),
        "interpretation_boundary": (
            "The benchmark contains one held-out developmental dataset. "
            "Cells and genes are not treated as independent biological replicates."
        ),
    }
    return pd.DataFrame.from_records(records), tolerance_summary, metadata


def audit_pbmc() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    test_cell_types = ["B cells", "CD8 T cells", "FCGR3A+ Monocytes"]
    counts = pd.read_csv(PBMC_COUNTS)
    required_count_columns = {
        "cell_type", "n_control_cells", "n_stimulated_cells",
        "split_role_control", "split_role_stimulated",
    }
    if not required_count_columns.issubset(counts.columns):
        raise ValueError("PBMC count table does not satisfy the release schema")
    counts = counts.set_index("cell_type").reindex(test_cell_types).reset_index()
    if counts[["n_control_cells", "n_stimulated_cells"]].isna().any().any():
        raise ValueError(f"PBMC control/stim counts are incomplete: {counts}")

    expression = pd.read_csv(PBMC_EXPRESSION)
    deg = pd.read_csv(PBMC_DEG)
    method_map = {
        "scGen": "scGen",
        "PertDiffU-test": "PerturbLDM",
        "PertDiffU-ctrl": "PerturbLDM-ctrl",
    }
    expression = expression.loc[expression["method"].isin(method_map)].copy()
    deg = deg.loc[deg["method"].isin(method_map)].copy()
    expression["method"] = expression["method"].map(method_map)
    deg["method"] = deg["method"].map(method_map)

    expression_long = expression.melt(
        id_vars=["cell_type", "method"],
        value_vars=["Pearson", "Spearman", "R2", "RMSE", "MAE"],
        var_name="metric",
        value_name="value",
    )
    deg_long = deg.melt(
        id_vars=["cell_type", "method"],
        value_vars=["Recall", "Precision", "Specificity", "F1"],
        var_name="metric",
        value_name="value",
    )
    deg_long["metric"] = "DEG_" + deg_long["metric"]
    combined = pd.concat([expression_long, deg_long], ignore_index=True)

    records: list[dict[str, object]] = []
    directions = {
        "Pearson": "higher",
        "Spearman": "higher",
        "R2": "higher",
        "RMSE": "lower",
        "MAE": "lower",
        "DEG_Recall": "higher",
        "DEG_Precision": "higher",
        "DEG_Specificity": "higher",
        "DEG_F1": "higher",
    }
    for metric, metric_frame in combined.groupby("metric", sort=True):
        wide = metric_frame.pivot(index="cell_type", columns="method", values="value")
        for method in ["PerturbLDM", "PerturbLDM-ctrl"]:
            difference = (
                wide[method] - wide["scGen"]
                if directions[metric] == "higher"
                else wide["scGen"] - wide[method]
            )
            for cell_type, oriented_difference in difference.items():
                records.append(
                    {
                        "comparison": f"{method} vs scGen",
                        "cell_type": cell_type,
                        "metric": metric,
                        "direction_oriented": "positive means PerturbLDM variant better",
                        "oriented_difference": float(oriented_difference),
                        "analysis_unit": "held-out lineage",
                        "method_level_inference": "descriptive only (n=3 lineages)",
                    }
                )

    metadata = {
        "cell_count_source": str(PBMC_COUNTS.relative_to(RELEASE_ROOT)),
        "expression_metric_source": str(PBMC_EXPRESSION.relative_to(RELEASE_ROOT)),
        "deg_metric_source": str(PBMC_DEG.relative_to(RELEASE_ROOT)),
        "test_cell_types": test_cell_types,
        "interpretation_boundary": (
            "Three held-out lineages are insufficient for useful task-level "
            "method inference. DEG/GO adjusted P values define within-profile "
            "features and do not test superiority over scGen."
        ),
    }
    return counts, pd.DataFrame.from_records(records), metadata


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    colon_errors, colon_tolerance, colon_metadata = audit_colon()
    pbmc_counts, pbmc_comparisons, pbmc_metadata = audit_pbmc()

    colon_errors.to_csv(OUT / "colon_feature_level_error_summary.csv", index=False)
    colon_tolerance.to_csv(OUT / "colon_error_tolerance_summary.csv", index=False)
    pbmc_counts.to_csv(OUT / "pbmc_evaluation_cell_counts.csv", index=False)
    pbmc_comparisons.to_csv(
        OUT / "pbmc_descriptive_method_comparisons.csv", index=False
    )
    (OUT / "colon_pbmc_audit_metadata.json").write_text(
        json.dumps(
            {"colon": colon_metadata, "pbmc": pbmc_metadata},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("Colon feature-level uncertainty:")
    print(colon_errors.to_string(index=False))
    print("\nColon tolerance summary:")
    print(colon_tolerance.to_string(index=False))
    print("\nPBMC evaluation cell counts:")
    print(pbmc_counts.to_string(index=False))


if __name__ == "__main__":
    main()
