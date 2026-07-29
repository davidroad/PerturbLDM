#!/usr/bin/env python3
"""Deterministic statistical audit for manuscript-facing Tahoe comparisons.

The scientific unit is the held-out treatment condition for the absolute-profile
benchmark and the predefined drug-dose or cell-line-dose group for the context-
specificity benchmark. Cells and genes are not treated as biological replicates.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_ndtr
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "statistics_revision_20260717"
SEED = 20260717
N_BOOTSTRAP = 2000
BOOTSTRAP_CHUNK = 64

ACTIVE_METRICS = ROOT / (
    "supplementary/benchmark/tahoe/metrics/corrected_test_pred_mean_metrics/"
    "ACTIVE_all_methods_condition_metrics_cpa_updated.csv"
)
PERTURBLDM_METRICS = ROOT / (
    "supplementary/benchmark/tahoe/metrics/corrected_test_pred_mean_metrics/"
    "perturbldm_condition_metrics_r2_chatterjee_corrected_clean.csv"
)
SIMPLE_METRICS = ROOT / (
    "supplementary/benchmark/tahoe/metrics/simple_baseline_pred_mean_metrics/"
    "simple_mean_baseline_condition_metrics.csv"
)
CONTEXT_IMPROVEMENTS = ROOT / (
    "supplementary/benchmark/tahoe/main_figure2/"
    "tahoe_context_specificity_paired_candidate_v1_improvements.csv"
)


def bootstrap_median_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    """Percentile bootstrap CI for the median of paired improvements."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    medians = np.empty(N_BOOTSTRAP, dtype=float)
    for start in range(0, N_BOOTSTRAP, BOOTSTRAP_CHUNK):
        stop = min(start + BOOTSTRAP_CHUNK, N_BOOTSTRAP)
        indices = rng.integers(
            0, values.size, size=(stop - start, values.size), dtype=np.int32
        )
        medians[start:stop] = np.median(values[indices], axis=1)
    return tuple(np.quantile(medians, [0.025, 0.975]))


def signed_rank_summary(values: np.ndarray) -> dict[str, float | int | str]:
    """Two-sided paired Wilcoxon summary with stable log-P reporting."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    nonzero = values[values != 0]
    if nonzero.size == 0:
        return {
            "n": int(values.size),
            "n_nonzero": 0,
            "wilcoxon_statistic": 0.0,
            "wilcoxon_z": 0.0,
            "p_value": 1.0,
            "log10_p_value": 0.0,
            "rank_biserial": 0.0,
            "test": "two-sided paired Wilcoxon signed-rank (asymptotic)",
        }

    result = wilcoxon(
        nonzero,
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        method="approx",
    )
    z_value = float(getattr(result, "zstatistic", math.nan))
    p_value = float(result.pvalue)
    if math.isfinite(z_value):
        log_p_natural = math.log(2.0) + float(log_ndtr(-abs(z_value)))
        log10_p = log_p_natural / math.log(10.0)
    elif p_value > 0:
        log10_p = math.log10(p_value)
    else:
        log10_p = -math.inf

    ranks = rankdata(np.abs(nonzero), method="average")
    w_positive = float(ranks[nonzero > 0].sum())
    w_negative = float(ranks[nonzero < 0].sum())
    rank_biserial = (w_positive - w_negative) / (w_positive + w_negative)
    return {
        "n": int(values.size),
        "n_nonzero": int(nonzero.size),
        "wilcoxon_statistic": float(result.statistic),
        "wilcoxon_z": z_value,
        "p_value": p_value,
        "log10_p_value": log10_p,
        "rank_biserial": rank_biserial,
        "test": "two-sided paired Wilcoxon signed-rank (asymptotic)",
    }


def bh_adjust_log10(log10_p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment in log space to avoid P-value underflow."""
    log10_p_values = np.asarray(log10_p_values, dtype=float)
    m = log10_p_values.size
    order = np.argsort(log10_p_values)
    sorted_values = log10_p_values[order]
    ranks = np.arange(1, m + 1, dtype=float)
    adjusted_sorted = sorted_values + np.log10(m / ranks)
    for index in range(m - 2, -1, -1):
        adjusted_sorted[index] = min(adjusted_sorted[index], adjusted_sorted[index + 1])
    adjusted_sorted = np.minimum(adjusted_sorted, 0.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def display_p(log10_p: float) -> str:
    if not math.isfinite(log10_p):
        return "<1e-300"
    if log10_p < -300:
        return "<1e-300"
    p_value = 10.0**log10_p
    return f"{p_value:.3g}"


def validate_unique_conditions(frame: pd.DataFrame, label: str) -> None:
    if frame["condition_key"].duplicated().any():
        duplicated = frame.loc[
            frame["condition_key"].duplicated(), "condition_key"
        ].head(5)
        raise ValueError(f"{label} contains duplicate condition keys: {duplicated.tolist()}")


def canonical_dose(value: object) -> str:
    """Repair known historical CSV/Excel dose encodings and return a numeric key."""
    text = str(value).strip()
    repaired = {
        "0-05": "0.05",
        "0-5": "0.5",
        "5-0": "5.0",
        "May-00": "5.0",
    }.get(text, text)
    try:
        numeric = float(repaired)
    except ValueError as error:
        raise ValueError(f"Unrecognised Tahoe dose encoding: {value!r}") from error
    return f"{numeric:g}"


def canonical_drug(value: object) -> str:
    text = str(value).strip()
    return {
        "Trametinib (DMSO-TF solvate)": "Trametinib (DMSO_TF solvate)",
    }.get(text, text)


def add_condition_key(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["dose_key"] = frame["dose"].map(canonical_dose)
    frame["condition_key"] = (
        frame["drug"].map(canonical_drug)
        + "\x1f"
        + frame["dose_key"]
        + "\x1f"
        + frame["cellname"].astype(str)
    )
    return frame


def infer_cell_identifier_mapping(
    perturb: pd.DataFrame, cpa: pd.DataFrame
) -> tuple[dict[str, str], pd.DataFrame]:
    """Infer and verify a CVCL-to-display-name bijection from condition incidence.

    Each of the 47 cell lines has a characteristic set of observed drug-dose
    combinations. A global one-to-one assignment maximises the Jaccard overlap
    between those sets without using metric values or row order. Exact matches,
    assigned overlap and the next-best margin are retained for audit.
    """
    perturb = perturb.copy()
    cpa = cpa.copy()
    perturb["dose_key"] = perturb["dose"].map(canonical_dose)
    cpa["dose_key"] = cpa["dose"].map(canonical_dose)

    def signatures(frame: pd.DataFrame) -> dict[str, frozenset[tuple[str, str]]]:
        return {
            str(cell): frozenset(
                zip(group["drug"].map(canonical_drug), group["dose_key"])
            )
            for cell, group in frame.groupby("cellname", sort=True)
        }

    display_signatures = signatures(perturb)
    cpa_signatures = signatures(cpa)
    cpa_cells = sorted(cpa_signatures)
    display_cells = sorted(display_signatures)
    if len(cpa_cells) != len(display_cells):
        raise ValueError(
            "CPA and PerturbLDM cell-line counts differ: "
            f"{len(cpa_cells)} vs {len(display_cells)}"
        )

    scores = np.empty((len(cpa_cells), len(display_cells)), dtype=float)
    intersections = np.empty_like(scores, dtype=int)
    unions = np.empty_like(scores, dtype=int)
    for row, cpa_cell in enumerate(cpa_cells):
        left = cpa_signatures[cpa_cell]
        for column, display_cell in enumerate(display_cells):
            right = display_signatures[display_cell]
            intersection = len(left & right)
            union = len(left | right)
            intersections[row, column] = intersection
            unions[row, column] = union
            scores[row, column] = intersection / union if union else 1.0

    row_indices, column_indices = linear_sum_assignment(-scores)
    mapping = {
        cpa_cells[row]: display_cells[column]
        for row, column in zip(row_indices, column_indices)
    }
    quality_records: list[dict[str, object]] = []
    for row, column in zip(row_indices, column_indices):
        row_scores = scores[row]
        alternatives = np.delete(row_scores, column)
        next_best = float(alternatives.max()) if alternatives.size else math.nan
        assigned = float(scores[row, column])
        quality_records.append(
            {
                "source_cell_identifier": cpa_cells[row],
                "cell_line": display_cells[column],
                "jaccard_drug_dose_incidence": assigned,
                "intersection_size": int(intersections[row, column]),
                "union_size": int(unions[row, column]),
                "exact_incidence_match": bool(assigned == 1.0),
                "next_best_jaccard": next_best,
                "assigned_margin": assigned - next_best,
                "mapping_basis": (
                    "global one-to-one maximum Jaccard assignment of drug-dose incidence sets"
                ),
            }
        )
    quality = pd.DataFrame.from_records(quality_records).sort_values(
        "source_cell_identifier"
    )
    minimum_score = float(quality["jaccard_drug_dose_incidence"].min())
    minimum_margin = float(quality["assigned_margin"].min())
    if minimum_score < 0.90 or minimum_margin <= 0:
        raise ValueError(
            "Cell-line incidence assignment did not meet audit thresholds: "
            f"minimum Jaccard={minimum_score:.6f}, minimum margin={minimum_margin:.6f}; "
            f"lowest rows={quality.nsmallest(5, 'jaccard_drug_dose_incidence').to_dict('records')}"
        )
    return mapping, quality


def load_absolute_profile_data() -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    csv_options = {"dtype": {"dose": "string"}, "low_memory": False}
    active = pd.read_csv(ACTIVE_METRICS, **csv_options)
    perturb = pd.read_csv(PERTURBLDM_METRICS, **csv_options)
    simple = pd.read_csv(SIMPLE_METRICS, **csv_options)

    active_counts = active.groupby("method").size().sort_index().to_dict()
    perturb_counts = perturb.groupby("method").size().sort_index().to_dict()
    simple_counts = simple.groupby("method_canonical").size().sort_index().to_dict()

    cpa_for_mapping = active.loc[active["method"].eq("CPA")].copy()
    cell_mapping, mapping_frame = infer_cell_identifier_mapping(perturb, cpa_for_mapping)

    # The dedicated corrected file is authoritative for PerturbLDM and avoids
    # relying on the historical raw label ("Diffusion") in the all-method file.
    perturb = perturb[
        ["drug", "dose", "cellname", "R2", "Pearson_r", "Spearman_r", "MAE"]
    ].copy()
    perturb["method_canonical"] = "PerturbLDM"
    perturb = add_condition_key(perturb)
    validate_unique_conditions(perturb, "PerturbLDM")

    selected_active = active.loc[
        active["method"].isin(["CPA", "chemCPA"]),
        ["drug", "dose", "cellname", "method", "R2", "Pearson_r", "Spearman_r", "MAE"],
    ].copy()
    selected_active = selected_active.rename(columns={"method": "method_canonical"})
    selected_active["cellname"] = selected_active["cellname"].map(cell_mapping)
    if selected_active["cellname"].isna().any():
        missing_codes = active.loc[
            active["method"].isin(["CPA", "chemCPA"])
            & ~active["cellname"].isin(cell_mapping),
            "cellname",
        ].drop_duplicates().tolist()
        raise ValueError(f"Unmapped CPA/chemCPA cell identifiers: {missing_codes}")
    selected_active = add_condition_key(selected_active)

    additive = simple.loc[
        simple["method_canonical"].eq("AdditiveMean"),
        ["drug", "dose", "cellname", "method_canonical", "R2", "Pearson_r", "Spearman_r", "MAE"],
    ].copy()
    additive = add_condition_key(additive)

    long_frame = pd.concat([perturb, selected_active, additive], ignore_index=True)
    for method, method_frame in long_frame.groupby("method_canonical"):
        validate_unique_conditions(method_frame, method)

    expected_methods = {"PerturbLDM", "AdditiveMean", "CPA", "chemCPA"}
    observed_methods = set(long_frame["method_canonical"])
    if observed_methods != expected_methods:
        raise ValueError(
            f"Expected methods {sorted(expected_methods)}, observed {sorted(observed_methods)}"
        )

    method_condition_sets = {
        method: set(frame["condition_key"])
        for method, frame in long_frame.groupby("method_canonical")
    }
    shared_conditions = set.intersection(*method_condition_sets.values())
    if len(shared_conditions) != 13_942:
        counts = {method: len(keys) for method, keys in method_condition_sets.items()}
        mismatch_examples = {
            method: {
                "not_in_shared_count": len(keys - shared_conditions),
                "not_in_shared_examples": sorted(keys - shared_conditions)[:10],
            }
            for method, keys in method_condition_sets.items()
        }
        raise ValueError(
            f"Expected 13,942 shared conditions, found {len(shared_conditions)}; "
            f"counts={counts}; mismatches={mismatch_examples}"
        )
    long_frame = long_frame[long_frame["condition_key"].isin(shared_conditions)].copy()

    metadata = {
        "active_method_counts": active_counts,
        "perturbldm_method_counts": perturb_counts,
        "simple_baseline_counts": simple_counts,
        "shared_condition_count": len(shared_conditions),
        "condition_key": "drug + numeric dose + mapped cell-line identifier",
        "cell_identifier_mapping_method": (
            "global one-to-one maximum Jaccard assignment of drug-dose incidence sets; "
            "metric values and row order unused"
        ),
        "cell_identifier_mapping_count": len(cell_mapping),
        "cell_identifier_mapping_minimum_jaccard": float(
            mapping_frame["jaccard_drug_dose_incidence"].min()
        ),
        "cell_identifier_mapping_minimum_margin": float(
            mapping_frame["assigned_margin"].min()
        ),
        "known_dose_repairs": {
            "0-05": "0.05",
            "0-5": "0.5",
            "5-0": "5.0",
            "May-00": "5.0",
        },
        "known_drug_name_repairs": {
            "leading_or_trailing_whitespace": "stripped",
            "Trametinib (DMSO-TF solvate)": "Trametinib (DMSO_TF solvate)",
        },
    }
    return long_frame, metadata, mapping_frame


def absolute_profile_statistics(long_frame: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "R2": "higher",
        "Pearson_r": "higher",
        "Spearman_r": "higher",
        "MAE": "lower",
    }
    baselines = ["AdditiveMean", "CPA", "chemCPA"]
    records: list[dict[str, object]] = []

    for metric_index, (metric, direction) in enumerate(metrics.items()):
        wide = long_frame.pivot(
            index="condition_key", columns="method_canonical", values=metric
        ).sort_index()
        if wide.isna().any().any():
            raise ValueError(f"Missing paired values in {metric}")
        for baseline_index, baseline in enumerate(baselines):
            if direction == "higher":
                improvement = (
                    wide["PerturbLDM"].to_numpy() - wide[baseline].to_numpy()
                )
            else:
                improvement = (
                    wide[baseline].to_numpy() - wide["PerturbLDM"].to_numpy()
                )
            test = signed_rank_summary(improvement)
            ci_low, ci_high = bootstrap_median_ci(
                improvement, SEED + 100 * metric_index + baseline_index
            )
            records.append(
                {
                    "comparison_family": "Tahoe absolute profiles: 3 baselines x 4 metrics",
                    "comparison": f"PerturbLDM vs {baseline}",
                    "baseline": baseline,
                    "metric": metric,
                    "direction_oriented": f"positive means PerturbLDM {direction}/better",
                    "analysis_unit": "held-out drug-dose-cell-line condition",
                    "paired": True,
                    "median_improvement": float(np.median(improvement)),
                    "median_improvement_ci95_low": ci_low,
                    "median_improvement_ci95_high": ci_high,
                    "mean_improvement": float(np.mean(improvement)),
                    "perturbldm_better_fraction": float(np.mean(improvement > 0)),
                    **test,
                }
            )

    result = pd.DataFrame.from_records(records)
    result["log10_p_bh"] = bh_adjust_log10(result["log10_p_value"].to_numpy())
    result["p_report"] = result["log10_p_value"].map(display_p)
    result["p_bh_report"] = result["log10_p_bh"].map(display_p)
    result["bootstrap_replicates"] = N_BOOTSTRAP
    result["bootstrap_seed"] = SEED
    return result


def context_specificity_statistics() -> tuple[pd.DataFrame, dict[str, int]]:
    frame = pd.read_csv(CONTEXT_IMPROVEMENTS)
    records: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    grouped = frame.groupby(["analysis", "analysis_label", "metric"], sort=True)
    for index, ((analysis, analysis_label, metric), group) in enumerate(grouped):
        values = group["improvement_vs_additive"].to_numpy(dtype=float)
        key = f"{analysis}|{metric}"
        counts[key] = int(values.size)
        test = signed_rank_summary(values)
        ci_low, ci_high = bootstrap_median_ci(values, SEED + 1000 + index)
        records.append(
            {
                "comparison_family": "Tahoe context specificity: 2 strata x 2 metrics",
                "analysis": analysis,
                "analysis_label": analysis_label,
                "metric": metric,
                "comparison": "PerturbLDM vs AdditiveMean",
                "direction_oriented": "positive means PerturbLDM better",
                "analysis_unit": (
                    "predefined drug-dose group" if "same_drug" in analysis
                    else "predefined cell-line-dose group"
                ),
                "paired": True,
                "median_improvement": float(np.median(values)),
                "median_improvement_ci95_low": ci_low,
                "median_improvement_ci95_high": ci_high,
                "mean_improvement": float(np.mean(values)),
                "perturbldm_better_fraction": float(np.mean(values > 0)),
                **test,
            }
        )

    result = pd.DataFrame.from_records(records)
    if len(result) != 4:
        raise ValueError(f"Expected four context-specificity tests, found {len(result)}")
    result["log10_p_bh"] = bh_adjust_log10(result["log10_p_value"].to_numpy())
    result["p_report"] = result["log10_p_value"].map(display_p)
    result["p_bh_report"] = result["log10_p_bh"].map(display_p)
    result["bootstrap_replicates"] = N_BOOTSTRAP
    result["bootstrap_seed"] = SEED
    return result, counts


def initial_audit_table() -> pd.DataFrame:
    rows = [
        {
            "task": "Tahoe",
            "comparison": "absolute-expression PerturbLDM vs AdditiveMean/CPA/chemCPA",
            "analysis_unit": "held-out condition",
            "n": 13942,
            "status": "complete",
            "reporting": "paired effect, bootstrap CI, two-sided Wilcoxon, BH across 12 tests",
            "reason": "raw paired condition-level values are present and condition keys match",
        },
        {
            "task": "Tahoe",
            "comparison": "matched-control effects vs AdditiveMean",
            "analysis_unit": "held-out condition",
            "n": 13942,
            "status": "descriptive_only",
            "reporting": "paired-win fraction and median/mean improvement",
            "reason": "package contains only pairwise summaries, not PerturbLDM paired vectors",
        },
        {
            "task": "Tahoe",
            "comparison": "context specificity vs AdditiveMean",
            "analysis_unit": "predefined drug-dose or cell-line-dose group",
            "n": "898 or 131 per stratum and metric",
            "status": "complete",
            "reporting": "group-level paired effect, bootstrap CI, two-sided Wilcoxon, BH across 4 tests",
            "reason": "group-level improvement vectors are present",
        },
        {
            "task": "Tahoe",
            "comparison": "signed Hallmark profile vs AdditiveMean",
            "analysis_unit": "held-out condition",
            "n": 13942,
            "status": "descriptive_only",
            "reporting": "paired-win fraction and median/mean improvement",
            "reason": "package contains only pairwise summaries, not PerturbLDM paired vectors",
        },
        {
            "task": "Tahoe",
            "comparison": "MMD-RBF and OT vs CPA/chemCPA",
            "analysis_unit": "held-out condition",
            "n": 13942,
            "status": "descriptive_only",
            "reporting": "paired-win fraction and median/mean reduction",
            "reason": "package contains only pairwise summaries, not raw paired vectors",
        },
        {
            "task": "Tahoe",
            "comparison": "Goserelin and Bortezomib examples",
            "analysis_unit": "single selected condition",
            "n": 1,
            "status": "illustrative",
            "reporting": "values and explicit case-study boundary; no inferential test",
            "reason": "single-condition examples are not method-level replicates",
        },
        {
            "task": "PANACEA",
            "comparison": "active broad-MoA top-1 endpoint",
            "analysis_unit": "evaluable query drug",
            "n": 27,
            "status": "complete_existing",
            "reporting": "14/27, expected 39.6%, one-sided Poisson-binomial P=0.121; descriptive",
            "reason": "active text and active Figure 3 use this endpoint",
        },
        {
            "task": "PANACEA",
            "comparison": "curated fine-MoA ssGSEA top-3 endpoint",
            "analysis_unit": "query (48) and drug (24)",
            "n": "48 queries; 24 drugs",
            "status": "version_conflict",
            "reporting": "do not merge into active story until endpoint provenance and figure version are resolved",
            "reason": "different labels, feature space, subset, endpoint and panel mapping from active Figure 3",
        },
        {
            "task": "Fetal colon",
            "comparison": "PerturbLDM vs Squidiff",
            "analysis_unit": "one held-out developmental dataset",
            "n": "800 genes; 1,529 cells",
            "status": "descriptive_only",
            "reporting": "dataset-level metrics and per-gene error distribution; avoid biological-replicate P values",
            "reason": "genes/cells are features/samples within one retrospective task, not independent biological replicates",
        },
        {
            "task": "PBMC",
            "comparison": "PerturbLDM/PerturbLDM-ctrl vs scGen",
            "analysis_unit": "held-out lineage",
            "n": 3,
            "status": "descriptive_only",
            "reporting": "per-lineage values and means; DEG/GO adjusted P values do not test method superiority",
            "reason": "three lineages provide insufficient task-level units for useful method inference",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    long_frame, metadata, cell_mapping = load_absolute_profile_data()
    absolute = absolute_profile_statistics(long_frame)
    context, context_counts = context_specificity_statistics()
    audit = initial_audit_table()

    absolute.to_csv(OUT / "tahoe_absolute_paired_statistics.csv", index=False)
    context.to_csv(OUT / "tahoe_context_specificity_statistics.csv", index=False)
    audit.to_csv(OUT / "comparison_statistical_audit.csv", index=False)
    cell_mapping.to_csv(OUT / "tahoe_cell_line_identifier_mapping.csv", index=False)

    run_metadata = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "seed": SEED,
        "bootstrap_replicates": N_BOOTSTRAP,
        "bootstrap_interval": "two-sided percentile 95% CI",
        "test": "two-sided paired Wilcoxon signed-rank, asymptotic normal approximation",
        "multiplicity": {
            "absolute_profile_family": "Benjamini-Hochberg across 12 prespecified tests",
            "context_specificity_family": "Benjamini-Hochberg across 4 prespecified tests",
        },
        "absolute_profile_sources": [
            str(ACTIVE_METRICS.relative_to(ROOT)),
            str(PERTURBLDM_METRICS.relative_to(ROOT)),
            str(SIMPLE_METRICS.relative_to(ROOT)),
        ],
        "context_specificity_source": str(CONTEXT_IMPROVEMENTS.relative_to(ROOT)),
        "absolute_profile_validation": metadata,
        "context_group_counts": context_counts,
        "interpretation_boundary": (
            "Conditions and predefined groups are the analysis units. Cells and genes "
            "are not treated as independent biological replicates."
        ),
    }
    (OUT / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("Absolute-profile method counts:")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print("\nAbsolute-profile statistics:")
    print(
        absolute[
            [
                "comparison",
                "metric",
                "n",
                "median_improvement",
                "median_improvement_ci95_low",
                "median_improvement_ci95_high",
                "perturbldm_better_fraction",
                "rank_biserial",
                "p_bh_report",
            ]
        ].to_string(index=False)
    )
    print("\nContext-specificity statistics:")
    print(
        context[
            [
                "analysis_label",
                "metric",
                "n",
                "median_improvement",
                "median_improvement_ci95_low",
                "median_improvement_ci95_high",
                "perturbldm_better_fraction",
                "rank_biserial",
                "p_bh_report",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
