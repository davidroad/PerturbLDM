#!/usr/bin/env python3
"""Freeze and analyse Tahoe condition-paired effect/program/distribution vectors.

Scientific contract
-------------------
* One row/pair is one held-out drug-dose-cell-line condition.
* Effect endpoints are matched-control-relative upstream quantities.
* Hallmark endpoints are signed program-effect scores/profiles, not enrichment.
* Distribution endpoints are condition-paired MMD/OT distances.
* Positive ``improvement`` always means PerturbLDM is better.

The script is location-relative and writes only beside itself in the staging
workspace. Canonical upstream files are read-only. Exact byte copies of the
two full canonical vector tables are frozen under ``source_vectors/`` before
analysis, and their SHA-256 hashes must match the upstream files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.special import log_ndtr
from scipy.stats import rankdata, wilcoxon


SEED = 20260718
N_BOOTSTRAP = 2000
BOOTSTRAP_CHUNK = 64
EXPECTED_CONDITIONS = 13942


def find_project_root(start):
    """Find the project root without relying on a machine-specific path."""
    for candidate in [start] + list(start.parents):
        if (candidate / "reproducibility").is_dir() and (candidate / "manuscript").is_dir():
            return candidate
    raise RuntimeError("Could not locate project root containing reproducibility/ and manuscript/")


OUT = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(OUT)
SOURCE_VECTOR_DIR = OUT / "source_vectors"

EFFECT_SOURCE = PROJECT_ROOT / (
    "reproducibility/ly/tahoe_working/effect_program_baselines/"
    "effect_program_condition_metrics.csv"
)
DIST_SOURCE = PROJECT_ROOT / (
    "reproducibility/ly/tahoe_working/distribution_metric_audit_only/"
    "all_methods_distribution_metrics_condition_aligned.csv"
)
DIST_ALIGNMENT_AUDIT = PROJECT_ROOT / (
    "reproducibility/ly/tahoe_working/distribution_metric_audit_only/"
    "distribution_metric_condition_alignment_audit.csv"
)
DIST_ACTIVE_UPSTREAM = PROJECT_ROOT / (
    "reproducibility/ly/tahoe_working/metric_verification/cpa_updated_extract/"
    "ACTIVE_all_methods_distribution_metrics_cpa_updated.csv"
)

MANUSCRIPT_SUMMARY_SOURCES = [
    PROJECT_ROOT / (
        "reproducibility/supplementary/benchmark/tahoe/main_figure2/"
        "Fig2D_effect_fidelity_baselines_v3_summary.csv"
    ),
    PROJECT_ROOT / (
        "reproducibility/supplementary/benchmark/tahoe/main_figure2/"
        "Fig2D_effect_fidelity_vs_additive_v3_pairwise.csv"
    ),
    PROJECT_ROOT / (
        "reproducibility/supplementary/benchmark/tahoe/main_figure2/"
        "tahoe_hallmark_program_fidelity_candidate_v2_pairwise.csv"
    ),
    PROJECT_ROOT / (
        "reproducibility/supplementary/benchmark/tahoe/main_figure2/"
        "Fig2Dist_mmd_ot_summary_v1.csv"
    ),
    PROJECT_ROOT / (
        "reproducibility/supplementary/benchmark/tahoe/main_figure2/"
        "Fig2Dist_mmd_ot_pairwise_v1.csv"
    ),
]

FROZEN_EFFECT = SOURCE_VECTOR_DIR / "effect_program_condition_metrics.csv"
FROZEN_DIST = SOURCE_VECTOR_DIR / "all_methods_distribution_metrics_condition_aligned.csv"
FROZEN_DIST_AUDIT = SOURCE_VECTOR_DIR / "distribution_metric_condition_alignment_audit.csv"


def relative(path):
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_dimensions(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = sum(1 for _ in reader)
    return rows, len(header)


def freeze_exact(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(source), str(destination))
    source_hash = sha256_file(source)
    frozen_hash = sha256_file(destination)
    if source_hash != frozen_hash:
        raise ValueError(
            "Frozen vector copy hash mismatch: {} != {} for {}".format(
                source_hash, frozen_hash, relative(source)
            )
        )
    return source_hash


def require_columns(frame, required, label):
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError("{} is missing required columns: {}".format(label, missing))


def require_finite(frame, columns, label):
    array = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(array).all():
        bad = np.argwhere(~np.isfinite(array))[:10].tolist()
        raise ValueError("{} contains non-finite endpoint values at {}".format(label, bad))


def validate_effect_vectors(frame):
    required = [
        "drug",
        "dose",
        "cellname",
        "condition_id",
        "method",
        "effect_mae",
        "effect_pearson_all_genes",
        "effect_spearman_all_genes",
        "pathway_score_mae",
        "pathway_profile_pearson",
    ]
    require_columns(frame, required, "effect/program source")
    if frame[["condition_id", "method"]].isna().any().any():
        raise ValueError("effect/program source contains missing condition_id or method")
    if frame.duplicated(["condition_id", "method"]).any():
        duplicated = frame.loc[
            frame.duplicated(["condition_id", "method"], keep=False),
            ["condition_id", "method"],
        ].head(10)
        raise ValueError("duplicate effect/program keys: {}".format(duplicated.to_dict("records")))

    counts = frame.groupby("method").size().sort_index().to_dict()
    for method in ["PerturbLDM", "AdditiveMean"]:
        if counts.get(method) != EXPECTED_CONDITIONS:
            raise ValueError(
                "Expected {} {} effect/program rows; found {}".format(
                    EXPECTED_CONDITIONS, method, counts.get(method)
                )
            )

    selected = frame.loc[frame["method"].isin(["PerturbLDM", "AdditiveMean"])].copy()
    sets = {
        method: set(group["condition_id"].astype(str))
        for method, group in selected.groupby("method")
    }
    if sets["PerturbLDM"] != sets["AdditiveMean"]:
        raise ValueError(
            "Effect/program condition mismatch: PerturbLDM-only={}, Additive-only={}".format(
                len(sets["PerturbLDM"] - sets["AdditiveMean"]),
                len(sets["AdditiveMean"] - sets["PerturbLDM"]),
            )
        )

    metadata_nunique = selected.groupby("condition_id")[["drug", "dose", "cellname"]].nunique(dropna=False)
    if (metadata_nunique > 1).any().any():
        raise ValueError("Effect/program condition metadata differ between paired methods")

    endpoints = [
        "effect_mae",
        "effect_pearson_all_genes",
        "effect_spearman_all_genes",
        "pathway_score_mae",
        "pathway_profile_pearson",
    ]
    require_finite(selected, endpoints, "selected effect/program vectors")
    return selected, counts, sets["PerturbLDM"]


def validate_distribution_vectors(frame):
    required = [
        "method_canonical",
        "condition_id",
        "MMD_RBF",
        "Wasserstein_OT",
        "n_real",
        "n_pred",
    ]
    require_columns(frame, required, "distribution source")
    if frame[["condition_id", "method_canonical"]].isna().any().any():
        raise ValueError("distribution source contains missing condition_id or method")
    if frame.duplicated(["condition_id", "method_canonical"]).any():
        duplicated = frame.loc[
            frame.duplicated(["condition_id", "method_canonical"], keep=False),
            ["condition_id", "method_canonical"],
        ].head(10)
        raise ValueError("duplicate distribution keys: {}".format(duplicated.to_dict("records")))

    expected_methods = {"PerturbLDM", "CPA", "chemCPA"}
    observed_methods = set(frame["method_canonical"].astype(str))
    if observed_methods != expected_methods:
        raise ValueError(
            "Expected distribution methods {}; observed {}".format(
                sorted(expected_methods), sorted(observed_methods)
            )
        )
    counts = frame.groupby("method_canonical").size().sort_index().to_dict()
    if any(counts.get(method) != EXPECTED_CONDITIONS for method in expected_methods):
        raise ValueError("Unexpected distribution method counts: {}".format(counts))

    sets = {
        method: set(group["condition_id"].astype(str))
        for method, group in frame.groupby("method_canonical")
    }
    shared = set.intersection(*sets.values())
    if len(shared) != EXPECTED_CONDITIONS or any(values != shared for values in sets.values()):
        mismatch = {method: len(values - shared) for method, values in sets.items()}
        raise ValueError(
            "Distribution condition alignment mismatch: shared={}, nonshared={}".format(
                len(shared), mismatch
            )
        )

    require_finite(frame, ["MMD_RBF", "Wasserstein_OT", "n_real", "n_pred"], "distribution vectors")
    if set(frame["n_real"].astype(int)) != {500} or set(frame["n_pred"].astype(int)) != {500}:
        raise ValueError(
            "Expected n_real=n_pred=500 for every distribution row; observed n_real={}, n_pred={}".format(
                sorted(frame["n_real"].unique().tolist()),
                sorted(frame["n_pred"].unique().tolist()),
            )
        )
    return counts, shared


def bootstrap_median_ci(values):
    """Two-sided percentile CI for the median paired improvement.

    The generator is reset to the declared seed for each endpoint so every
    endpoint uses the same deterministic condition-resampling index scheme.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Bootstrap input must be a finite one-dimensional vector")
    rng = np.random.default_rng(SEED)
    medians = np.empty(N_BOOTSTRAP, dtype=float)
    for start in range(0, N_BOOTSTRAP, BOOTSTRAP_CHUNK):
        stop = min(start + BOOTSTRAP_CHUNK, N_BOOTSTRAP)
        indices = rng.integers(
            0, values.size, size=(stop - start, values.size), dtype=np.int32
        )
        medians[start:stop] = np.median(values[indices], axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def signed_rank_summary(values):
    """Two-sided paired Wilcoxon signed-rank test with stable log-P output."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Wilcoxon input must be a finite one-dimensional vector")
    nonzero = values[values != 0]
    if nonzero.size == 0:
        return {
            "n_pairs": int(values.size),
            "n_nonzero": 0,
            "n_ties_zero": int(values.size),
            "wilcoxon_statistic": 0.0,
            "wilcoxon_z": 0.0,
            "p_value": 1.0,
            "log10_p_value": 0.0,
            "rank_biserial": 0.0,
        }

    try:
        result = wilcoxon(
            nonzero,
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
            method="approx",
        )
    except TypeError:
        # SciPy <=1.6 calls this argument ``mode``.
        result = wilcoxon(
            nonzero,
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
            mode="approx",
        )

    ranks = rankdata(np.abs(nonzero), method="average")
    w_positive = float(ranks[nonzero > 0].sum())
    w_negative = float(ranks[nonzero < 0].sum())
    statistic = min(w_positive, w_negative)
    count = int(nonzero.size)
    mean_null = count * (count + 1.0) * 0.25
    variance_numerator = count * (count + 1.0) * (2.0 * count + 1.0)
    _, tie_sizes = np.unique(np.abs(nonzero), return_counts=True)
    tie_sizes = tie_sizes[tie_sizes > 1].astype(float)
    if tie_sizes.size:
        variance_numerator -= 0.5 * np.sum(tie_sizes * (tie_sizes * tie_sizes - 1.0))
    standard_error = math.sqrt(variance_numerator / 24.0)
    z_value = (statistic - mean_null) / standard_error
    log_p_natural = math.log(2.0) + float(log_ndtr(-abs(z_value)))
    log10_p_value = log_p_natural / math.log(10.0)
    rank_biserial = (w_positive - w_negative) / (w_positive + w_negative)

    if not math.isclose(float(result.statistic), statistic, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            "Manual and SciPy Wilcoxon statistics differ: {} vs {}".format(
                statistic, result.statistic
            )
        )
    return {
        "n_pairs": int(values.size),
        "n_nonzero": count,
        "n_ties_zero": int(values.size - count),
        "wilcoxon_statistic": float(result.statistic),
        "wilcoxon_z": float(z_value),
        "p_value": float(result.pvalue),
        "log10_p_value": float(log10_p_value),
        "rank_biserial": float(rank_biserial),
    }


def bh_adjust_log10(log10_p_values):
    values = np.asarray(log10_p_values, dtype=float)
    count = values.size
    order = np.argsort(values)
    sorted_values = values[order]
    adjusted_sorted = sorted_values + np.log10(count / np.arange(1, count + 1, dtype=float))
    for index in range(count - 2, -1, -1):
        adjusted_sorted[index] = min(adjusted_sorted[index], adjusted_sorted[index + 1])
    adjusted_sorted = np.minimum(adjusted_sorted, 0.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def display_p(log10_p):
    if not math.isfinite(log10_p) or log10_p < -300:
        return "<1e-300"
    return "{:.3g}".format(10.0 ** log10_p)


def method_wide(frame, endpoint, method_column):
    wide = frame.pivot(index="condition_id", columns=method_column, values=endpoint).sort_index()
    if wide.isna().any().any():
        raise ValueError("Missing paired values after pivot for {}".format(endpoint))
    return wide


def build_effect_and_program_pairs(effect_selected):
    metadata = (
        effect_selected.loc[
            effect_selected["method"].eq("PerturbLDM"),
            ["condition_id", "drug", "dose", "cellname"],
        ]
        .drop_duplicates("condition_id")
        .set_index("condition_id")
        .sort_index()
    )

    effect_specs = [
        ("effect_pearson_all_genes", "higher"),
        ("effect_spearman_all_genes", "higher"),
        ("effect_mae", "lower"),
    ]
    program_specs = [
        ("pathway_profile_pearson", "higher"),
        ("pathway_score_mae", "lower"),
    ]

    outputs = {}
    for label, specs in [("effect", effect_specs), ("hallmark_program", program_specs)]:
        paired = metadata.copy()
        for endpoint, direction in specs:
            wide = method_wide(effect_selected, endpoint, "method")
            paired["PerturbLDM__{}".format(endpoint)] = wide["PerturbLDM"]
            paired["AdditiveMean__{}".format(endpoint)] = wide["AdditiveMean"]
            if direction == "higher":
                improvement = wide["PerturbLDM"] - wide["AdditiveMean"]
            else:
                improvement = wide["AdditiveMean"] - wide["PerturbLDM"]
            paired["improvement__{}".format(endpoint)] = improvement
        outputs[label] = paired.reset_index()
    return outputs, effect_specs, program_specs


def build_distribution_pairs(distribution):
    metadata_columns = [
        column
        for column in [
            "condition_id",
            "cell_line",
            "mapped_cellname",
            "normalized_drug",
            "normalized_dose",
            "n_real",
            "n_pred",
        ]
        if column in distribution.columns
    ]
    metadata = (
        distribution.loc[
            distribution["method_canonical"].eq("PerturbLDM"), metadata_columns
        ]
        .drop_duplicates("condition_id")
        .set_index("condition_id")
        .sort_index()
    )
    records = []
    for baseline in ["CPA", "chemCPA"]:
        paired = metadata.copy()
        paired.insert(0, "baseline", baseline)
        for endpoint in ["MMD_RBF", "Wasserstein_OT"]:
            wide = method_wide(distribution, endpoint, "method_canonical")
            paired["PerturbLDM__{}".format(endpoint)] = wide["PerturbLDM"]
            paired["baseline__{}".format(endpoint)] = wide[baseline]
            paired["improvement__{}".format(endpoint)] = wide[baseline] - wide["PerturbLDM"]
        records.append(paired.reset_index())
    return pd.concat(records, ignore_index=True)


def summary_record(family, endpoint, label, direction, baseline, reference_values, baseline_values):
    reference_values = np.asarray(reference_values, dtype=float)
    baseline_values = np.asarray(baseline_values, dtype=float)
    if direction == "higher":
        improvement = reference_values - baseline_values
        direction_text = "positive = PerturbLDM higher/better"
    elif direction == "lower":
        improvement = baseline_values - reference_values
        direction_text = "positive = PerturbLDM lower/better"
    else:
        raise ValueError("Unknown endpoint direction: {}".format(direction))
    test = signed_rank_summary(improvement)
    ci_low, ci_high = bootstrap_median_ci(improvement)
    return {
        "comparison_family": family,
        "comparison": "PerturbLDM vs {}".format(baseline),
        "reference_method": "PerturbLDM",
        "baseline_method": baseline,
        "endpoint": endpoint,
        "endpoint_label": label,
        "endpoint_direction": direction,
        "improvement_definition": direction_text,
        "analysis_unit": "held-out drug-dose-cell-line condition",
        "paired": True,
        "reference_mean": float(np.mean(reference_values)),
        "baseline_mean": float(np.mean(baseline_values)),
        "reference_median": float(np.median(reference_values)),
        "baseline_median": float(np.median(baseline_values)),
        "median_improvement": float(np.median(improvement)),
        "median_improvement_ci95_low": ci_low,
        "median_improvement_ci95_high": ci_high,
        "mean_improvement": float(np.mean(improvement)),
        "perturbldm_better_count": int(np.sum(improvement > 0)),
        "perturbldm_better_fraction": float(np.mean(improvement > 0)),
        "zero_difference_fraction": float(np.mean(improvement == 0)),
        **test,
    }


def compute_statistics(effect_selected, distribution, effect_specs, program_specs):
    records = []
    labels = {
        "effect_pearson_all_genes": "all-gene matched-control effect Pearson",
        "effect_spearman_all_genes": "all-gene matched-control effect Spearman",
        "effect_mae": "matched-control effect MAE",
        "pathway_profile_pearson": "signed Hallmark effect-profile Pearson",
        "pathway_score_mae": "signed Hallmark effect-score MAE",
        "MMD_RBF": "MMD-RBF distribution distance",
        "Wasserstein_OT": "OT Wasserstein distribution distance",
    }

    for endpoint, direction in effect_specs:
        wide = method_wide(effect_selected, endpoint, "method")
        records.append(
            summary_record(
                "matched_control_effects",
                endpoint,
                labels[endpoint],
                direction,
                "AdditiveMean",
                wide["PerturbLDM"].to_numpy(),
                wide["AdditiveMean"].to_numpy(),
            )
        )
    for endpoint, direction in program_specs:
        wide = method_wide(effect_selected, endpoint, "method")
        records.append(
            summary_record(
                "signed_hallmark_program_effects",
                endpoint,
                labels[endpoint],
                direction,
                "AdditiveMean",
                wide["PerturbLDM"].to_numpy(),
                wide["AdditiveMean"].to_numpy(),
            )
        )
    for endpoint in ["MMD_RBF", "Wasserstein_OT"]:
        wide = method_wide(distribution, endpoint, "method_canonical")
        for baseline in ["CPA", "chemCPA"]:
            records.append(
                summary_record(
                    "distribution_fidelity",
                    endpoint,
                    labels[endpoint],
                    "lower",
                    baseline,
                    wide["PerturbLDM"].to_numpy(),
                    wide[baseline].to_numpy(),
                )
            )

    result = pd.DataFrame.from_records(records)
    result["log10_p_bh"] = np.nan
    for family, indices in result.groupby("comparison_family", sort=False).groups.items():
        adjusted = bh_adjust_log10(result.loc[indices, "log10_p_value"].to_numpy())
        result.loc[indices, "log10_p_bh"] = adjusted
    result["p_report"] = result["log10_p_value"].map(display_p)
    result["p_bh_report"] = result["log10_p_bh"].map(display_p)
    result["test"] = "two-sided paired Wilcoxon signed-rank; asymptotic normal approximation; zero_method=wilcox; no continuity correction"
    result["multiplicity"] = result["comparison_family"].map(
        {
            "matched_control_effects": "Benjamini-Hochberg across 3 effect endpoints",
            "signed_hallmark_program_effects": "Benjamini-Hochberg across 2 program endpoints",
            "distribution_fidelity": "Benjamini-Hochberg across 4 baseline-endpoint comparisons",
        }
    )
    result["bootstrap_interval"] = "two-sided percentile 95% CI for median paired improvement"
    result["bootstrap_replicates"] = N_BOOTSTRAP
    result["bootstrap_seed"] = SEED
    return result


def write_csv(frame, path, sep=","):
    frame.to_csv(
        path,
        index=False,
        sep=sep,
        float_format="%.17g",
        line_terminator="\n",
    )


def make_manifest(effect, distribution, effect_counts, dist_counts):
    sources = [
        {
            "role": "canonical_upstream_full_vector",
            "path": relative(EFFECT_SOURCE),
            "paired_key": "condition_id + method",
            "methods": ";".join(sorted(effect_counts)),
            "endpoints": "effect_mae;effect_pearson_all_genes;effect_spearman_all_genes;pathway_score_mae;pathway_profile_pearson",
            "key_validation": "unique condition_id-method; PerturbLDM/AdditiveMean sets identical",
            "upstream_match": "self",
        },
        {
            "role": "frozen_exact_full_vector",
            "path": relative(FROZEN_EFFECT),
            "paired_key": "condition_id + method",
            "methods": ";".join(sorted(effect_counts)),
            "endpoints": "effect_mae;effect_pearson_all_genes;effect_spearman_all_genes;pathway_score_mae;pathway_profile_pearson",
            "key_validation": "unique condition_id-method; PerturbLDM/AdditiveMean sets identical",
            "upstream_match": relative(EFFECT_SOURCE),
        },
        {
            "role": "canonical_upstream_full_vector",
            "path": relative(DIST_SOURCE),
            "paired_key": "condition_id + method_canonical",
            "methods": ";".join(sorted(dist_counts)),
            "endpoints": "MMD_RBF;Wasserstein_OT;n_real;n_pred",
            "key_validation": "unique condition_id-method; all three method sets identical; n_real=n_pred=500",
            "upstream_match": "self",
        },
        {
            "role": "frozen_exact_full_vector",
            "path": relative(FROZEN_DIST),
            "paired_key": "condition_id + method_canonical",
            "methods": ";".join(sorted(dist_counts)),
            "endpoints": "MMD_RBF;Wasserstein_OT;n_real;n_pred",
            "key_validation": "unique condition_id-method; all three method sets identical; n_real=n_pred=500",
            "upstream_match": relative(DIST_SOURCE),
        },
        {
            "role": "canonical_alignment_audit",
            "path": relative(DIST_ALIGNMENT_AUDIT),
            "paired_key": "method",
            "methods": "CPA;PerturbLDM;chemCPA",
            "endpoints": "row/key counts;n_real range;n_pred range",
            "key_validation": "canonical audit reports 13,942 unique aligned conditions per method",
            "upstream_match": "self",
        },
        {
            "role": "frozen_exact_alignment_audit",
            "path": relative(FROZEN_DIST_AUDIT),
            "paired_key": "method",
            "methods": "CPA;PerturbLDM;chemCPA",
            "endpoints": "row/key counts;n_real range;n_pred range",
            "key_validation": "byte-identical copy",
            "upstream_match": relative(DIST_ALIGNMENT_AUDIT),
        },
        {
            "role": "active_distribution_upstream_provenance",
            "path": relative(DIST_ACTIVE_UPSTREAM),
            "paired_key": "method-specific source key normalized by canonical aligned table",
            "methods": "diffusion;cpa;chemcpa",
            "endpoints": "MMD_RBF;Wasserstein_OT",
            "key_validation": "CPA updated active source; canonical aligned table is used for analysis",
            "upstream_match": "self",
        },
    ]
    for summary_path in MANUSCRIPT_SUMMARY_SOURCES:
        sources.append(
            {
                "role": "manuscript_summary_reconciliation_source",
                "path": relative(summary_path),
                "paired_key": "summary-specific",
                "methods": "summary table",
                "endpoints": "manuscript-facing aggregate values",
                "key_validation": "used only for reconciliation; not used as paired inferential input",
                "upstream_match": "self",
            }
        )

    records = []
    for item in sources:
        path = PROJECT_ROOT / item["path"]
        rows, columns = csv_dimensions(path)
        match = item["upstream_match"]
        hash_matches = True
        if match not in ["self", ""]:
            hash_matches = sha256_file(path) == sha256_file(PROJECT_ROOT / match)
        records.append(
            {
                **item,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "data_rows": rows,
                "columns": columns,
                "sha256_matches_upstream": bool(hash_matches),
                "status": "frozen_and_validated" if hash_matches else "hash_mismatch",
            }
        )
    manifest = pd.DataFrame.from_records(records)
    if not manifest["sha256_matches_upstream"].all():
        raise ValueError("At least one frozen vector does not match its canonical upstream source")
    return manifest


def summary_row(summary, family, endpoint, baseline):
    row = summary.loc[
        summary["comparison_family"].eq(family)
        & summary["endpoint"].eq(endpoint)
        & summary["baseline_method"].eq(baseline)
    ]
    if len(row) != 1:
        raise ValueError(
            "Expected one summary row for {} {} {}; found {}".format(
                family, endpoint, baseline, len(row)
            )
        )
    return row.iloc[0]


def rounding_match(value, reported, decimals):
    return round(float(value), decimals) == round(float(reported), decimals)


def build_validation_markdown(summary, effect_counts, dist_counts, cross_module_match):
    checks = []

    def add_check(quantity, computed, reported, decimals, note=""):
        status = "PASS" if rounding_match(computed, reported, decimals) else "FAIL"
        checks.append(
            {
                "quantity": quantity,
                "computed": float(computed),
                "manuscript": reported,
                "rule": "round to {} decimals".format(decimals),
                "status": status,
                "note": note,
            }
        )

    effect_pearson = summary_row(summary, "matched_control_effects", "effect_pearson_all_genes", "AdditiveMean")
    effect_spearman = summary_row(summary, "matched_control_effects", "effect_spearman_all_genes", "AdditiveMean")
    effect_mae = summary_row(summary, "matched_control_effects", "effect_mae", "AdditiveMean")
    program_pearson = summary_row(summary, "signed_hallmark_program_effects", "pathway_profile_pearson", "AdditiveMean")
    program_mae = summary_row(summary, "signed_hallmark_program_effects", "pathway_score_mae", "AdditiveMean")

    add_check("Effect Pearson median, PerturbLDM", effect_pearson["reference_median"], 0.8446, 4)
    add_check("Effect Pearson median, Additive", effect_pearson["baseline_median"], 0.7561, 4)
    add_check("Effect Spearman median, PerturbLDM", effect_spearman["reference_median"], 0.7524, 4)
    add_check("Effect Spearman median, Additive", effect_spearman["baseline_median"], 0.6702, 4)
    add_check("Effect MAE mean, PerturbLDM", effect_mae["reference_mean"], 0.0203, 4)
    add_check("Effect MAE mean, Additive", effect_mae["baseline_mean"], 0.0256, 4)
    add_check("Effect Pearson PerturbLDM-better fraction", effect_pearson["perturbldm_better_fraction"], 0.952, 3)
    add_check("Effect Spearman PerturbLDM-better fraction", effect_spearman["perturbldm_better_fraction"], 0.980, 3)
    add_check("Effect MAE PerturbLDM-better fraction", effect_mae["perturbldm_better_fraction"], 0.990, 3)
    add_check("Hallmark profile Pearson PerturbLDM-better fraction", program_pearson["perturbldm_better_fraction"], 0.9436, 4)
    add_check("Hallmark score MAE PerturbLDM-better fraction", program_mae["perturbldm_better_fraction"], 0.8705, 4)

    for baseline, mmd_mean, ot_mean in [
        ("CPA", 0.2426, 60.262),
        ("chemCPA", 0.2197, 59.434),
    ]:
        mmd = summary_row(summary, "distribution_fidelity", "MMD_RBF", baseline)
        ot = summary_row(summary, "distribution_fidelity", "Wasserstein_OT", baseline)
        if baseline == "CPA":
            add_check("MMD mean, PerturbLDM", mmd["reference_mean"], 0.2063, 4)
            add_check("OT mean, PerturbLDM", ot["reference_mean"], 58.846, 3)
        add_check("MMD mean, {}".format(baseline), mmd["baseline_mean"], mmd_mean, 4)
        add_check("OT mean, {}".format(baseline), ot["baseline_mean"], ot_mean, 3)
        checks.append(
            {
                "quantity": "MMD lower fraction, PerturbLDM vs {}".format(baseline),
                "computed": float(mmd["perturbldm_better_fraction"]),
                "manuscript": ">0.999",
                "rule": "strictly greater than 0.999",
                "status": "PASS" if float(mmd["perturbldm_better_fraction"]) > 0.999 else "FAIL",
                "note": "",
            }
        )
        checks.append(
            {
                "quantity": "OT lower fraction, PerturbLDM vs {}".format(baseline),
                "computed": float(ot["perturbldm_better_fraction"]),
                "manuscript": ">0.999",
                "rule": "strictly greater than 0.999",
                "status": "PASS" if float(ot["perturbldm_better_fraction"]) > 0.999 else "FAIL",
                "note": "",
            }
        )

    check_frame = pd.DataFrame.from_records(checks)
    overall = "PASS" if check_frame["status"].eq("PASS").all() else "FAIL"

    lines = [
        "# Validation: Tahoe effect, signed Hallmark program and distribution statistics",
        "",
        "Overall validation: **{}**".format(overall),
        "",
        "## Alignment and integrity",
        "",
        "- Effect/program source rows by method: `{}`.".format(json.dumps(effect_counts, sort_keys=True)),
        "- Distribution source rows by method: `{}`.".format(json.dumps(dist_counts, sort_keys=True)),
        "- Unique paired conditions per required method: `{}`.".format(EXPECTED_CONDITIONS),
        "- Cross-module condition-ID set match (effect/program versus distribution): `{}`.".format(cross_module_match),
        "- Duplicate required keys: `0`.",
        "- Missing or non-finite selected endpoint values: `0`.",
        "- Distribution cells per condition: `n_real = n_pred = 500` for every method-condition row.",
        "- Frozen full-vector copies are byte-identical to canonical upstream sources; see `source_vector_manifest.tsv`.",
        "",
        "## Manuscript-value reconciliation",
        "",
        "Verification below tests the rounding/display values currently used in Results. It does not turn a selected endpoint into an independent biological replicate.",
        "",
        "| Quantity | Computed | Manuscript/display value | Check | Status |",
        "|---|---:|---:|---|---|",
    ]
    for row in checks:
        lines.append(
            "| {} | {:.12g} | {} | {} | {} |".format(
                row["quantity"], row["computed"], row["manuscript"], row["rule"], row["status"]
            )
        )

    lines.extend(
        [
            "",
            "## Statistical provenance",
            "",
            "- Analysis unit: one held-out drug-dose-cell-line condition (`n=13,942` paired conditions per comparison).",
            "- Test: two-sided paired Wilcoxon signed-rank, asymptotic normal approximation, zeros removed, no continuity correction.",
            "- Effect estimate: direction-oriented median paired improvement; positive always favours PerturbLDM.",
            "- Interval: 2,000 condition-level bootstrap resamples, two-sided percentile 95% CI; seed 20260718 reset for each endpoint.",
            "- Multiplicity: BH separately across three matched-control effect endpoints, two signed Hallmark program endpoints and four distribution comparisons.",
            "- Cells and genes are not used as independent inferential replicates.",
            "- Signed Hallmark effect score/profile is a program-level perturbation-effect endpoint, not enrichment.",
            "",
            "## Outputs",
            "",
            "- `tahoe_paired_statistics.csv`: nine inferential comparisons.",
            "- `tahoe_effect_paired_condition_differences.csv`: three effect endpoints for every condition.",
            "- `tahoe_hallmark_paired_condition_differences.csv`: two signed Hallmark program endpoints for every condition.",
            "- `tahoe_distribution_paired_condition_differences.csv`: MMD/OT differences for CPA and chemCPA for every condition.",
            "",
        ]
    )
    return "\n".join(lines), overall, check_frame


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    required_sources = [
        EFFECT_SOURCE,
        DIST_SOURCE,
        DIST_ALIGNMENT_AUDIT,
        DIST_ACTIVE_UPSTREAM,
        *MANUSCRIPT_SUMMARY_SOURCES,
    ]
    missing = [relative(path) for path in required_sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required canonical inputs: {}".format(missing))

    effect_hash = freeze_exact(EFFECT_SOURCE, FROZEN_EFFECT)
    dist_hash = freeze_exact(DIST_SOURCE, FROZEN_DIST)
    dist_audit_hash = freeze_exact(DIST_ALIGNMENT_AUDIT, FROZEN_DIST_AUDIT)

    read_options = {"low_memory": False, "float_precision": "round_trip"}
    effect = pd.read_csv(FROZEN_EFFECT, **read_options)
    distribution = pd.read_csv(FROZEN_DIST, **read_options)
    effect_selected, effect_counts, effect_conditions = validate_effect_vectors(effect)
    dist_counts, dist_conditions = validate_distribution_vectors(distribution)
    if effect_conditions != dist_conditions:
        raise ValueError(
            "Cross-module condition-ID mismatch: effect-only={}, distribution-only={}".format(
                len(effect_conditions - dist_conditions), len(dist_conditions - effect_conditions)
            )
        )
    cross_module_match = True

    pair_outputs, effect_specs, program_specs = build_effect_and_program_pairs(effect_selected)
    distribution_pairs = build_distribution_pairs(distribution)
    summary = compute_statistics(
        effect_selected, distribution, effect_specs, program_specs
    )

    effect_pair_path = OUT / "tahoe_effect_paired_condition_differences.csv"
    program_pair_path = OUT / "tahoe_hallmark_paired_condition_differences.csv"
    dist_pair_path = OUT / "tahoe_distribution_paired_condition_differences.csv"
    summary_path = OUT / "tahoe_paired_statistics.csv"
    write_csv(pair_outputs["effect"], effect_pair_path)
    write_csv(pair_outputs["hallmark_program"], program_pair_path)
    write_csv(distribution_pairs, dist_pair_path)
    write_csv(summary, summary_path)

    validation_text, validation_status, validation_checks = build_validation_markdown(
        summary, effect_counts, dist_counts, cross_module_match
    )
    validation_path = OUT / "validation.md"
    validation_path.write_text(validation_text, encoding="utf-8")
    write_csv(validation_checks, OUT / "manuscript_value_reconciliation.csv")
    if validation_status != "PASS":
        raise ValueError("At least one manuscript summary reconciliation check failed")

    manifest = make_manifest(effect, distribution, effect_counts, dist_counts)
    manifest_path = OUT / "source_vector_manifest.tsv"
    write_csv(manifest, manifest_path, sep="\t")

    output_paths = [
        FROZEN_EFFECT,
        FROZEN_DIST,
        FROZEN_DIST_AUDIT,
        effect_pair_path,
        program_pair_path,
        dist_pair_path,
        summary_path,
        OUT / "manuscript_value_reconciliation.csv",
        manifest_path,
        validation_path,
    ]
    output_hashes = {relative(path): sha256_file(path) for path in output_paths}
    input_paths = [EFFECT_SOURCE, DIST_SOURCE, DIST_ALIGNMENT_AUDIT, DIST_ACTIVE_UPSTREAM]
    input_hashes = {relative(path): sha256_file(path) for path in input_paths}

    metadata = {
        "analysis_date": "2026-07-18",
        "script": relative(Path(__file__)),
        "script_sha256": sha256_file(Path(__file__)),
        "working_directory_independent": True,
        "scientific_question": {
            "matched_control_effects": "Does PerturbLDM improve condition-level matched-control effect fidelity over AdditiveMean?",
            "signed_hallmark_program_effects": "Does PerturbLDM improve signed Hallmark program-effect fidelity over AdditiveMean?",
            "distribution_fidelity": "Does PerturbLDM reduce condition-level MMD-RBF and OT distances relative to CPA and chemCPA?",
        },
        "analysis_unit": "held-out drug-dose-cell-line condition",
        "n_paired_conditions": EXPECTED_CONDITIONS,
        "pairing_key": "condition_id",
        "effect_semantics": "upstream matched-control-relative effect metrics",
        "program_semantics": "signed Hallmark effect score/profile; not enrichment",
        "distribution_semantics": "condition-paired distances with n_real=n_pred=500",
        "improvement_orientation": "positive values always mean PerturbLDM is better",
        "bootstrap": {
            "replicates": N_BOOTSTRAP,
            "seed": SEED,
            "unit": "condition",
            "interval": "two-sided percentile 95% CI of median paired improvement",
            "seed_strategy": "reset numpy Generator(seed=20260718) for each endpoint so all endpoints use the same deterministic resampling-index scheme",
        },
        "test": {
            "name": "paired Wilcoxon signed-rank",
            "sidedness": "two-sided",
            "approximation": "asymptotic normal",
            "zero_method": "wilcox",
            "continuity_correction": False,
        },
        "multiplicity": {
            "matched_control_effects": "Benjamini-Hochberg across 3 endpoints",
            "signed_hallmark_program_effects": "Benjamini-Hochberg across 2 endpoints",
            "distribution_fidelity": "Benjamini-Hochberg across 4 baseline-endpoint comparisons",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "validation": {
            "status": validation_status,
            "effect_method_counts": effect_counts,
            "distribution_method_counts": dist_counts,
            "cross_module_condition_id_match": cross_module_match,
            "duplicate_required_keys": 0,
            "missing_or_nonfinite_selected_values": 0,
            "frozen_effect_sha256_matches_upstream": effect_hash == sha256_file(FROZEN_EFFECT),
            "frozen_distribution_sha256_matches_upstream": dist_hash == sha256_file(FROZEN_DIST),
            "frozen_distribution_audit_sha256_matches_upstream": dist_audit_hash == sha256_file(FROZEN_DIST_AUDIT),
        },
        "canonical_input_sha256": input_hashes,
        "staged_output_sha256": output_hashes,
        "interpretation_boundary": "Conditions are the inferential units. Cells and genes are not treated as independent biological replicates; extremely small P-values must be reported with effect estimates and confidence intervals.",
    }
    (OUT / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("VALIDATION_STATUS={}".format(validation_status))
    print("PAIRED_CONDITIONS={}".format(EXPECTED_CONDITIONS))
    print("CROSS_MODULE_CONDITION_MATCH={}".format(cross_module_match))
    print("STATISTICS_ROWS={}".format(len(summary)))
    print(
        summary[
            [
                "comparison_family",
                "baseline_method",
                "endpoint",
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
