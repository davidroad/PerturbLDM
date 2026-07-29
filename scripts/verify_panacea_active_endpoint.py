#!/usr/bin/env python3
"""Verify the active (2026-05-19) PANACEA top-1 endpoint.

The superseded top-3 fine-MoA/ssGSEA analysis under
supplementary/benchmark/panacea is intentionally not used.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "statistics_revision_20260717"
ACTIVE_DIR = (
    ROOT / "supplementary/benchmark/panacea_moa_concordance_20260519"
)
QUERY_TABLE = (
    ACTIVE_DIR
    / "supplementary_source_tables/panacea_query_level_neighbor_table.csv"
)
PERMUTATION_TABLE = (
    ACTIVE_DIR
    / "supplementary_source_tables/panel_c_random_label_summary.csv"
)


def poisson_binomial_distribution(probabilities: np.ndarray) -> np.ndarray:
    distribution = np.zeros(probabilities.size + 1, dtype=np.float64)
    distribution[0] = 1.0
    for probability in probabilities:
        updated = np.zeros_like(distribution)
        updated[:-1] += distribution[:-1] * (1.0 - probability)
        updated[1:] += distribution[:-1] * probability
        distribution = updated
    if not math.isclose(float(distribution.sum()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Poisson-binomial probabilities do not sum to one")
    return distribution


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / n
    denominator = 1.0 + z * z / n
    center = (proportion + z * z / (2.0 * n)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n))
        / denominator
    )
    return center - half_width, center + half_width


def main() -> None:
    queries = pd.read_csv(QUERY_TABLE)
    permutations = pd.read_csv(PERMUTATION_TABLE)

    n_queries = int(len(queries))
    successes = int(queries["top1_moa_hit"].astype(bool).sum())
    probabilities = queries["random_top1_hit"].to_numpy(dtype=float)
    distribution = poisson_binomial_distribution(probabilities)
    one_sided_p = float(distribution[successes:].sum())
    expected_successes = float(probabilities.sum())
    expected_rate = float(probabilities.mean())
    ci_low, ci_high = wilson_interval(successes, n_queries)

    if n_queries != 27 or successes != 14:
        raise ValueError(
            f"Expected the active 14/27 endpoint, found {successes}/{n_queries}"
        )
    if round(one_sided_p, 3) != 0.121:
        raise ValueError(
            f"Recomputed Poisson-binomial P does not reproduce 0.121: {one_sided_p}"
        )

    permutation_row = permutations.loc[
        permutations["group"].eq("all")
        & permutations["metric"].eq("top1_hit")
    ]
    if len(permutation_row) != 1:
        raise ValueError("Expected one all-drug top-1 permutation row")
    permutation_row = permutation_row.iloc[0]

    record = {
        "analysis_version": "active PANACEA MoA concordance staged 2026-05-19",
        "endpoint": "Broad Hub MoA top-1 neighbour concordance",
        "analysis_unit": "evaluable query drug",
        "n_queries": n_queries,
        "successes": successes,
        "observed_rate": successes / n_queries,
        "wilson95_low": ci_low,
        "wilson95_high": ci_high,
        "mean_query_label_prevalence": expected_rate,
        "expected_successes": expected_successes,
        "poisson_binomial_p_greater": one_sided_p,
        "poisson_binomial_sidedness": "one-sided (greater)",
        "permutation_null_mean": float(permutation_row["null_mean"]),
        "permutation_empirical_p_ge_observed": float(
            permutation_row["empirical_p_ge_observed"]
        ),
        "permutation_count": int(permutation_row["n_permutations"]),
        "interpretation": "descriptive; combined endpoint did not reach 0.05",
        "multiplicity": "single combined top-1 endpoint; subgroup and ranking metrics descriptive",
        "superseded_endpoint_excluded": (
            "top-3 curated fine-MoA ssGSEA analysis in "
            "supplementary/benchmark/panacea (deprecated 2026-05-19)"
        ),
    }

    pd.DataFrame([record]).to_csv(
        OUT / "panacea_active_top1_statistics.csv", index=False
    )
    (OUT / "panacea_active_top1_verification.json").write_text(
        json.dumps(
            {
                **record,
                "source_query_table": str(QUERY_TABLE.relative_to(ROOT)),
                "source_permutation_table": str(PERMUTATION_TABLE.relative_to(ROOT)),
                "calculation": (
                    "dynamic-programming Poisson-binomial probability mass; "
                    "tail summed from 14 through 27 successes"
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
