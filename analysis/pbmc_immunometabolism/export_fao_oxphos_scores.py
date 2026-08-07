#!/usr/bin/env python3
"""Export FAO--OXPHOS source tables for PBMC response evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import energy_distance, ks_2samp, wasserstein_distance


CELL_TYPES = ["B cells", "CD8 T cells", "FCGR3A+ Monocytes"]
PROFILES = ["Measured", "PerturbLDM", "scGen"]
SEED = 20260806

# MSigDB C5 GO:BP GOBP_FATTY_ACID_BETA_OXIDATION (GO:0006635),
# version 2025.1.Hs. Only genes represented in the supplied feature space are used.
FAO_GENES = [
    "ACOT8", "ACAA2", "ECI2", "SLC25A17", "SLC27A2", "CPT1A", "CPT1B",
    "CPT2", "CRAT", "ALDH1L2", "ECI1", "DECR1", "ECH1", "ECHS1",
    "EHHADH", "MTLN", "AKT1", "AKT2", "ETFA", "ETFB", "ETFDH", "ABCD1",
    "FABP1", "TYSND1", "ABCD2", "MLYCD", "AMACR", "ETFBKMT", "DECR2",
    "GCDH", "MCAT", "ACAA1", "HSD17B10", "HADHA", "HADHB", "HADH",
    "ACACB", "HSD17B4", "ACADL", "ACADM", "ACADS", "IRS1", "ACADVL",
    "IVD", "ACAT1", "LEP", "PLIN5", "ACOX1", "PEX7", "PPARA", "PPARD",
    "CROT", "AUH", "ECHDC2", "ACOXL", "ECHDC1", "BDH2", "ABCD3", "ABCD4",
    "PEX2", "PEX5", "SCP2", "TWIST1", "ACAD10", "ACOX2", "ACOX3", "SESN2",
    "LONP2", "ACAD11", "MFSD2A", "ABCB11", "IRS2", "ADIPOQ",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score measured, PerturbLDM and scGen PBMC profiles with the same "
            "expression-matched reference-gene pool."
        )
    )
    parser.add_argument("--train-h5ad", required=True, type=Path)
    parser.add_argument("--test-h5ad", required=True, type=Path)
    parser.add_argument("--scgen-h5ad", required=True, type=Path)
    parser.add_argument("--hallmark-gmt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cell-type-column", default="cell.type")
    parser.add_argument("--stimulation-column", default="stim")
    parser.add_argument("--control-label", default="ctrl")
    parser.add_argument("--prediction-key", default="cf_expr")
    return parser.parse_args()


def dense(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def read_gmt(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip().split("\t")
            result[fields[0]] = fields[2:]
    return result


def profile_adata(matrix, cell_types, var_names, profile: str) -> ad.AnnData:
    obs = pd.DataFrame(
        {"cell_type": np.asarray(cell_types, dtype=str), "profile": profile}
    )
    return ad.AnnData(
        X=dense(matrix),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(var_names.astype(str))),
    )


def composite_ratios(summary: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    def program_wide(program: str) -> pd.DataFrame:
        return (
            summary.loc[summary["program"].eq(program)]
            .pivot(index="cell_type", columns="profile", values="matched_control_effect")
            .loc[CELL_TYPES]
        )

    composite = (program_wide("FAO") + program_wide("OXPHOS")) / 2
    effect_ratio = (
        (composite["PerturbLDM"] - composite["Measured"]).abs()
        / (composite["scGen"] - composite["Measured"]).abs()
    )

    selected = cells.loc[cells["program"].isin(["FAO", "OXPHOS"])].copy()
    selected["cell_index"] = selected.groupby(
        ["cell_type", "profile", "program"]
    ).cumcount()
    wide = selected.pivot(
        index=["cell_type", "profile", "cell_index"],
        columns="program",
        values="score",
    )
    wide["composite"] = (wide["FAO"] + wide["OXPHOS"]) / 2

    w1_ratios = []
    for cell_type in CELL_TYPES:
        measured = wide.xs(
            (cell_type, "Measured"), level=("cell_type", "profile")
        )["composite"]
        perturbldm = wide.xs(
            (cell_type, "PerturbLDM"), level=("cell_type", "profile")
        )["composite"]
        scgen = wide.xs(
            (cell_type, "scGen"), level=("cell_type", "profile")
        )["composite"]
        w1_ratios.append(
            wasserstein_distance(measured, perturbldm)
            / wasserstein_distance(measured, scgen)
        )

    return pd.DataFrame(
        {
            "cell_type": CELL_TYPES,
            "composite_effect_error_ratio": effect_ratio.to_numpy(),
            "composite_score_wasserstein_ratio": w1_ratios,
        }
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = ad.read_h5ad(args.train_h5ad)
    test = ad.read_h5ad(args.test_h5ad)
    scgen = ad.read_h5ad(args.scgen_h5ad)

    var_names = pd.Index(test.var_names.astype(str))
    for name, obj in [("train", train), ("scGen", scgen)]:
        if not var_names.equals(pd.Index(obj.var_names.astype(str))):
            raise ValueError(f"Gene order mismatch for {name}")
    if args.prediction_key not in test.obsm:
        raise KeyError(f"Missing test.obsm[{args.prediction_key!r}]")
    if test.obsm[args.prediction_key].shape != test.shape:
        raise ValueError("PerturbLDM prediction matrix does not match test shape")

    train_type = train.obs[args.cell_type_column].astype(str).to_numpy()
    train_stim = train.obs[args.stimulation_column].astype(str).to_numpy()
    test_type = test.obs[args.cell_type_column].astype(str).to_numpy()
    scgen_type = scgen.obs[args.cell_type_column].astype(str).to_numpy()

    parts = []
    for cell_type in CELL_TYPES:
        mask = (train_type == cell_type) & (train_stim == args.control_label)
        if not mask.any():
            raise ValueError(f"No matched controls for {cell_type}")
        parts.append(
            profile_adata(train.X[mask], train_type[mask], var_names, "Matched control")
        )
    for profile, matrix, types in [
        ("Measured", test.X, test_type),
        ("PerturbLDM", test.obsm[args.prediction_key], test_type),
        ("scGen", scgen.X, scgen_type),
    ]:
        for cell_type in CELL_TYPES:
            mask = types == cell_type
            if not mask.any():
                raise ValueError(f"No {profile} cells for {cell_type}")
            parts.append(profile_adata(matrix[mask], types[mask], var_names, profile))

    combined = ad.concat(parts, join="inner", merge="same", index_unique="-")
    hallmark = read_gmt(args.hallmark_gmt)
    programs = {
        "FAO": FAO_GENES,
        "OXPHOS": hallmark["HALLMARK_OXIDATIVE_PHOSPHORYLATION"],
    }
    overlaps = {}
    for label, source_genes in programs.items():
        genes = [gene for gene in source_genes if gene in combined.var_names]
        if not genes:
            raise ValueError(f"No feature-space genes overlap {label}")
        overlaps[label] = len(genes)
        sc.tl.score_genes(
            combined,
            genes,
            score_name=label,
            ctrl_size=50,
            n_bins=25,
            random_state=SEED,
            use_raw=False,
        )

    score_rows = []
    summary_rows = []
    distribution_rows = []
    for cell_type in CELL_TYPES:
        control_mask = combined.obs["cell_type"].eq(cell_type) & combined.obs[
            "profile"
        ].eq("Matched control")
        for program in programs:
            control_mean = float(combined.obs.loc[control_mask, program].mean())
            measured_scores = combined.obs.loc[
                combined.obs["cell_type"].eq(cell_type)
                & combined.obs["profile"].eq("Measured"),
                program,
            ].to_numpy(dtype=float)
            for profile in PROFILES:
                mask = combined.obs["cell_type"].eq(cell_type) & combined.obs[
                    "profile"
                ].eq(profile)
                values = combined.obs.loc[mask, program].to_numpy(dtype=float)
                score_rows.extend(
                    {
                        "cell_type": cell_type,
                        "profile": profile,
                        "program": program,
                        "score": float(value),
                    }
                    for value in values
                )
                effect = float(values.mean() - control_mean)
                summary_rows.append(
                    {
                        "cell_type": cell_type,
                        "profile": profile,
                        "program": program,
                        "n_genes": overlaps[program],
                        "matched_control_effect": effect,
                        "cell_sd": float(values.std(ddof=1)),
                    }
                )
                if profile != "Measured":
                    distribution_rows.append(
                        {
                            "cell_type": cell_type,
                            "program": program,
                            "method": profile,
                            "n_cells": len(values),
                            "effect_score": effect,
                            "wasserstein": float(
                                wasserstein_distance(measured_scores, values)
                            ),
                            "energy": float(energy_distance(measured_scores, values)),
                            "ks": float(ks_2samp(measured_scores, values).statistic),
                        }
                    )

    summary = pd.DataFrame(summary_rows)
    cells = pd.DataFrame(score_rows)
    distribution = pd.DataFrame(distribution_rows)
    ratios = composite_ratios(summary, cells)

    summary.to_csv(
        args.output_dir / "pbmc_fao_oxphos_pathway_scores.csv", index=False
    )
    cells.to_csv(args.output_dir / "pbmc_fao_oxphos_cell_scores.csv", index=False)
    distribution.to_csv(
        args.output_dir / "pbmc_fao_oxphos_distribution_metrics.csv", index=False
    )
    ratios.to_csv(
        args.output_dir / "pbmc_fao_oxphos_composite_fidelity_ratios.csv",
        index=False,
    )
    print(ratios.to_string(index=False))


if __name__ == "__main__":
    main()
