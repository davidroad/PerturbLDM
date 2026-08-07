#!/usr/bin/env python3
"""Export measured-reference UMAP coordinates used in Supplementary Fig. S13a."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA


CELL_TYPES = ["B cells", "CD8 T cells", "FCGR3A+ Monocytes"]
PROFILES = ["Measured", "PerturbLDM", "scGen"]
SEED = 20260806


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-h5ad", required=True, type=Path)
    parser.add_argument("--scgen-h5ad", required=True, type=Path)
    parser.add_argument("--cell-scores", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cell-type-column", default="cell.type")
    parser.add_argument("--prediction-key", default="cf_expr")
    return parser.parse_args()


def dense(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def make_profile(matrix, obs, var_names, profile, cell_type_column):
    result = ad.AnnData(
        X=dense(matrix),
        obs=obs[[cell_type_column]].copy(),
        var=pd.DataFrame(index=var_names.copy()),
    )
    result.obs["profile"] = profile
    return result


def main() -> None:
    args = parse_args()
    test = ad.read_h5ad(args.test_h5ad)
    scgen = ad.read_h5ad(args.scgen_h5ad)
    if not test.var_names.equals(scgen.var_names):
        raise ValueError("Gene order mismatch between measured and scGen profiles")
    if args.prediction_key not in test.obsm:
        raise KeyError(f"Missing test.obsm[{args.prediction_key!r}]")

    measured = make_profile(
        test.X, test.obs, test.var_names, "Measured", args.cell_type_column
    )
    perturbldm = make_profile(
        test.obsm[args.prediction_key],
        test.obs,
        test.var_names,
        "PerturbLDM",
        args.cell_type_column,
    )
    scgen_profile = make_profile(
        scgen.X, scgen.obs, scgen.var_names, "scGen", args.cell_type_column
    )
    combined = ad.concat(
        [measured, perturbldm, scgen_profile],
        join="inner",
        merge="same",
        index_unique="-",
    )

    scores = pd.read_csv(args.cell_scores)
    scores = scores.loc[scores["program"].eq("OXPHOS")].copy()
    combined.obs["OXPHOS score"] = np.nan
    for profile in PROFILES:
        for cell_type in CELL_TYPES:
            target = combined.obs["profile"].eq(profile) & combined.obs[
                args.cell_type_column
            ].astype(str).eq(cell_type)
            values = scores.loc[
                scores["profile"].eq(profile)
                & scores["cell_type"].eq(cell_type),
                "score",
            ].to_numpy()
            if len(values) != int(target.sum()):
                raise ValueError(
                    f"OXPHOS-score row mismatch for {profile}, {cell_type}: "
                    f"{len(values)} != {int(target.sum())}"
                )
            combined.obs.loc[target, "OXPHOS score"] = values
    if combined.obs["OXPHOS score"].isna().any():
        raise ValueError("Missing OXPHOS scores after row-aligned assignment")

    pca = PCA(n_components=30, random_state=SEED)
    measured_pca = pca.fit_transform(dense(measured.X))
    perturbldm_pca = pca.transform(dense(perturbldm.X))
    scgen_pca = pca.transform(dense(scgen_profile.X))
    mapper = umap.UMAP(
        n_neighbors=15,
        min_dist=0.3,
        metric="euclidean",
        random_state=SEED,
        transform_seed=SEED,
        n_jobs=1,
    ).fit(measured_pca)
    coordinates = np.vstack(
        [mapper.embedding_, mapper.transform(perturbldm_pca), mapper.transform(scgen_pca)]
    )

    table = pd.DataFrame(
        {
            "row_id": [f"pbmc_{index:06d}" for index in range(combined.n_obs)],
            "UMAP1": coordinates[:, 0],
            "UMAP2": coordinates[:, 1],
            "profile": combined.obs["profile"].to_numpy(),
            "cell_type": combined.obs[args.cell_type_column].astype(str).to_numpy(),
            "OXPHOS_score": combined.obs["OXPHOS score"].to_numpy(),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
