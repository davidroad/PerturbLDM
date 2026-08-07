# PBMC FAO–OXPHOS analysis

This module contains portable source-data export and plotting code for
manuscript Fig. 5h and Supplementary Fig. S13. The analysis evaluates
matched-control-relative fatty-acid beta-oxidation (FAO) and oxidative
phosphorylation (OXPHOS) transcriptomic scores in the three held-out PBMC
lineages.

## Released source tables

The paired Figshare package stores the four compact inputs under
`source_data/pbmc/`:

- `pbmc_fao_oxphos_pathway_scores.csv`: lineage, profile and programme score
  summaries used for the effect panels.
- `pbmc_fao_oxphos_composite_fidelity_ratios.csv`: PerturbLDM error divided
  by scGen error for the composite effect and per-cell score distribution.
- `pbmc_fao_oxphos_cell_scores.csv`: anonymous per-cell FAO and OXPHOS
  scores used to recompute one-dimensional Wasserstein distances.
- `pbmc_fao_oxphos_distribution_metrics.csv`: component-level Wasserstein,
  energy and Kolmogorov--Smirnov summaries.
- `pbmc_oxphos_umap_coordinates.csv`: measured-reference embedding
  coordinates and OXPHOS scores used for Supplementary Fig. S13a.

These are transcriptomic pathway scores and do not measure metabolic flux.

## Recreate the displayed panels

Install the repository with the `full` optional dependencies before running these scripts.

```bash
python analysis/pbmc_immunometabolism/plot_fig5h.py \
  --source-dir /path/to/figshare/source_data/pbmc \
  --output-dir outputs/pbmc_immunometabolism

python analysis/pbmc_immunometabolism/plot_suppfig_s13.py \
  --source-dir /path/to/figshare/source_data/pbmc \
  --output-dir outputs/pbmc_immunometabolism
```

## Re-export scores from staged prediction objects

The full export requires the measured PBMC training and held-out objects, the
PerturbLDM prediction matrix in `test.obsm["cf_expr"]`, the aligned scGen
prediction object and an MSigDB Hallmark GMT file.

```bash
python analysis/pbmc_immunometabolism/export_fao_oxphos_scores.py \
  --train-h5ad external_inputs/pbmc/train_adata_final_diffU.h5ad \
  --test-h5ad external_inputs/pbmc/test_adata_final_diffU.h5ad \
  --scgen-h5ad external_inputs/pbmc/scgen_predictions.h5ad \
  --hallmark-gmt external_inputs/gene_sets/h.all.v2024.1.Hs.symbols.gmt \
  --output-dir outputs/pbmc_immunometabolism/source_data

python analysis/pbmc_immunometabolism/export_oxphos_umap_coordinates.py \
  --test-h5ad external_inputs/pbmc/test_adata_final_diffU.h5ad \
  --scgen-h5ad external_inputs/pbmc/scgen_predictions.h5ad \
  --cell-scores outputs/pbmc_immunometabolism/source_data/pbmc_fao_oxphos_cell_scores.csv \
  --output outputs/pbmc_immunometabolism/source_data/pbmc_oxphos_umap_coordinates.csv
```

The UMAP is fitted to measured stimulated profiles and used to transform the
PerturbLDM and scGen profiles. The fixed seed, 30-component PCA, 15-neighbour
UMAP and score-gene settings match the manuscript Methods. The released coordinate table is authoritative for exact panel recreation because UMAP coordinates can vary across software versions.
