# Tahoe random-split benchmark

This directory contains the code for the 13,942-condition Tahoe benchmark in manuscript Fig. 2.

## Active analysis

The fixed condition split in `splits/condition_assignments_seed42.csv`
contains 29,277 fitting, 3,252 validation and 13,942 external-test conditions.
All learned comparators use this condition membership; external-test responses
are excluded from hyperparameter and checkpoint selection.

- `baseline/final/`: MLP and random-forest fitting, validation selection and external-test inference.
- `CPA/final/`: condition-disjoint CPA training, inference and distribution metrics.
- `chemCPA/`: chemCPA training, inference and distribution metrics.
- `PerturbLDM/`: PerturbLDM distribution-metric code.
- `plotting/`: Fig. 2b and Supplementary Fig. S3 plotting wrappers.

## Data boundary

GitHub is code-focused. Stage prepared Tahoe objects under
`external_inputs/tahoe/` as documented in the active method READMEs. Raw H5AD
files, checkpoints, condition-mean arrays and per-cell prediction objects are
not included.

The paired Figshare package contains the condition-level metric tables, model-selection histories and manuscript source data.

## Recreate Fig. 2b

```bash
python plotting/create_fig2b_absolute_expression_benchmark.py \
  --learned-table <figshare_dir>/source_data/tahoe/fig2b_absolute_expression/learned_methods_condition_metrics.csv.gz \
  --simple-table <figshare_dir>/source_data/tahoe/fig2b_absolute_expression/simple_mean_baseline_condition_metrics.csv.gz \
  --expected-summary <figshare_dir>/source_data/tahoe/main_figure2/Fig2C_condition_benchmark_boxplot_v5_summary.csv \
  --outdir outputs/fig2b
```

## Recreate Supplementary Fig. S3e-h

```bash
python plotting/create_suppfig_s3_model_selection.py \
  --source-dir <figshare_dir>/source_data/tahoe/suppfig_s3_model_selection \
  --output-dir outputs/suppfig_s3
```

These panels document model-specific selection. Their numerical loss scales
are not compared across models.

The condition-level source tables contain 13,942 rows for each reported method.
