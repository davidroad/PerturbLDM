# Tahoe random-split benchmark code

This directory contains the code and compact provenance for the active
13,942-condition Tahoe benchmark in manuscript Fig. 2.

## Active analysis

The metadata-only split in `manifests/cpa_condition_assignments_seed42.csv`
contains 29,277 fitting, 3,252 validation and 13,942 external-test conditions.
All learned comparators use this condition membership; external-test responses
are excluded from hyperparameter and checkpoint selection.

- `baseline/active_20260729/`: active MLP and random-forest fitting, validation
  selection and gated external-test inference.
- `CPA/active_20260729/`: corrected condition-disjoint CPA training, selected
  checkpoint, inference and distribution metrics.
- `chemCPA/`: frozen chemCPA training and inference lineage.
- `plotting/`: relative-path figure and selection-diagnostic wrappers.

Older scripts remain under the method-level `scripts/` directories and are
labelled `legacy` in each `SCRIPT_MANIFEST.csv`; they are retained for
provenance and are not the producer of the active refreshed values.

## Data boundary

GitHub is code-focused. Stage prepared Tahoe objects under
`external_inputs/tahoe/` as documented in the active method READMEs. Raw H5AD
files, checkpoints, condition-mean arrays and per-cell prediction objects are
not included.

The paired Figshare package contains the compact condition-level metric tables,
model-selection histories and manuscript source data. The split manifest is
included here because it is lightweight metadata required to reproduce exact
fitting, validation and external-test membership.

## Recreate Fig. 2b

```bash
python plotting/create_fig2b_absolute_expression_benchmark.py \
  --learned-table figshare/inputs/plotting_inputs/tahoe/fig2b_absolute_expression/learned_methods_condition_metrics.csv.gz \
  --simple-table figshare/inputs/plotting_inputs/tahoe/fig2b_absolute_expression/simple_mean_baseline_condition_metrics.csv.gz \
  --expected-summary figshare/inputs/plotting_inputs/tahoe/main_figure2/Fig2C_condition_benchmark_boxplot_v5_summary.csv \
  --outdir figshare/derived/validation/fig2b_absolute_expression
```

The historical source filename contains `Fig2C`; the final manuscript places
this component in Fig. 2b.

## Recreate Supplementary Fig. S3e-h

```bash
python plotting/create_suppfig_s3_selection_diagnostics.py \
  --source-dir figshare/inputs/plotting_inputs/tahoe/suppfig_s3_selection_diagnostics \
  --output-dir figshare/derived/validation/suppfig_s3_selection_diagnostics
```

These panels document model-specific selection. Their numerical loss scales
are not compared across models.

## Verify result lineage

```bash
python plotting/verify_result_lineage.py \
  --learned-table figshare/inputs/plotting_inputs/tahoe/fig2b_absolute_expression/learned_methods_condition_metrics.csv.gz \
  --baseline-raw figshare/inputs/plotting_inputs/tahoe/fig2b_absolute_expression/baseline_raw_per_condition.csv.gz \
  --cpa-json figshare/inputs/plotting_inputs/tahoe/fig2b_absolute_expression/cpa_merged_metrics_by_condition.json.gz \
  --cell-line-map figshare/inputs/plotting_inputs/tahoe/fig2b_absolute_expression/cellline2name.json \
  --output figshare/derived/validation/fig2b_absolute_expression/result_lineage_validation.json
```

The active result tables contain 13,942 rows for each reported method. The
lineage check compares 27,884 final MLP/RF rows and 13,942 final CPA rows
condition by condition. Full end-to-end regeneration additionally requires the external prepared Tahoe
objects and recorded model checkpoints.
