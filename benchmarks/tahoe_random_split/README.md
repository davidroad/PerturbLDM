# Tahoe random-split benchmark code

This directory contains the code lineage consistent with the current
13,942-condition Tahoe absolute-expression benchmark in Fig. 2b.

## Version decision

`script.zip` is a legacy development archive. Its May-June 2025 baseline code
evaluates matched-control-relative deltas, and its split JSON describes
4,674/10,906 single-dose conditions rather than the current
32,529/13,942-condition, three-dose benchmark. It is not the Fig. 2b producer.

The release-facing code is:

- `baseline/`: the July 2025 `save_expression` RF/MLP program. Its 41,826
  original condition rows (MLP, RF and matched-control/no-effect baseline) match
  the released rows after identifier normalization and CSV rounding.
- `CPA/`: the Gaussian CPA training, April 2026 inference and CPA-row update
  chain used by the active CPA-updated table.
- `chemCPA/`: the optimized random-counterfactual workflow, not the older
  chemCPA draft in `script.zip`.
- `plotting/`: a standalone relative-path wrapper replacing the historical
  v5 -> v3 -> v2 plotting chain, plus a result-lineage verifier.

## Data boundary

GitHub is code-only. Exact derived condition-level tables and the original
RF/MLP and CPA metric outputs are in the paired Figshare package under
`inputs/plotting_inputs/tahoe/fig2b_absolute_expression/`. Raw Tahoe H5AD files
and model checkpoints are not copied.

## Recreate the summary and panel

```bash
python github/benchmarks/tahoe_random_split/plotting/create_fig2b_absolute_expression_benchmark.py \
  --learned-table figshare/inputs/plotting_inputs/tahoe/fig2b_absolute_expression/learned_methods_condition_metrics.csv.gz \
  --simple-table figshare/inputs/plotting_inputs/tahoe/fig2b_absolute_expression/simple_mean_baseline_condition_metrics.csv.gz \
  --expected-summary figshare/inputs/plotting_inputs/tahoe/main_figure2/Fig2C_condition_benchmark_boxplot_v5_summary.csv \
  --outdir figshare/derived/validation/fig2b_absolute_expression
```

The historical filename says `Fig2C`; the final manuscript places this
component in Fig. 2b.

## Recreate Supplementary Fig. S3e-h

The compact fitting/validation histories document model-specific selection for MLP, random forest, CPA and chemCPA. The loss scales are not compared across models.

```bash
python github/benchmarks/tahoe_random_split/plotting/create_suppfig_s3_selection_diagnostics.py \
  --source-dir figshare/inputs/plotting_inputs/tahoe/suppfig_s3_selection_diagnostics \
  --output-dir figshare/derived/validation/suppfig_s3_selection_diagnostics
```

## Verify result lineage

```bash
python github/benchmarks/tahoe_random_split/plotting/verify_result_lineage.py \
  --learned-table figshare/inputs/plotting_inputs/tahoe/fig2b_absolute_expression/learned_methods_condition_metrics.csv.gz \
  --baseline-raw figshare/inputs/plotting_inputs/tahoe/fig2b_absolute_expression/baseline_raw_per_condition.csv.gz \
  --cpa-json figshare/inputs/plotting_inputs/tahoe/fig2b_absolute_expression/cpa_merged_metrics_by_condition.json.gz \
  --output figshare/derived/validation/fig2b_absolute_expression/result_lineage_validation.json
```

The RF/MLP chain and CPA condition metrics are verified. End-to-end model
regeneration still requires external prepared Tahoe inputs and checkpoint
objects. The active chemCPA checkpoint and its completed run lineage have been
verified and frozen in the private reproducibility archive, but the checkpoint
is not distributed in this public package and the exact solved runtime
environment remains an external provenance boundary.
