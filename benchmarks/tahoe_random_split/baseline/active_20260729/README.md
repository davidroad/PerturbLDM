# Active MLP and random-forest Tahoe benchmark

This directory contains the active condition-level MLP and random-forest
lineage used for the 13,942-condition external test results in manuscript
Fig. 2b. The scientific unit is one drug-dose-cell-line condition.

## Split and inputs

The fixed split is recorded in
`../../manifests/cpa_condition_assignments_seed42.csv`:

- 29,277 fitting conditions;
- 3,252 validation conditions;
- 13,942 external-test conditions.

Stage the three prepared Tahoe objects under
`../../external_inputs/tahoe/`, or provide their paths explicitly to
`scripts/build_condition_cache.py`. Raw H5AD files and checkpoints are not
stored in GitHub.

## Active models

- MLP: 14,211 -> 512 -> 256 -> 13,784, Adam, learning rate 0.0001,
  batch size 1,024, dropout 0 and no weight decay. Epoch 99 (one-based) was
  selected by minimum validation delta-expression MSE.
- Random forest: 300 trees, maximum depth 10, square-root feature sampling,
  bootstrap sampling and 1,500 fitting-derived matched-control genes.
  `min_samples_leaf=5` was selected from 5, 20 and 50 by validation
  delta-expression MSE.

`config/active_model_contract.json` is the compact release authority for these
settings. The scripts retain the integrity checks used during the frozen run.
External-test responses are opened only by the gated OOD inference program
after model selection.

## Cache construction

```bash
python scripts/build_condition_cache.py \
  --datasets train control \
  --output-root cache/train_control \
  --train-h5ad ../../external_inputs/tahoe/train_adata_processed.h5ad \
  --control-h5ad ../../external_inputs/tahoe/control_adata_processed.h5ad
```

The large condition-mean arrays and fitted model objects remain external
artifacts. Compact selection histories used in Supplementary Fig. S3e-f are
distributed with the manuscript source data.
