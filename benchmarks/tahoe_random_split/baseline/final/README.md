# MLP and random-forest Tahoe benchmark

This directory contains the MLP and random-forest workflows used for the 13,942-condition external test results in manuscript Fig. 2b. The unit of analysis is one drug-dose-cell-line condition.

## Split and inputs

The fixed split is recorded in `../../splits/condition_assignments_seed42.csv`:

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

The selected parameters are recorded in `config/model_config.json`, and the compact search space is recorded in `config/hyperparameter_search.json`. External-test responses are evaluated only after model selection.

## Cache construction

```bash
python scripts/build_condition_cache.py \
  --datasets train control \
  --output-root cache/train_control \
  --train-h5ad ../../external_inputs/tahoe/train_adata_processed.h5ad \
  --control-h5ad ../../external_inputs/tahoe/control_adata_processed.h5ad
```

The remaining scripts tune the MLP and random forest on the validation split and evaluate both selected models on the external test split. Selection histories used in Supplementary Fig. S3e-f are distributed with the manuscript source data.
