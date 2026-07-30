# Active corrected-split CPA Tahoe benchmark

This directory contains the corrected condition-disjoint CPA training,
inference and distribution-metric lineage used in manuscript Fig. 2b and
Fig. 2h.

## Split

CPA used the same condition membership as the other learned comparators:
29,277 fitting, 3,252 validation and 13,942 external-test conditions. The exact
metadata-only assignment is in
`../../manifests/cpa_condition_assignments_seed42.csv`.

## Active training contract

- Gaussian reconstruction loss;
- CPU training with seed 0;
- batch size 131,072 and learning rate 0.001;
- adversarial regularisation 200 and penalty 400;
- at most 20 epochs with validation after every epoch;
- early stopping on the package `cpa_metric`, patience 4 and minimum
  improvement 0.0001;
- epoch 4 (one-based) selected from the eight completed epochs.

The checkpoint itself is external and is represented by its recorded SHA256 in
`provenance/checkpoint_selection.json`.

## Relative input staging

The training script reads
`../../external_inputs/tahoe/cpa_preprocessed/adata_all_concatenated_random.h5ad`
by default. Override this with `CPA_PREPROCESSED_H5AD`. Inference reads the
prepared test and control objects under `../../external_inputs/tahoe/`; use
`TAHOE_TEST_H5AD`, `TAHOE_CONTROL_H5AD` and `CPA_MODEL_PATH` to override those
relative defaults.

Compact training and validation histories used in Supplementary Fig. S3g are
distributed with the manuscript source data. Large H5AD objects, per-cell
predictions and checkpoints are not included in GitHub.
