# Full-data RF model-selection audit

Status: **PASS**

- Analysis unit: drug-dose-cell-line condition.
- Target: treated-condition mean minus matched cell-line control mean.
- Formal internal training set: 29,277 conditions; membership SHA256 `b7cc23d2a04dd75f90b8d20ea3637c86fbd2a40ed2d7d643adcb1925345f2642`.
- Fixed validation set: 3,252 conditions; membership SHA256 `0e39c54f22109408c37e1fadf44443dd72d4db4ccec710f7a9d13c62de4aa3e9`.
- Input design: 1,927 features; 13,784 gene outputs.
- Fixed parameters: 300 trees, maximum depth 10, `sqrt` feature sampling, bootstrap enabled and random state 42.
- Compared `min_samples_leaf`: 5, 20 and 50.
- Selection metric: pooled formal-validation matched-control-relative delta-expression MSE.
- Strict argmin: `min_samples_leaf=5`.
- Selected validation delta-expression MSE: 0.00338418427605.
- Selected model SHA256: `bd29033a7e12d5fc64b0bbad871b83b69241e9d5098df533436eb9389091d433`.
- All three saved models were reloaded and their fitted dimensions, tree counts and fixed parameters were checked.
- External OOD responses were not used for this selection.

The earlier `results/rf/selection.json` remains unchanged and is retained as
upstream provenance. The versioned JSON published with this audit is
authoritative for the RF hyperparameter and model used in subsequent OOD
evaluation.
