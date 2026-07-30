# Full-data MLP model-selection audit

Status: **PASS**

- Analysis unit: drug-dose-cell-line condition.
- Formal internal training set: 29,277 conditions.
- Fixed validation set: 3,252 conditions; membership SHA256 `0e39c54f22109408c37e1fadf44443dd72d4db4ccec710f7a9d13c62de4aa3e9`.
- Selection metric: pooled validation delta-expression MSE.
- Learning-rate candidates at dropout 0: 0.0001, 0.0005 and 0.001.
- Dropout candidates at learning rate 0.0001: 0, 0.1, 0.2 and 0.3.
- Strict joint optimum: learning rate 0.0001, dropout 0.
- Architecture: 14,211 -> 512 -> 256 -> 13,784.
- Best epoch: 99 (one-based; zero-based index 98).
- Validation delta-expression MSE: 0.00231497557461.
- Validation median condition-level delta Pearson: 0.704657128543.
- Checkpoint SHA256: `031629f971a9fb8b04bee5811b21d9541a8ff2c7f4917551af33858285b6b944`.
- External OOD responses were not used for model selection.

The earlier development-search record remains preserved as upstream provenance.
This versioned record is authoritative for the full-data validation confirmation
and the checkpoint to be used in subsequent OOD evaluation.
