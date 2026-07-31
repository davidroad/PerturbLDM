# Tahoe cell-line/drug/dose experiment

This folder contains the Tahoe-specific entry points for the PerturbLDM cell-line/drug/dose response reconstruction experiment.

Run these scripts from the PerturbLDM release root, or use the configurable
shell wrappers in this folder. The `--data_root` argument should point directly
to one processed Tahoe split folder containing `collection/` and `processed/`.

Typical order:

```bash
python -u examples/cellline_drug_dose_tahoe/train_latent_model.py \
  --data_root external_inputs/tahoe/Random_7_3_Jun6

python -u examples/cellline_drug_dose_tahoe/run_latent_inference.py \
  --data_root external_inputs/tahoe/Random_7_3_Jun6 \
  --latent_dir train_results/Random_7_3_Jun6/<latent-run-dir>

python -u examples/cellline_drug_dose_tahoe/train_diffusionmlp_model.py \
  --data_root external_inputs/tahoe/Random_7_3_Jun6 \
  --latent_dir train_results/Random_7_3_Jun6/<latent-run-dir>

python -u examples/cellline_drug_dose_tahoe/run_diff_inference.py \
  --data_root external_inputs/tahoe/Random_7_3_Jun6 \
  --diff_dir train_results/Random_7_3_Jun6/<diffusion-run-dir> \
  --num_samples 50

python -u examples/cellline_drug_dose_tahoe/decode_diffusion_latents_and_metrics.py \
  --data_root external_inputs/tahoe/Random_7_3_Jun6 \
  --diff_dir train_results/Random_7_3_Jun6/<diffusion-run-dir>

python -u examples/cellline_drug_dose_tahoe/compute_simple_baselines.py \
  --data_root external_inputs/tahoe/Random_7_3_Jun6 \
  --output_dir benchmark_simple_baselines
```

The simple baseline benchmark is independent of PerturbLDM training and uses no
test-set information when estimating marginal means:

- `MatchedCtrl`: matched control mean expression for the same cell line.
- `CellLineMean`: mean of training perturbation-condition means from the same cell line.
- `DrugDoseMean`: mean of training perturbation-condition means from the same drug-dose pair.
- `AdditiveMean`: `CellLineMean + DrugDoseMean - GlobalMean`, using training-split marginals.

The shell wrappers are execution templates rather than archived records of a
particular fitted checkpoint. Set `DATA_DIR` before running
`train_latent.sh`, and set both `DATA_DIR` and `LATENT_DIR` before running
`train_diffusion.sh`. Saved run configurations and checkpoint metadata should
be retained alongside outputs when adapting these templates.

`run_diff_inference.py` reads held-out condition names from `test_metadf.csv`.
Use `--save_latent_steps` only when reverse-diffusion trajectory snapshots are
needed, because that output is much larger than the final latent samples.
`decode_diffusion_latents_and_metrics.py` saves mean predicted expression by
default; add `--save_cf_expr` only when per-cell counterfactual expression arrays
are required.

The `external_inputs/` paths above are staging locations and are not distributed with the code repository. Large objects remain in their authoritative data repository or local controlled storage.
