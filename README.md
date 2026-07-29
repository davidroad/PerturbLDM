# PerturbLDM manuscript reproducibility code

This code-only release accompanies **PerturbLDM: conditional latent diffusion for modelling single-cell perturbation responses**. It separates portable code and compact configuration from large single-cell matrices, checkpoints and prediction objects.

## Repository layout

- `PerturbLDM/`: core latent VAE and conditional diffusion implementation, active colon/PBMC configurations and runnable examples.
- `PerturbLDM/examples/pbmc_ifnb/`: PBMC IFN-beta response-transfer example, exact hold-out contract and read-only input audit.
- `PerturbLDM/examples/cellline_drug_dose_tahoe/`: Tahoe training and inference entry points.
- `benchmarks/tahoe_random_split/`: MLP, random-forest, CPA and chemCPA benchmark code plus figure wrappers.
- `scripts/`: compact manuscript-statistics and audit utilities.
- `environment/`: recovered dependency records.
- `provenance/`: panel-index structural validation.

## PBMC example

Stage the processed GSE96583-derived object under the documented relative path, validate its annotations and split, then run the full experiment:

```bash
python PerturbLDM/examples/pbmc_ifnb/audit_input.py \
  --input PerturbLDM/external_inputs/pbmc/pbmc_IFN_filtered.h5ad

cd PerturbLDM
python -u examples/exp_pbmc_ifn.py \
  --basedir external_inputs/pbmc \
  --output_dir outputs/pbmc_ifnb \
  --topgenenum 2000 --subtype mse --cuda_id 0
```

See `PerturbLDM/examples/pbmc_ifnb/README.md` for the biological task, preprocessing and interpretation boundaries.

## Tahoe benchmark figures

The Tahoe code and paired Figshare source tables recreate the absolute-expression benchmark and Supplementary Fig. S3e-h selection diagnostics. Commands are documented in `benchmarks/tahoe_random_split/README.md`.

## Data and model boundary

Raw and processed H5AD files, model checkpoints, latent tensors and per-cell predictions are not part of GitHub. Relative staging names, byte sizes and recovered hashes for the PBMC and fetal-colon objects are listed in `PerturbLDM/input_contracts/external_input_manifest.tsv`. Tahoe objects follow the contracts documented in the benchmark release.

The code is released under the MIT License. Citation metadata and final public deposit identifiers will be added before the repository is made public.
