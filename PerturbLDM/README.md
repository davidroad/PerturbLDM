# PerturbLDM core code

This directory contains a manuscript-facing release snapshot of the core PerturbLDM implementation used for latent VAE training, conditional latent diffusion training, and inference.

## Scope

Included:

- VAE latent model architecture and training utilities.
- Conditional denoising MLP architecture and diffusion training utilities.
- Tahoe cell-line/drug/dose latent reconstruction, diffusion inference, and metric utilities.
- Example scripts for PBMC and colon-development transfer experiments.

Not included:

- Raw single-cell data, processed atlas folders, checkpoints, exploratory plots, deprecated metrics, or internal analysis outputs.

## Files

- `LatentModelFinal.py`: VAE encoder/decoder used to map expression profiles into latent space and reconstruct expression.
- `DenoisingMLPFinal.py`: conditional denoising MLP modules for latent diffusion.
- `util_latent.py`: latent model dataset and training loop.
- `util_diff.py`: Tahoe condition parsing, diffusion dataset construction, and diffusion training loop.
- `util_inference_diff.py`: batch diffusion inference utilities.
- `util_metrics.py`: expression and effect metric helpers.
- `util_plot.py`: lightweight plotting and UMAP helper functions used by example scripts.
- `examples/`: manuscript experiment entry points and small release helpers.
- `examples/cellline_drug_dose_tahoe/`: Tahoe cell-line/drug/dose experiment scripts.
- `examples/cellline_drug_dose_tahoe/train_latent_model.py`: Tahoe VAE training entry point.
- `examples/cellline_drug_dose_tahoe/train_diffusionmlp_model.py`: Tahoe conditional latent diffusion training entry point.
- `examples/cellline_drug_dose_tahoe/run_latent_inference.py`: VAE reconstruction inference for held-out Tahoe conditions.
- `examples/cellline_drug_dose_tahoe/run_diff_inference.py`: diffusion inference for all held-out Tahoe conditions.
- `examples/cellline_drug_dose_tahoe/decode_diffusion_latents_and_metrics.py`: decode diffusion-generated latents and export condition-level metrics.
- `examples/cellline_drug_dose_tahoe/compute_simple_baselines.py`: train-only simple baseline benchmark for MatchedCtrl, CellLineMean, DrugDoseMean, and AdditiveMean.
- `examples/exp_pbmc_ifn.py`: PBMC IFN transfer experiment.
- `examples/exp_colon_development.py`: colon-development transfer experiment.
- `examples/prepare_drug_embedding_tensor.py`: optional helper for aligning an external drug-embedding dictionary to PerturbLDM drug IDs.

## Expected Tahoe processed-data layout

The Tahoe entry points expect `--data_root` to point directly to one processed Tahoe split folder:

```text
<data_root>/
  collection/
    train_adata.h5ad
    train_metadf.csv
    test_adata.h5ad
    test_metadf.csv
    control_adata.h5ad
    control_metadf.csv
  processed/
    cond2id.json
```

For example, `--data_root external_inputs/tahoe/Random_7_3_Jun6`.

## Minimal commands

Run commands from this directory unless adapting imports in a separate package.

Train the VAE:

```bash
python -u examples/cellline_drug_dose_tahoe/train_latent_model.py \
  --data_root external_inputs/tahoe/Random_7_3_Jun6 \
  --save_dir train_results
```

The VAE input dimension is inferred from `collection/train_adata.h5ad` unless `--input_dim` is provided explicitly.

Train the conditional latent diffusion model:

```bash
python -u examples/cellline_drug_dose_tahoe/train_diffusionmlp_model.py \
  --data_root external_inputs/tahoe/Random_7_3_Jun6 \
  --latent_dir train_results/Random_7_3_Jun6/<latent-run-dir> \
  --prediction_type v_prediction
```

The diffusion training script infers `drug_vocab_size` from `processed/cond2id.json` and `ctrl_dim` from `collection/control_adata.h5ad` unless these values are provided explicitly.

Optional: align an external drug-embedding dictionary before training a frozen/pretrained drug-embedding variant:

```bash
python -u examples/prepare_drug_embedding_tensor.py \
  --cond2id_path external_inputs/tahoe/Random_7_3_Jun6/processed/cond2id.json \
  --embedding_pkl external_inputs/embeddings/drug_embeddings.pkl \
  --output_path Random_7_3_Jun6_drug_emb_pretrain_tensor.pt
```

Then pass the tensor to diffusion training with `--drug_emb_pretrained_path`.

Run all-condition diffusion inference:

```bash
python -u examples/cellline_drug_dose_tahoe/run_diff_inference.py \
  --data_root external_inputs/tahoe/Random_7_3_Jun6 \
  --diff_dir train_results/Random_7_3_Jun6/<diffusion-run-dir> \
  --num_samples 50
```

Use `--save_latent_steps` only when reverse-diffusion trajectories are needed;
the all-condition latent-step tensor can be large.

Decode latent outputs and compute condition-level metrics:

```bash
python -u examples/cellline_drug_dose_tahoe/decode_diffusion_latents_and_metrics.py \
  --data_root external_inputs/tahoe/Random_7_3_Jun6 \
  --diff_dir train_results/Random_7_3_Jun6/<diffusion-run-dir>
```

Per-cell counterfactual expression arrays are not saved by default. Add
`--save_cf_expr` only when those large arrays are required.

Compute train-only simple baseline controls:

```bash
python -u examples/cellline_drug_dose_tahoe/compute_simple_baselines.py \
  --data_root external_inputs/tahoe/Random_7_3_Jun6 \
  --output_dir benchmark_simple_baselines
```

The simple baseline script evaluates `MatchedCtrl`, `CellLineMean`, `DrugDoseMean`, and `AdditiveMean`. `CellLineMean`, `DrugDoseMean`, and `AdditiveMean` use only training-split perturbation-condition means; test data are used only as observed ground truth during metric computation.

## PBMC IFN-beta example

The PBMC example is documented in `examples/pbmc_ifnb/`. It records the exact stimulated-cell hold-out rule and links the fitted model configurations to the full training entry point:

```bash
python -u examples/exp_pbmc_ifn.py \
  --basedir external_inputs/pbmc \
  --output_dir outputs/pbmc_ifnb \
  --topgenenum 2000 --subtype mse --cuda_id 0
```

## Notes

- PerturbLDM uses a VAE to reconstruct expression profiles in latent space and a conditional diffusion model to generate latent responses from conditions.
- The diffusion scripts support `v_prediction`, `epsilon`, and `sample` prediction types through `--prediction_type`.
- The code snapshot supports method inspection and reproduction with the datasets cited in the manuscript.
- All command examples use repository-relative paths. Raw or processed H5ADs, checkpoints and latent tensors are staged outside the repository.
