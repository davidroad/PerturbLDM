# PBMC IFN-beta response-transfer example

This example reproduces the manuscript task that asks whether IFN-beta response programmes learned from measured myeloid and lymphoid cell types can generate unmeasured stimulated states in other cell types from the same immune compartments.

## Experimental split

The processed PBMC object is derived from Kang et al. (GSE96583) and must contain `cell.type` and `stim` in `adata.obs`.

- IFN-beta-stimulated B cells, CD8 T cells and FCGR3A+ monocytes are held out.
- The unstimulated (`ctrl`) cells from those same cell types remain in the fitting data and provide their observed cellular backgrounds.
- Other measured cell-type and stimulation combinations remain in the fitting data.
- Megakaryocytes are excluded before model fitting.

The exact rule is recorded in `split_contract.tsv`.

## How to read the example

The script is organised as a single end-to-end workflow:

1. **Input and split:** validate `cell.type` and `stim`, remove megakaryocytes and construct the response-transfer hold-out above.
2. **Shared feature space:** preprocess expression and keep one ordered gene set for fitting, generation and evaluation.
3. **Latent model:** fit the VAE and encode the fitting cells into 128-dimensional latent representations.
4. **Conditional diffusion:** learn latent denoising conditioned on cell type and stimulation state.
5. **Two inference paths:** `PerturbLDM` starts from noise, whereas `PerturbLDM-ctrl` starts from observed control latents and applies controlled noising before denoising.
6. **Outputs:** decode generated latents, retain row/condition alignment and write checkpoints, predictions and condition-level metrics.

Inline comments in `exp_pbmc_ifn.py` mark these stages. The installed
`python -m PerturbLDM.examples.test_pbmc_training` workflow applies the same
biological hold-out in a compact, CPU-safe configuration. It writes losses,
diagnostic metrics, figures and checkpoints, but is an execution test rather
than manuscript reproduction.

## Full manuscript workflow

Run from the `PerturbLDM` repository directory:

```bash
python -u examples/exp_pbmc_ifn.py \
  --basedir external_inputs/pbmc \
  --output_dir outputs/pbmc_ifnb \
  --topgenenum 2000 \
  --subtype mse \
  --cuda_id 0
```

The full script trains the latent model and conditional denoiser, generates the three held-out stimulated states and writes condition-level metrics and prediction objects. The fitted model settings are supplied in `configs/pbmc_active_run/`.

## Compact post-install test

```bash
python -m PerturbLDM.examples.test_pbmc_training \
  --input /path/to/pbmc_IFN_filtered.h5ad \
  --output-dir outputs/pbmc_example \
  --device cpu
```

The processed H5AD is not bundled with the package and must contain
`cell.type` and `stim` in `adata.obs`.

## Preprocessing

The workflow filters cells with fewer than 10 detected genes, library-size normalises to 10,000 counts, applies `log1p`, removes genes detected in fewer than 1% of cells and retains 2,000 highly variable genes.

The primary experiment conditions generation on cell type and stimulation state. The control-initialised variant reported as `PerturbLDM-ctrl` is also implemented in `exp_pbmc_ifn.py`.

## Data

The processed input is derived from Kang et al. (GSE96583) and is staged at `external_inputs/pbmc/pbmc_IFN_filtered.h5ad`. Model checkpoints and prediction objects are written to the selected output directory.
