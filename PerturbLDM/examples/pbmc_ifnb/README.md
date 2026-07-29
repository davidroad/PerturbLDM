# PBMC IFN-beta response-transfer example

This example reproduces the manuscript task that asks whether IFN-beta response programmes learned from measured myeloid and lymphoid cell types can generate unmeasured stimulated states in other cell types from the same immune compartments.

## Experimental split

The processed PBMC object is derived from Kang et al. (GSE96583) and must contain `cell.type` and `stim` in `adata.obs`.

- IFN-beta-stimulated B cells, CD8 T cells and FCGR3A+ monocytes are held out.
- The unstimulated (`ctrl`) cells from those same cell types remain in the fitting data and provide their observed cellular backgrounds.
- Other measured cell-type and stimulation combinations remain in the fitting data.
- Megakaryocytes are excluded before model fitting.

The exact rule is recorded in `split_contract.tsv`. Validate a staged object before training:

```bash
python examples/pbmc_ifnb/audit_input.py \
  --input external_inputs/pbmc/pbmc_IFN_filtered.h5ad \
  --summary-out outputs/pbmc_ifnb/input_audit.json
```

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

The full script trains the latent model and conditional denoiser, generates the three held-out stimulated states and writes condition-level metrics and prediction objects. The exact recovered manuscript-active model settings are also supplied in `configs/pbmc_active_run/`.

## Preprocessing and interpretation boundary

The manuscript run filters cells with fewer than 10 detected genes, library-size normalises to 10,000 counts, applies `log1p`, removes genes detected in fewer than 1% of cells and retains 2,000 highly variable genes. The historical 2,000-gene set was selected on the complete archived task object, including held-out cells; the comparison is therefore transductive and descriptive rather than a strict prospective feature-selection benchmark.

The primary experiment conditions generation only on cell type and stimulation state. The control-initialised strength scan in `exp_pbmc_ifn.py` is a post hoc sensitivity analysis (`PerturbLDM-ctrl` in the manuscript), not a separately prespecified primary model. The release does not claim donor-held-out validation.

## External objects

The processed H5AD, checkpoints and prediction objects are deliberately excluded from GitHub. Their repository-relative staging paths, byte sizes and SHA256 identities are listed in `../../input_contracts/external_input_manifest.tsv`.
