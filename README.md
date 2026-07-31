# PerturbLDM

Code accompanying **PerturbLDM: conditional latent diffusion for modelling single-cell perturbation responses**.

## Overview

<a href="docs/figures/fig1_PerturbLDM.pdf">
  <img src="docs/figures/fig1_PerturbLDM.png" alt="Overview of the PerturbLDM framework and study design" width="100%">
</a>

**Figure 1 | PerturbLDM framework and study design.** **a,** PerturbLDM encodes single-cell expression into a latent space and learns a conditional diffusion process that generates response states from cellular context, perturbation, dose and control-state information. Tahoe-100M provides the atlas-scale foundation for model fitting and quantitative benchmarking. **b,** In PANACEA, shared anchor conditions align assay platforms so that Tahoe-derived response profiles can be transferred to an external experimental context and used to organise drug-response hypotheses. **c,** In fetal colon, models fitted to early and mature enterocyte states generate the held-out developmental transition. **d,** In PBMCs, response programmes learned across measured immune populations generate held-out, lineage-specific IFN-beta-stimulated states.

## Repository layout

- `PerturbLDM/`: core latent VAE and conditional diffusion implementation, colon/PBMC configurations and runnable examples.
- `PerturbLDM/examples/pbmc_ifnb/`: PBMC IFN-beta response-transfer example and exact hold-out definition.
- `PerturbLDM/examples/cellline_drug_dose_tahoe/`: Tahoe training and inference entry points.
- `benchmarks/tahoe_random_split/`: MLP, random-forest, CPA and chemCPA benchmark code plus figure wrappers.
- `docs/figures/`: the manuscript overview figure in PDF and README-preview formats.

## Installation

Python 3.10 or 3.11 is recommended. Create an isolated environment and install
PerturbLDM directly from GitHub:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install \
  "perturbldm[pbmc] @ git+https://github.com/davidroad/PerturbLDM.git"
python -c "import PerturbLDM; print(PerturbLDM.__file__)"
```

The explicit PyTorch command installs the CPU wheel used by the smoke test.
GPU users should instead install the PyTorch build matching their CUDA runtime
before installing PerturbLDM.

While the repository is private, the installing machine must have authenticated
GitHub access. For example, `gh auth login` followed by `gh auth setup-git`
configures Git credentials for command-line installation.

Some managed clusters export another environment's PyTorch libraries through
`LD_LIBRARY_PATH`. If `import torch` reports a missing C-extension symbol even
though `pip check` passes, verify the isolated environment without that inherited
path:

```bash
env -u LD_LIBRARY_PATH python -c "import torch; print(torch.__version__)"
```

Use the same prefix for the smoke test on such a cluster:

```bash
env -u LD_LIBRARY_PATH perturbldm-pbmc-smoke \
  --input /path/to/pbmc_IFN_filtered.h5ad \
  --output-dir outputs/pbmc_smoke \
  --device cpu
```

## Five-minute PBMC diffusion example

The installed command below runs a standalone CPU-safe end-to-end example on
the Kang et al. PBMC object. It is designed to complete within five minutes on
a conventional CPU; actual time depends on hardware.

```bash
python -m PerturbLDM.examples.test_pbmc_training \
  --input /path/to/pbmc_IFN_filtered.h5ad \
  --output-dir outputs/pbmc_example \
  --device cpu
```

The Python test prints eight numbered stages, periodic latent-model and
diffusion losses, condition-level diagnostic correlations and the absolute
output directory. A completed run ends with:

```text
[Step 8/8] Validate the run and report completion
[SUCCESS] PerturbLDM test training completed.
Outputs: /absolute/path/to/outputs/pbmc_example
```

Any uncaught error or failed finite-value/shape check prints `[FAILED]` and
returns a non-zero exit status. The equivalent installed command is
`perturbldm-pbmc-example`.

The command validates the required `cell.type` and `stim` metadata, applies the
response-transfer hold-out of IFN-beta-stimulated B cells, CD8 T cells and
FCGR3A+ monocytes, fits compact latent and conditional-diffusion models and
generates the held-out states. It writes `pbmc_example_summary.json`,
`training_losses.png`, `prediction_diagnostics.png`, `training_history.csv`,
`condition_metrics.csv`, the selected genes, cell-level held-out predictions,
condition-mean profiles, run configuration and both model checkpoints. The
summary indexes every artifact and includes shape and finite-value checks plus
diagnostic whole-state and matched-control-relative metrics.

This is a self-contained teaching and execution example, not manuscript
reproduction or benchmark evidence. Its fixed defaults use all available cells
after the response-transfer split (11,842 fitting and 1,710 held-out cells in
the supplied object), 1,000 training-selected HVGs, a 64-dimensional latent
space, 40 latent-model epochs, 160 diffusion epochs and 50 inference steps. It
does not run hyperparameter search or cross-validation. The verified default
run completed in 116 seconds with 1.84 GB peak memory on the reference CPU
server; runtime varies with hardware. The legacy `perturbldm-pbmc-smoke`
command is retained as an alias.

## Full PBMC example

Stage the processed GSE96583-derived object under the documented relative path, then run the experiment:

```bash
cd PerturbLDM
python -u examples/exp_pbmc_ifn.py \
  --basedir external_inputs/pbmc \
  --output_dir outputs/pbmc_ifnb \
  --topgenenum 2000 --subtype mse --cuda_id 0
```

See `PerturbLDM/examples/pbmc_ifnb/README.md` for the biological task, split and preprocessing.

## Tahoe benchmark figures

The Tahoe code and paired Figshare source tables recreate the absolute-expression benchmark and Supplementary Fig. S3e-h selection diagnostics. Commands are documented in `benchmarks/tahoe_random_split/README.md`.

## Data

Raw and processed H5AD files are obtained from the sources cited in the manuscript and staged outside the repository. Model checkpoints, latent tensors and per-cell predictions are generated by the released workflows.

The code is released under the MIT License.
