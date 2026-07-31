"""Run the post-install PerturbLDM PBMC training test.

After installing ``perturbldm[pbmc]``, run:

    python -m PerturbLDM.examples.test_pbmc_training \
        --input /path/to/pbmc_IFN_filtered.h5ad \
        --output-dir outputs/pbmc_example \
        --device cpu

The command prints each workflow step and ends with ``[SUCCESS]`` only when
training, generation, metrics, figures, checkpoints and output validation all
complete successfully.
"""

from PerturbLDM.pbmc_smoke import main


if __name__ == "__main__":
    raise SystemExit(main())
