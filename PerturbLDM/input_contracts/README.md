# External input contract

The repository contains code, exact small configuration files and input
contracts. It deliberately does not contain raw or processed H5AD files,
checkpoints, latent tensors or complete prediction objects.

Stage external objects under the repository-relative names in
`external_input_manifest.tsv`, or pass equivalent relative paths explicitly to
the command-line entry points. Verify both byte size and SHA256 before using a
locally retained object.

Large-object policy:

- GitHub: code, configuration, environment records and templates only.
- Figshare: manuscript figures and minimum sufficient panel-level source tables.
- GEO/Mendeley/Tahoe data service or controlled local storage: raw/processed
  matrices, checkpoints and large prediction objects.

The manifest records hashes for the recovered manuscript-active colon and PBMC
objects without publishing the objects themselves. A recorded hash establishes
object identity; it does not replace the accession, preprocessing or sample-map
contract.


A runnable PBMC entry point, exact stimulated-cell hold-out rules and a read-only input audit are provided in `../examples/pbmc_ifnb/`.

The PBMC and colon experiment scripts preserve the historical feature-selection
scope used for the manuscript runs. Because features were selected on the full
archived task object, these comparisons remain transductive and descriptive;
the release does not relabel them as strict prospective hold-outs.
