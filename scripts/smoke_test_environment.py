#!/usr/bin/env python3
"""Validate the repaired PerturbLDM analysis environment and one H5AD input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import anndata
    import diffusers
    import numpy
    import pandas
    import rdkit
    import scanpy
    import scipy
    import sklearn
    import torch

    if not args.h5ad.is_file():
        raise FileNotFoundError(args.h5ad)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the repaired environment")

    x = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
    cuda_checksum = float((x @ x.T).sum().item())
    adata = anndata.read_h5ad(args.h5ad, backed="r")
    try:
        h5ad_shape = [int(adata.n_obs), int(adata.n_vars)]
        obs_columns = sorted(map(str, adata.obs.columns))
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    payload = {
        "status": "PASS",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "scanpy": scanpy.__version__,
            "anndata": anndata.__version__,
            "diffusers": diffusers.__version__,
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "rdkit": rdkit.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "cuda": {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "device_0": torch.cuda.get_device_name(0),
            "matrix_checksum": cuda_checksum,
        },
        "input": {
            "path_recorded_as": args.h5ad.name,
            "bytes": args.h5ad.stat().st_size,
            "sha256": sha256(args.h5ad),
            "shape": h5ad_shape,
            "obs_columns": obs_columns,
        },
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("ENVIRONMENT_SMOKE_TEST=PASS")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
