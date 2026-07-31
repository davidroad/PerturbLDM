#!/usr/bin/env python3
"""Stream sparse H5ADs into auditable condition-level mean-expression caches."""

from __future__ import annotations

import argparse
import json
import os
import resource
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


RUN = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = RUN.parents[1]
DEFAULT_SPLIT = BENCHMARK_ROOT / "splits/condition_assignments_seed42.csv"
DEFAULT_INPUTS = {
    "train": BENCHMARK_ROOT / "external_inputs/tahoe/train_adata_processed.h5ad",
    "test": BENCHMARK_ROOT / "external_inputs/tahoe/test_adata_processed.h5ad",
    "control": BENCHMARK_ROOT / "external_inputs/tahoe/control_adata_processed.h5ad",
}
DOSE_ID_COMPONENTS = {0.05: "0-05", 0.5: "0-5", 5.0: "5-0"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def decode(values) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def input_stat(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def categorical(group: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    require(group.attrs.get("encoding-type") == "categorical", "expected categorical H5AD field")
    return decode(group["categories"][:]), np.asarray(group["codes"][:])


def dose_component(value: float) -> str:
    numeric = float(value)
    require(numeric in DOSE_ID_COMPONENTS, f"unsupported dose for condition ID: {numeric}")
    return DOSE_ID_COMPONENTS[numeric]


def normalize_cell_line(value: object) -> str:
    return str(value).strip().replace("CVCL_", "CVCL-")


def normalize_drug(value: object) -> str:
    return str(value).strip().replace("DMSO_TF", "DMSO-TF")


def condition_metadata(handle: h5py.File, source: str, split_manifest: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    condition_categories, condition_codes = categorical(handle["obs/CondID"])
    require(np.min(condition_codes) >= 0, "missing condition code")
    unique_codes, first_rows = np.unique(condition_codes, return_index=True)
    require(
        np.array_equal(unique_codes, np.arange(len(condition_categories))),
        "condition categories are not fully represented",
    )

    cell_categories, cell_codes = categorical(handle["obs/cell_line"])
    drug_categories, drug_codes = categorical(handle["obs/drug"])
    doses = np.asarray(handle["obs/dose"][:], dtype=np.float64)

    cells = cell_categories[cell_codes[first_rows]]
    drugs = drug_categories[drug_codes[first_rows]]
    condition_ids = np.asarray(
        [
            f"{normalize_cell_line(cell)}_"
            f"{normalize_drug(drug)}_"
            f"{dose_component(dose)}"
            for cell, drug, dose in zip(cells, drugs, doses[first_rows])
        ],
        dtype=object,
    )
    metadata = pd.DataFrame(
        {
            "condition_code": unique_codes.astype(np.int32),
            "CondID": condition_categories[unique_codes],
            "condition_id": condition_ids,
            "cell_line": cells,
            "drug": drugs,
            "dose": doses[first_rows],
            "first_source_row": first_rows.astype(np.int64),
        }
    )
    require(metadata["condition_id"].is_unique, f"duplicate normalized IDs in {source}")

    manifest = split_manifest.set_index("condition_id")
    expected_original = "train" if source == "train" else "test"
    missing = set(metadata["condition_id"]).difference(manifest.index)
    require(
        not missing,
        f"{source} conditions absent from split manifest: {len(missing)}; "
        f"examples={sorted(missing)[:10]}",
    )
    aligned = manifest.loc[metadata["condition_id"]]
    require(aligned["original_split"].eq(expected_original).all(), f"{source} original split mismatch")
    metadata["search_split"] = aligned["cpa_split"].to_numpy()
    expected_count = 32_529 if source == "train" else 13_942
    require(len(metadata) == expected_count, f"{source} condition count={len(metadata)}")
    if source == "train":
        require(metadata["search_split"].value_counts().to_dict() == {"train": 29_277, "valid": 3_252}, "train/valid counts differ")
    else:
        require(metadata["search_split"].eq("ood").all(), "external test is not exclusively OOD")
    return metadata, condition_codes


def choose_condition_subset(metadata: pd.DataFrame, per_split_limit: int | None) -> pd.DataFrame:
    if per_split_limit is None:
        return metadata.sort_values("condition_code").reset_index(drop=True)
    if set(metadata["search_split"].unique()) == {"train", "valid"}:
        train_pool = metadata.loc[metadata["search_split"].eq("train")].sort_values(
            "first_source_row"
        )
        valid = metadata.loc[metadata["search_split"].eq("valid")].sort_values(
            "first_source_row"
        ).head(per_split_limit)
        covering_train = []
        for drug in valid["drug"].astype(str).unique():
            candidates = train_pool.loc[train_pool["drug"].astype(str).eq(drug)]
            require(len(candidates) > 0, f"no internal-training coverage for dry-run drug: {drug}")
            covering_train.append(candidates.head(1))
        for cell_line in valid["cell_line"].astype(str).unique():
            candidates = train_pool.loc[train_pool["cell_line"].astype(str).eq(cell_line)]
            require(
                len(candidates) > 0,
                f"no internal-training coverage for dry-run cell line: {cell_line}",
            )
            covering_train.append(candidates.head(1))
        train = pd.concat(covering_train + [train_pool.head(per_split_limit)], ignore_index=True)
        train = train.drop_duplicates("condition_code")
        output = pd.concat([train, valid], ignore_index=True)
        return output.sort_values("condition_code").reset_index(drop=True)
    selected = []
    for split in sorted(metadata["search_split"].unique()):
        block = metadata.loc[metadata["search_split"].eq(split)].sort_values("first_source_row")
        selected.append(block.head(per_split_limit))
    output = pd.concat(selected, ignore_index=True).sort_values("condition_code").reset_index(drop=True)
    require(len(output) > 0, "empty condition subset")
    return output


def csr_rows(group: h5py.Group, start: int, stop: int, n_genes: int) -> sparse.csr_matrix:
    indptr = np.asarray(group["indptr"][start : stop + 1], dtype=np.int64)
    lower = int(indptr[0])
    upper = int(indptr[-1])
    data = np.asarray(group["data"][lower:upper], dtype=np.float32)
    indices = np.asarray(group["indices"][lower:upper], dtype=np.int64)
    indptr -= lower
    return sparse.csr_matrix((data, indices, indptr), shape=(stop - start, n_genes))


def aggregate_groups(
    handle: h5py.File,
    group_codes: np.ndarray,
    selected_codes: np.ndarray,
    output_dir: Path,
    output_name: str,
    chunk_rows: int,
) -> tuple[Path, Path, dict]:
    x = handle["X"]
    require(x.attrs.get("encoding-type") == "csr_matrix", "X must be CSR")
    n_rows, n_genes = [int(value) for value in x.attrs["shape"]]
    require(len(group_codes) == n_rows, "group-code row count differs from X")
    output_dir.mkdir(parents=True, exist_ok=True)
    final_means = output_dir / f"{output_name}.npy"
    final_counts = output_dir / f"{output_name}_counts.npy"
    progress_path = output_dir / f"{output_name}_progress.json"
    partial_means = output_dir / f".{output_name}.partial.npy"
    partial_counts = output_dir / f".{output_name}_counts.partial.npy"

    if final_means.is_file() and final_counts.is_file():
        counts = np.load(final_counts)
        return final_means, final_counts, {"status": "already_complete", "groups": len(counts)}

    code_to_output = np.full(int(group_codes.max()) + 1, -1, dtype=np.int32)
    code_to_output[selected_codes] = np.arange(len(selected_codes), dtype=np.int32)
    if progress_path.is_file() and partial_means.is_file() and partial_counts.is_file():
        progress = json.loads(progress_path.read_text())
        require(progress["selected_codes"] == selected_codes.astype(int).tolist(), "resume selection changed")
        next_row = int(progress["next_row"])
        sums = np.lib.format.open_memmap(partial_means, mode="r+")
        counts = np.lib.format.open_memmap(partial_counts, mode="r+")
    else:
        next_row = 0
        sums = np.lib.format.open_memmap(
            partial_means, mode="w+", dtype=np.float32, shape=(len(selected_codes), n_genes)
        )
        counts = np.lib.format.open_memmap(
            partial_counts, mode="w+", dtype=np.int64, shape=(len(selected_codes),)
        )
        sums[:] = 0
        counts[:] = 0
        sums.flush()
        counts.flush()

    for start in range(next_row, n_rows, chunk_rows):
        stop = min(n_rows, start + chunk_rows)
        chunk_codes = group_codes[start:stop]
        mapped = code_to_output[chunk_codes]
        active = mapped >= 0
        if np.any(active):
            matrix = csr_rows(x, start, stop, n_genes)[active]
            active_codes = mapped[active]
            unique_output, inverse = np.unique(active_codes, return_inverse=True)
            grouping = sparse.csr_matrix(
                (
                    np.ones(len(inverse), dtype=np.float32),
                    (inverse, np.arange(len(inverse), dtype=np.int64)),
                ),
                shape=(len(unique_output), len(inverse)),
            )
            local_sums = (grouping @ matrix).toarray().astype(np.float32, copy=False)
            sums[unique_output, :] = sums[unique_output, :] + local_sums
            counts[unique_output] = counts[unique_output] + np.bincount(
                inverse, minlength=len(unique_output)
            )
        sums.flush()
        counts.flush()
        write_json_atomic(
            progress_path,
            {
                "next_row": stop,
                "n_rows": n_rows,
                "selected_codes": selected_codes.astype(int).tolist(),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
        )

    require(np.all(np.asarray(counts) > 0), "one or more selected groups have no cells")
    for start in range(0, len(selected_codes), 256):
        stop = min(len(selected_codes), start + 256)
        sums[start:stop, :] = sums[start:stop, :] / np.asarray(counts[start:stop])[:, None]
    sums.flush()
    counts.flush()
    del sums, counts
    partial_means.replace(final_means)
    partial_counts.replace(final_counts)
    progress_path.unlink(missing_ok=True)
    return final_means, final_counts, {
        "status": "complete",
        "groups": int(len(selected_codes)),
        "source_rows": n_rows,
        "genes": n_genes,
    }


def build_response_cache(
    source: str,
    input_path: Path,
    split_manifest: pd.DataFrame,
    output_root: Path,
    chunk_rows: int,
    per_split_limit: int | None,
) -> dict:
    before = input_stat(input_path)
    with h5py.File(input_path, "r") as handle:
        metadata, condition_codes = condition_metadata(handle, source, split_manifest)
        selected = choose_condition_subset(metadata, per_split_limit)
        selected_codes = selected["condition_code"].to_numpy(dtype=np.int32)
        out = output_root / source
        means, counts, summary = aggregate_groups(
            handle, condition_codes, selected_codes, out, "response_means", chunk_rows
        )
        selected.to_csv(out / "metadata.csv", index=False)
        genes = decode(handle["var/gene_name"][:])
        pd.DataFrame({"gene": genes}).to_csv(output_root / "genes.csv", index=False)
    after = input_stat(input_path)
    require(before == after, f"raw {source} H5AD changed during read")
    return {
        "source": source,
        "input": before,
        "input_unchanged": True,
        "means": str(means),
        "counts": str(counts),
        "metadata": str(out / "metadata.csv"),
        **summary,
    }


def build_control_cache(input_path: Path, output_root: Path, chunk_rows: int) -> dict:
    before = input_stat(input_path)
    with h5py.File(input_path, "r") as handle:
        cell_categories, cell_codes = categorical(handle["obs/cell_line"])
        selected_codes = np.arange(len(cell_categories), dtype=np.int32)
        out = output_root / "control"
        means, counts, summary = aggregate_groups(
            handle, cell_codes, selected_codes, out, "control_means", chunk_rows
        )
        metadata = pd.DataFrame(
            {
                "cell_line_code": selected_codes,
                "cell_line": cell_categories,
                "normalized_cell_line": [
                    normalize_cell_line(value) for value in cell_categories
                ],
            }
        )
        require(metadata["normalized_cell_line"].is_unique, "duplicate control cell lines")
        metadata.to_csv(out / "metadata.csv", index=False)
        if not (output_root / "genes.csv").is_file():
            genes = decode(handle["var/gene_name"][:])
            pd.DataFrame({"gene": genes}).to_csv(output_root / "genes.csv", index=False)
    after = input_stat(input_path)
    require(before == after, "raw control H5AD changed during read")
    return {
        "source": "control",
        "input": before,
        "input_unchanged": True,
        "means": str(means),
        "counts": str(counts),
        "metadata": str(out / "metadata.csv"),
        **summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["train", "control", "test"], default=["train", "control"])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--train-h5ad", type=Path, default=DEFAULT_INPUTS["train"])
    parser.add_argument("--test-h5ad", type=Path, default=DEFAULT_INPUTS["test"])
    parser.add_argument("--control-h5ad", type=Path, default=DEFAULT_INPUTS["control"])
    parser.add_argument("--chunk-rows", type=int, default=20_000)
    parser.add_argument("--condition-limit-per-split", type=int)
    parser.add_argument("--allow-test-response", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.chunk_rows > 0, "chunk rows must be positive")
    if "test" in args.datasets:
        require(args.allow_test_response, "test response access requires --allow-test-response")
        selection_files = [RUN / "results/mlp/selection.json", RUN / "results/rf/selection.json"]
        require(all(path.is_file() for path in selection_files), "MLP and RF selections must be frozen before test caching")
    args.output_root.mkdir(parents=True, exist_ok=True)
    split_manifest = pd.read_csv(args.split_manifest, dtype=str)
    require(split_manifest["condition_id"].is_unique, "split manifest condition IDs are not unique")
    input_paths = {
        "train": args.train_h5ad,
        "test": args.test_h5ad,
        "control": args.control_h5ad,
    }
    reports = []
    for dataset in args.datasets:
        if dataset == "control":
            reports.append(build_control_cache(input_paths[dataset], args.output_root, args.chunk_rows))
        else:
            reports.append(
                build_response_cache(
                    dataset,
                    input_paths[dataset],
                    split_manifest,
                    args.output_root,
                    args.chunk_rows,
                    args.condition_limit_per_split,
                )
            )
    genes = pd.read_csv(args.output_root / "genes.csv")
    require(len(genes) == 13_784 and genes["gene"].is_unique, "gene contract failed")
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CACHE_OK",
        "output_root": str(args.output_root),
        "datasets": reports,
        "gene_count": len(genes),
        "split_manifest": str(args.split_manifest),
        "condition_id_normalization": {
            "cell_line": "trim leading/trailing whitespace; replace CVCL_ with CVCL-",
            "drug": "trim leading/trailing whitespace; replace DMSO_TF with DMSO-TF; preserve meaningful internal spaces",
            "dose_components": {str(key): value for key, value in DOSE_ID_COMPONENTS.items()},
            "match_rule": "exact one-to-one match after normalization; no fuzzy matching or silent dropping",
        },
        "test_response_accessed": "test" in args.datasets,
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    write_json_atomic(args.output_root / "cache_manifest.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
