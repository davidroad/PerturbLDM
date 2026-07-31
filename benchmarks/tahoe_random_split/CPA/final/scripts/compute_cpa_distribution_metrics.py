#!/usr/bin/env python3
"""Compute CPA distribution metrics in parallel without splitting the test H5AD.

The external test H5AD remains a single read-only source.  The parent process
loads only observation metadata and the CSR row pointer.  Forked workers open
the same H5AD read-only and materialize only the rows needed for one condition.
Each worker processes one existing per-cell-line CPA prediction H5AD at a time.

The scientific metric contract is unchanged from
``cpa_random_counterfactor_distribution_corrected_split.py``:

* condition-specific MMD sigma from real cells only (seed 42);
* unbiased RBF MMD;
* the existing flattened-value E-distance implementation;
* 128-projection sliced Wasserstein;
* exact POT EMD with ``ot_reg=None``;
* the same 5,000-cell and 2,000-cell caps.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# One numerical thread per worker prevents nested oversubscription.  Parallelism
# is across independent cell-line workers.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["MKL_DYNAMIC"] = "FALSE"

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from chemcpa_distribution_metrics import (
    _median_heuristic_sigma_from_real_only,
    compute_condition_distribution_metrics,
)


METRICS = ["MMD_RBF", "E_distance", "Wasserstein_Sliced", "Wasserstein_OT"]
EXPECTED_FULL_CONDITIONS = 13_942
EXPECTED_FULL_CELL_LINES = 47

# Read-only globals inherited by forked workers.
_TEST_PATH: Path | None = None
_TEST_ROWS: dict[str, np.ndarray] = {}
_TEST_INDPTR: np.ndarray | None = None
_TEST_N_VARS = 0
_TEST_HANDLE: h5py.File | None = None
_TEST_X: h5py.Group | None = None
_SHARD_DIR: Path | None = None
_FORCE = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_cell_line(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.strip()
        .str.replace("CVCL_", "CVCL-", regex=False)
    )


def normalize_drug(values: pd.Series) -> pd.Series:
    # This matches the historical distribution workflow.  The authoritative
    # OOD manifest contains no drug name with a literal underscore.
    return values.astype(str).str.strip().str.replace("_", "-", regex=False)


def normalize_dose(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(".", "-", regex=False)


def condition_id(cell_line: object, drug: object, dose_str: object) -> str:
    return f"{cell_line}_{drug}_{dose_str}"


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_cell_line_from_prediction(path: Path) -> str:
    name = path.name
    prefix = "cpa_inference_"
    suffix = ".h5ad"
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"unexpected inference filename: {name}")
    return name[len(prefix) : -len(suffix)].replace("_", "-")


def load_ood_manifest(path: Path, selected_cell_lines: set[str] | None) -> set[str]:
    manifest = pd.read_csv(path, dtype=str)
    required = {"condition_id", "cpa_split"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"manifest lacks columns: {required - set(manifest.columns)}")
    values = set(manifest.loc[manifest["cpa_split"].eq("ood"), "condition_id"])
    if selected_cell_lines:
        values = {
            value
            for value in values
            if value.split("_", 1)[0] in selected_cell_lines
        }
    if not values:
        raise ValueError("selected OOD condition set is empty")
    return values


def build_test_row_index(test_path: Path, expected: set[str]) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """Read test metadata only and map each selected condition to CSR rows."""
    started = time.time()
    backed = ad.read_h5ad(test_path, backed="r")
    try:
        obs = backed.obs
        frame = pd.DataFrame(
            {
                "cell_line": normalize_cell_line(obs["cell_line"]),
                "drug": normalize_drug(obs["drug"]),
                "dose_str": normalize_dose(obs["dose"]),
            }
        )
        grouped = frame.groupby(
            ["cell_line", "drug", "dose_str"],
            sort=False,
            observed=True,
        ).indices
        rows = {
            condition_id(*key): np.asarray(indices, dtype=np.int64)
            for key, indices in grouped.items()
            if condition_id(*key) in expected
        }
    finally:
        backed.file.close()

    missing = expected - set(rows)
    extra = set(rows) - expected
    if missing or extra:
        raise AssertionError(
            f"test row-index mismatch: missing={len(missing)} extra={len(extra)}"
        )

    with h5py.File(test_path, "r") as handle:
        matrix = handle["X"]
        if not isinstance(matrix, h5py.Group):
            raise TypeError("expected CSR-encoded test X")
        if matrix.attrs.get("encoding-type") != "csr_matrix":
            raise TypeError(f"unexpected test X encoding: {dict(matrix.attrs)}")
        shape = tuple(int(v) for v in matrix.attrs["shape"])
        indptr = matrix["indptr"][:]

    if len(indptr) != shape[0] + 1:
        raise AssertionError("CSR indptr length does not match test matrix rows")
    print(
        f"INDEX_READY conditions={len(rows)} test_rows={shape[0]} "
        f"genes={shape[1]} seconds={time.time() - started:.1f}",
        flush=True,
    )
    del frame, grouped
    gc.collect()
    return rows, indptr, shape[1]


def worker_initializer() -> None:
    global _TEST_HANDLE, _TEST_X
    if _TEST_PATH is None:
        raise RuntimeError("test path was not initialized before fork")
    _TEST_HANDLE = h5py.File(_TEST_PATH, "r")
    matrix = _TEST_HANDLE["X"]
    if not isinstance(matrix, h5py.Group):
        raise TypeError("worker expected CSR-encoded test X")
    _TEST_X = matrix


def consecutive_runs(rows: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive row runs from sorted, unique row indices."""
    if len(rows) == 0:
        return []
    breaks = np.flatnonzero(np.diff(rows) != 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(rows) - 1]
    return [(int(rows[start]), int(rows[end])) for start, end in zip(starts, ends)]


def read_test_rows(rows: np.ndarray) -> np.ndarray:
    """Materialize selected CSR rows without loading or splitting the test H5AD."""
    if _TEST_X is None or _TEST_INDPTR is None:
        raise RuntimeError("worker test matrix is not initialized")
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) == 0:
        return np.empty((0, _TEST_N_VARS), dtype=np.float32)
    if np.any(np.diff(rows) <= 0):
        rows = np.unique(rows)

    lengths = _TEST_INDPTR[rows + 1] - _TEST_INDPTR[rows]
    local_indptr = np.empty(len(rows) + 1, dtype=np.int64)
    local_indptr[0] = 0
    np.cumsum(lengths, out=local_indptr[1:])

    data_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    for first_row, last_row in consecutive_runs(rows):
        start = int(_TEST_INDPTR[first_row])
        end = int(_TEST_INDPTR[last_row + 1])
        data_parts.append(_TEST_X["data"][start:end])
        index_parts.append(_TEST_X["indices"][start:end])

    data = data_parts[0] if len(data_parts) == 1 else np.concatenate(data_parts)
    indices = index_parts[0] if len(index_parts) == 1 else np.concatenate(index_parts)
    matrix = sparse.csr_matrix(
        (data, indices, local_indptr),
        shape=(len(rows), _TEST_N_VARS),
    )
    return matrix.toarray().astype(np.float32, copy=False)


def shard_paths(cell_line: str) -> tuple[Path, Path]:
    if _SHARD_DIR is None:
        raise RuntimeError("shard directory is not initialized")
    safe = cell_line.replace("-", "_")
    return (
        _SHARD_DIR / f"{safe}.csv",
        _SHARD_DIR / f"{safe}.meta.json",
    )


def valid_existing_shard(
    csv_path: Path,
    meta_path: Path,
    source_path: Path,
    expected_conditions: set[str],
) -> bool:
    if _FORCE or not csv_path.is_file() or not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text())
        stat = source_path.stat()
        if (
            meta["source_prediction_bytes"] != stat.st_size
            or meta["source_prediction_mtime_ns"] != stat.st_mtime_ns
        ):
            return False
        data = pd.read_csv(csv_path, low_memory=False)
        return (
            len(data) == len(expected_conditions)
            and set(data["condition"]) == expected_conditions
            and data["condition"].nunique() == len(data)
            and data["status"].eq("success").all()
            and all(pd.to_numeric(data[m], errors="coerce").notna().all() for m in METRICS)
            and data["Wasserstein_OT_type"].eq("ot_emd").all()
        )
    except Exception:
        return False


def process_cell_line(task: tuple[str, str, tuple[str, ...]]) -> dict[str, Any]:
    source_text, cell_line, expected_tuple = task
    source_path = Path(source_text)
    expected_conditions = set(expected_tuple)
    csv_path, meta_path = shard_paths(cell_line)
    if valid_existing_shard(
        csv_path,
        meta_path,
        source_path,
        expected_conditions,
    ):
        return {
            "cell_line": cell_line,
            "conditions": len(expected_conditions),
            "seconds": 0.0,
            "status": "reused",
            "csv": str(csv_path),
        }

    started = time.time()
    backed = ad.read_h5ad(source_path, backed="r")
    try:
        obs = backed.obs.copy()
        obs["cell_line"] = normalize_cell_line(obs["cell_line"])
        obs["drug"] = normalize_drug(obs["drug"])
        obs["dose_str"] = normalize_dose(obs["dose"])
        prediction = np.asarray(backed.obsm["CPA_pred"], dtype=np.float32)
    finally:
        backed.file.close()

    if prediction.shape[1] != _TEST_N_VARS:
        raise AssertionError(
            f"{cell_line}: prediction genes={prediction.shape[1]} test genes={_TEST_N_VARS}"
        )
    pred_groups_raw = obs.groupby(
        ["cell_line", "drug", "dose_str"],
        sort=False,
        observed=True,
    ).indices
    pred_groups = {
        condition_id(*key): np.asarray(indices, dtype=np.int64)
        for key, indices in pred_groups_raw.items()
    }
    if set(pred_groups) != expected_conditions:
        raise AssertionError(
            f"{cell_line}: prediction condition mismatch "
            f"missing={len(expected_conditions - set(pred_groups))} "
            f"extra={len(set(pred_groups) - expected_conditions)}"
        )

    results: list[dict[str, Any]] = []
    for index, cid in enumerate(sorted(expected_conditions), start=1):
        left, dose_str = cid.rsplit("_", 1)
        cid_cell_line, drug = left.split("_", 1)
        real_expr = read_test_rows(_TEST_ROWS[cid])
        pred_expr = prediction[pred_groups[cid]]
        sigma = _median_heuristic_sigma_from_real_only(
            real_expr,
            max_samples=2000,
            rng=42,
        )
        result = compute_condition_distribution_metrics(
            real_expr=real_expr,
            pred_expr=pred_expr,
            condition_name=cid,
            cell_line=cid_cell_line,
            drug=drug,
            dose=dose_str,
            subsample=5000,
            rng=42,
            mmd_sigma=sigma,
            sw_projections=128,
            sw_grid_size=400,
            ot_reg=None,
            ot_subsample=2000,
        )
        result["mmd_sigma_used"] = sigma
        results.append(result)
        if index == 1 or index % 25 == 0 or index == len(expected_conditions):
            print(
                f"WORKER_PROGRESS cell_line={cell_line} "
                f"condition={index}/{len(expected_conditions)}",
                flush=True,
            )
        del real_expr, pred_expr

    frame = pd.DataFrame(results)
    if (
        len(frame) != len(expected_conditions)
        or set(frame["condition"]) != expected_conditions
        or not frame["status"].eq("success").all()
        or not frame["Wasserstein_OT_type"].eq("ot_emd").all()
    ):
        raise AssertionError(f"{cell_line}: completed shard failed its integrity checks")
    atomic_write_csv(frame, csv_path)
    source_stat = source_path.stat()
    elapsed = time.time() - started
    meta = {
        "timestamp_utc": utc_now(),
        "cell_line": cell_line,
        "conditions": len(frame),
        "seconds": elapsed,
        "source_prediction": str(source_path),
        "source_prediction_bytes": source_stat.st_size,
        "source_prediction_mtime_ns": source_stat.st_mtime_ns,
        "test_h5ad": str(_TEST_PATH),
        "metric_contract": {
            "rng": 42,
            "mmd_sigma": "real-only median heuristic",
            "mmd_unbiased": True,
            "subsample": 5000,
            "sw_projections": 128,
            "sw_grid_size": 400,
            "ot_reg": None,
            "ot_subsample": 2000,
            "ot_type": "exact EMD",
        },
        "worker_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    atomic_write_json(meta, meta_path)
    del prediction, obs, pred_groups_raw, pred_groups, frame, results
    gc.collect()
    return {
        "cell_line": cell_line,
        "conditions": len(expected_conditions),
        "seconds": elapsed,
        "status": "computed",
        "csv": str(csv_path),
        "peak_rss_kib": meta["worker_peak_rss_kib"],
    }


def build_tasks(
    inference_dir: Path,
    expected: set[str],
    selected_cell_lines: set[str] | None,
) -> list[tuple[str, str, tuple[str, ...]]]:
    by_cell_line: dict[str, list[str]] = {}
    for cid in expected:
        by_cell_line.setdefault(cid.split("_", 1)[0], []).append(cid)
    files = sorted(inference_dir.glob("cpa_inference_*.h5ad"))
    file_map = {parse_cell_line_from_prediction(path): path for path in files}
    cell_lines = sorted(selected_cell_lines or set(by_cell_line))
    missing_files = set(cell_lines) - set(file_map)
    if missing_files:
        raise FileNotFoundError(f"missing inference H5ADs: {sorted(missing_files)}")
    tasks = [
        (
            str(file_map[cell_line]),
            cell_line,
            tuple(sorted(by_cell_line[cell_line])),
        )
        for cell_line in cell_lines
    ]
    return tasks


def merge_and_audit(
    tasks: list[tuple[str, str, tuple[str, ...]]],
    output_root: Path,
    expected: set[str],
) -> pd.DataFrame:
    frames = []
    for _, cell_line, _ in tasks:
        csv_path, _ = shard_paths(cell_line)
        frames.append(pd.read_csv(csv_path, low_memory=False))
    merged = pd.concat(frames, ignore_index=True)
    if len(merged) != len(expected):
        raise AssertionError(f"merged rows={len(merged)} expected={len(expected)}")
    if merged["condition"].nunique() != len(merged):
        raise AssertionError("merged output contains duplicate conditions")
    if set(merged["condition"]) != expected:
        raise AssertionError("merged condition set differs from selected OOD manifest")
    if not merged["status"].eq("success").all():
        raise AssertionError("merged output contains failed conditions")
    for metric in METRICS:
        values = pd.to_numeric(merged[metric], errors="coerce")
        if not values.notna().all() or not np.isfinite(values).all():
            raise AssertionError(f"merged output has missing/non-finite {metric}")
    if not merged["Wasserstein_OT_type"].eq("ot_emd").all():
        raise AssertionError("merged output contains non-exact OT")

    merged = merged.sort_values("condition").reset_index(drop=True)
    atomic_write_csv(merged, output_root / "global_condition_metrics.csv")

    cellline_rows = []
    for cell_line, data in merged.groupby("cell_line", sort=True):
        row: dict[str, Any] = {
            "cell_line": cell_line,
            "status": "completed",
            "n_conditions": len(data),
            "n_successful_conditions": int(data["status"].eq("success").sum()),
            "total_cells_analyzed": int(data["n_pred_cells"].sum()),
            "total_real_cells": int(data["n_real_cells"].sum()),
        }
        for metric in METRICS:
            values = pd.to_numeric(data[metric])
            row[f"avg_{metric}"] = float(values.mean())
            row[f"std_{metric}"] = float(values.std(ddof=0))
        cellline_rows.append(row)
    atomic_write_csv(
        pd.DataFrame(cellline_rows),
        output_root / "global_cellline_metrics.csv",
    )
    return merged


def compare_reference(
    merged: pd.DataFrame,
    reference_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    reference = pd.read_csv(reference_path, low_memory=False).set_index("condition")
    current = merged.set_index("condition")
    missing = set(current.index) - set(reference.index)
    if missing:
        raise AssertionError(f"reference lacks {len(missing)} selected conditions")
    rows = []
    all_close = True
    for metric in METRICS + ["mmd_sigma_used"]:
        old = pd.to_numeric(reference.loc[current.index, metric]).to_numpy()
        new = pd.to_numeric(current[metric]).to_numpy()
        difference = np.abs(new - old)
        close = np.isclose(new, old, rtol=1e-7, atol=1e-10, equal_nan=False)
        metric_close = bool(close.all())
        all_close = all_close and metric_close
        rows.append(
            {
                "metric": metric,
                "all_close_rtol_1e-7_atol_1e-10": metric_close,
                "max_abs_difference": float(difference.max()),
                "mean_abs_difference": float(difference.mean()),
            }
        )
    report = {
        "timestamp_utc": utc_now(),
        "reference": str(reference_path),
        "conditions_compared": len(current),
        "all_metrics_close": all_close,
        "metric_comparisons": rows,
    }
    atomic_write_json(report, output_path)
    if not all_close:
        raise AssertionError("parallel implementation differs from reference metrics")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-h5ad", type=Path, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cell-lines", nargs="*")
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reference-csv", type=Path)
    args = parser.parse_args()

    selected = set(args.cell_lines) if args.cell_lines else None
    expected = load_ood_manifest(args.manifest, selected)
    if not args.allow_subset:
        if len(expected) != EXPECTED_FULL_CONDITIONS:
            raise AssertionError(
                f"full run requires {EXPECTED_FULL_CONDITIONS} OOD conditions; "
                f"selected={len(expected)}"
            )

    global _TEST_PATH, _TEST_ROWS, _TEST_INDPTR, _TEST_N_VARS, _SHARD_DIR, _FORCE
    _TEST_PATH = args.test_h5ad.resolve()
    _SHARD_DIR = args.output_root.resolve() / "shards"
    _FORCE = args.force
    args.output_root.mkdir(parents=True, exist_ok=True)
    _SHARD_DIR.mkdir(parents=True, exist_ok=True)

    _TEST_ROWS, _TEST_INDPTR, _TEST_N_VARS = build_test_row_index(
        _TEST_PATH,
        expected,
    )
    tasks = build_tasks(args.inference_dir, expected, selected)
    if not args.allow_subset and len(tasks) != EXPECTED_FULL_CELL_LINES:
        raise AssertionError(
            f"full run requires {EXPECTED_FULL_CELL_LINES} cell lines; tasks={len(tasks)}"
        )
    workers = max(1, min(args.workers, len(tasks)))
    print(
        f"RUN_START conditions={len(expected)} cell_lines={len(tasks)} "
        f"workers={workers} test_mode=single_read_only_h5ad",
        flush=True,
    )

    started = time.time()
    context = mp.get_context("fork")
    worker_reports: list[dict[str, Any]] = []
    with context.Pool(processes=workers, initializer=worker_initializer) as pool:
        for report in pool.imap_unordered(process_cell_line, tasks):
            worker_reports.append(report)
            print(
                f"SHARD_DONE cell_line={report['cell_line']} "
                f"conditions={report['conditions']} status={report['status']} "
                f"seconds={report['seconds']:.1f}",
                flush=True,
            )

    merged = merge_and_audit(tasks, args.output_root, expected)
    reference_report = None
    if args.reference_csv:
        reference_report = compare_reference(
            merged,
            args.reference_csv,
            args.output_root / "reference_equivalence_audit.json",
        )

    run_report = {
        "timestamp_utc": utc_now(),
        "status": "completed",
        "conditions": len(merged),
        "cell_lines": len(tasks),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "test_h5ad": str(_TEST_PATH),
        "test_h5ad_mode": "single read-only CSR source; no split-data copies",
        "inference_dir": str(args.inference_dir.resolve()),
        "output_root": str(args.output_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "worker_reports": sorted(worker_reports, key=lambda value: value["cell_line"]),
        "reference_equivalence": reference_report,
    }
    atomic_write_json(run_report, args.output_root / "parallel_distribution_run.json")
    print(
        f"RUN_COMPLETE conditions={len(merged)} cell_lines={len(tasks)} "
        f"seconds={run_report['elapsed_seconds']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
