#!/usr/bin/env python3
"""Audit the processed PBMC input against the manuscript split contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad


TARGET_CELL_TYPES = ("B cells", "CD8 T cells", "FCGR3A+ Monocytes")
EXCLUDED_CELL_TYPES = ("Megakaryocytes",)
REQUIRED_OBS_COLUMNS = ("cell.type", "stim")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate PBMC annotations and report the exact PerturbLDM hold-out split."
    )
    parser.add_argument("--input", required=True, type=Path, help="Processed PBMC H5AD")
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional JSON destination; the H5AD is never modified.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    adata = ad.read_h5ad(args.input, backed="r")
    missing = [column for column in REQUIRED_OBS_COLUMNS if column not in adata.obs]
    if missing:
        raise SystemExit(f"Missing required adata.obs columns: {missing}")

    cell_type = adata.obs["cell.type"].astype(str)
    stimulation = adata.obs["stim"].astype(str)
    included = ~cell_type.isin(EXCLUDED_CELL_TYPES)
    held_out = included & cell_type.isin(TARGET_CELL_TYPES) & stimulation.eq("stim")
    fitting = included & ~held_out
    excluded = ~included

    issues: list[str] = []
    observed_stimulation_labels = sorted(stimulation.unique().tolist())
    for label in ("ctrl", "stim"):
        if label not in observed_stimulation_labels:
            issues.append(f"required stimulation label absent: {label}")

    targets: list[dict[str, object]] = []
    for target in TARGET_CELL_TYPES:
        target_mask = cell_type.eq(target)
        ctrl_count = int((target_mask & stimulation.eq("ctrl") & fitting).sum())
        stim_count = int((target_mask & stimulation.eq("stim") & held_out).sum())
        if ctrl_count == 0:
            issues.append(f"no retained ctrl cells for target: {target}")
        if stim_count == 0:
            issues.append(f"no held-out stim cells for target: {target}")
        targets.append(
            {
                "cell_type": target,
                "fitting_ctrl_cells": ctrl_count,
                "held_out_stim_cells": stim_count,
            }
        )

    if int((fitting & held_out).sum()) != 0:
        issues.append("fitting and held-out masks overlap")
    if int((fitting | held_out | excluded).sum()) != int(adata.n_obs):
        issues.append("split masks do not cover every cell")

    grouped = (
        adata.obs.assign(
            audit_split=[
                "excluded" if is_excluded else "held_out" if is_held_out else "fit"
                for is_excluded, is_held_out in zip(excluded, held_out)
            ]
        )
        .groupby(["audit_split", "cell.type", "stim"], observed=True)
        .size()
        .reset_index(name="cells")
    )

    report = {
        "input": str(args.input),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "required_obs_columns": list(REQUIRED_OBS_COLUMNS),
        "observed_stimulation_labels": observed_stimulation_labels,
        "target_cell_types": list(TARGET_CELL_TYPES),
        "excluded_cell_types": list(EXCLUDED_CELL_TYPES),
        "fitting_cells": int(fitting.sum()),
        "held_out_cells": int(held_out.sum()),
        "excluded_cells": int(excluded.sum()),
        "target_summary": targets,
        "combination_counts": grouped.to_dict(orient="records"),
        "feature_selection_scope": "complete archived task object (transductive)",
        "issues": issues,
        "passed": not issues,
    }

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(rendered + "\n", encoding="utf-8")
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
