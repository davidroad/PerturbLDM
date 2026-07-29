#!/usr/bin/env python3
"""Validate curated release provenance and refresh its compact audit summary.

The 2026-07-24 release index is curated evidence. This script intentionally
does not rebuild it from an older private draft, because doing so would erase
the recovered Tahoe benchmark lineage.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


CANDIDATE = Path(__file__).resolve().parents[2]
PROVENANCE = CANDIDATE / "figshare" / "provenance"
INDEX = PROVENANCE / "figure_panel_reproducibility_index.tsv"
GAPS = PROVENANCE / "previous_missing_item_resolution.tsv"
VALID_STATUSES = {
    "reproducible",
    "reproducible_with_external_inputs",
    "metrics_and_output_only",
    "wrapper_needed",
    "external_input",
    "not_reconstructable",
}
REQUIRED_COLUMNS = {
    "figure_panel",
    "analysis_object",
    "unit_of_analysis",
    "upstream_accession",
    "private_upstream_object",
    "released_metrics",
    "reproduce_script",
    "released_output",
    "statistical_method",
    "status",
    "notes",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate() -> dict[str, object]:
    panels = read_tsv(INDEX)
    gaps = read_tsv(GAPS)
    issues: list[str] = []
    if not panels:
        issues.append("panel index is empty")
    else:
        missing = sorted(REQUIRED_COLUMNS.difference(panels[0]))
        if missing:
            issues.append(f"panel index missing columns: {missing}")
    panel_ids = [row.get("figure_panel", "") for row in panels]
    duplicates = sorted({panel for panel in panel_ids if panel_ids.count(panel) > 1})
    if duplicates:
        issues.append(f"duplicate panel IDs: {duplicates}")
    invalid = sorted({row.get("status", "") for row in panels}.difference(VALID_STATUSES))
    if invalid:
        issues.append(f"invalid panel statuses: {invalid}")
    if not any(row.get("figure_panel") == "Fig2b" and
               row.get("status") == "reproducible_with_external_inputs"
               for row in panels):
        issues.append("Fig2b recovered benchmark status is absent")
    if not any(row.get("item_id") == "T05" and
               row.get("current_status") == "found_verified_and_staged"
               for row in gaps):
        issues.append("T05 recovered benchmark record is absent")

    counts = Counter(row["status"] for row in panels)
    summary = [
        {"measure": "indexed_panels_and_tables", "value": str(len(panels))},
        {"measure": "active_main_figure_panels", "value": str(sum(p.startswith("Fig") for p in panel_ids))},
        {"measure": "active_supplementary_figure_panels", "value": str(sum(p.startswith("S") for p in panel_ids))},
        {"measure": "supplementary_tables", "value": str(sum(p.startswith("Table") for p in panel_ids))},
    ]
    summary.extend(
        {"measure": f"status_{status}", "value": str(count)}
        for status, count in sorted(counts.items())
    )
    write_tsv(PROVENANCE / "release_audit_summary.tsv", ["measure", "value"], summary)
    report = {
        "root": ".",
        "indexed_panels_and_tables": len(panels),
        "status_counts": dict(sorted(counts.items())),
        "issues": issues,
        "passed": not issues,
    }
    (PROVENANCE / "STRUCTURAL_VALIDATION_20260724.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    report = validate()
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
