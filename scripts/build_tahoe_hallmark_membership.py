#!/usr/bin/env python3
"""Materialize the Tahoe Hallmark gene membership used for signed-mean scoring."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gmt", required=True, type=Path)
    parser.add_argument("--gene-order", required=True, type=Path)
    parser.add_argument("--output-membership", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--minimum-overlap", type=int, default=10)
    args = parser.parse_args()

    genes = json.loads(args.gene_order.read_text())
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    membership_rows = []
    summary_rows = []

    with args.gmt.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            pathway, source, *members = fields
            overlap = [(gene_to_index[gene], gene) for gene in members if gene in gene_to_index]
            retained = len(overlap) >= args.minimum_overlap
            summary_rows.append(
                {
                    "pathway": pathway,
                    "source": source,
                    "original_gene_count": len(members),
                    "overlap_gene_count": len(overlap),
                    "overlap_fraction": len(overlap) / len(members),
                    "retained_min_overlap_10": retained,
                }
            )
            if retained:
                weight = 1.0 / len(overlap)
                for gene_index, gene in overlap:
                    membership_rows.append(
                        {
                            "pathway": pathway,
                            "gene": gene,
                            "gene_index_zero_based": gene_index,
                            "signed_mean_weight": weight,
                        }
                    )

    args.output_membership.parent.mkdir(parents=True, exist_ok=True)
    with args.output_membership.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pathway", "gene", "gene_index_zero_based", "signed_mean_weight"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(membership_rows)

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.output_summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)

    print(
        f"wrote {len(membership_rows)} retained gene-set memberships "
        f"across {sum(row['retained_min_overlap_10'] for row in summary_rows)} gene sets"
    )


if __name__ == "__main__":
    main()
