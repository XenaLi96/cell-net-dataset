from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import ensure_parent, read_cell_expression  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a common gene list from the first available cell in each sample."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--out",
        default="PATH/cache/scaling_law/common_genes.json",
    )
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument(
        "--min-sample-fraction",
        type=float,
        default=1.0,
        help="Fraction of samples a gene must appear in. 1.0 means strict intersection.",
    )
    return parser.parse_args()


def read_first_expr_paths(manifest_path: str | Path, selected_samples: list[str] | None):
    selected = set(selected_samples or [])
    sample_to_expr_path = {}
    with open(manifest_path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample", "expr_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        for row in reader:
            sample = row["sample"]
            if selected and sample not in selected:
                continue
            sample_to_expr_path.setdefault(sample, row["expr_path"])
            if selected and set(sample_to_expr_path) == selected:
                break

    if selected:
        missing = sorted(selected - set(sample_to_expr_path))
        if missing:
            raise ValueError(f"Selected samples not found in manifest: {missing}")
    return sample_to_expr_path


def main() -> None:
    args = parse_args()
    if args.min_sample_fraction <= 0 or args.min_sample_fraction > 1:
        raise ValueError("--min-sample-fraction must be in (0, 1]")

    sample_to_genes = {}
    sample_to_expr_path = read_first_expr_paths(args.manifest, args.samples)
    if not sample_to_expr_path:
        raise RuntimeError("No manifest rows selected.")

    for sample, expr_path in sorted(sample_to_expr_path.items()):
        _, genes = read_cell_expression(expr_path)
        sample_to_genes[sample] = genes
        print(f"[genes] {sample}: {len(genes)} genes from {expr_path}")

    samples = sorted(sample_to_genes)
    min_count = max(1, int(len(samples) * args.min_sample_fraction + 0.999999))
    counts = {}
    for genes in sample_to_genes.values():
        for gene in set(genes):
            counts[gene] = counts.get(gene, 0) + 1

    first_sample = samples[0]
    ordered = [
        gene
        for gene in sample_to_genes[first_sample]
        if counts.get(gene, 0) >= min_count
    ]
    if not ordered:
        raise RuntimeError(
            "No genes passed the requested sample-frequency threshold. "
            "Try a smaller --min-sample-fraction or split samples by panel."
        )

    ensure_parent(args.out)
    with open(args.out, "w") as handle:
        json.dump(
            {
                "genes": ordered,
                "samples": samples,
                "min_sample_fraction": args.min_sample_fraction,
                "min_sample_count": min_count,
                "source_expr_paths": sample_to_expr_path,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"[done] wrote {len(ordered)} genes to {args.out}")


if __name__ == "__main__":
    main()
