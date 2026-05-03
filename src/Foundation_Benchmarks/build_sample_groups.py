from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import ensure_parent, read_cell_expression  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group samples by gene panel for same-panel scaling-law runs."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--out",
        default="PATH/cache/scaling_law/sample_groups.json",
    )
    return parser.parse_args()


def read_manifest_sample_info(path: str | Path) -> dict[str, dict]:
    sample_info: dict[str, dict] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample", "expr_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        for row in reader:
            sample = row["sample"]
            info = sample_info.setdefault(
                sample,
                {"n_cells": 0, "first_expr_path": row["expr_path"]},
            )
            info["n_cells"] += 1
    return sample_info


def gene_hash(genes: list[str]) -> str:
    joined = "\n".join(genes).encode("utf-8")
    return hashlib.sha1(joined).hexdigest()


def add_group(groups: dict, mode: str, group_name: str, sample: str, info: dict) -> None:
    group = groups.setdefault(mode, {}).setdefault(
        group_name,
        {
            "name": group_name,
            "mode": mode,
            "samples": [],
            "n_samples": 0,
            "n_cells": 0,
            "n_genes": info["n_genes"],
            "gene_hash": info["gene_hash"],
        },
    )
    group["samples"].append(sample)
    group["samples"].sort()
    group["n_samples"] = len(group["samples"])
    group["n_cells"] += info["n_cells"]


def main() -> None:
    args = parse_args()
    sample_info = read_manifest_sample_info(args.manifest)
    if not sample_info:
        raise RuntimeError("No samples found in manifest.")

    for sample, info in sorted(sample_info.items()):
        _, genes = read_cell_expression(info["first_expr_path"])
        info["n_genes"] = len(genes)
        info["gene_hash"] = gene_hash(genes)
        info["gene_hash_short"] = info["gene_hash"][:8]
        info["first_genes"] = genes[:5]
        print(
            f"[sample] {sample}: cells={info['n_cells']} genes={info['n_genes']} "
            f"hash={info['gene_hash_short']}"
        )

    groups: dict[str, dict] = {}
    for sample, info in sorted(sample_info.items()):
        exact_name = f"panel_{info['n_genes']}_{info['gene_hash_short']}"
        count_name = f"genes_{info['n_genes']}"
        add_group(groups, "exact_panel", exact_name, sample, info)
        add_group(groups, "gene_count", count_name, sample, info)

    payload = {
        "manifest": str(args.manifest),
        "samples": sample_info,
        "groups": groups,
    }
    ensure_parent(args.out)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"[done] wrote sample groups to {args.out}")


if __name__ == "__main__":
    main()
