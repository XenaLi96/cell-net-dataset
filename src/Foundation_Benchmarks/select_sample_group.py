from __future__ import annotations

import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a sample group from sample_groups.json.")
    parser.add_argument("--groups", required=True)
    parser.add_argument("--mode", choices=["exact_panel", "gene_count"], default="exact_panel")
    parser.add_argument("--group-name", default=None)
    parser.add_argument("--rank", type=int, default=1, help="1 means largest group.")
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument(
        "--field",
        choices=["name", "samples", "n_samples", "n_cells", "n_genes"],
        default="samples",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.groups) as handle:
        payload = json.load(handle)
    groups = payload["groups"].get(args.mode, {})
    if not groups:
        raise RuntimeError(f"No groups found for mode {args.mode}")

    if args.group_name:
        if args.group_name not in groups:
            raise ValueError(
                f"Group {args.group_name} not found. Available groups: {sorted(groups)}"
            )
        selected = groups[args.group_name]
    else:
        ranked = [
            group
            for group in groups.values()
            if group["n_samples"] >= args.min_samples
        ]
        if not ranked:
            raise RuntimeError(
                f"No {args.mode} groups have at least {args.min_samples} samples. "
                "Use --min-samples 1, SAMPLE_GROUP_MODE=gene_count, or pass SAMPLES explicitly."
            )
        ranked = sorted(
            ranked,
            key=lambda group: (group["n_samples"], group["n_cells"], group["n_genes"]),
            reverse=True,
        )
        if args.rank < 1 or args.rank > len(ranked):
            raise ValueError(f"--rank must be in [1, {len(ranked)}]")
        selected = ranked[args.rank - 1]

    if args.field == "samples":
        print(" ".join(selected["samples"]))
    else:
        print(selected[args.field])


if __name__ == "__main__":
    main()
