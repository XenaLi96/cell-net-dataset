#!/usr/bin/env python
"""Validate campaign embedding HDF5 files before launching evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-sheet", required=True, type=Path)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--gene-root", required=True, type=Path)
    parser.add_argument("--embedding-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model", default="conch")
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Optional sample subset; default validates the full sample sheet.",
    )
    return parser.parse_args()


def decode(values) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    failures: list[str] = []
    samples = pd.read_csv(args.sample_sheet)["sample"].astype(str).tolist()
    if args.samples:
        requested = set(args.samples)
        unknown = sorted(requested - set(samples))
        if unknown:
            raise ValueError(f"Unknown requested samples: {unknown}")
        samples = [sample for sample in samples if sample in requested]
    for sample in samples:
        manifest_path = args.manifest_root / f"{sample}.csv"
        gene_path = args.gene_root / f"{sample}.txt"
        embedding_path = args.embedding_root / args.model / f"{sample}.h5"
        expected_cells = pd.read_csv(
            manifest_path, usecols=["cell_id"], dtype={"cell_id": str}
        )["cell_id"].tolist()
        expected_genes = [
            line.strip()
            for line in gene_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        messages: list[str] = []
        n_cells = n_genes = embedding_dim = 0
        finite_embeddings = finite_expressions = False
        if not embedding_path.exists():
            messages.append("missing HDF5")
        else:
            try:
                with h5py.File(embedding_path, "r") as handle:
                    required = {
                        "embeddings",
                        "expressions",
                        "cell_ids",
                        "gene_names",
                        "image_paths",
                    }
                    missing = sorted(required - set(handle))
                    if missing:
                        messages.append(f"missing datasets: {missing}")
                    else:
                        cell_ids = decode(handle["cell_ids"][:])
                        genes = decode(handle["gene_names"][:])
                        n_cells = len(cell_ids)
                        n_genes = len(genes)
                        embedding_dim = int(handle["embeddings"].shape[1])
                        if cell_ids != expected_cells:
                            messages.append("cell IDs/order differ from manifest")
                        if genes != expected_genes:
                            messages.append("genes/order differ from target list")
                        if handle["embeddings"].shape[0] != len(expected_cells):
                            messages.append("embedding row count mismatch")
                        if handle["expressions"].shape != (
                            len(expected_cells),
                            len(expected_genes),
                        ):
                            messages.append("expression shape mismatch")
                        if len(handle["image_paths"]) != len(expected_cells):
                            messages.append("image path row count mismatch")
                        finite_embeddings = bool(
                            np.isfinite(handle["embeddings"][:]).all()
                        )
                        finite_expressions = bool(
                            np.isfinite(handle["expressions"][:]).all()
                        )
                        if not finite_embeddings:
                            messages.append("non-finite embeddings")
                        if not finite_expressions:
                            messages.append("non-finite expressions")
                    if str(handle.attrs.get("model", "")) != args.model:
                        messages.append("model attribute mismatch")
            except Exception as exc:  # record corrupt/truncated HDF5 explicitly
                messages.append(f"{type(exc).__name__}: {exc}")
        valid = not messages
        if not valid:
            failures.append(f"{sample}: {'; '.join(messages)}")
        rows.append(
            {
                "sample": sample,
                "embedding_file": str(Path(args.model) / embedding_path.name),
                "n_cells": n_cells,
                "n_genes": n_genes,
                "embedding_dim": embedding_dim,
                "finite_embeddings": finite_embeddings,
                "finite_expressions": finite_expressions,
                "valid": valid,
                "message": "; ".join(messages),
            }
        )
        print(f"[{'ok' if valid else 'fail'}] {sample}: {'; '.join(messages)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out_dir / "embedding_validation.csv", index=False)
    summary = {
        "model": args.model,
        "n_expected": len(samples),
        "n_valid": int(frame["valid"].sum()),
        "n_invalid": int((~frame["valid"]).sum()),
        "failures": failures,
    }
    (args.out_dir / "embedding_validation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} invalid embedding files: {failures[:3]}"
        )
    print(f"[done] validated {len(samples)} embedding files")


if __name__ == "__main__":
    main()
