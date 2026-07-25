#!/usr/bin/env python
"""Subset a compatible frozen-encoder HDF5 cache to a campaign manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--gene-list", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-model", default="conch")
    return parser.parse_args()


def decode(values) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"cell_id": str})
    requested_cells = manifest["cell_id"].astype(str).tolist()
    requested_genes = [
        line.strip()
        for line in args.gene_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    with h5py.File(args.source, "r") as source:
        model = str(source.attrs.get("model", ""))
        if model != args.expected_model:
            raise ValueError(
                f"Source model {model!r} does not match {args.expected_model!r}"
            )
        source_cells = decode(source["cell_ids"][:])
        source_genes = decode(source["gene_names"][:])
        cell_lookup = {cell_id: idx for idx, cell_id in enumerate(source_cells)}
        gene_lookup = {gene: idx for idx, gene in enumerate(source_genes)}
        missing_cells = [cell for cell in requested_cells if cell not in cell_lookup]
        missing_genes = [gene for gene in requested_genes if gene not in gene_lookup]
        if missing_cells or missing_genes:
            raise ValueError(
                f"Missing {len(missing_cells)} cells and {len(missing_genes)} genes"
            )

        cell_idx = np.asarray([cell_lookup[cell] for cell in requested_cells])
        gene_idx = np.asarray([gene_lookup[gene] for gene in requested_genes])
        # h5py requires monotone fancy indices, so load the compatible arrays
        # once and then restore the deterministic campaign order in NumPy.
        embeddings = np.asarray(source["embeddings"][:], dtype=np.float32)[cell_idx]
        expressions = np.asarray(source["expressions"][:], dtype=np.float32)[
            np.ix_(cell_idx, gene_idx)
        ]
        source_paths = decode(source["image_paths"][:])
        image_paths = [source_paths[idx] for idx in cell_idx]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    string_type = h5py.string_dtype(encoding="utf-8")
    with h5py.File(args.output, "w") as output:
        output.create_dataset("embeddings", data=embeddings, compression="gzip")
        output.create_dataset("expressions", data=expressions, compression="gzip")
        output.create_dataset(
            "cell_ids", data=np.asarray(requested_cells, dtype=string_type)
        )
        output.create_dataset(
            "gene_names", data=np.asarray(requested_genes, dtype=string_type)
        )
        output.create_dataset(
            "image_paths", data=np.asarray(image_paths, dtype=string_type)
        )
        output.attrs["model"] = args.expected_model
        output.attrs["encoder"] = args.expected_model
        # Store only file names so shared HDF5 artifacts do not reveal local
        # directory layouts or user names.
        output.attrs["source_cache_file"] = args.source.name
        output.attrs["subset_manifest_file"] = args.manifest.name
    print(
        f"[done] {args.output}: cells={len(requested_cells)} "
        f"genes={len(requested_genes)}"
    )


if __name__ == "__main__":
    main()
