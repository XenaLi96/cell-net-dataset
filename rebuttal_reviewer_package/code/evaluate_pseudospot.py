#!/usr/bin/env python
"""Compare cell-level targets with pseudo-spot aggregation on Xenium."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_cross_sample import fit_image_model
from evaluate_spatial import (
    add_cell_types,
    align_metadata,
    load_embedding,
    metric_rows,
    spatial_split,
    summarize,
    top_variable_genes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--encoder", default="conch")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cell-types", type=Path, default=None)
    parser.add_argument(
        "--mixing-manifest",
        type=Path,
        default=None,
        help=(
            "Optional full-cell coordinate table used only for physical-bin "
            "cell-count and cell-type-mixing estimates."
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument(
        "--spot-microns", nargs="+", type=float, default=[8.0, 16.0, 55.0]
    )
    parser.add_argument(
        "--pixels-per-micron",
        type=float,
        default=1.0,
        help=(
            "Coordinate units per micron. Xenium x/y centroids are already "
            "reported in microns, so the campaign value is 1."
        ),
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--buffer-fraction", type=float, default=0.05)
    parser.add_argument("--max-target-genes", type=int, default=200)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    return parser.parse_args()


def spot_ids(coords: np.ndarray, bin_pixels: float) -> np.ndarray:
    # Xenium coordinates already live in a fixed image-pixel coordinate frame.
    # Anchor bins at zero rather than deriving an origin from all cells, which
    # keeps grid construction independent of the held-out partition.
    grid = np.floor(coords / bin_pixels).astype(np.int64)
    return np.char.add(
        np.char.add(grid[:, 0].astype(str), "_"), grid[:, 1].astype(str)
    )


def aggregate_spots(
    x: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    local_ids = ids[indices]
    unique_ids, inverse = np.unique(local_ids, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float32)
    x_spot = np.zeros((len(unique_ids), x.shape[1]), dtype=np.float32)
    y_spot = np.zeros((len(unique_ids), y.shape[1]), dtype=np.float32)
    np.add.at(x_spot, inverse, x[indices])
    np.add.at(y_spot, inverse, y[indices])
    x_spot /= counts[:, None]
    y_spot /= counts[:, None]
    lookup = {spot_id: idx for idx, spot_id in enumerate(unique_ids)}
    return x_spot, y_spot, unique_ids, lookup


def mixing_record(
    ids: np.ndarray,
    indices: np.ndarray,
    cell_types: np.ndarray,
    scale: float,
    seed: int,
) -> dict:
    frame = pd.DataFrame(
        {
            "spot": ids[indices],
            "cell_type": cell_types[indices],
        }
    )
    grouped = frame.groupby("spot", sort=False)["cell_type"]
    n_types = grouped.nunique()
    sizes = grouped.size()
    mixed_spots = n_types > 1
    mixed_cells = int(sizes[mixed_spots].sum()) if mixed_spots.any() else 0
    return {
        "seed": seed,
        "spot_microns": scale,
        "n_test_cells": len(indices),
        "n_test_spots": len(n_types),
        "median_cells_per_spot": float(sizes.median()),
        "p95_cells_per_spot": float(sizes.quantile(0.95)),
        "fraction_spots_mixed_cell_type": float(mixed_spots.mean()),
        "fraction_cells_in_mixed_spots": float(mixed_cells / max(len(indices), 1)),
    }


def load_mixing_metadata(
    path: Path | None,
    fallback: pd.DataFrame,
    cell_types_path: Path | None,
) -> pd.DataFrame:
    if path is None:
        return fallback[["cell_id", "coord_x", "coord_y", "cell_type"]].copy()
    frame = pd.read_csv(
        path,
        dtype={"cell_id": str},
        usecols=lambda column: column
        in {"cell_id", "coord_x", "coord_y", "x_centroid", "y_centroid"},
    )
    if "coord_x" not in frame:
        frame["coord_x"] = frame["x_centroid"]
        frame["coord_y"] = frame["y_centroid"]
    frame = frame.dropna(subset=["cell_id", "coord_x", "coord_y"])
    return add_cell_types(frame, cell_types_path)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load_embedding(args.embedding)
    data, meta = align_metadata(data, args.manifest, args.sample)
    meta = add_cell_types(meta, args.cell_types)
    coords = meta[["coord_x", "coord_y"]].to_numpy(dtype=np.float32)
    mixing_meta = load_mixing_metadata(
        args.mixing_manifest, meta, args.cell_types
    )
    mixing_coords = mixing_meta[["coord_x", "coord_y"]].to_numpy(
        dtype=np.float32
    )
    mixing_cell_types = mixing_meta["cell_type"].astype(str).to_numpy()

    summaries: list[dict] = []
    gene_frames: list[pd.DataFrame] = []
    mixing: list[dict] = []
    split_records: list[dict] = []

    for seed in args.seeds:
        train_idx, test_idx, split_info = spatial_split(
            coords,
            seed,
            test_fraction=args.test_fraction,
            buffer_fraction=args.buffer_fraction,
        )
        gene_idx = top_variable_genes(data["y"][train_idx], args.max_target_genes)
        genes = [data["genes"][idx] for idx in gene_idx]
        y_raw = data["y"][:, gene_idx]
        y_log = np.log1p(np.clip(y_raw, 0, None)).astype(np.float32)

        cell_pred = fit_image_model(
            data["x"][train_idx],
            y_log[train_idx],
            data["x"][test_idx],
            pca_components=args.pca_components,
            ridge_alpha=args.ridge_alpha,
            seed=seed,
        )
        predictions: dict[str, np.ndarray] = {
            "cell_image": cell_pred,
            "train_mean": np.repeat(
                np.mean(y_log[train_idx], axis=0, keepdims=True),
                len(test_idx),
                axis=0,
            ),
        }

        for scale in args.spot_microns:
            bin_pixels = scale * args.pixels_per_micron
            ids = spot_ids(coords, bin_pixels)
            x_train_spot, y_train_spot, _, _ = aggregate_spots(
                data["x"], y_log, ids, train_idx
            )
            x_test_spot, _, _, test_lookup = aggregate_spots(
                data["x"], y_log, ids, test_idx
            )
            spot_pred = fit_image_model(
                x_train_spot,
                y_train_spot,
                x_test_spot,
                pca_components=args.pca_components,
                ridge_alpha=args.ridge_alpha,
                seed=seed,
            )
            broadcast_indices = np.asarray(
                [test_lookup[spot_id] for spot_id in ids[test_idx]], dtype=int
            )
            predictions[f"pseudo_spot_{scale:g}um"] = spot_pred[broadcast_indices]
            axis_index = 0 if split_info["axis"] == "x" else 1
            if split_info["side"] == "high":
                mixing_test_idx = np.flatnonzero(
                    mixing_coords[:, axis_index] >= split_info["cut"]
                )
            else:
                mixing_test_idx = np.flatnonzero(
                    mixing_coords[:, axis_index] <= split_info["cut"]
                )
            mixing_ids = spot_ids(mixing_coords, bin_pixels)
            record = mixing_record(
                mixing_ids,
                mixing_test_idx,
                mixing_cell_types,
                scale=scale,
                seed=seed,
            )
            record["density_source"] = (
                "full_mixing_manifest"
                if args.mixing_manifest is not None
                else "embedding_manifest"
            )
            record["n_available_cells"] = len(mixing_meta)
            mixing.append(record)

        gene_sets = {
            "top50_hvg": np.arange(min(50, len(genes))),
            "top200_hvg": np.arange(len(genes)),
        }
        for model, pred in predictions.items():
            metrics = metric_rows(y_raw[test_idx], y_log[test_idx], pred, genes)
            metrics.insert(0, "model", model)
            metrics.insert(0, "seed", seed)
            metrics.insert(0, "sample", args.sample)
            metrics.insert(0, "encoder", args.encoder)
            gene_frames.append(metrics)
            for gene_set, indices in gene_sets.items():
                summaries.append(
                    {
                        "sample": args.sample,
                        "encoder": args.encoder,
                        "seed": seed,
                        "split_axis": split_info["axis"],
                        "split_side": split_info["side"],
                        "model": model,
                        "gene_set": gene_set,
                        "n_train_cells": len(train_idx),
                        "n_test_cells": len(test_idx),
                        "coordinate_units_per_micron": args.pixels_per_micron,
                        **summarize(
                            metrics.iloc[indices],
                            y_log[test_idx][:, indices],
                            pred[:, indices],
                        ),
                    }
                )
        split_info["selected_genes"] = genes
        split_records.append(split_info)

    pd.DataFrame(summaries).to_csv(args.out_dir / "summary.csv", index=False)
    pd.concat(gene_frames, ignore_index=True).to_csv(
        args.out_dir / "gene_metrics.csv", index=False
    )
    pd.DataFrame(mixing).to_csv(args.out_dir / "spot_mixing.csv", index=False)
    with (args.out_dir / "run_info.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "sample": args.sample,
                "encoder": args.encoder,
                "spot_microns": args.spot_microns,
                "coordinate_units_per_micron": args.pixels_per_micron,
                "mixing_manifest": (
                    args.mixing_manifest.name
                    if args.mixing_manifest is not None
                    else None
                ),
                "splits": split_records,
            },
            handle,
            indent=2,
        )
    print(f"[done] pseudo-spot evaluation for {args.sample}")


if __name__ == "__main__":
    main()
