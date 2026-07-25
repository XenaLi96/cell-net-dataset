#!/usr/bin/env python
"""Read-only Visium HD polygon-to-2um-bin assignment sensitivity audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.prepared import prep


BARCODE_FORMAT = "s_002um_{:05d}_{:05d}-1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--example-id", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-cells", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--barcode-cache", type=Path, default=None)
    return parser.parse_args()


def manifest_row(path: Path, example_id: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["example_id"] == example_id:
                return row
    raise ValueError(f"Example {example_id} not found in {path}")


def resolve_inputs(row: dict[str, str]) -> tuple[Path, Path, Path]:
    root = Path(row["dataset_dir"])
    microscope = row.get("microscope_image_path", "")
    candidates: list[Path] = []
    if microscope:
        stem = Path(microscope).stem
        candidates.extend(
            [
                root / stem / stem / "cell_detection" / "cells.json",
                root / stem / "cell_detection" / "cells.json",
            ]
        )
    for child in root.iterdir():
        if child.is_dir():
            candidates.extend(
                [
                    child / "cell_detection" / "cells.json",
                    child / child.name / "cell_detection" / "cells.json",
                ]
            )
    cells_json = next(
        (path for path in candidates if path.exists() and path.is_file()), Path("")
    )
    h5_path = (
        root / "binned_outputs" / "square_002um" / "filtered_feature_bc_matrix.h5"
    )
    positions = (
        root
        / "binned_outputs"
        / "square_002um"
        / "spatial"
        / "tissue_positions.parquet"
    )
    for label, path in (
        ("cells.json", cells_json),
        ("2um H5", h5_path),
        ("2um positions", positions),
    ):
        if not path or not path.exists() or not path.is_file():
            raise FileNotFoundError(f"{label} is not readable for {row['example_id']}: {path}")
    return cells_json, h5_path, positions


def iter_json_array(path: Path, key: str = "cells", chunk_size: int = 1 << 20):
    """Stream objects from a named top-level JSON array without ijson."""
    decoder = json.JSONDecoder()
    marker = f'"{key}"'
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                raise ValueError(f"Could not find JSON array {key!r} in {path}")
            buffer += chunk
            marker_pos = buffer.find(marker)
            if marker_pos < 0:
                buffer = buffer[-len(marker) - 8 :]
                continue
            array_pos = buffer.find("[", marker_pos + len(marker))
            while array_pos < 0:
                chunk = handle.read(chunk_size)
                if not chunk:
                    raise ValueError(f"Could not find opening array for {key!r}")
                buffer += chunk
                array_pos = buffer.find("[", marker_pos + len(marker))
            buffer = buffer[array_pos + 1 :]
            break

        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = handle.read(chunk_size)
                if not chunk:
                    raise
                buffer += chunk
                continue
            yield value
            buffer = buffer[end:]


def reservoir_cells(path: Path, n: int, seed: int) -> tuple[list[dict], int]:
    rng = np.random.default_rng(seed)
    reservoir: list[dict] = []
    total = 0
    for total, cell in enumerate(iter_json_array(path), start=1):
        if len(reservoir) < n:
            reservoir.append(cell)
        else:
            replacement = int(rng.integers(0, total))
            if replacement < n:
                reservoir[replacement] = cell
        if total % 50_000 == 0:
            print(f"[cells] streamed {total:,}", flush=True)
    return reservoir, total


def parse_barcode_keys(barcodes, cache_path: Path | None) -> tuple[np.ndarray, np.ndarray]:
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path) as cached:
            return cached["sorted_keys"], cached["sorted_columns"]
    keys = np.empty(len(barcodes), dtype=np.uint64)
    chunk_size = 500_000
    for start in range(0, len(barcodes), chunk_size):
        end = min(start + chunk_size, len(barcodes))
        chunk = barcodes[start:end]
        rows = np.fromiter(
            (int(value[8:13]) for value in chunk),
            dtype=np.int32,
            count=len(chunk),
        )
        cols = np.fromiter(
            (int(value[14:19]) for value in chunk),
            dtype=np.int32,
            count=len(chunk),
        )
        keys[start:end] = (
            rows.astype(np.uint64) << np.uint64(32)
        ) | cols.astype(np.uint64)
        print(f"[barcodes] parsed {end:,}/{len(barcodes):,}", flush=True)
    order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[order]
    sorted_columns = order.astype(np.int64)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            sorted_keys=sorted_keys,
            sorted_columns=sorted_columns,
        )
    return sorted_keys, sorted_columns


def load_affine(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wanted = {(0, 0), (0, 1), (1, 0)}
    hits: dict[tuple[int, int], np.ndarray] = {}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=5000,
        columns=[
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ],
    ):
        for row in batch.to_pylist():
            key = (row["array_row"], row["array_col"])
            if key in wanted:
                hits[key] = np.asarray(
                    [row["pxl_col_in_fullres"], row["pxl_row_in_fullres"]],
                    dtype=float,
                )
        if len(hits) == 3:
            break
    if len(hits) != 3:
        raise RuntimeError(f"Could not recover affine anchors from {path}")
    origin = hits[(0, 0)]
    affine_matrix = np.column_stack(
        [hits[(1, 0)] - origin, hits[(0, 1)] - origin]
    )
    return origin, affine_matrix, np.linalg.inv(affine_matrix)


def query_columns(
    polygon: Polygon,
    sorted_keys: np.ndarray,
    sorted_columns: np.ndarray,
) -> np.ndarray:
    if polygon.is_empty:
        return np.empty(0, dtype=np.int64)
    prepared = prep(polygon)
    min_col = max(0, int(np.floor(polygon.bounds[0] - 0.5)))
    max_col = int(np.ceil(polygon.bounds[2] + 0.5))
    min_row = max(0, int(np.floor(polygon.bounds[1] - 0.5)))
    max_row = int(np.ceil(polygon.bounds[3] + 0.5))
    if max_col < min_col or max_row < min_row:
        return np.empty(0, dtype=np.int64)
    if (max_col - min_col + 1) * (max_row - min_row + 1) > 10_000:
        return np.empty(0, dtype=np.int64)
    cols, rows = np.meshgrid(
        np.arange(min_col, max_col + 1),
        np.arange(min_row, max_row + 1),
    )
    points = np.column_stack([cols.ravel(), rows.ravel()])
    hit = np.fromiter(
        (
            prepared.intersects(box(col - 0.5, row - 0.5, col + 0.5, row + 0.5))
            for col, row in points
        ),
        dtype=bool,
        count=len(points),
    )
    points = points[hit]
    if len(points) == 0:
        return np.empty(0, dtype=np.int64)
    keys = (
        points[:, 1].astype(np.uint64) << np.uint64(32)
    ) | points[:, 0].astype(np.uint64)
    positions = np.searchsorted(sorted_keys, keys)
    valid = positions < len(sorted_keys)
    matched = np.zeros(len(keys), dtype=bool)
    valid_idx = np.flatnonzero(valid)
    matched[valid_idx] = sorted_keys[positions[valid_idx]] == keys[valid_idx]
    return np.unique(sorted_columns[positions[matched]])


def aggregate_expression(
    columns: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(columns) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
    gene_parts = []
    value_parts = []
    for column in columns:
        start, end = int(indptr[column]), int(indptr[column + 1])
        if end > start:
            gene_parts.append(indices[start:end])
            value_parts.append(values[start:end])
    if not gene_parts:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
    genes = np.concatenate(gene_parts).astype(np.int32, copy=False)
    counts = np.concatenate(value_parts).astype(np.float64, copy=False)
    order = np.argsort(genes, kind="mergesort")
    genes = genes[order]
    counts = counts[order]
    unique, starts = np.unique(genes, return_index=True)
    return unique, np.add.reduceat(counts, starts)


def sparse_cosine(
    genes_a: np.ndarray,
    counts_a: np.ndarray,
    genes_b: np.ndarray,
    counts_b: np.ndarray,
) -> float:
    denom = float(np.linalg.norm(counts_a) * np.linalg.norm(counts_b))
    if denom == 0:
        return float("nan")
    common, ia, ib = np.intersect1d(
        genes_a, genes_b, assume_unique=True, return_indices=True
    )
    if len(common) == 0:
        return 0.0
    return float(np.dot(counts_a[ia], counts_b[ib]) / denom)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.union1d(a, b)
    if len(union) == 0:
        return float("nan")
    return float(len(np.intersect1d(a, b, assume_unique=True)) / len(union))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    row = manifest_row(args.manifest, args.example_id)
    cells_path, h5_path, positions_path = resolve_inputs(row)
    print(f"[inputs] cells={cells_path}", flush=True)
    print(f"[inputs] h5={h5_path}", flush=True)
    print(f"[inputs] positions={positions_path}", flush=True)

    cells, total_cells = reservoir_cells(cells_path, args.max_cells, args.seed)
    print(
        f"[sample] reservoir={len(cells):,} total_streamed={total_cells:,}",
        flush=True,
    )
    origin, affine_matrix, inverse_affine = load_affine(positions_path)

    cache = args.barcode_cache
    with h5py.File(h5_path, "r") as handle:
        matrix = handle["matrix"]
        sorted_keys, sorted_columns = parse_barcode_keys(
            matrix["barcodes"], cache
        )
        indptr = np.asarray(matrix["indptr"][:], dtype=np.int64)
        gene_indices = np.asarray(matrix["indices"][:], dtype=np.int32)
        values = np.asarray(matrix["data"][:], dtype=np.float64)

    records: list[dict] = []
    variants = {
        "erode_1um": lambda poly: poly.buffer(-0.5),
        "dilate_1um": lambda poly: poly.buffer(0.5),
        "shift_x_1um": lambda poly: affinity.translate(poly, xoff=0.5),
        "shift_y_1um": lambda poly: affinity.translate(poly, yoff=0.5),
    }

    for index, cell in enumerate(cells):
        contour = np.asarray(cell.get("contour", []), dtype=float)
        if contour.ndim != 2 or contour.shape[0] < 3:
            continue
        row_col = (contour - origin) @ inverse_affine.T
        polygon = Polygon(np.column_stack([row_col[:, 1], row_col[:, 0]]))
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            continue
        canonical_columns = query_columns(polygon, sorted_keys, sorted_columns)
        canonical_genes, canonical_counts = aggregate_expression(
            canonical_columns, indptr, gene_indices, values
        )
        base = {
            "reservoir_index": index,
            "n_bins_canonical": len(canonical_columns),
            "umi_canonical": float(canonical_counts.sum()),
            "genes_canonical": len(canonical_genes),
        }
        for name, transform in variants.items():
            variant_columns = query_columns(
                transform(polygon), sorted_keys, sorted_columns
            )
            variant_genes, variant_counts = aggregate_expression(
                variant_columns, indptr, gene_indices, values
            )
            variant_umi = float(variant_counts.sum())
            base[f"n_bins_{name}"] = len(variant_columns)
            if len(canonical_columns) == 0:
                base[f"bin_jaccard_{name}"] = float("nan")
                base[f"expression_cosine_{name}"] = float("nan")
                base[f"umi_relative_change_{name}"] = float("nan")
            else:
                base[f"bin_jaccard_{name}"] = jaccard(
                    canonical_columns, variant_columns
                )
                base[f"expression_cosine_{name}"] = sparse_cosine(
                    canonical_genes,
                    canonical_counts,
                    variant_genes,
                    variant_counts,
                )
                denominator = max(float(canonical_counts.sum()), 1.0)
                base[f"umi_relative_change_{name}"] = (
                    variant_umi - float(canonical_counts.sum())
                ) / denominator
        records.append(base)
        if (index + 1) % 500 == 0:
            print(f"[audit] processed {index + 1:,}/{len(cells):,}", flush=True)

    metrics = pd.DataFrame(records)
    metrics.to_csv(args.out_dir / "cell_assignment_metrics.csv", index=False)
    assigned_metrics = metrics[metrics["n_bins_canonical"] > 0]
    summary: dict[str, object] = {
        "example_id": args.example_id,
        "dataset_name": row["dataset_name"],
        # Basenames preserve input provenance without exposing machine paths.
        "source_cells_file": cells_path.name,
        "source_h5_002um_file": h5_path.name,
        "source_positions_file": positions_path.name,
        "seed": args.seed,
        "n_cells_streamed": total_cells,
        "n_cells_reservoir": len(cells),
        "n_cells_audited": len(metrics),
        "n_cells_with_canonical_bins": len(assigned_metrics),
        "fraction_without_canonical_bins": float(
            (metrics["n_bins_canonical"] == 0).mean()
        ),
        "median_bins_canonical_assigned": float(
            assigned_metrics["n_bins_canonical"].median()
        ),
    }
    for name in variants:
        summary[f"median_bin_jaccard_{name}"] = float(
            assigned_metrics[f"bin_jaccard_{name}"].median()
        )
        summary[f"p05_bin_jaccard_{name}"] = float(
            assigned_metrics[f"bin_jaccard_{name}"].quantile(0.05)
        )
        summary[f"median_expression_cosine_{name}"] = float(
            assigned_metrics[f"expression_cosine_{name}"].median()
        )
        summary[f"median_abs_umi_relative_change_{name}"] = float(
            assigned_metrics[f"umi_relative_change_{name}"].abs().median()
        )
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
