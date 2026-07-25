#!/usr/bin/env python
"""Leakage-controlled Xenium evaluation with explicit spatial baselines."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cell-types", type=Path, default=None)
    parser.add_argument("--marker-list", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument(
        "--ratios", nargs="+", type=float, default=[0.1, 0.25, 1.0]
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--buffer-fraction", type=float, default=0.05)
    parser.add_argument("--max-target-genes", type=int, default=200)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def decode(values) -> list[str]:
    return [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in values
    ]


def load_embedding(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        return {
            "x": np.asarray(handle["embeddings"][:], dtype=np.float32),
            "y": np.asarray(handle["expressions"][:], dtype=np.float32),
            "cell_ids": decode(handle["cell_ids"][:]),
            "genes": decode(handle["gene_names"][:]),
        }


def align_metadata(data: dict, manifest_path: Path, sample: str) -> tuple[dict, pd.DataFrame]:
    manifest = pd.read_csv(
        manifest_path,
        dtype={"cell_id": str},
        usecols=lambda col: col
        in {
            "sample",
            "cell_id",
            "coord_x",
            "coord_y",
            "x_centroid",
            "y_centroid",
        },
    )
    if "sample" in manifest.columns:
        manifest = manifest[manifest["sample"] == sample].copy()
    if "coord_x" not in manifest:
        manifest["coord_x"] = manifest["x_centroid"]
        manifest["coord_y"] = manifest["y_centroid"]
    manifest = manifest.drop_duplicates("cell_id").set_index("cell_id")

    keep_h5: list[int] = []
    keep_ids: list[str] = []
    for idx, cell_id in enumerate(data["cell_ids"]):
        if cell_id in manifest.index:
            keep_h5.append(idx)
            keep_ids.append(cell_id)
    if len(keep_h5) < 100:
        raise RuntimeError(
            f"Only {len(keep_h5)} embedding cells overlap the manifest for {sample}"
        )
    aligned = manifest.loc[keep_ids].reset_index()
    data = {
        **data,
        "x": data["x"][keep_h5],
        "y": data["y"][keep_h5],
        "cell_ids": keep_ids,
    }
    return data, aligned


def add_cell_types(meta: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    meta = meta.copy()
    meta["cell_type"] = "unknown"
    if path is None or not path.exists():
        return meta
    types = pd.read_csv(path, dtype={"cell_id": str})
    candidate = "predicted_cell_type"
    if candidate not in types:
        candidates = [col for col in types if "cell" in col.lower() and "type" in col.lower()]
        if not candidates:
            return meta
        candidate = candidates[0]
    lookup = (
        types.drop_duplicates("cell_id")
        .set_index("cell_id")[candidate]
        .astype(str)
        .to_dict()
    )
    meta["cell_type"] = meta["cell_id"].map(lookup).fillna("unknown")
    return meta


def read_gene_list(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    first_line = path.open(encoding="utf-8").readline()
    if "\t" in first_line and "geneSymbol" in first_line:
        table = pd.read_csv(path, sep="\t", dtype=str)
        genes: set[str] = set()
        for value in table["geneSymbol"].dropna():
            genes.update(re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value))
        genes.discard("NA")
        return genes
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def spatial_split(
    coords: np.ndarray,
    seed: int,
    test_fraction: float,
    buffer_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Hold out one contiguous edge and discard an intervening buffer."""
    orientation = seed % 4
    axis_index = 0 if orientation in (0, 2) else 1
    high_side = orientation in (0, 1)
    axis = coords[:, axis_index].astype(float)
    span = max(float(np.nanmax(axis) - np.nanmin(axis)), 1.0)
    if high_side:
        cut = float(np.quantile(axis, 1.0 - test_fraction))
        test = axis >= cut
        train = axis < cut - buffer_fraction * span
        side = "high"
    else:
        cut = float(np.quantile(axis, test_fraction))
        test = axis <= cut
        train = axis > cut + buffer_fraction * span
        side = "low"
    train_idx = np.flatnonzero(train)
    test_idx = np.flatnonzero(test)
    if len(train_idx) < 100 or len(test_idx) < 50:
        raise RuntimeError(
            f"Degenerate spatial split seed={seed}: train={len(train_idx)} "
            f"test={len(test_idx)}"
        )
    info = {
        "seed": seed,
        "axis": "x" if axis_index == 0 else "y",
        "side": side,
        "cut": cut,
        "buffer_fraction_of_axis_span": buffer_fraction,
        "n_train_pool": len(train_idx),
        "n_test": len(test_idx),
    }
    return train_idx, test_idx, info


def subsample_train(indices: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    if ratio >= 1:
        return indices.copy()
    n = max(100, int(round(len(indices) * ratio)))
    n = min(n, len(indices))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=n, replace=False))


def top_variable_genes(y_train: np.ndarray, max_genes: int) -> np.ndarray:
    variance = np.var(np.log1p(np.clip(y_train, 0, None)), axis=0)
    valid = np.flatnonzero(np.isfinite(variance) & (variance > 0))
    order = valid[np.argsort(variance[valid])[::-1]]
    return order[: min(max_genes, len(order))]


def make_predictions(
    x: np.ndarray,
    y_log: np.ndarray,
    coords: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    pca_components: int,
    ridge_alpha: float,
    knn_k: int,
    seed: int,
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    train_mean = np.mean(y_log[train_idx], axis=0, keepdims=True)
    predictions["train_mean"] = np.repeat(train_mean, len(test_idx), axis=0)

    coord_model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=ridge_alpha)),
        ]
    )
    coord_model.fit(coords[train_idx], y_log[train_idx])
    predictions["coordinate"] = coord_model.predict(coords[test_idx]).astype(np.float32)

    n_components = min(
        pca_components,
        x.shape[1],
        max(1, len(train_idx) - 1),
    )
    image_model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=n_components,
                    svd_solver="randomized",
                    random_state=seed,
                ),
            ),
            ("ridge", Ridge(alpha=ridge_alpha)),
        ]
    )
    image_model.fit(x[train_idx], y_log[train_idx])
    predictions["image"] = image_model.predict(x[test_idx]).astype(np.float32)

    k = min(knn_k, len(train_idx))
    neighbor_model = NearestNeighbors(n_neighbors=k, algorithm="auto")
    neighbor_model.fit(coords[train_idx])
    neighbor_indices = neighbor_model.kneighbors(
        coords[test_idx], return_distance=False
    )
    predictions["spatial_knn"] = np.mean(
        y_log[train_idx][neighbor_indices], axis=1
    ).astype(np.float32)
    return predictions


def column_pearson(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    yc = y - np.mean(y, axis=0, keepdims=True)
    pc = pred - np.mean(pred, axis=0, keepdims=True)
    denom = np.sqrt(np.sum(yc * yc, axis=0) * np.sum(pc * pc, axis=0))
    out = np.full(y.shape[1], np.nan, dtype=float)
    np.divide(np.sum(yc * pc, axis=0), denom, out=out, where=denom > 0)
    return out


def row_pearson(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    yc = y - np.mean(y, axis=1, keepdims=True)
    pc = pred - np.mean(pred, axis=1, keepdims=True)
    denom = np.sqrt(np.sum(yc * yc, axis=1) * np.sum(pc * pc, axis=1))
    out = np.full(y.shape[0], np.nan, dtype=float)
    np.divide(np.sum(yc * pc, axis=1), denom, out=out, where=denom > 0)
    return out


def detection_metrics(y_raw: np.ndarray, pred_log: np.ndarray) -> dict[str, np.ndarray]:
    truth = y_raw > 0
    called = pred_log > np.log1p(0.5)
    tp = np.sum(truth & called, axis=0).astype(float)
    fn = np.sum(truth & ~called, axis=0).astype(float)
    tn = np.sum(~truth & ~called, axis=0).astype(float)
    fp = np.sum(~truth & called, axis=0).astype(float)
    sensitivity = np.divide(tp, tp + fn, out=np.full_like(tp, np.nan), where=(tp + fn) > 0)
    specificity = np.divide(tn, tn + fp, out=np.full_like(tn, np.nan), where=(tn + fp) > 0)
    f1 = np.divide(
        2 * tp,
        2 * tp + fp + fn,
        out=np.full_like(tp, np.nan),
        where=(2 * tp + fp + fn) > 0,
    )
    auprc = np.full(y_raw.shape[1], np.nan, dtype=float)
    for gene_idx in range(y_raw.shape[1]):
        if np.unique(truth[:, gene_idx]).size == 2:
            auprc[gene_idx] = average_precision_score(
                truth[:, gene_idx], pred_log[:, gene_idx]
            )
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "f1": f1,
        "auprc": auprc,
    }


def metric_rows(
    y_raw: np.ndarray,
    y_log: np.ndarray,
    pred_log: np.ndarray,
    genes: list[str],
) -> pd.DataFrame:
    pearson = column_pearson(y_log, pred_log)
    spearman = np.full(len(genes), np.nan, dtype=float)
    for idx in range(len(genes)):
        spearman[idx] = column_pearson(
            rankdata(y_log[:, idx])[:, None],
            rankdata(pred_log[:, idx])[:, None],
        )[0]
    mse = np.mean((y_log - pred_log) ** 2, axis=0)
    mae = np.mean(np.abs(y_log - pred_log), axis=0)
    denom = np.sum(
        (y_log - np.mean(y_log, axis=0, keepdims=True)) ** 2, axis=0
    )
    r2 = np.full(len(genes), np.nan, dtype=float)
    np.divide(
        denom - np.sum((y_log - pred_log) ** 2, axis=0),
        denom,
        out=r2,
        where=denom > 0,
    )
    detect = detection_metrics(y_raw, pred_log)
    result = pd.DataFrame(
        {
            "gene": genes,
            "pearson": pearson,
            "spearman": spearman,
            "mse_log1p": mse,
            "mae_log1p": mae,
            "r2": r2,
            **detect,
        }
    )
    return result


def summarize(
    metrics: pd.DataFrame,
    y_log: np.ndarray,
    pred_log: np.ndarray,
) -> dict:
    return {
        "n_genes": len(metrics),
        "pearson_gene_macro": float(metrics["pearson"].mean()),
        "spearman_gene_macro": float(metrics["spearman"].mean()),
        "r2_gene_macro": float(metrics["r2"].mean()),
        "rmse_log1p": float(np.sqrt(np.mean((y_log - pred_log) ** 2))),
        "mae_log1p": float(np.mean(np.abs(y_log - pred_log))),
        "pearson_cell_macro": float(np.nanmean(row_pearson(y_log, pred_log))),
        "sensitivity_gene_macro": float(metrics["sensitivity"].mean()),
        "specificity_gene_macro": float(metrics["specificity"].mean()),
        "balanced_accuracy_gene_macro": float(metrics["balanced_accuracy"].mean()),
        "f1_gene_macro": float(metrics["f1"].mean()),
        "auprc_gene_macro": float(metrics["auprc"].mean()),
    }


def cell_type_rows(
    cell_types: np.ndarray,
    y_log: np.ndarray,
    predictions: dict[str, np.ndarray],
    min_cells: int = 20,
) -> list[dict]:
    rows: list[dict] = []
    for cell_type in sorted(set(cell_types)):
        mask = cell_types == cell_type
        if int(mask.sum()) < min_cells:
            continue
        observed = np.mean(y_log[mask], axis=0, keepdims=True)
        for model, pred in predictions.items():
            expected = np.mean(pred[mask], axis=0, keepdims=True)
            rows.append(
                {
                    "cell_type": cell_type,
                    "n_cells": int(mask.sum()),
                    "model": model,
                    "pseudobulk_pearson": float(
                        column_pearson(observed.T, expected.T)[0]
                        if observed.shape[1] > 1
                        else np.nan
                    ),
                    "pseudobulk_rmse_log1p": float(
                        np.sqrt(np.mean((observed - expected) ** 2))
                    ),
                }
            )
    # Recompute Pearson across genes (one pseudobulk vector per type).
    for row in rows:
        cell_type = row["cell_type"]
        mask = cell_types == cell_type
        observed = np.mean(y_log[mask], axis=0)
        expected = np.mean(predictions[row["model"]][mask], axis=0)
        row["pseudobulk_pearson"] = float(
            row_pearson(observed[None, :], expected[None, :])[0]
        )
    return rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load_embedding(args.embedding)
    data, meta = align_metadata(data, args.manifest, args.sample)
    meta = add_cell_types(meta, args.cell_types)
    coords = meta[["coord_x", "coord_y"]].to_numpy(dtype=np.float32)
    marker_genes = {gene.upper() for gene in read_gene_list(args.marker_list)}

    all_summary: list[dict] = []
    all_gene_metrics: list[pd.DataFrame] = []
    all_cell_type: list[dict] = []
    split_records: list[dict] = []

    for seed in args.seeds:
        train_pool, test_idx, split_info = spatial_split(
            coords,
            seed=seed,
            test_fraction=args.test_fraction,
            buffer_fraction=args.buffer_fraction,
        )
        selected_genes = top_variable_genes(
            data["y"][train_pool], args.max_target_genes
        )
        if len(selected_genes) < 10:
            raise RuntimeError(f"Only {len(selected_genes)} variable genes in {args.sample}")
        genes = [data["genes"][idx] for idx in selected_genes]
        y_raw = data["y"][:, selected_genes]
        y_log = np.log1p(np.clip(y_raw, 0, None)).astype(np.float32)
        split_info["selected_genes"] = genes
        split_records.append(split_info)

        for ratio in args.ratios:
            train_idx = subsample_train(
                train_pool, ratio=ratio, seed=seed * 10_000 + int(ratio * 1_000)
            )
            predictions = make_predictions(
                x=data["x"],
                y_log=y_log,
                coords=coords,
                train_idx=train_idx,
                test_idx=test_idx,
                pca_components=args.pca_components,
                ridge_alpha=args.ridge_alpha,
                knn_k=args.knn_k,
                seed=seed,
            )
            gene_sets = {
                "top50_hvg": np.arange(min(50, len(genes))),
                "top200_hvg": np.arange(len(genes)),
            }
            marker_idx = np.asarray(
                [idx for idx, gene in enumerate(genes) if gene.upper() in marker_genes],
                dtype=int,
            )
            if len(marker_idx) >= 5:
                gene_sets["marker_hvg_overlap"] = marker_idx

            for model, pred in predictions.items():
                full_metrics = metric_rows(
                    y_raw[test_idx], y_log[test_idx], pred, genes
                )
                full_metrics.insert(0, "model", model)
                full_metrics.insert(0, "ratio", ratio)
                full_metrics.insert(0, "seed", seed)
                full_metrics.insert(0, "encoder", args.encoder)
                full_metrics.insert(0, "sample", args.sample)
                all_gene_metrics.append(full_metrics)

                for gene_set, indices in gene_sets.items():
                    subset_metrics = full_metrics.iloc[indices]
                    record = {
                        "sample": args.sample,
                        "encoder": args.encoder,
                        "seed": seed,
                        "split_axis": split_info["axis"],
                        "split_side": split_info["side"],
                        "ratio": ratio,
                        "model": model,
                        "gene_set": gene_set,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        **summarize(
                            subset_metrics,
                            y_log[test_idx][:, indices],
                            pred[:, indices],
                        ),
                    }
                    all_summary.append(record)

            if ratio >= 1.0:
                type_rows = cell_type_rows(
                    meta.iloc[test_idx]["cell_type"].astype(str).to_numpy(),
                    y_log[test_idx],
                    predictions,
                )
                for row in type_rows:
                    row.update(
                        {
                            "sample": args.sample,
                            "encoder": args.encoder,
                            "seed": seed,
                            "ratio": ratio,
                            "n_genes": len(genes),
                        }
                    )
                all_cell_type.extend(type_rows)

                if args.save_predictions:
                    np.savez_compressed(
                        args.out_dir / f"predictions_seed{seed}.npz",
                        cell_ids=np.asarray(data["cell_ids"], dtype=str)[test_idx],
                        genes=np.asarray(genes, dtype=str),
                        y_true_raw=y_raw[test_idx],
                        y_true_log1p=y_log[test_idx],
                        pred_image=predictions["image"],
                        pred_coordinate=predictions["coordinate"],
                        pred_spatial_knn=predictions["spatial_knn"],
                        pred_train_mean=predictions["train_mean"],
                        coord_x=coords[test_idx, 0],
                        coord_y=coords[test_idx, 1],
                        cell_type=meta.iloc[test_idx]["cell_type"].astype(str).to_numpy(),
                    )

    pd.DataFrame(all_summary).to_csv(args.out_dir / "summary.csv", index=False)
    pd.concat(all_gene_metrics, ignore_index=True).to_csv(
        args.out_dir / "gene_metrics.csv", index=False
    )
    pd.DataFrame(all_cell_type).to_csv(
        args.out_dir / "cell_type_metrics.csv", index=False
    )
    with (args.out_dir / "split_info.json").open("w", encoding="utf-8") as handle:
        json.dump(split_records, handle, indent=2)
    print(
        f"[done] {args.sample}/{args.encoder}: "
        f"{len(all_summary)} summary rows -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
