#!/usr/bin/env python
"""Same-organ leave-one-sample-out evaluation for Xenium."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_spatial import (
    add_cell_types,
    cell_type_rows,
    decode,
    metric_rows,
    read_gene_list,
    summarize,
    top_variable_genes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-root", required=True, type=Path)
    parser.add_argument("--sample-sheet", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--train-samples",
        nargs="+",
        default=None,
        help="Optional explicit source samples for a context-shift contrast.",
    )
    parser.add_argument("--encoder", default="conch")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cell-type-root", type=Path, default=None)
    parser.add_argument("--marker-list", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--ratios", nargs="+", type=float, default=[0.1, 0.25, 1.0]
    )
    parser.add_argument("--max-target-genes", type=int, default=200)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def load_h5(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        return {
            "x": np.asarray(handle["embeddings"][:], dtype=np.float32),
            "y": np.asarray(handle["expressions"][:], dtype=np.float32),
            "cell_ids": decode(handle["cell_ids"][:]),
            "genes": decode(handle["gene_names"][:]),
        }


def intersect_genes(samples: dict[str, dict]) -> list[str]:
    first = next(iter(samples.values()))
    common = set(first["genes"])
    for data in list(samples.values())[1:]:
        common.intersection_update(data["genes"])
    return [gene for gene in first["genes"] if gene in common]


def subset_genes(data: dict, genes: list[str]) -> np.ndarray:
    index = {gene: idx for idx, gene in enumerate(data["genes"])}
    return data["y"][:, [index[gene] for gene in genes]]


def balanced_subsample(
    source_names: np.ndarray, ratio: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for source in sorted(set(source_names)):
        indices = np.flatnonzero(source_names == source)
        n = len(indices) if ratio >= 1 else max(100, int(round(len(indices) * ratio)))
        n = min(n, len(indices))
        selected.append(np.sort(rng.choice(indices, size=n, replace=False)))
    return np.sort(np.concatenate(selected))


def fit_image_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    pca_components: int,
    ridge_alpha: float,
    seed: int,
) -> np.ndarray:
    n_components = min(pca_components, x_train.shape[1], len(x_train) - 1)
    model = Pipeline(
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
    model.fit(x_train, y_train)
    return model.predict(x_test).astype(np.float32)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sheet = pd.read_csv(args.sample_sheet)
    if args.target not in set(sheet["sample"]):
        raise ValueError(f"Unknown target {args.target}")
    target_row = sheet.set_index("sample").loc[args.target]
    peers = sheet[
        (sheet["species"] == target_row["species"])
        & (sheet["organ"] == target_row["organ"])
    ]["sample"].tolist()
    train_samples = (
        list(args.train_samples)
        if args.train_samples
        else [sample for sample in peers if sample != args.target]
    )
    invalid = sorted(set(train_samples) - set(sheet["sample"]))
    if invalid:
        raise ValueError(f"Unknown source samples: {invalid}")
    incompatible = sheet[
        sheet["sample"].isin(train_samples)
        & (
            (sheet["species"] != target_row["species"])
            | (sheet["organ"] != target_row["organ"])
        )
    ]["sample"].tolist()
    if incompatible:
        raise ValueError(
            f"Source samples do not match target species/organ: {incompatible}"
        )
    if not train_samples:
        raise RuntimeError(
            f"No same-organ source sample for target {args.target} "
            f"({target_row['species']}/{target_row['organ']})"
        )

    required_samples = train_samples + [args.target]
    sample_data = {
        sample: load_h5(args.embedding_root / args.encoder / f"{sample}.h5")
        for sample in required_samples
    }
    common_genes = intersect_genes(sample_data)
    if len(common_genes) < 10:
        raise RuntimeError(
            f"Only {len(common_genes)} shared genes among {peers}"
        )

    x_train_all = np.concatenate(
        [sample_data[sample]["x"] for sample in train_samples], axis=0
    )
    y_train_all_raw = np.concatenate(
        [subset_genes(sample_data[sample], common_genes) for sample in train_samples],
        axis=0,
    )
    source_names = np.concatenate(
        [
            np.repeat(sample, len(sample_data[sample]["cell_ids"]))
            for sample in train_samples
        ]
    )
    target = sample_data[args.target]
    x_test = target["x"]
    y_test_all_raw = subset_genes(target, common_genes)

    top_indices = top_variable_genes(
        y_train_all_raw, max_genes=args.max_target_genes
    )
    genes = [common_genes[idx] for idx in top_indices]
    y_train_all_raw = y_train_all_raw[:, top_indices]
    y_test_raw = y_test_all_raw[:, top_indices]
    y_train_all = np.log1p(np.clip(y_train_all_raw, 0, None)).astype(np.float32)
    y_test = np.log1p(np.clip(y_test_raw, 0, None)).astype(np.float32)
    marker_genes = {gene.upper() for gene in read_gene_list(args.marker_list)}

    target_meta = pd.DataFrame({"cell_id": target["cell_ids"]})
    type_path = (
        args.cell_type_root / f"{args.target}.csv"
        if args.cell_type_root is not None
        else None
    )
    target_meta = add_cell_types(target_meta, type_path)

    summaries: list[dict] = []
    gene_frames: list[pd.DataFrame] = []
    type_records: list[dict] = []
    run_records: list[dict] = []

    for seed in args.seeds:
        for ratio in args.ratios:
            train_idx = balanced_subsample(
                source_names, ratio=ratio, seed=seed * 10_000 + int(ratio * 1000)
            )
            image_pred = fit_image_model(
                x_train_all[train_idx],
                y_train_all[train_idx],
                x_test,
                pca_components=args.pca_components,
                ridge_alpha=args.ridge_alpha,
                seed=seed,
            )
            mean_pred = np.repeat(
                np.mean(y_train_all[train_idx], axis=0, keepdims=True),
                len(x_test),
                axis=0,
            )
            predictions = {"image": image_pred, "train_mean": mean_pred}
            gene_sets = {
                "top50_hvg": np.arange(min(50, len(genes))),
                "top200_hvg": np.arange(len(genes)),
            }
            marker_idx = np.asarray(
                [
                    idx
                    for idx, gene in enumerate(genes)
                    if gene.upper() in marker_genes
                ],
                dtype=int,
            )
            if len(marker_idx) >= 5:
                gene_sets["marker_hvg_overlap"] = marker_idx

            for model, pred in predictions.items():
                metrics = metric_rows(y_test_raw, y_test, pred, genes)
                metrics.insert(0, "model", model)
                metrics.insert(0, "ratio", ratio)
                metrics.insert(0, "seed", seed)
                metrics.insert(0, "target", args.target)
                metrics.insert(0, "organ", target_row["organ"])
                metrics.insert(0, "species", target_row["species"])
                metrics.insert(0, "encoder", args.encoder)
                gene_frames.append(metrics)
                for gene_set, indices in gene_sets.items():
                    summaries.append(
                        {
                            "encoder": args.encoder,
                            "species": target_row["species"],
                            "organ": target_row["organ"],
                            "target": args.target,
                            "train_samples": ";".join(train_samples),
                            "seed": seed,
                            "ratio": ratio,
                            "model": model,
                            "gene_set": gene_set,
                            "n_train": len(train_idx),
                            "n_test": len(x_test),
                            "n_shared_genes": len(common_genes),
                            **summarize(
                                metrics.iloc[indices],
                                y_test[:, indices],
                                pred[:, indices],
                            ),
                        }
                    )

            if ratio >= 1:
                rows = cell_type_rows(
                    target_meta["cell_type"].astype(str).to_numpy(),
                    y_test,
                    predictions,
                )
                for row in rows:
                    row.update(
                        {
                            "encoder": args.encoder,
                            "species": target_row["species"],
                            "organ": target_row["organ"],
                            "target": args.target,
                            "seed": seed,
                            "ratio": ratio,
                            "n_genes": len(genes),
                        }
                    )
                type_records.extend(rows)
                if args.save_predictions and seed == args.seeds[0]:
                    np.savez_compressed(
                        args.out_dir / "predictions.npz",
                        cell_ids=np.asarray(target["cell_ids"], dtype=str),
                        genes=np.asarray(genes, dtype=str),
                        y_true_raw=y_test_raw,
                        y_true_log1p=y_test,
                        pred_image=image_pred,
                        pred_train_mean=mean_pred,
                        cell_type=target_meta["cell_type"].astype(str).to_numpy(),
                    )
            run_records.append(
                {
                    "seed": seed,
                    "ratio": ratio,
                    "n_train": len(train_idx),
                    "n_test": len(x_test),
                }
            )

    pd.DataFrame(summaries).to_csv(args.out_dir / "summary.csv", index=False)
    pd.concat(gene_frames, ignore_index=True).to_csv(
        args.out_dir / "gene_metrics.csv", index=False
    )
    pd.DataFrame(type_records).to_csv(
        args.out_dir / "cell_type_metrics.csv", index=False
    )
    with (args.out_dir / "run_info.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "target": args.target,
                "train_samples": train_samples,
                "common_genes": common_genes,
                "selected_genes": genes,
                "runs": run_records,
            },
            handle,
            indent=2,
        )
    print(
        f"[done] target={args.target} train={train_samples} "
        f"shared={len(common_genes)} selected={len(genes)}"
    )


if __name__ == "__main__":
    main()
