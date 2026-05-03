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

from common import (  # noqa: E402
    MODEL_NAME,
    embedding_path,
    ensure_dir,
    parse_gene_list,
    parse_ratios,
    read_h5_strings,
    safe_pearson,
    safe_r2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset scaling-law evaluation.")
    parser.add_argument(
        "--embedding-root",
        default="PATH/cache/scaling_law/embeddings",
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument(
        "--out-dir",
        default="PATH/results/scaling_law/resnet50",
    )
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument(
        "--train-samples",
        nargs="*",
        default=None,
        help="Samples allowed in the training/internal-test pool. Useful for cross-panel tests.",
    )
    parser.add_argument("--external-samples", nargs="*", default=None)
    parser.add_argument("--external-sample-frac", type=float, default=0.2)
    parser.add_argument("--internal-test-frac", type=float, default=0.1)
    parser.add_argument("--ratios", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--pca-components", type=int, default=256)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--gene-list", default=None)
    parser.add_argument(
        "--eval-gene-list",
        default=None,
        help=(
            "Optional JSON or text gene list used only for metric aggregation. "
            "Predictions are still fit for all genes from --gene-list."
        ),
    )
    parser.add_argument(
        "--eval-top-genes",
        type=int,
        default=None,
        help=(
            "If set, aggregate summary metrics over the top N genes selected from "
            "the training pool. gene_metrics.csv still includes every predicted gene."
        ),
    )
    parser.add_argument(
        "--eval-top-gene-method",
        choices=["variance", "mean"],
        default="variance",
        help="Criterion for --eval-top-genes. Default matches highly-variable-gene style evaluation.",
    )
    parser.add_argument("--regime", default="scaling")
    parser.add_argument("--group-name", default="")
    parser.add_argument("--log1p", action="store_true")
    return parser.parse_args()


def discover_samples(
    embedding_root: str | Path,
    requested: list[str] | None,
    model_name: str,
) -> list[str]:
    model_dir = Path(embedding_root) / model_name
    samples = sorted(path.stem for path in model_dir.glob("*.h5"))
    if requested:
        requested_set = set(requested)
        samples = [sample for sample in samples if sample in requested_set]
    if not samples:
        raise RuntimeError(f"No embedding files found in {model_dir}")
    return samples


def choose_external_samples(
    samples: list[str],
    explicit: list[str] | None,
    frac: float,
    seed: int,
) -> list[str]:
    if explicit:
        external = sorted(explicit)
        missing = sorted(set(external) - set(samples))
        if missing:
            raise ValueError(f"External samples missing embeddings: {missing}")
        return external
    if frac <= 0 or len(samples) < 2:
        return []
    rng = np.random.default_rng(seed)
    n_external = int(round(len(samples) * frac))
    n_external = max(1, min(n_external, len(samples) - 1))
    return sorted(rng.choice(samples, size=n_external, replace=False).tolist())


def load_sample(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        return {
            "embeddings": np.asarray(handle["embeddings"][:], dtype=np.float32),
            "expressions": np.asarray(handle["expressions"][:], dtype=np.float32),
            "cell_ids": read_h5_strings(handle["cell_ids"]),
            "gene_names": read_h5_strings(handle["gene_names"]),
        }


def resolve_gene_order(sample_data: dict[str, dict], requested_genes: list[str] | None) -> list[str]:
    first_sample = next(iter(sample_data.values()))
    first_genes = first_sample["gene_names"]
    if requested_genes is not None:
        genes = requested_genes
    else:
        gene_sets = {sample: set(data["gene_names"]) for sample, data in sample_data.items()}
        genes = [gene for gene in first_genes if all(gene in names for names in gene_sets.values())]
    if not genes:
        raise RuntimeError("No common genes found across selected samples.")
    missing = {
        sample: sorted(set(genes) - set(data["gene_names"]))
        for sample, data in sample_data.items()
        if set(genes) - set(data["gene_names"])
    }
    if missing:
        raise ValueError(f"Requested genes are missing from some samples: {missing}")
    return genes


def concatenate_samples(sample_data: dict[str, dict], genes: list[str], log1p: bool):
    xs, ys, sample_labels, cell_ids = [], [], [], []
    for sample, data in sample_data.items():
        gene_to_idx = {gene: idx for idx, gene in enumerate(data["gene_names"])}
        gene_idx = [gene_to_idx[gene] for gene in genes]
        y = data["expressions"][:, gene_idx]
        if log1p:
            y = np.log1p(y)
        xs.append(data["embeddings"])
        ys.append(y.astype(np.float32))
        sample_labels.extend([sample] * len(y))
        cell_ids.extend(data["cell_ids"])
    return (
        np.vstack(xs).astype(np.float32),
        np.vstack(ys).astype(np.float32),
        np.asarray(sample_labels),
        np.asarray(cell_ids),
    )


def build_fixed_split(sample_labels: np.ndarray, external_samples: list[str], frac: float, seed: int):
    rng = np.random.default_rng(seed)
    by_sample = {
        sample: np.flatnonzero(sample_labels == sample)
        for sample in sorted(np.unique(sample_labels))
    }
    external_indices = []
    internal_indices = []
    train_pool_by_sample = {}

    for sample, indices in by_sample.items():
        if sample in external_samples:
            external_indices.append(indices)
            continue
        permuted = rng.permutation(indices)
        n_internal = int(round(len(permuted) * frac))
        n_internal = min(max(n_internal, 1), len(permuted) - 1) if len(permuted) > 1 else 0
        internal_indices.append(permuted[:n_internal])
        train_pool_by_sample[sample] = permuted[n_internal:]

    return {
        "external": np.concatenate(external_indices) if external_indices else np.array([], dtype=int),
        "internal": np.concatenate(internal_indices) if internal_indices else np.array([], dtype=int),
        "train_pool_by_sample": train_pool_by_sample,
    }


def subset_for_ratio(train_pool_by_sample: dict[str, np.ndarray], ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    selected = []
    for sample, indices in train_pool_by_sample.items():
        ordered = rng.permutation(indices)
        if ratio >= 1:
            n_take = len(ordered)
        else:
            n_take = int(round(len(ordered) * ratio))
            n_take = max(1, min(n_take, len(ordered)))
        selected.append(ordered[:n_take])
    if not selected:
        raise RuntimeError("No training samples available after split.")
    return np.concatenate(selected)


def evaluate_split(y_true: np.ndarray, y_pred: np.ndarray, genes: list[str]):
    rows = []
    for idx, gene in enumerate(genes):
        target = y_true[:, idx]
        pred = y_pred[:, idx]
        rows.append(
            {
                "gene": gene,
                "pearson": safe_pearson(target, pred),
                "r2": safe_r2(target, pred),
                "mse": float(np.mean((target - pred) ** 2)),
            }
        )
    return rows


def select_eval_genes(
    genes: list[str],
    y: np.ndarray,
    selection_idx: np.ndarray,
    eval_genes: list[str] | None,
    top_n: int | None,
    method: str,
) -> dict:
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    if eval_genes is not None:
        missing = [gene for gene in eval_genes if gene not in gene_to_idx]
        if missing:
            raise ValueError(f"--eval-gene-list contains genes absent from targets: {missing[:10]}")
        selected_genes = eval_genes
        scores = {gene: float("nan") for gene in selected_genes}
        mode = "gene_list"
    elif top_n is not None:
        if top_n <= 0:
            raise ValueError("--eval-top-genes must be positive.")
        top_n = min(top_n, len(genes))
        source = y[selection_idx] if len(selection_idx) else y
        if method == "variance":
            raw_scores = np.nanvar(source.astype(np.float64), axis=0)
        elif method == "mean":
            raw_scores = np.nanmean(source.astype(np.float64), axis=0)
        else:
            raise ValueError(f"Unsupported eval top gene method: {method}")
        sortable_scores = np.where(np.isfinite(raw_scores), raw_scores, -np.inf)
        order = np.argsort(-sortable_scores, kind="stable")[:top_n]
        selected_genes = [genes[idx] for idx in order]
        scores = {genes[idx]: float(raw_scores[idx]) for idx in order}
        mode = f"top_{top_n}_{method}"
    else:
        selected_genes = list(genes)
        scores = {gene: float("nan") for gene in selected_genes}
        mode = "all"

    rank = {gene: idx + 1 for idx, gene in enumerate(selected_genes)}
    return {
        "genes": selected_genes,
        "gene_set": set(selected_genes),
        "rank": rank,
        "scores": scores,
        "mode": mode,
        "method": method if top_n is not None and eval_genes is None else mode,
    }


def annotate_eval_gene_rows(rows: list[dict], eval_selection: dict) -> list[dict]:
    eval_gene_set = eval_selection["gene_set"]
    rank = eval_selection["rank"]
    scores = eval_selection["scores"]
    annotated = []
    for row in rows:
        gene = row["gene"]
        annotated.append(
            {
                **row,
                "eval_gene": gene in eval_gene_set,
                "eval_gene_rank": rank.get(gene, np.nan),
                "eval_gene_score": scores.get(gene, np.nan),
            }
        )
    return annotated


def summarize_gene_rows(rows: list[dict], eval_only: bool = True) -> dict:
    frame = pd.DataFrame(rows)
    if eval_only and "eval_gene" in frame.columns:
        frame = frame[frame["eval_gene"]]
    if frame.empty:
        raise RuntimeError("No genes available for metric summary.")
    return {
        "pearson_mean": float(np.nanmean(frame["pearson"])),
        "pearson_median": float(np.nanmedian(frame["pearson"])),
        "r2_mean": float(np.nanmean(frame["r2"])),
        "mse_mean": float(np.nanmean(frame["mse"])),
    }


def fit_predict(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    pca_components: int,
    ridge_alpha: float,
    seed: int,
):
    n_components = min(pca_components, x.shape[1], len(train_idx) - 1)
    if n_components < 1:
        raise RuntimeError(f"Need at least two training rows for PCA, got {len(train_idx)}")
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=seed)),
        ]
    )
    x_train = pipe.fit_transform(x[train_idx])
    x_test = pipe.transform(x[test_idx])
    reg = Ridge(alpha=ridge_alpha)
    reg.fit(x_train, y[train_idx])
    return reg.predict(x_test), n_components


def main() -> None:
    args = parse_args()
    ratios = parse_ratios(args.ratios)

    if args.train_samples:
        requested_samples = set(args.train_samples or [])
        requested_samples.update(args.external_samples or [])
    else:
        requested_samples = set(args.samples or [])
        requested_samples.update(args.external_samples or [])
    samples = discover_samples(
        args.embedding_root,
        sorted(requested_samples) if requested_samples else None,
        args.model_name,
    )

    if args.train_samples:
        train_samples = sorted(args.train_samples)
        missing_train = sorted(set(train_samples) - set(samples))
        if missing_train:
            raise ValueError(f"Train samples missing embeddings: {missing_train}")
        external_samples = sorted(args.external_samples or [])
        missing_external = sorted(set(external_samples) - set(samples))
        if missing_external:
            raise ValueError(f"External samples missing embeddings: {missing_external}")
    else:
        external_samples = choose_external_samples(
            samples, args.external_samples, args.external_sample_frac, args.split_seed
        )
        train_samples = [sample for sample in samples if sample not in external_samples]

    print(f"[samples] train/internal samples: {train_samples}")
    print(f"[samples] external samples: {external_samples}")

    sample_data = {
        sample: load_sample(embedding_path(args.embedding_root, sample, args.model_name))
        for sample in samples
    }
    genes = resolve_gene_order(sample_data, parse_gene_list(args.gene_list))
    x, y, sample_labels, cell_ids = concatenate_samples(sample_data, genes, args.log1p)
    print(f"[data] X={x.shape} Y={y.shape} genes={len(genes)}")

    split = build_fixed_split(
        sample_labels,
        external_samples=external_samples,
        frac=args.internal_test_frac,
        seed=args.split_seed,
    )
    eval_selection_idx = np.concatenate(list(split["train_pool_by_sample"].values()))
    eval_selection = select_eval_genes(
        genes=genes,
        y=y,
        selection_idx=eval_selection_idx,
        eval_genes=parse_gene_list(args.eval_gene_list),
        top_n=args.eval_top_genes,
        method=args.eval_top_gene_method,
    )
    print(
        f"[eval] summary genes={len(eval_selection['genes'])}/{len(genes)} "
        f"mode={eval_selection['mode']}"
    )

    ensure_dir(args.out_dir)
    with open(Path(args.out_dir) / "split_info.json", "w") as handle:
        json.dump(
            {
                "regime": args.regime,
                "group_name": args.group_name,
                "samples": samples,
                "train_samples": train_samples,
                "external_samples": external_samples,
                "internal_test_frac": args.internal_test_frac,
                "split_seed": args.split_seed,
                "n_internal": int(len(split["internal"])),
                "n_external": int(len(split["external"])),
                "n_train_pool": int(sum(len(v) for v in split["train_pool_by_sample"].values())),
                "n_genes": len(genes),
                "n_eval_genes": len(eval_selection["genes"]),
                "eval_gene_mode": eval_selection["mode"],
                "eval_top_genes": args.eval_top_genes,
                "eval_top_gene_method": args.eval_top_gene_method,
                "eval_gene_list": args.eval_gene_list,
                "log1p": args.log1p,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    with open(Path(args.out_dir) / "eval_genes.json", "w") as handle:
        json.dump(
            {
                "genes": eval_selection["genes"],
                "mode": eval_selection["mode"],
                "top_n": args.eval_top_genes,
                "method": args.eval_top_gene_method,
                "selection_source": "train_pool",
                "n_total_genes": len(genes),
                "scores": eval_selection["scores"],
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    summary_rows = []
    gene_rows = []
    for seed in args.seeds:
        for ratio in ratios:
            train_idx = subset_for_ratio(split["train_pool_by_sample"], ratio, seed)
            eval_sets = [("internal", split["internal"])]
            if len(split["external"]):
                eval_sets.append(("external", split["external"]))

            for split_name, test_idx in eval_sets:
                if len(test_idx) == 0:
                    continue
                preds, n_components = fit_predict(
                    x=x,
                    y=y,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    pca_components=args.pca_components,
                    ridge_alpha=args.ridge_alpha,
                    seed=seed,
                )
                rows = annotate_eval_gene_rows(
                    evaluate_split(y[test_idx], preds, genes),
                    eval_selection,
                )
                summary = summarize_gene_rows(rows)
                base = {
                    "model": args.model_name,
                    "regime": args.regime,
                    "group_name": args.group_name,
                    "seed": seed,
                    "ratio": ratio,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "split_type": split_name,
                    "pca_components": int(n_components),
                    "ridge_alpha": args.ridge_alpha,
                    "n_genes": len(genes),
                    "n_eval_genes": len(eval_selection["genes"]),
                    "eval_gene_mode": eval_selection["mode"],
                    "eval_top_genes": args.eval_top_genes,
                    "eval_top_gene_method": args.eval_top_gene_method,
                }
                summary_rows.append({**base, **summary})
                for row in rows:
                    gene_rows.append({**base, **row})
                print(
                    f"[result] regime={args.regime} seed={seed} ratio={ratio:g} split={split_name} "
                    f"n_train={len(train_idx)} pearson_mean={summary['pearson_mean']:.4f} "
                    f"eval_genes={len(eval_selection['genes'])}"
                )

    pd.DataFrame(summary_rows).to_csv(Path(args.out_dir) / "summary.csv", index=False)
    pd.DataFrame(gene_rows).to_csv(Path(args.out_dir) / "gene_metrics.csv", index=False)
    print(f"[done] wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
