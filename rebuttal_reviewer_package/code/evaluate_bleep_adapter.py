#!/usr/bin/env python
"""Xenium adaptation of the official BLEEP contrastive/retrieval algorithm.

The official BLEEP projection head, symmetric soft-target contrastive loss, and
top-50 reference-expression retrieval are preserved. Frozen CONCH features are
used in place of re-running BLEEP's ResNet-50 image encoder so that the
cell-versus-spot target, not an encoder mismatch, is the controlled variable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_spatial import (  # noqa: E402
    align_metadata,
    load_embedding,
    metric_rows,
    read_gene_list,
    spatial_split,
    summarize,
    top_variable_genes,
)


OFFICIAL_BLEEP_COMMIT = "23959670c7407510c1716a173184bdcc18833a35"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--marker-list", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--buffer-fraction", type=float, default=0.05)
    parser.add_argument("--max-target-genes", type=int, default=200)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


class ProjectionHead(nn.Module):
    """Projection head copied structurally from official BLEEP modules.py."""

    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        projected = self.projection(values)
        hidden = self.gelu(projected)
        hidden = self.fc(hidden)
        hidden = self.dropout(hidden)
        return self.layer_norm(hidden + projected)


class BleepAdapter(nn.Module):
    def __init__(
        self,
        image_dim: int,
        expression_dim: int,
        projection_dim: int,
        dropout: float,
        temperature: float,
    ):
        super().__init__()
        self.image_projection = ProjectionHead(
            image_dim, projection_dim, dropout
        )
        self.spot_projection = ProjectionHead(
            expression_dim, projection_dim, dropout
        )
        self.temperature = temperature

    @staticmethod
    def soft_cross_entropy(
        predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return (-targets * predictions.log_softmax(dim=-1)).sum(dim=1)

    def forward(
        self, image_features: torch.Tensor, expression_features: torch.Tensor
    ) -> torch.Tensor:
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(expression_features)
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        image_similarity = image_embeddings @ image_embeddings.T
        spot_similarity = spot_embeddings @ spot_embeddings.T
        targets = functional.softmax(
            ((image_similarity + spot_similarity) / 2) / self.temperature,
            dim=-1,
        )
        spot_loss = self.soft_cross_entropy(logits, targets)
        image_loss = self.soft_cross_entropy(logits.T, targets.T)
        return ((image_loss + spot_loss) / 2).mean()


def train_model(
    x: np.ndarray,
    expression_input: np.ndarray,
    train_idx: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[BleepAdapter, list[float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)
    model = BleepAdapter(
        image_dim=x.shape[1],
        expression_dim=expression_input.shape[1],
        projection_dim=args.projection_dim,
        dropout=args.dropout,
        temperature=args.temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    dataset = TensorDataset(
        torch.from_numpy(x[train_idx]).float(),
        torch.from_numpy(expression_input[train_idx]).float(),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )
    losses: list[float] = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        seen = 0
        for image_batch, expression_batch in loader:
            image_batch = image_batch.to(device)
            expression_batch = expression_batch.to(device)
            loss = model(image_batch, expression_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(image_batch)
            seen += len(image_batch)
        epoch_loss = total / max(seen, 1)
        losses.append(epoch_loss)
        print(
            f"[train] seed={seed} epoch={epoch + 1}/{args.epochs} "
            f"loss={epoch_loss:.6f}",
            flush=True,
        )
    return model, losses


def project(
    head: nn.Module,
    values: np.ndarray,
    device: str,
    batch_size: int = 1024,
) -> torch.Tensor:
    outputs = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).float()
            outputs.append(head(batch.to(device)).cpu())
    return torch.cat(outputs, dim=0)


def retrieve_expression(
    model: BleepAdapter,
    x: np.ndarray,
    expression_input: np.ndarray,
    y_log: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    top_k: int,
    device: str,
) -> np.ndarray:
    reference = project(
        model.spot_projection, expression_input[train_idx], device
    )
    query = project(model.image_projection, x[test_idx], device)
    reference = functional.normalize(reference, p=2, dim=-1)
    query = functional.normalize(query, p=2, dim=-1)
    k = min(top_k, len(reference))
    indices = torch.topk(query @ reference.T, k=k, dim=1).indices.numpy()
    return np.mean(y_log[train_idx][indices], axis=1).astype(np.float32)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load_embedding(args.embedding)
    data, meta = align_metadata(data, args.manifest, args.sample)
    coords = meta[["coord_x", "coord_y"]].to_numpy(dtype=np.float32)
    marker_genes = {gene.upper() for gene in read_gene_list(args.marker_list)}

    summaries: list[dict] = []
    gene_frames: list[pd.DataFrame] = []
    runs: list[dict] = []
    for seed in args.seeds:
        train_idx, test_idx, split_info = spatial_split(
            coords,
            seed,
            test_fraction=args.test_fraction,
            buffer_fraction=args.buffer_fraction,
        )
        selected = top_variable_genes(
            data["y"][train_idx], args.max_target_genes
        )
        genes = [data["genes"][idx] for idx in selected]
        y_raw = data["y"][:, selected]
        y_log = np.log1p(np.clip(y_raw, 0, None)).astype(np.float32)
        mean = np.mean(y_log[train_idx], axis=0, keepdims=True)
        std = np.std(y_log[train_idx], axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        expression_input = ((y_log - mean) / std).astype(np.float32)

        model, losses = train_model(
            data["x"], expression_input, train_idx, args, seed
        )
        pred = retrieve_expression(
            model,
            data["x"],
            expression_input,
            y_log,
            train_idx,
            test_idx,
            top_k=args.top_k,
            device=args.device,
        )
        metrics = metric_rows(y_raw[test_idx], y_log[test_idx], pred, genes)
        metrics.insert(0, "model", "bleep_conch_adapter")
        metrics.insert(0, "seed", seed)
        metrics.insert(0, "sample", args.sample)
        gene_frames.append(metrics)
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
        for gene_set, indices in gene_sets.items():
            summaries.append(
                {
                    "sample": args.sample,
                    "encoder": "conch",
                    "method": "bleep_conch_adapter",
                    "seed": seed,
                    "split_axis": split_info["axis"],
                    "split_side": split_info["side"],
                    "gene_set": gene_set,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "top_k": min(args.top_k, len(train_idx)),
                    **summarize(
                        metrics.iloc[indices],
                        y_log[test_idx][:, indices],
                        pred[:, indices],
                    ),
                }
            )
        runs.append(
            {
                **split_info,
                "selected_genes": genes,
                "epoch_losses": losses,
            }
        )

    pd.DataFrame(summaries).to_csv(args.out_dir / "summary.csv", index=False)
    pd.concat(gene_frames, ignore_index=True).to_csv(
        args.out_dir / "gene_metrics.csv", index=False
    )
    with (args.out_dir / "run_info.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": "BLEEP Xenium/CONCH adapter",
                "official_bleep_commit": OFFICIAL_BLEEP_COMMIT,
                "controlled_changes": [
                    "frozen CONCH cell-image features replace BLEEP ResNet-50 features",
                    "Xenium cell expression replaces Visium spot expression",
                    "contiguous spatial holdouts replace random spot split",
                ],
                "preserved_components": [
                    "official residual projection-head structure",
                    "official symmetric soft-target contrastive loss",
                    "official L2-normalized top-50 expression retrieval",
                    "official four epochs, AdamW lr=1e-3, weight_decay=1e-3",
                ],
                "runs": runs,
            },
            handle,
            indent=2,
        )
    print(f"[done] BLEEP adapter {args.sample} -> {args.out_dir}")


if __name__ == "__main__":
    main()

