from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = "PATH"
DEFAULT_RATIOS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
MODEL_NAME = "resnet50"


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def decode_array(values: Iterable[object]) -> list[str]:
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def read_cell_expression(expr_path: str | Path) -> tuple[np.ndarray, list[str]]:
    with h5py.File(expr_path, "r") as handle:
        expression = np.asarray(handle["gene_expression"][:], dtype=np.float32)
        gene_names = decode_array(handle["gene_names"][:])
    if expression.ndim != 1:
        expression = expression.reshape(-1)
    if len(expression) != len(gene_names):
        raise ValueError(
            f"Expression/gene name length mismatch in {expr_path}: "
            f"{len(expression)} vs {len(gene_names)}"
        )
    return expression, gene_names


def h5_string_dtype():
    return h5py.string_dtype(encoding="utf-8")


def read_h5_strings(dataset) -> list[str]:
    return decode_array(dataset[:])


def load_manifest(path: str | Path) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_parent(path)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def parse_ratios(values: list[str] | None) -> list[float]:
    if not values:
        return list(DEFAULT_RATIOS)
    ratios = [float(value) for value in values]
    bad = [ratio for ratio in ratios if ratio <= 0 or ratio > 1]
    if bad:
        raise ValueError(f"Ratios must be in (0, 1], got {bad}")
    return ratios


def parse_gene_list(path: str | None) -> list[str] | None:
    if path is None:
        return None
    with open(path) as handle:
        if path.endswith(".json"):
            data = json.load(handle)
            if isinstance(data, dict) and "genes" in data:
                return [str(gene) for gene in data["genes"]]
            if isinstance(data, list):
                return [str(gene) for gene in data]
            raise ValueError(f"Unsupported gene JSON format in {path}")
        return [line.strip() for line in handle if line.strip()]


def safe_pearson(target: np.ndarray, pred: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    target_centered = target - target.mean()
    pred_centered = pred - pred.mean()
    denom = np.sqrt(np.sum(target_centered**2) * np.sum(pred_centered**2))
    if denom == 0:
        return float("nan")
    return float(np.sum(target_centered * pred_centered) / denom)


def safe_r2(target: np.ndarray, pred: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    denom = np.sum((target - target.mean()) ** 2)
    if denom == 0:
        return float("nan")
    return float(1 - np.sum((target - pred) ** 2) / denom)


def existing_sample_dirs(data_root: str | Path) -> list[str]:
    patch_root = Path(data_root) / "cell_patches_150"
    return sorted(
        item.name
        for item in patch_root.iterdir()
        if item.is_dir() and item.name != "compressed"
    )


def manifest_path_for_sample(data_root: str | Path, sample: str) -> Path:
    return Path(data_root) / "cell_data_aligned_with_patchid" / f"{sample}.csv.gz"


def cell_dir(data_root: str | Path, sample: str, cell_id: str) -> Path:
    return Path(data_root) / "cell_patches_150" / sample / cell_id


def embedding_path(
    embedding_root: str | Path,
    sample: str,
    model_name: str = MODEL_NAME,
) -> Path:
    return Path(embedding_root) / model_name / f"{sample}.h5"
