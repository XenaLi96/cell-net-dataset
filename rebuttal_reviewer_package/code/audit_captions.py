#!/usr/bin/env python
"""Audit available generated captions for label leakage and templating."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\*\*image pair \d+\*\*\s*:\s*", "", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"[^a-z]+", " ", text)
    return " ".join(text.split())


def remove_label(text: str, label: str) -> str:
    cleaned = normalize(text)
    label_tokens = normalize(label)
    if label_tokens:
        cleaned = re.sub(
            rf"\b{re.escape(label_tokens)}s?\b", " ", cleaned, flags=re.IGNORECASE
        )
    return " ".join(cleaned.split())


def classify_text(
    text: pd.Series, labels: pd.Series, seed: int
) -> dict[str, float]:
    counts = labels.value_counts()
    keep_classes = counts[counts >= 3].index
    keep = labels.isin(keep_classes)
    x = text[keep].astype(str).to_numpy()
    y = labels[keep].astype(str).to_numpy()
    min_count = int(pd.Series(y).value_counts().min())
    n_splits = min(5, min_count)
    if n_splits < 2 or len(np.unique(y)) < 2:
        return {
            "n_examples": len(y),
            "n_classes": len(np.unique(y)),
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "majority_accuracy": float("nan"),
        }
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = cross_val_predict(model, x, y, cv=folds, n_jobs=1)
    majority = float(pd.Series(y).value_counts(normalize=True).max())
    return {
        "n_examples": len(y),
        "n_classes": len(np.unique(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "majority_accuracy": majority,
    }


def audit_frame(frame: pd.DataFrame, source: str, seed: int) -> tuple[dict, pd.DataFrame]:
    required = {"caption", "cell-type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} misses columns {sorted(missing)}")
    frame = frame.dropna(subset=["caption", "cell-type"]).copy()
    frame["normalized"] = frame["caption"].map(normalize)
    frame["label_normalized"] = frame["cell-type"].map(normalize)
    frame["label_removed"] = [
        remove_label(text, label)
        for text, label in zip(frame["caption"], frame["cell-type"])
    ]
    frame["explicit_label_mention"] = [
        bool(label and re.search(rf"\b{re.escape(label)}s?\b", text))
        for text, label in zip(frame["normalized"], frame["label_normalized"])
    ]
    frame["has_image_pair_number"] = frame["caption"].astype(str).str.contains(
        r"\*\*Image Pair \d+\*\*", regex=True
    )
    frame["word_count"] = frame["normalized"].str.split().str.len()
    frame["exact_duplicate_after_normalization"] = frame["normalized"].duplicated(
        keep=False
    )
    frame["duplicate_after_label_removal"] = frame["label_removed"].duplicated(
        keep=False
    )

    full_classifier = classify_text(frame["normalized"], frame["cell-type"], seed)
    sanitized_classifier = classify_text(
        frame["label_removed"], frame["cell-type"], seed
    )
    summary = {
        "source": source,
        "n_captions": len(frame),
        "n_cell_types": int(frame["cell-type"].nunique()),
        "fraction_explicit_cell_type_mention": float(
            frame["explicit_label_mention"].mean()
        ),
        "fraction_image_pair_number": float(frame["has_image_pair_number"].mean()),
        "fraction_exact_duplicate_normalized": float(
            frame["exact_duplicate_after_normalization"].mean()
        ),
        "fraction_duplicate_after_label_removal": float(
            frame["duplicate_after_label_removal"].mean()
        ),
        "median_word_count": float(frame["word_count"].median()),
        "p05_word_count": float(frame["word_count"].quantile(0.05)),
        "p95_word_count": float(frame["word_count"].quantile(0.95)),
        "cell_type_from_full_caption": full_classifier,
        "cell_type_after_explicit_label_removal": sanitized_classifier,
    }
    by_type = (
        frame.groupby("cell-type", dropna=False)
        .agg(
            n=("caption", "size"),
            explicit_label_fraction=("explicit_label_mention", "mean"),
            normalized_duplicate_fraction=(
                "exact_duplicate_after_normalization",
                "mean",
            ),
            median_words=("word_count", "median"),
        )
        .reset_index()
    )
    by_type.insert(0, "source", source)
    return summary, by_type


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    type_frames: list[pd.DataFrame] = []
    for path in args.inputs:
        frame = pd.read_csv(path)
        summary, by_type = audit_frame(frame, path.stem, args.seed)
        summaries.append(summary)
        type_frames.append(by_type)
        print(
            f"[audit] {path.name}: n={summary['n_captions']} "
            f"types={summary['n_cell_types']}"
        )
    with (args.out_dir / "caption_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, sort_keys=True)
    pd.json_normalize(summaries).to_csv(
        args.out_dir / "caption_audit_summary.csv", index=False
    )
    pd.concat(type_frames, ignore_index=True).to_csv(
        args.out_dir / "caption_audit_by_cell_type.csv", index=False
    )
    print(f"[done] wrote caption audit to {args.out_dir}")


if __name__ == "__main__":
    main()

