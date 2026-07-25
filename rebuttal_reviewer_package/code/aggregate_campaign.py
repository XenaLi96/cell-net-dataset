#!/usr/bin/env python
"""Aggregate campaign outputs into reviewer-facing tables, figures, and report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon


ENCODERS = [
    "resnet50",
    "conch",
    "ctranspath",
    "hoptimus0",
    "phikon",
    "uni",
    "uni2",
]
CROSS_TARGETS = [
    "FFHO026",
    "HOCD032",
    "HOCW023",
    "HBCA041",
    "HBCH024",
    "HBES040",
    "HLCD033",
    "HLCX022",
    "XTNL005",
    "HPDA034",
    "HPXM007",
    "PCHM006",
    "HSDX009",
    "HSDX009_2",
    "HSPD012",
    "PDHS029",
    "WMPP017",
    "WMPW025",
]
PSEUDOSPOT_SAMPLES = [
    "FFMB028",
    "HLCX022",
    "FFHO026",
    "HBCH024",
    "HSDX009",
    "WMPP017",
]
FULL_DENSITY_SAMPLES = ["HHDX011", "HLCX022"]
SKIN_CONTRASTS = [
    "normal_replication_1_to_2",
    "normal_to_melanoma_addon",
    "normal_to_melanoma_prime",
    "melanoma_to_normal_1",
    "melanoma_to_normal_2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--sample-sheet", required=True, type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def read_many(paths: list[Path], label_from_parent: str | None = None) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        frame = pd.read_csv(path)
        if label_from_parent is not None:
            frame[label_from_parent] = path.parent.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def bootstrap_ci(
    values: np.ndarray, reps: int, seed: int
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def paired_tests(
    frame: pd.DataFrame,
    index_cols: list[str],
    metric: str,
    reference: str = "image",
) -> pd.DataFrame:
    pivot = frame.pivot_table(index=index_cols, columns="model", values=metric)
    rows = []
    if reference not in pivot:
        return pd.DataFrame()
    for comparator in sorted(set(pivot.columns) - {reference}):
        pair = pivot[[reference, comparator]].dropna()
        if len(pair) == 0:
            continue
        diff = pair[reference] - pair[comparator]
        try:
            statistic, pvalue = wilcoxon(diff)
        except ValueError:
            statistic, pvalue = float("nan"), 1.0
        rows.append(
            {
                "reference": reference,
                "comparator": comparator,
                "metric": metric,
                "n_pairs": len(pair),
                "mean_difference": float(diff.mean()),
                "median_difference": float(diff.median()),
                "wilcoxon_statistic": float(statistic),
                "wilcoxon_pvalue": float(pvalue),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # Holm family-wise correction across the comparator set.
    order = np.argsort(result["wilcoxon_pvalue"].to_numpy(dtype=float))
    adjusted = np.empty(len(result), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (len(result) - rank) * float(
            result.iloc[idx]["wilcoxon_pvalue"]
        )
        running = max(running, candidate)
        adjusted[idx] = min(running, 1.0)
    result["wilcoxon_pvalue_holm"] = adjusted
    return result


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(rows)


def expected_status(root: Path, samples: list[str]) -> pd.DataFrame:
    checks: list[dict] = []

    def add(family: str, item: str, path: Path) -> None:
        checks.append(
            {
                "family": family,
                "item": item,
                "summary_path": str(path.relative_to(root)),
                "complete": path.exists() and path.stat().st_size > 0,
            }
        )

    for sample in samples:
        add(
            "xenium_spatial",
            sample,
            root / "results" / "xenium_spatial" / "conch" / sample / "summary.csv",
        )
        add(
            "bleep_adapter",
            sample,
            root
            / "results"
            / "xenium_bleep_adapter"
            / sample
            / "summary.csv",
        )
    for encoder in ENCODERS:
        add(
            "encoder_spatial",
            encoder,
            root
            / "results"
            / "encoder_spatial"
            / "HHDX011"
            / encoder
            / "summary.csv",
        )
    for target in CROSS_TARGETS:
        add(
            "cross_sample",
            target,
            root
            / "results"
            / "xenium_cross_sample"
            / "conch"
            / target
            / "summary.csv",
        )
    for sample in PSEUDOSPOT_SAMPLES:
        add(
            "pseudospot",
            sample,
            root
            / "results"
            / "xenium_pseudospot"
            / "conch"
            / sample
            / "summary.csv",
        )
    for contrast in SKIN_CONTRASTS:
        add(
            "skin_context",
            contrast,
            root / "results" / "skin_context" / contrast / "summary.csv",
        )
    for example in ("d", "e", "g"):
        add(
            "hd_assignment",
            example,
            root / "results" / "hd_assignment" / example / "summary.json",
        )
    add(
        "caption_audit",
        "HD_a",
        root / "results" / "caption_audit" / "caption_audit_summary.csv",
    )
    return pd.DataFrame(checks)


def main() -> None:
    args = parse_args()
    root = args.campaign_root
    table_dir = root / "tables"
    figure_dir = root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    sheet = pd.read_csv(args.sample_sheet)
    samples = sheet["sample"].tolist()

    status = expected_status(root, samples)
    status.to_csv(table_dir / "campaign_status.csv", index=False)
    status_summary = (
        status.groupby("family")["complete"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "n_complete", "count": "n_expected"})
        .reset_index()
    )
    status_summary.to_csv(table_dir / "campaign_status_summary.csv", index=False)

    encoder = read_many(
        [
            root
            / "results"
            / "encoder_spatial"
            / "HHDX011"
            / name
            / "summary.csv"
            for name in ENCODERS
        ]
    )
    encoder_table = pd.DataFrame()
    if not encoder.empty:
        encoder.to_csv(table_dir / "encoder_all_runs.csv", index=False)
        encoder_primary = encoder[
            (encoder["ratio"] == 1)
            & (encoder["gene_set"] == "top50_hvg")
        ].copy()
        metrics = [
            "pearson_gene_macro",
            "spearman_gene_macro",
            "r2_gene_macro",
            "pearson_cell_macro",
            "f1_gene_macro",
            "auprc_gene_macro",
        ]
        encoder_table = (
            encoder_primary.groupby(["encoder", "model"])[metrics]
            .agg(["mean", "std"])
            .reset_index()
        )
        encoder_table.columns = [
            "_".join([str(part) for part in column if part])
            if isinstance(column, tuple)
            else column
            for column in encoder_table.columns
        ]
        encoder_table.to_csv(table_dir / "encoder_primary.csv", index=False)
        tests = []
        for name, group in encoder_primary.groupby("encoder"):
            test = paired_tests(
                group, ["seed"], "pearson_gene_macro", reference="image"
            )
            test.insert(0, "encoder", name)
            tests.append(test)
        pd.concat(tests, ignore_index=True).to_csv(
            table_dir / "encoder_paired_tests.csv", index=False
        )
        scaling = (
            encoder[
                (encoder["model"] == "image")
                & (encoder["gene_set"] == "top50_hvg")
            ]
            .groupby(["encoder", "ratio"])["pearson_gene_macro"]
            .agg(["mean", "std"])
            .reset_index()
        )
        scaling.to_csv(table_dir / "encoder_training_size.csv", index=False)

        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        image_encoder = encoder_primary[encoder_primary["model"] == "image"]
        sns.barplot(
            data=image_encoder,
            x="encoder",
            y="pearson_gene_macro",
            errorbar="sd",
            ax=ax,
            color="#4472C4",
        )
        ax.set(xlabel="", ylabel="Gene-macro Pearson", title="Spatially held-out HHDX011")
        ax.tick_params(axis="x", rotation=30)
        save_figure(fig, figure_dir / "encoder_spatial")

        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        sns.lineplot(
            data=scaling,
            x="ratio",
            y="mean",
            hue="encoder",
            marker="o",
            ax=ax,
        )
        ax.set(
            xlabel="Fraction of spatial training pool",
            ylabel="Gene-macro Pearson",
            title="Training-size sensitivity",
        )
        save_figure(fig, figure_dir / "encoder_training_size")

    spatial = read_many(
        [
            root
            / "results"
            / "xenium_spatial"
            / "conch"
            / sample
            / "summary.csv"
            for sample in samples
        ]
    )
    spatial_primary = pd.DataFrame()
    marker_by_sample = pd.DataFrame()
    cell_type_by_sample = pd.DataFrame()
    if not spatial.empty:
        spatial = spatial.merge(
            sheet[["sample", "species", "organ", "condition"]],
            on="sample",
            how="left",
        )
        spatial.to_csv(table_dir / "xenium_spatial_all_runs.csv", index=False)
        spatial_primary = spatial[
            (spatial["ratio"] == 1)
            & (spatial["gene_set"] == "top50_hvg")
        ].copy()
        sample_means = (
            spatial_primary.groupby(
                ["sample", "species", "organ", "condition", "model"]
            )[
                [
                    "pearson_gene_macro",
                    "spearman_gene_macro",
                    "r2_gene_macro",
                    "pearson_cell_macro",
                    "f1_gene_macro",
                    "auprc_gene_macro",
                ]
            ]
            .mean()
            .reset_index()
        )
        sample_means.to_csv(table_dir / "xenium_spatial_by_sample.csv", index=False)
        macro_rows = []
        for model, group in sample_means.groupby("model"):
            for metric in (
                "pearson_gene_macro",
                "spearman_gene_macro",
                "pearson_cell_macro",
                "f1_gene_macro",
            ):
                mean, low, high = bootstrap_ci(
                    group[metric].to_numpy(), args.bootstrap_reps, args.seed
                )
                macro_rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "n_samples": len(group),
                        "macro_mean": mean,
                        "bootstrap_95_low": low,
                        "bootstrap_95_high": high,
                    }
                )
        pd.DataFrame(macro_rows).to_csv(
            table_dir / "xenium_spatial_macro_bootstrap.csv", index=False
        )
        spatial_test_frame = (
            spatial_primary.groupby(["sample", "model"])[
                "pearson_gene_macro"
            ]
            .mean()
            .reset_index()
        )
        paired_tests(
            spatial_test_frame,
            ["sample"],
            "pearson_gene_macro",
            reference="image",
        ).to_csv(table_dir / "xenium_spatial_paired_tests.csv", index=False)
        organ = (
            sample_means.groupby(["species", "organ", "model"])[
                "pearson_gene_macro"
            ]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        organ.to_csv(table_dir / "xenium_spatial_by_organ.csv", index=False)

        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        sns.boxplot(
            data=sample_means,
            x="model",
            y="pearson_gene_macro",
            order=["train_mean", "coordinate", "spatial_knn", "image"],
            ax=ax,
        )
        sns.stripplot(
            data=sample_means,
            x="model",
            y="pearson_gene_macro",
            order=["train_mean", "coordinate", "spatial_knn", "image"],
            color="black",
            alpha=0.45,
            size=3,
            ax=ax,
        )
        ax.set(xlabel="", ylabel="Gene-macro Pearson", title="30-sample Xenium breadth")
        ax.tick_params(axis="x", rotation=20)
        save_figure(fig, figure_dir / "xenium_spatial_models")

        marker_primary = spatial[
            (spatial["ratio"] == 1)
            & (spatial["gene_set"] == "marker_hvg_overlap")
        ].copy()
        if not marker_primary.empty:
            marker_by_sample = (
                marker_primary.groupby(
                    ["sample", "species", "organ", "condition", "model"]
                )[
                    [
                        "pearson_gene_macro",
                        "spearman_gene_macro",
                        "f1_gene_macro",
                        "auprc_gene_macro",
                        "n_genes",
                    ]
                ]
                .mean()
                .reset_index()
            )
            marker_by_sample.to_csv(
                table_dir / "xenium_marker_calibration_by_sample.csv",
                index=False,
            )
            paired_tests(
                marker_by_sample,
                ["sample"],
                "pearson_gene_macro",
                reference="image",
            ).to_csv(
                table_dir / "xenium_marker_calibration_paired_tests.csv",
                index=False,
            )

        cell_type = read_many(
            [
                root
                / "results"
                / "xenium_spatial"
                / "conch"
                / sample
                / "cell_type_metrics.csv"
                for sample in samples
            ]
        )
        if not cell_type.empty:
            cell_type = cell_type.merge(
                sheet[["sample", "species", "organ", "condition"]],
                on="sample",
                how="left",
            )
            cell_type.to_csv(
                table_dir / "xenium_cell_type_calibration_all_runs.csv",
                index=False,
            )
            cell_type_by_sample = (
                cell_type.groupby(
                    ["sample", "species", "organ", "condition", "model"]
                )
                .agg(
                    n_cell_types=("cell_type", "nunique"),
                    pseudobulk_pearson=("pseudobulk_pearson", "mean"),
                    pseudobulk_rmse_log1p=("pseudobulk_rmse_log1p", "mean"),
                )
                .reset_index()
            )
            cell_type_by_sample.to_csv(
                table_dir / "xenium_cell_type_calibration_by_sample.csv",
                index=False,
            )
            paired_tests(
                cell_type_by_sample,
                ["sample"],
                "pseudobulk_rmse_log1p",
                reference="image",
            ).to_csv(
                table_dir / "xenium_cell_type_calibration_rmse_tests.csv",
                index=False,
            )
            fig, ax = plt.subplots(figsize=(6.2, 3.8))
            sns.boxplot(
                data=cell_type_by_sample,
                x="model",
                y="pseudobulk_rmse_log1p",
                order=["train_mean", "coordinate", "spatial_knn", "image"],
                ax=ax,
            )
            ax.set(
                xlabel="",
                ylabel="Cell-type pseudobulk RMSE (log1p)",
                title="Biological calibration by predicted cell type",
            )
            ax.tick_params(axis="x", rotation=20)
            save_figure(fig, figure_dir / "xenium_cell_type_calibration")

    bleep = read_many(
        [
            root
            / "results"
            / "xenium_bleep_adapter"
            / sample
            / "summary.csv"
            for sample in samples
        ]
    )
    if not bleep.empty:
        bleep.to_csv(table_dir / "bleep_adapter_all_runs.csv", index=False)
        bleep_primary = bleep[bleep["gene_set"] == "top50_hvg"].copy()
        bleep_sample = (
            bleep_primary.groupby(["sample", "method"])[
                [
                    "pearson_gene_macro",
                    "spearman_gene_macro",
                    "pearson_cell_macro",
                    "f1_gene_macro",
                ]
            ]
            .mean()
            .reset_index()
        )
        bleep_sample.to_csv(table_dir / "bleep_adapter_by_sample.csv", index=False)
        if not spatial_primary.empty:
            ridge = spatial_primary[spatial_primary["model"] == "image"][
                ["sample", "seed", "pearson_gene_macro", "f1_gene_macro"]
            ].rename(
                columns={
                    "pearson_gene_macro": "ridge_pearson_gene_macro",
                    "f1_gene_macro": "ridge_f1_gene_macro",
                }
            )
            comparison = bleep_primary.merge(
                ridge, on=["sample", "seed"], how="inner"
            )
            comparison["pearson_difference_bleep_minus_ridge"] = (
                comparison["pearson_gene_macro"]
                - comparison["ridge_pearson_gene_macro"]
            )
            comparison["f1_difference_bleep_minus_ridge"] = (
                comparison["f1_gene_macro"] - comparison["ridge_f1_gene_macro"]
            )
            comparison.to_csv(
                table_dir / "bleep_adapter_vs_ridge_paired.csv", index=False
            )
            comparison_by_sample = (
                comparison.groupby("sample")[
                    [
                        "pearson_gene_macro",
                        "ridge_pearson_gene_macro",
                        "f1_gene_macro",
                        "ridge_f1_gene_macro",
                    ]
                ]
                .mean()
                .reset_index()
            )
            comparison_by_sample.to_csv(
                table_dir / "bleep_adapter_vs_ridge_by_sample.csv", index=False
            )
            tests = []
            for metric, ridge_metric in (
                ("pearson_gene_macro", "ridge_pearson_gene_macro"),
                ("f1_gene_macro", "ridge_f1_gene_macro"),
            ):
                pair = comparison_by_sample[[metric, ridge_metric]].dropna()
                if pair.empty:
                    continue
                try:
                    statistic, pvalue = wilcoxon(
                        pair[metric] - pair[ridge_metric]
                    )
                except ValueError:
                    statistic, pvalue = float("nan"), 1.0
                tests.append(
                    {
                        "metric": metric,
                        "n_pairs": len(pair),
                        "mean_bleep": float(pair[metric].mean()),
                        "mean_ridge": float(pair[ridge_metric].mean()),
                        "mean_difference_bleep_minus_ridge": float(
                            (pair[metric] - pair[ridge_metric]).mean()
                        ),
                        "wilcoxon_statistic": float(statistic),
                        "wilcoxon_pvalue": float(pvalue),
                    }
                )
            pd.DataFrame(tests).to_csv(
                table_dir / "bleep_adapter_vs_ridge_tests.csv", index=False
            )
            plot_bleep = bleep_primary[
                ["sample", "seed", "pearson_gene_macro"]
            ].copy()
            plot_bleep["method_plot"] = "BLEEP adapter"
            plot_ridge = ridge.rename(
                columns={"ridge_pearson_gene_macro": "pearson_gene_macro"}
            )[["sample", "seed", "pearson_gene_macro"]]
            plot_ridge["method_plot"] = "Ridge on CONCH"
            fig, ax = plt.subplots(figsize=(4.8, 3.8))
            sns.boxplot(
                data=pd.concat([plot_ridge, plot_bleep], ignore_index=True),
                x="method_plot",
                y="pearson_gene_macro",
                ax=ax,
            )
            ax.set(
                xlabel="",
                ylabel="Gene-macro Pearson",
                title="Xenium BLEEP-adapter comparison",
            )
            save_figure(fig, figure_dir / "bleep_adapter_comparison")

    cross = read_many(
        [
            root
            / "results"
            / "xenium_cross_sample"
            / "conch"
            / target
            / "summary.csv"
            for target in CROSS_TARGETS
        ]
    )
    if not cross.empty:
        cross.to_csv(table_dir / "cross_sample_all_runs.csv", index=False)
        cross_primary = cross[
            (cross["ratio"] == 1)
            & (cross["gene_set"] == "top50_hvg")
        ]
        cross_target = (
            cross_primary.groupby(["species", "organ", "target", "model"])[
                [
                    "pearson_gene_macro",
                    "spearman_gene_macro",
                    "pearson_cell_macro",
                    "f1_gene_macro",
                ]
            ]
            .mean()
            .reset_index()
        )
        cross_target.to_csv(table_dir / "cross_sample_by_target.csv", index=False)
        cross_test_frame = (
            cross_primary.groupby(["target", "model"])[
                "pearson_gene_macro"
            ]
            .mean()
            .reset_index()
        )
        paired_tests(
            cross_test_frame,
            ["target"],
            "pearson_gene_macro",
            reference="image",
        ).to_csv(table_dir / "cross_sample_paired_tests.csv", index=False)
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        image_cross = cross_target[cross_target["model"] == "image"]
        sns.boxplot(
            data=image_cross,
            x="organ",
            y="pearson_gene_macro",
            ax=ax,
            color="#70AD47",
        )
        sns.stripplot(
            data=image_cross,
            x="organ",
            y="pearson_gene_macro",
            ax=ax,
            color="black",
            size=4,
        )
        ax.set(xlabel="", ylabel="Gene-macro Pearson", title="Same-organ held-out sample")
        ax.tick_params(axis="x", rotation=30)
        save_figure(fig, figure_dir / "cross_sample_by_organ")

        cross_type = read_many(
            [
                root
                / "results"
                / "xenium_cross_sample"
                / "conch"
                / target
                / "cell_type_metrics.csv"
                for target in CROSS_TARGETS
            ]
        )
        if not cross_type.empty:
            cross_type.to_csv(
                table_dir / "cross_sample_cell_type_all_runs.csv", index=False
            )
            (
                cross_type.groupby(["species", "organ", "target", "model"])
                .agg(
                    n_cell_types=("cell_type", "nunique"),
                    pseudobulk_pearson=("pseudobulk_pearson", "mean"),
                    pseudobulk_rmse_log1p=("pseudobulk_rmse_log1p", "mean"),
                )
                .reset_index()
                .to_csv(
                    table_dir / "cross_sample_cell_type_by_target.csv",
                    index=False,
                )
            )

    pseudo = read_many(
        [
            root
            / "results"
            / "xenium_pseudospot"
            / "conch"
            / sample
            / "summary.csv"
            for sample in PSEUDOSPOT_SAMPLES
        ]
    )
    pseudo_table = pd.DataFrame()
    if not pseudo.empty:
        pseudo.to_csv(table_dir / "pseudospot_all_runs.csv", index=False)
        pseudo_primary = pseudo[pseudo["gene_set"] == "top50_hvg"].copy()
        pseudo_primary["resolution_microns"] = pseudo_primary["model"].map(
            lambda model: (
                0.0
                if model == "cell_image"
                else (
                    float(model.split("_")[-1].replace("um", ""))
                    if model.startswith("pseudo_spot")
                    else np.nan
                )
            )
        )
        pseudo_table = (
            pseudo_primary.groupby(["sample", "model", "resolution_microns"])[
                [
                    "pearson_gene_macro",
                    "spearman_gene_macro",
                    "pearson_cell_macro",
                    "f1_gene_macro",
                ]
            ]
            .mean()
            .reset_index()
        )
        pseudo_table.to_csv(table_dir / "pseudospot_by_sample.csv", index=False)
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        plot_data = pseudo_table[pseudo_table["model"] != "train_mean"]
        sns.lineplot(
            data=plot_data,
            x="resolution_microns",
            y="pearson_gene_macro",
            units="sample",
            estimator=None,
            marker="o",
            alpha=0.7,
            ax=ax,
        )
        ax.set(
            xlabel="Aggregation width (µm; 0 = cell target)",
            ylabel="Gene-macro Pearson",
            title="Cell versus pseudo-spot resolution",
        )
        save_figure(fig, figure_dir / "pseudospot_resolution")
        mixing = read_many(
            [
                root
                / "results"
                / "xenium_pseudospot"
                / "conch"
                / sample
                / "spot_mixing.csv"
                for sample in PSEUDOSPOT_SAMPLES
            ],
            label_from_parent="sample",
        )
        if not mixing.empty:
            mixing.to_csv(table_dir / "pseudospot_mixing_all_runs.csv", index=False)
            mixing_summary = (
                mixing.groupby(
                    ["sample", "density_source", "spot_microns"]
                )[
                    [
                        "median_cells_per_spot",
                        "p95_cells_per_spot",
                        "fraction_spots_mixed_cell_type",
                        "fraction_cells_in_mixed_spots",
                    ]
                ]
                .mean()
                .reset_index()
            )
            mixing_summary.to_csv(
                table_dir / "pseudospot_mixing_by_sample.csv", index=False
            )

    full_pseudo_paths = [
        root
        / "results"
        / "xenium_pseudospot_full_density"
        / "conch"
        / sample
        / "summary.csv"
        for sample in FULL_DENSITY_SAMPLES
    ]
    full_pseudo = read_many(full_pseudo_paths)
    full_pseudo_table = pd.DataFrame()
    if not full_pseudo.empty:
        full_pseudo.to_csv(
            table_dir / "pseudospot_full_density_all_runs.csv", index=False
        )
        full_primary = full_pseudo[
            full_pseudo["gene_set"] == "top50_hvg"
        ].copy()
        full_primary["resolution_microns"] = full_primary["model"].map(
            lambda model: (
                0.0
                if model == "cell_image"
                else (
                    float(model.split("_")[-1].replace("um", ""))
                    if model.startswith("pseudo_spot")
                    else np.nan
                )
            )
        )
        full_pseudo_table = (
            full_primary.groupby(["sample", "model", "resolution_microns"])[
                [
                    "pearson_gene_macro",
                    "spearman_gene_macro",
                    "pearson_cell_macro",
                    "f1_gene_macro",
                    "n_train_cells",
                    "n_test_cells",
                ]
            ]
            .mean()
            .reset_index()
        )
        full_pseudo_table.to_csv(
            table_dir / "pseudospot_full_density_summary.csv", index=False
        )
        full_mixing = read_many(
            [path.parent / "spot_mixing.csv" for path in full_pseudo_paths],
            label_from_parent="sample",
        )
        if not full_mixing.empty:
            full_mixing.to_csv(
                table_dir / "pseudospot_full_density_mixing.csv", index=False
            )
        fig, ax = plt.subplots(figsize=(5.8, 3.8))
        sns.lineplot(
            data=full_pseudo_table[
                full_pseudo_table["model"] != "train_mean"
            ],
            x="resolution_microns",
            y="pearson_gene_macro",
            hue="sample",
            marker="o",
            ax=ax,
        )
        ax.set(
            xlabel="Aggregation width (µm; 0 = cell target)",
            ylabel="Gene-macro Pearson",
            title="Full-density resolution controls",
        )
        save_figure(fig, figure_dir / "pseudospot_full_density")

    skin = read_many(
        [
            root / "results" / "skin_context" / name / "summary.csv"
            for name in SKIN_CONTRASTS
        ],
        label_from_parent="contrast",
    )
    if not skin.empty:
        skin.to_csv(table_dir / "skin_context_all_runs.csv", index=False)
        skin_primary = skin[
            (skin["ratio"] == 1)
            & (skin["gene_set"] == "top50_hvg")
            & (skin["model"] == "image")
        ]
        skin_table = (
            skin_primary.groupby("contrast")[
                [
                    "pearson_gene_macro",
                    "spearman_gene_macro",
                    "pearson_cell_macro",
                    "f1_gene_macro",
                ]
            ]
            .agg(["mean", "std"])
            .reset_index()
        )
        skin_table.columns = [
            "_".join([str(part) for part in column if part])
            if isinstance(column, tuple)
            else column
            for column in skin_table.columns
        ]
        skin_table.to_csv(table_dir / "skin_context_primary.csv", index=False)
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        sns.barplot(
            data=skin_primary,
            x="contrast",
            y="pearson_gene_macro",
            errorbar="sd",
            color="#ED7D31",
            ax=ax,
        )
        ax.set(xlabel="", ylabel="Gene-macro Pearson", title="Skin context transfer")
        ax.tick_params(axis="x", rotation=30)
        save_figure(fig, figure_dir / "skin_context")

    hd_rows = []
    for example in ("d", "e", "g"):
        path = root / "results" / "hd_assignment" / example / "summary.json"
        if path.exists():
            hd_rows.append(json.loads(path.read_text(encoding="utf-8")))
    hd = pd.DataFrame(hd_rows)
    if not hd.empty:
        hd.to_csv(table_dir / "hd_assignment_summary.csv", index=False)
        long_rows = []
        for row in hd_rows:
            for variant in (
                "erode_1um",
                "dilate_1um",
                "shift_x_1um",
                "shift_y_1um",
            ):
                long_rows.append(
                    {
                        "example_id": row["example_id"],
                        "dataset_name": row["dataset_name"],
                        "variant": variant,
                        "median_bin_jaccard": row[
                            f"median_bin_jaccard_{variant}"
                        ],
                        "median_expression_cosine": row[
                            f"median_expression_cosine_{variant}"
                        ],
                    }
                )
        hd_long = pd.DataFrame(long_rows)
        hd_long.to_csv(table_dir / "hd_assignment_variants.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
        sns.barplot(
            data=hd_long,
            x="example_id",
            y="median_bin_jaccard",
            hue="variant",
            ax=axes[0],
        )
        sns.barplot(
            data=hd_long,
            x="example_id",
            y="median_expression_cosine",
            hue="variant",
            ax=axes[1],
        )
        axes[0].set(title="Assigned-bin stability", xlabel="HD example")
        axes[1].set(title="Expression stability", xlabel="HD example")
        axes[0].legend_.remove()
        axes[1].legend(title="Perturbation", fontsize=7)
        save_figure(fig, figure_dir / "hd_assignment_sensitivity")

    caption_path = (
        root / "results" / "caption_audit" / "caption_audit_summary.csv"
    )
    caption = (
        pd.read_csv(caption_path)
        if caption_path.exists() and caption_path.stat().st_size > 0
        else pd.DataFrame()
    )
    if not caption.empty:
        caption.to_csv(table_dir / "caption_audit_summary.csv", index=False)

    # Freeze checksums for the portable analysis scripts. The reviewer package
    # uses ``code`` while a full campaign workspace may use ``scripts``.
    script_dir = root / "scripts"
    if not script_dir.exists():
        script_dir = root / "code"
    checksum_rows = []
    for path in sorted(
        list(script_dir.glob("*.py"))
    ):
        checksum_rows.append(
            {
                "path": str(path.relative_to(root.parent)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    pd.DataFrame(checksum_rows).to_csv(
        table_dir / "campaign_code_checksums.csv", index=False
    )

    complete_total = int(status["complete"].sum())
    expected_total = len(status)
    report = [
        "# Rebuttal experiment campaign summary",
        "",
        f"Completed result units: **{complete_total}/{expected_total}**.",
        "",
        "## Status by family",
        "",
        markdown_table(status_summary),
        "",
        "## Reviewer concern coverage",
        "",
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "Concern": "Visium HD bin-to-cell validity",
                        "Evidence": "3-dataset, 9,000-cell boundary/assignment audit",
                    },
                    {
                        "Concern": "Breadth and heterogeneity",
                        "Evidence": (
                            f"{len(samples)} Xenium samples; "
                            f"{sheet['organ'].nunique()} organ labels; "
                            f"{sheet['condition'].nunique()} conditions; "
                            f"{sheet['species'].nunique()} species"
                        ),
                    },
                    {
                        "Concern": "Local interpolation/autocorrelation",
                        "Evidence": (
                            "4 contiguous edge holdouts, 5% spatial buffer, "
                            "coordinate/KNN/mean controls"
                        ),
                    },
                    {
                        "Concern": "Low-correlation biological meaning",
                        "Evidence": (
                            "marker/HVG metrics and predicted-cell-type "
                            "pseudobulk calibration"
                        ),
                    },
                    {
                        "Concern": "Cell versus spot resolution",
                        "Evidence": (
                            "8/16/55 µm pseudo-spots plus full-density "
                            "HLCX022 control"
                        ),
                    },
                    {
                        "Concern": "Method breadth and BLEEP clarity",
                        "Evidence": (
                            "7 frozen encoders and an explicitly labeled "
                            "official-algorithm BLEEP adapter"
                        ),
                    },
                    {
                        "Concern": "Generated text validity/leakage",
                        "Evidence": "caption label-mention and label-recovery audit",
                    },
                    {
                        "Concern": "Context generalization",
                        "Evidence": (
                            "same-organ leave-one-sample-out and 5 normal/"
                            "melanoma skin transfers"
                        ),
                    },
                ]
            )
        ),
        "",
        (
            "Target provenance is separated explicitly: Xenium targets are "
            "native per-cell transcript counts, whereas Visium HD cell targets "
            "are derived by segmentation/bin assignment and are therefore "
            "reported with the assignment audit."
        ),
        "",
    ]
    if not encoder.empty:
        image = encoder[
            (encoder["ratio"] == 1)
            & (encoder["gene_set"] == "top50_hvg")
            & (encoder["model"] == "image")
        ]
        best = (
            image.groupby("encoder")["pearson_gene_macro"].mean().sort_values(
                ascending=False
            )
        )
        report.extend(
            [
                "## Encoder spatial evaluation",
                "",
                (
                    f"Best mean gene-macro Pearson on HHDX011: "
                    f"**{best.index[0]} = {best.iloc[0]:.3f}** "
                    f"(four contiguous edge holdouts)."
                ),
                "",
            ]
        )
    if not spatial_primary.empty:
        sample_means = (
            spatial_primary.groupby(["sample", "model"])["pearson_gene_macro"]
            .mean()
            .reset_index()
        )
        model_means = (
            sample_means.groupby("model")["pearson_gene_macro"]
            .mean()
            .sort_values(ascending=False)
        )
        report.extend(
            [
                "## Xenium breadth",
                "",
                (
                    "Macro mean gene-Pearson across available samples: "
                    + ", ".join(
                        f"{name}={value:.3f}"
                        for name, value in model_means.items()
                    )
                    + "."
                ),
                "",
            ]
        )
    if not bleep.empty:
        bleep_mean = float(
            bleep[
                (bleep["gene_set"] == "top50_hvg")
                & (bleep["method"] == "bleep_conch_adapter")
            ]["pearson_gene_macro"].mean()
        )
        report.extend(
            [
                "## BLEEP algorithm adapter",
                "",
                (
                    "Mean gene-macro Pearson for the official-loss/top-50 "
                    f"BLEEP adapter across available runs: **{bleep_mean:.3f}**."
                ),
                "",
            ]
        )
    if not marker_by_sample.empty:
        marker_mean = float(
            marker_by_sample[marker_by_sample["model"] == "image"][
                "pearson_gene_macro"
            ].mean()
        )
        report.extend(
            [
                "## Biological calibration",
                "",
                (
                    "Mean image-model Pearson on train-selected marker/HVG "
                    f"overlap across available samples: **{marker_mean:.3f}**."
                ),
                "",
            ]
        )
    if not cell_type_by_sample.empty:
        calibration = (
            cell_type_by_sample.groupby("model")["pseudobulk_rmse_log1p"]
            .mean()
            .sort_values()
        )
        report.extend(
            [
                (
                    "Mean predicted-cell-type pseudobulk RMSE (lower is better): "
                    + ", ".join(
                        f"{model}={value:.3f}"
                        for model, value in calibration.items()
                    )
                    + "."
                ),
                "",
            ]
        )
    if not pseudo_table.empty:
        pseudo_macro = (
            pseudo_table.groupby("model")["pearson_gene_macro"]
            .mean()
            .sort_values(ascending=False)
        )
        report.extend(
            [
                "## Cell versus pseudo-spot resolution",
                "",
                (
                    "Subsample-matched mean gene-Pearson across available "
                    "samples: "
                    + ", ".join(
                        f"{model}={value:.3f}"
                        for model, value in pseudo_macro.items()
                    )
                    + "."
                ),
                "",
            ]
        )
    if not full_pseudo_table.empty:
        full_text = []
        for sample, group in full_pseudo_table.groupby("sample"):
            values = group.set_index("model")["pearson_gene_macro"]
            full_text.append(
                f"{sample}: "
                + ", ".join(
                    f"{model}={value:.3f}"
                    for model, value in values.items()
                )
            )
        report.extend(
            [
                (
                    "Full-density controls: " + "; ".join(full_text)
                    + "."
                ),
                "",
            ]
        )
    if not hd.empty:
        failure = ", ".join(
            f"{row.example_id}={row.fraction_without_canonical_bins:.1%}"
            for row in hd.itertuples()
        )
        report.extend(
            [
                "## Visium HD assignment audit",
                "",
                f"Fraction of sampled CellViT cells receiving no valid 2 µm bin: {failure}.",
                "",
            ]
        )
    if not caption.empty:
        row = caption.iloc[0]
        report.extend(
            [
                "## Generated-caption leakage audit",
                "",
                (
                    f"Audited {int(row['n_captions'])} readable captions; "
                    f"{row['fraction_explicit_cell_type_mention']:.1%} explicitly "
                    "contained the supplied cell-type label. Cell-type recovery "
                    "after removing the literal label remained "
                    f"{row['cell_type_after_explicit_label_removal.accuracy']:.3f} "
                    "accuracy."
                ),
                "",
            ]
        )
    report.extend(
        [
            "## Interpretation constraints",
            "",
            "- All target genes are selected from the training partition only.",
            "- Spatial tests use contiguous edge holdouts plus a 5% coordinate-span buffer.",
            "- External source-data inputs are treated as read-only.",
            "- The BLEEP comparison is explicitly an algorithm adapter using the official projection/loss/retrieval design with frozen CONCH inputs, not a claim that the original spot-level checkpoint is cell-native.",
            "- An ovarian age rerun and exact STBoost rerun remain unavailable unless their inaccessible/missing artifacts are supplied.",
            "",
        ]
    )
    (root / "REBUTTAL_EXPERIMENT_SUMMARY.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(status_summary.to_string(index=False))
    print(f"[done] tables={table_dir} figures={figure_dir}")


if __name__ == "__main__":
    main()
