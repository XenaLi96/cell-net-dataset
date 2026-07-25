#!/usr/bin/env python
"""Build deterministic, spatially stratified Xenium campaign manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# These labels are resolved from the source collection/H&E filenames, not from
# the abbreviated sample IDs.
SAMPLE_META = {
    "FFHO026": ("human", "ovary", "ovarian adenocarcinoma", "fresh_frozen"),
    "FFMB028": ("mouse", "brain", "brain hemisphere", "fresh_frozen"),
    "FFMC016": ("mouse", "colon", "colon", "fresh_frozen"),
    "HBCA041": ("human", "breast", "breast IDC", "ffpe"),
    "HBCD036": ("human", "brain", "glioblastoma", "ffpe"),
    "HBCH024": ("human", "breast", "breast cancer", "ffpe"),
    "HBES040": ("human", "breast", "breast IDC", "ffpe"),
    "HCCD035": ("human", "colorectal", "colorectal cancer", "ffpe"),
    "HCCW027": ("human", "cervix", "cervical cancer", "ffpe"),
    "HHDX011": ("human", "heart", "non-diseased", "ffpe"),
    "HKPD015": ("human", "kidney", "mixed kidney regions", "ffpe"),
    "HLCD033": ("human", "lung", "lung cancer", "ffpe"),
    "HLCX022": ("human", "lung", "lung cancer", "ffpe"),
    "HLDX010": ("human", "liver", "liver cancer", "ffpe"),
    "HOCD032": ("human", "ovary", "ovarian cancer", "ffpe"),
    "HOCW023": ("human", "ovary", "ovarian cancer", "ffpe"),
    "HPDA034": ("human", "pancreas", "ductal adenocarcinoma", "ffpe"),
    "HPXM007": ("human", "pancreas", "pancreas", "ffpe"),
    "HSDX009": ("human", "skin", "non-diseased section 1", "ffpe"),
    "HSDX009_2": ("human", "skin", "non-diseased section 2", "ffpe"),
    "HSPD012": ("human", "skin", "melanoma", "ffpe"),
    "HTDX008": ("human", "tonsil", "reactive follicular hyperplasia", "ffpe"),
    "MBDC037": ("mouse", "bone", "femur", "decalcified"),
    "PCHM006": ("human", "pancreas", "pancreatic cancer", "ffpe"),
    "PDHL031": ("human", "lymph_node", "reactive lymph node", "ffpe"),
    "PDHP030": ("human", "prostate", "prostate", "ffpe"),
    "PDHS029": ("human", "skin", "skin", "ffpe"),
    "WMPP017": ("mouse", "whole_pup", "whole pup", "ffpe"),
    "WMPW025": ("mouse", "whole_pup", "whole pup", "ffpe"),
    "XTNL005": ("human", "lung", "lung cancer", "ffpe"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-cells", type=int, default=12_000)
    parser.add_argument(
        "--max-target-genes",
        type=int,
        default=600,
        help="Per-sample extraction cap; all small-panel genes are retained.",
    )
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="Root of the processed Xenium data tree.",
    )
    parser.add_argument(
        "--raw-root",
        required=True,
        type=Path,
        help="Root of the source Xenium data tree.",
    )
    parser.add_argument(
        "--marker-file",
        type=Path,
        default=Path("cell_type_predict/Human_markers.txt"),
    )
    return parser.parse_args()


def decode(values) -> list[str]:
    return [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in values
    ]


def read_genes(expr_path: Path) -> list[str]:
    with h5py.File(expr_path, "r") as handle:
        return decode(handle["gene_names"][:])


def gene_hash(genes: list[str]) -> str:
    return hashlib.sha256("\n".join(genes).encode("utf-8")).hexdigest()


def spatial_sample(
    frame: pd.DataFrame, n: int, grid_size: int, seed: int
) -> pd.DataFrame:
    """Round-robin through occupied spatial grid cells before filling."""
    frame = frame.dropna(subset=["x_centroid", "y_centroid"]).copy()
    if len(frame) <= n:
        return frame.sort_values("cell_id").reset_index(drop=True)

    x = frame["x_centroid"].to_numpy(dtype=float)
    y = frame["y_centroid"].to_numpy(dtype=float)
    x_span = max(float(np.nanmax(x) - np.nanmin(x)), 1.0)
    y_span = max(float(np.nanmax(y) - np.nanmin(y)), 1.0)
    gx = np.minimum(
        ((x - np.nanmin(x)) / x_span * grid_size).astype(int), grid_size - 1
    )
    gy = np.minimum(
        ((y - np.nanmin(y)) / y_span * grid_size).astype(int), grid_size - 1
    )

    rng = np.random.default_rng(seed)
    frame["_grid"] = gx + grid_size * gy
    frame["_random"] = rng.random(len(frame))
    frame = frame.sort_values(["_grid", "_random", "cell_id"])
    frame["_round"] = frame.groupby("_grid", sort=False).cumcount()
    # Sorting by within-grid rank implements balanced round-robin selection.
    selected = (
        frame.sort_values(["_round", "_grid", "_random"])
        .head(n)
        .drop(columns=["_grid", "_random", "_round"])
    )
    return selected.sort_values("cell_id").reset_index(drop=True)


def parse_marker_genes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    table = pd.read_csv(path, sep="\t", dtype=str)
    genes: set[str] = set()
    for value in table.get("geneSymbol", pd.Series(dtype=str)).dropna():
        # Marker entries include comma-separated symbols and bracketed groups.
        genes.update(re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value))
    genes.discard("NA")
    return genes


def first_existing_expression(
    rows: pd.DataFrame, data_root: Path, sample: str
) -> Path:
    for cell_id in rows["cell_id"].astype(str).head(100):
        path = (
            data_root
            / "cell_patches_150"
            / sample
            / cell_id
            / "cell_gene_expression.h5"
        )
        if path.exists():
            return path
    raise FileNotFoundError(f"No readable expression file found for {sample}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    manifest_dir = out_dir / "manifests" / "by_sample"
    gene_dir = out_dir / "genes"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    gene_dir.mkdir(parents=True, exist_ok=True)

    combined_rows: list[pd.DataFrame] = []
    sample_records: list[dict] = []
    sample_genes: dict[str, list[str]] = {}
    marker_universe = parse_marker_genes(args.marker_file)

    for sample_index, (sample, labels) in enumerate(SAMPLE_META.items()):
        species, organ, condition, preservation = labels
        aligned_path = (
            args.data_root / "cell_data_aligned_with_patchid" / f"{sample}.csv.gz"
        )
        if not aligned_path.exists():
            raise FileNotFoundError(aligned_path)

        usecols = [
            "cell_id",
            "x_centroid",
            "y_centroid",
            "patch_id",
            "patch_position",
        ]
        aligned = pd.read_csv(
            aligned_path,
            usecols=lambda column: column in usecols,
            dtype={"cell_id": str},
        )
        missing = {"cell_id", "x_centroid", "y_centroid"} - set(aligned.columns)
        if missing:
            raise ValueError(f"{sample} aligned CSV misses {sorted(missing)}")

        selected = spatial_sample(
            aligned,
            n=args.max_cells,
            grid_size=args.grid_size,
            seed=args.seed + sample_index,
        )
        expr_example = first_existing_expression(selected, args.data_root, sample)
        genes = read_genes(expr_example)
        sample_genes[sample] = genes

        base = args.data_root / "cell_patches_150" / sample
        selected.insert(0, "sample", sample)
        selected["species"] = species
        selected["organ"] = organ
        selected["condition"] = condition
        selected["coord_x"] = selected["x_centroid"].astype(float)
        selected["coord_y"] = selected["y_centroid"].astype(float)
        selected["image_path"] = selected["cell_id"].map(
            lambda cell_id: str(base / str(cell_id) / "cell_image.tif")
        )
        selected["expr_path"] = selected["cell_id"].map(
            lambda cell_id: str(base / str(cell_id) / "cell_gene_expression.h5")
        )
        selected["metadata_path"] = selected["cell_id"].map(
            lambda cell_id: str(base / str(cell_id) / "metadata.csv.gz")
        )

        type_path = args.data_root / "cell_type_prediction" / f"{sample}.csv"
        metrics_path = args.raw_root / sample / "metrics_summary.csv"
        source_name = ""
        metric_cells = np.nan
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path, nrows=1)
            source_name = str(metrics.iloc[0].get("run_name", ""))
            metric_cells = metrics.iloc[0].get("num_cells_detected", np.nan)

        sample_record = {
            "sample": sample,
            "species": species,
            "organ": organ,
            "condition": condition,
            "preservation": preservation,
            "source_name": source_name,
            "n_cells_metrics": metric_cells,
            "n_cells_aligned": len(aligned),
            "n_cells_selected": len(selected),
            "n_genes_panel": len(genes),
            "gene_sha256": gene_hash(genes),
            "cell_type_file": str(type_path) if type_path.exists() else "",
            "aligned_file": str(aligned_path),
            "example_expression": str(expr_example),
        }
        sample_records.append(sample_record)

        selected.to_csv(manifest_dir / f"{sample}.csv", index=False)
        combined_rows.append(selected)
        print(
            f"[sample] {sample}: aligned={len(aligned):,} "
            f"selected={len(selected):,} genes={len(genes):,}"
        )

    sample_sheet = pd.DataFrame(sample_records)
    sample_sheet.to_csv(out_dir / "sample_sheet.csv", index=False)
    combined = pd.concat(combined_rows, ignore_index=True)
    combined.to_csv(out_dir / "manifests" / "xenium_core.csv", index=False)

    marker_upper = {gene.upper() for gene in marker_universe}
    organ_common: dict[tuple[str, str], list[str]] = {}
    organ_dir = gene_dir / "by_organ"
    sample_target_dir = gene_dir / "by_sample"
    organ_dir.mkdir(parents=True, exist_ok=True)
    sample_target_dir.mkdir(parents=True, exist_ok=True)
    target_inventory: dict[str, dict] = {}

    for (species, organ), group in sample_sheet.groupby(["species", "organ"]):
        organ_samples = group["sample"].tolist()
        shared = set(sample_genes[organ_samples[0]])
        for sample in organ_samples[1:]:
            shared.intersection_update(sample_genes[sample])
        ordered_shared = [
            gene for gene in sample_genes[organ_samples[0]] if gene in shared
        ]
        organ_common[(species, organ)] = ordered_shared
        organ_path = organ_dir / f"{species}_{organ}.txt"
        organ_path.write_text(
            "\n".join(ordered_shared) + "\n", encoding="utf-8"
        )
        print(
            f"[organ genes] {species}/{organ}: samples={len(organ_samples)} "
            f"common={len(ordered_shared):,}"
        )

    for sample, labels in SAMPLE_META.items():
        species, organ = labels[:2]
        panel = sample_genes[sample]
        if len(panel) <= args.max_target_genes:
            targets = list(panel)
        else:
            required = set(organ_common[(species, organ)])
            targets = [gene for gene in panel if gene in required]
            seen = set(targets)
            for gene in panel:
                if gene.upper() in marker_upper and gene not in seen:
                    targets.append(gene)
                    seen.add(gene)
            for gene in panel:
                if len(targets) >= args.max_target_genes:
                    break
                if gene not in seen:
                    targets.append(gene)
                    seen.add(gene)
            targets = targets[: args.max_target_genes]
        target_path = sample_target_dir / f"{sample}.txt"
        target_path.write_text("\n".join(targets) + "\n", encoding="utf-8")
        target_inventory[sample] = {
            "n_panel": len(panel),
            "n_targets": len(targets),
            "n_organ_common_retained": sum(
                gene in set(organ_common[(species, organ)]) for gene in targets
            ),
            "n_markers": sum(gene.upper() in marker_upper for gene in targets),
            "sha256": gene_hash(targets),
        }

    gene_payload: dict[str, dict] = {}
    for species in sorted(sample_sheet["species"].unique()):
        species_samples = sample_sheet.loc[
            sample_sheet["species"] == species, "sample"
        ].tolist()
        common = set(sample_genes[species_samples[0]])
        for sample in species_samples[1:]:
            common.intersection_update(sample_genes[sample])
        ordered = [gene for gene in sample_genes[species_samples[0]] if gene in common]
        marker_common = [
            gene for gene in ordered if gene.upper() in {g.upper() for g in marker_universe}
        ]
        (gene_dir / f"{species}_common.txt").write_text(
            "\n".join(ordered) + "\n", encoding="utf-8"
        )
        (gene_dir / f"{species}_marker_common.txt").write_text(
            "\n".join(marker_common) + ("\n" if marker_common else ""),
            encoding="utf-8",
        )
        gene_payload[species] = {
            "samples": species_samples,
            "n_common": len(ordered),
            "n_marker_common": len(marker_common),
            "sha256": gene_hash(ordered),
        }
        print(
            f"[genes] {species}: common={len(ordered):,} "
            f"marker_common={len(marker_common):,}"
        )

    with (gene_dir / "gene_inventory.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "species_common": gene_payload,
                "sample_targets": target_inventory,
                "max_target_genes": args.max_target_genes,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    provenance = {
        "seed": args.seed,
        "max_cells_per_sample": args.max_cells,
        "max_target_genes": args.max_target_genes,
        "grid_size": args.grid_size,
        # Directory labels are retained for provenance without recording
        # machine-specific absolute paths in shareable configuration files.
        "data_root_label": args.data_root.name,
        "raw_root_label": args.raw_root.name,
        "n_samples": len(sample_sheet),
        "n_selected_total": int(sample_sheet["n_cells_selected"].sum()),
        "manifest_sha256": hashlib.sha256(
            (out_dir / "manifests" / "xenium_core.csv").read_bytes()
        ).hexdigest(),
    }
    with (out_dir / "preparation.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
    print(f"[done] wrote campaign inputs to {out_dir}")


if __name__ == "__main__":
    main()
