from __future__ import annotations

import argparse
import csv
import gzip
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    cell_dir,
    ensure_parent,
    existing_sample_dirs,
    manifest_path_for_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cell-level manifest for Xenium scaling-law experiments."
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--out",
        default="PATH/cache/scaling_law/manifest.csv",
    )
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument(
        "--max-cells-per-sample",
        type=int,
        default=None,
        help="For smoke tests, keep only the first N cells from each sample metadata file.",
    )
    parser.add_argument(
        "--validate-files",
        action="store_true",
        help="Check image/expression/metadata files exist. Useful for smoke tests but slow at full scale.",
    )
    return parser.parse_args()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return open(path, newline="")


def iter_sample_rows(path: Path, max_cells: int | None):
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "cell_id" not in reader.fieldnames:
            raise ValueError(f"{path} does not contain a cell_id column")
        keep = [col for col in ["patch_id", "patch_position"] if col in reader.fieldnames]
        for idx, row in enumerate(reader):
            if max_cells is not None and idx >= max_cells:
                break
            yield row["cell_id"], {col: row.get(col, "") for col in keep}


def build_manifest(args: argparse.Namespace) -> int:
    samples = args.samples or existing_sample_dirs(args.data_root)
    out_path = Path(args.out)
    if out_path.suffix == ".parquet":
        raise ValueError("This first version writes CSV manifests; use an .csv output path.")

    ensure_parent(out_path)
    fieldnames = [
        "sample",
        "cell_id",
        "image_path",
        "expr_path",
        "metadata_path",
        "patch_id",
        "patch_position",
    ]

    total_rows = 0
    with open(out_path, "w", newline="") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        writer.writeheader()

        for sample in samples:
            sample_meta = manifest_path_for_sample(args.data_root, sample)
            if not sample_meta.exists():
                print(f"[skip] {sample}: missing {sample_meta}")
                continue

            sample_rows = 0
            for cell_id, extra in iter_sample_rows(sample_meta, args.max_cells_per_sample):
                base = cell_dir(args.data_root, sample, cell_id)
                image_path = base / "cell_image.tif"
                expr_path = base / "cell_gene_expression.h5"
                metadata_path = base / "metadata.csv.gz"

                if args.validate_files and not (
                    image_path.exists() and expr_path.exists() and metadata_path.exists()
                ):
                    continue

                writer.writerow(
                    {
                        "sample": sample,
                        "cell_id": cell_id,
                        "image_path": str(image_path),
                        "expr_path": str(expr_path),
                        "metadata_path": str(metadata_path),
                        "patch_id": extra.get("patch_id", ""),
                        "patch_position": extra.get("patch_position", ""),
                    }
                )
                sample_rows += 1
                total_rows += 1

            print(f"[manifest] {sample}: {sample_rows} cells")

    if total_rows == 0:
        raise RuntimeError("No manifest rows were produced.")
    return total_rows


def main() -> None:
    args = parse_args()
    total_rows = build_manifest(args)
    print(f"[done] wrote {total_rows} rows to {args.out}")


if __name__ == "__main__":
    main()
