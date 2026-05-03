from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tifffile
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    MODEL_NAME,
    embedding_path,
    ensure_dir,
    h5_string_dtype,
    load_manifest,
    parse_gene_list,
    read_cell_expression,
)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class TorchOnlyResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * Bottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ResNet-50 embeddings from Xenium cells.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--out-root",
        default="PATH/cache/scaling_law/embeddings",
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--weights-path", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--gene-list",
        default=None,
        help="Optional JSON or text gene list. If set, only these genes are cached as targets.",
    )
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Only for pipeline smoke tests when pretrained ResNet dependencies/weights are unavailable.",
    )
    parser.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip")
    return parser.parse_args()


def load_manifest_subset(path: str | Path, samples: list[str] | None):
    if not samples:
        return load_manifest(path)

    path = str(path)
    sample_set = set(samples)
    if not path.endswith(".csv"):
        manifest = load_manifest(path)
        return manifest[manifest["sample"].isin(sample_set)].copy()

    chunks = []
    for chunk in pd.read_csv(path, chunksize=100_000):
        filtered = chunk[chunk["sample"].isin(sample_set)]
        if not filtered.empty:
            chunks.append(filtered.copy())
    if not chunks:
        raise RuntimeError(f"No manifest rows found for samples: {sorted(sample_set)}")
    return pd.concat(chunks, ignore_index=True)


def load_image_tensor(path: str) -> torch.Tensor:
    image = tifffile.imread(path)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0, 1) * 255
    image = image.astype(np.uint8)
    pil_image = Image.fromarray(image).convert("RGB").resize((224, 224), Image.BILINEAR)
    array = np.asarray(pil_image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(array).permute(2, 0, 1)


def try_torchvision_resnet50(pretrained: bool):
    try:
        import torchvision.models as models
    except Exception as exc:
        raise RuntimeError(f"torchvision unavailable: {exc}") from exc

    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)
    model.fc = nn.Identity()
    return model


def try_timm_resnet50(pretrained: bool):
    try:
        import timm
    except Exception as exc:
        raise RuntimeError(f"timm unavailable: {exc}") from exc
    return timm.create_model("resnet50", pretrained=pretrained, num_classes=0, global_pool="avg")


def build_model(args: argparse.Namespace) -> nn.Module:
    pretrained = not args.no_pretrained

    if args.weights_path:
        model = TorchOnlyResNet50()
        state_dict = torch.load(args.weights_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        cleaned = {
            key.replace("module.", "").replace("encoder.", ""): value
            for key, value in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"[weights] loaded {args.weights_path}")
        print(f"[weights] missing={len(missing)} unexpected={len(unexpected)}")
        return model

    errors = []
    for builder in (try_torchvision_resnet50, try_timm_resnet50):
        try:
            model = builder(pretrained)
            print(f"[model] using {builder.__name__}, pretrained={pretrained}")
            return model
        except Exception as exc:
            errors.append(str(exc))

    if args.allow_random_init:
        print("[warn] using randomly initialized torch-only ResNet-50; do not use these metrics as results")
        return TorchOnlyResNet50()

    joined = "\n  - ".join(errors)
    raise RuntimeError(
        "Could not build a pretrained ResNet-50. Install torchvision/timm, provide "
        "--weights-path, or use --allow-random-init for smoke tests only.\n"
        f"Attempts:\n  - {joined}"
    )


def write_sample_h5(
    out_path: Path,
    embeddings: np.ndarray,
    expressions: np.ndarray,
    cell_ids: list[str],
    gene_names: list[str],
    image_paths: list[str],
    compression: str,
) -> None:
    ensure_dir(out_path.parent)
    h5_compression = None if compression == "none" else compression
    with h5py.File(out_path, "w") as handle:
        handle.create_dataset("embeddings", data=embeddings, compression=h5_compression)
        handle.create_dataset("expressions", data=expressions, compression=h5_compression)
        handle.create_dataset("cell_ids", data=np.asarray(cell_ids, dtype=h5_string_dtype()))
        handle.create_dataset("gene_names", data=np.asarray(gene_names, dtype=h5_string_dtype()))
        handle.create_dataset("image_paths", data=np.asarray(image_paths, dtype=h5_string_dtype()))
        handle.attrs["model"] = MODEL_NAME


def extract_sample(df, model: nn.Module, args: argparse.Namespace, out_path: Path) -> None:
    model.eval()
    embeddings = []
    expressions = []
    cell_ids = []
    image_paths = []
    requested_genes = parse_gene_list(args.gene_list)
    requested_gene_set = set(requested_genes) if requested_genes else None
    gene_names = requested_genes
    gene_indices = None

    for start in range(0, len(df), args.batch_size):
        batch = df.iloc[start : start + args.batch_size]
        images = torch.stack([load_image_tensor(path) for path in batch["image_path"]])
        images = images.to(args.device, non_blocking=True)

        with torch.no_grad():
            batch_embeddings = model(images).detach().cpu().numpy().astype(np.float32)
        embeddings.append(batch_embeddings)

        for row in batch.itertuples(index=False):
            expression, source_genes = read_cell_expression(row.expr_path)
            if requested_gene_set is None:
                if gene_names is None:
                    gene_names = source_genes
                elif source_genes != gene_names:
                    raise ValueError(
                        f"Gene names changed within sample {row.sample}, cell {row.cell_id}."
                    )
                expressions.append(expression)
            else:
                if gene_indices is None:
                    source_gene_to_idx = {gene: idx for idx, gene in enumerate(source_genes)}
                    missing = sorted(requested_gene_set - set(source_gene_to_idx))
                    if missing:
                        raise ValueError(
                            f"{row.sample} is missing {len(missing)} requested genes. "
                            f"First missing genes: {missing[:10]}"
                        )
                    gene_indices = [source_gene_to_idx[gene] for gene in requested_genes]
                expressions.append(expression[gene_indices])

            if requested_gene_set is None and source_genes != gene_names:
                raise ValueError(
                    f"Gene names changed within sample {row.sample}, cell {row.cell_id}."
                )
            cell_ids.append(str(row.cell_id))
            image_paths.append(str(row.image_path))

        print(f"[extract] {batch.iloc[0]['sample']}: {min(start + len(batch), len(df))}/{len(df)}")

    if gene_names is None:
        raise RuntimeError("No rows to extract.")

    write_sample_h5(
        out_path=out_path,
        embeddings=np.vstack(embeddings),
        expressions=np.vstack(expressions).astype(np.float32),
        cell_ids=cell_ids,
        gene_names=gene_names,
        image_paths=image_paths,
        compression=args.compression,
    )
    print(f"[done] wrote {out_path}")


def main() -> None:
    args = parse_args()
    manifest = load_manifest_subset(args.manifest, args.samples)
    if args.samples:
        manifest = manifest[manifest["sample"].isin(args.samples)].copy()
    samples = sorted(manifest["sample"].unique())
    if not samples:
        raise RuntimeError("No samples selected.")

    model = build_model(args).to(args.device)
    for sample in samples:
        out_path = embedding_path(args.out_root, sample, args.model_name)
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {sample}: {out_path} exists")
            continue
        sample_df = manifest[manifest["sample"] == sample].reset_index(drop=True)
        print(f"[sample] {sample}: {len(sample_df)} cells")
        extract_sample(sample_df, model, args, out_path)


if __name__ == "__main__":
    main()
