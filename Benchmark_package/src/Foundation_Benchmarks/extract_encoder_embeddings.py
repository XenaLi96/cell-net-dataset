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

SELF_BENCHMARKS_DIR = SCRIPT_DIR.parent
OLD_DIR = SELF_BENCHMARKS_DIR / "old"

from common import (  # noqa: E402
    embedding_path,
    ensure_dir,
    h5_string_dtype,
    load_manifest,
    parse_gene_list,
    read_cell_expression,
)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
HOPTIMUS_MEAN = np.array([0.707223, 0.578729, 0.703617], dtype=np.float32)
HOPTIMUS_STD = np.array([0.211883, 0.230117, 0.177517], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract embeddings for histology encoders.")
    parser.add_argument(
        "--encoder",
        required=True,
        choices=["conch", "ctranspath", "hoptimus0", "phikon", "uni", "uni2"],
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Directory/result model name. Defaults to --encoder.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--out-root",
        default="PATH/cache/scaling_law/embeddings",
    )
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument(
        "--gene-list",
        default=None,
        help="Optional JSON or text gene list. If set, only these genes are cached as targets.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="For HuggingFace-backed models, avoid network access and use local cache/checkpoints.",
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


def load_pil_image(path: str) -> Image.Image:
    image = tifffile.imread(path)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if np.issubdtype(image.dtype, np.floating):
        if float(np.nanmax(image)) <= 1.0:
            image = np.clip(image, 0, 1) * 255
        else:
            image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    return Image.fromarray(image).convert("RGB")


def preprocess_pil(
    image: Image.Image,
    image_size: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor:
    image = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - mean) / std
    return torch.from_numpy(array).permute(2, 0, 1)


def flatten_features(features: torch.Tensor) -> torch.Tensor:
    if isinstance(features, (tuple, list)):
        features = features[0]
    if features.ndim > 2:
        features = torch.flatten(features, 1)
    return features


class TensorEncoder:
    def __init__(
        self,
        model: nn.Module,
        device: str,
        image_size: int = 224,
        mean: np.ndarray = IMAGENET_MEAN,
        std: np.ndarray = IMAGENET_STD,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def prepare(self, paths: list[str]):
        images = [
            preprocess_pil(load_pil_image(path), self.image_size, self.mean, self.std)
            for path in paths
        ]
        return torch.stack(images).to(self.device, non_blocking=True)

    def encode(self, batch) -> torch.Tensor:
        with torch.no_grad():
            return flatten_features(self.model(batch))


class ConchEncoder:
    def __init__(self, checkpoint_path: str, device: str) -> None:
        conch_dir = SELF_BENCHMARKS_DIR / "CONCH"
        if str(conch_dir) not in sys.path:
            sys.path.insert(0, str(conch_dir))
        from conch.open_clip_custom import create_model_from_pretrained

        self.model, self.preprocess = create_model_from_pretrained(
            "conch_ViT-B-16",
            checkpoint_path=checkpoint_path,
        )
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device

    def prepare(self, paths: list[str]):
        images = [self.preprocess(load_pil_image(path)) for path in paths]
        return torch.stack(images).to(self.device, non_blocking=True)

    def encode(self, batch) -> torch.Tensor:
        with torch.no_grad():
            return flatten_features(
                self.model.encode_image(batch, proj_contrast=False, normalize=False)
            )


class CTransPathEncoder:
    def __init__(self, checkpoint_path: str, device: str) -> None:
        hest_src = SELF_BENCHMARKS_DIR.parent / "benchmarks" / "HEST" / "src"
        if str(hest_src) not in sys.path:
            sys.path.insert(0, str(hest_src))
        from hest.bench.cpath_model_zoo.ctranspath.ctran import ctranspath

        model = ctranspath(img_size=224)
        model.head = nn.Identity()

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if "attn_mask" not in str(key)
        }

        self.model = model.to(device)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(f"[weights] loaded {checkpoint_path}")
        print(f"[weights] missing={len(missing)} unexpected={len(unexpected)}")
        self.model.eval()
        self.device = device

    def prepare(self, paths: list[str]):
        images = [
            preprocess_pil(load_pil_image(path), 224, IMAGENET_MEAN, IMAGENET_STD)
            for path in paths
        ]
        return torch.stack(images).to(self.device, non_blocking=True)

    def encode(self, batch) -> torch.Tensor:
        with torch.no_grad():
            return flatten_features(self.model(batch))


class PhikonEncoder:
    def __init__(
        self,
        checkpoint_path: str | None,
        device: str,
        local_files_only: bool,
    ) -> None:
        from transformers import AutoImageProcessor, ViTConfig, ViTModel

        self.device = device
        self.processor = None
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                "owkin/phikon",
                local_files_only=local_files_only,
            )
            self.model = ViTModel.from_pretrained(
                "owkin/phikon",
                add_pooling_layer=False,
                local_files_only=local_files_only,
            )
            print("[model] loaded owkin/phikon via transformers")
        except Exception as exc:
            if checkpoint_path is None:
                raise RuntimeError(
                    "Could not load owkin/phikon from cache/HuggingFace and no "
                    "--checkpoint-path was provided."
                ) from exc
            print(f"[warn] transformers from_pretrained failed, using local ViTConfig: {exc}")
            config = ViTConfig(
                image_size=224,
                patch_size=16,
                num_channels=3,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                qkv_bias=True,
            )
            self.model = ViTModel(config, add_pooling_layer=False)
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print(f"[weights] loaded {checkpoint_path}")
            print(f"[weights] missing={len(missing)} unexpected={len(unexpected)}")
        self.model = self.model.to(device)
        self.model.eval()

    def prepare(self, paths: list[str]):
        images = [load_pil_image(path) for path in paths]
        if self.processor is not None:
            return images
        tensors = [
            preprocess_pil(image, 224, IMAGENET_MEAN, IMAGENET_STD)
            for image in images
        ]
        return torch.stack(tensors).to(self.device, non_blocking=True)

    def encode(self, batch) -> torch.Tensor:
        with torch.no_grad():
            if self.processor is not None:
                inputs = self.processor(batch, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
            else:
                outputs = self.model(pixel_values=batch)
            return outputs.last_hidden_state[:, 0, :]


def clean_state_dict(state_dict):
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return {
        str(key).replace("module.", "").replace("encoder.", ""): value
        for key, value in state_dict.items()
    }


def build_hoptimus0(args: argparse.Namespace):
    import timm

    checkpoint_path = args.checkpoint_path or str(OLD_DIR / "hoptimus0.bin")
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0",
        pretrained=False,
        init_values=1e-5,
        dynamic_img_size=False,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(clean_state_dict(state_dict), strict=False)
    print(f"[weights] loaded {checkpoint_path}")
    print(f"[weights] missing={len(missing)} unexpected={len(unexpected)}")
    return TensorEncoder(model, args.device, image_size=224, mean=HOPTIMUS_MEAN, std=HOPTIMUS_STD)


def build_uni(args: argparse.Namespace):
    import timm

    checkpoint_path = args.checkpoint_path or str(OLD_DIR / "uni.bin")
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(clean_state_dict(state_dict), strict=False)
    print(f"[weights] loaded {checkpoint_path}")
    print(f"[weights] missing={len(missing)} unexpected={len(unexpected)}")
    return TensorEncoder(model, args.device)


def build_uni2(args: argparse.Namespace):
    import timm

    checkpoint_path = args.checkpoint_path or str(OLD_DIR / "uni_2.bin")
    timm_kwargs = {
        "model_name": "vit_giant_patch14_224",
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model(pretrained=False, **timm_kwargs)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(clean_state_dict(state_dict), strict=False)
    print(f"[weights] loaded {checkpoint_path}")
    print(f"[weights] missing={len(missing)} unexpected={len(unexpected)}")
    return TensorEncoder(model, args.device)


def build_encoder(args: argparse.Namespace):
    if args.encoder == "conch":
        return ConchEncoder(args.checkpoint_path or str(OLD_DIR / "CONCH.bin"), args.device)
    if args.encoder == "ctranspath":
        return CTransPathEncoder(args.checkpoint_path or str(OLD_DIR / "ctranspath.pth"), args.device)
    if args.encoder == "hoptimus0":
        return build_hoptimus0(args)
    if args.encoder == "phikon":
        return PhikonEncoder(
            args.checkpoint_path or str(OLD_DIR / "Phikon.bin"),
            args.device,
            args.local_files_only,
        )
    if args.encoder == "uni":
        return build_uni(args)
    if args.encoder == "uni2":
        return build_uni2(args)
    raise ValueError(f"Unsupported encoder: {args.encoder}")


def write_sample_h5(
    out_path: Path,
    embeddings: np.ndarray,
    expressions: np.ndarray,
    cell_ids: list[str],
    gene_names: list[str],
    image_paths: list[str],
    model_name: str,
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
        handle.attrs["model"] = model_name
        handle.attrs["encoder"] = model_name


def extract_sample(df, encoder, args: argparse.Namespace, out_path: Path) -> None:
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
        prepared = encoder.prepare([str(path) for path in batch["image_path"]])

        batch_embeddings = encoder.encode(prepared).detach().cpu().numpy().astype(np.float32)
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
        model_name=args.model_name,
        compression=args.compression,
    )
    print(f"[done] wrote {out_path}")


def main() -> None:
    args = parse_args()
    args.model_name = args.model_name or args.encoder
    manifest = load_manifest_subset(args.manifest, args.samples)
    if args.samples:
        manifest = manifest[manifest["sample"].isin(args.samples)].copy()
    samples = sorted(manifest["sample"].unique())
    if not samples:
        raise RuntimeError("No samples selected.")

    print(f"[encoder] {args.encoder}")
    print(f"[model_name] {args.model_name}")
    encoder = build_encoder(args)
    for sample in samples:
        out_path = embedding_path(args.out_root, sample, args.model_name)
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {sample}: {out_path} exists")
            continue
        sample_df = manifest[manifest["sample"] == sample].reset_index(drop=True)
        print(f"[sample] {sample}: {len(sample_df)} cells")
        extract_sample(sample_df, encoder, args, out_path)


if __name__ == "__main__":
    main()
