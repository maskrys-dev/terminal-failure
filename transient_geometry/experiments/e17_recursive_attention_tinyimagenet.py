"""
E17: weight-tied Universal-Transformer-style recursive attention on frozen
ConvNeXt-Tiny spatial features for Tiny ImageNet-200.

The immutable protocol/configs are under:
  results/modern_iterative_panel/protocol/

Typical execution:
  .venv/Scripts/python.exe -u transient_geometry/experiments/e17_recursive_attention_tinyimagenet.py features
  .venv/Scripts/python.exe -u transient_geometry/experiments/e17_recursive_attention_tinyimagenet.py smoke
  .venv/Scripts/python.exe -u transient_geometry/experiments/e17_recursive_attention_tinyimagenet.py train --seed 101
  .venv/Scripts/python.exe -u transient_geometry/experiments/e17_recursive_attention_tinyimagenet.py evaluate --seed 101
  .venv/Scripts/python.exe -u transient_geometry/experiments/e17_recursive_attention_tinyimagenet.py probe --seed 101
  .venv/Scripts/python.exe -u transient_geometry/experiments/e17_recursive_attention_tinyimagenet.py aggregate

The triggered matched-horizon control uses --variant t32 and is rejected
unless the frozen trigger file says to continue.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_MPL_CONFIG_DIR = ROOT / "tmp" / "matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import models


PANEL_DIR = ROOT / "results" / "modern_iterative_panel"
PROTOCOL_DIR = PANEL_DIR / "protocol"
FROZEN_HASH_PATH = PROTOCOL_DIR / "frozen_hashes.json"
CONFIG_PATHS = {
    "t8": PROTOCOL_DIR / "attention_t8_config.json",
    "t32": PROTOCOL_DIR / "attention_t32_control_config.json",
}
OUTPUT_DIRS = {
    "t8": PANEL_DIR / "attention_t8",
    "t32": PANEL_DIR / "attention_t32_control",
}
DATA_ROOT = ROOT / "data" / "tiny-imagenet-200"
SOURCE_PATH = Path(__file__).resolve()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
torch.set_float32_matmul_precision("high")


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing raw artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def write_text_new(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing raw artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_bytes_new(path: Path, content: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing raw artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def amp_dtype() -> torch.dtype | None:
    if DEVICE.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_context():
    dtype = amp_dtype()
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def precision_name() -> str:
    dtype = amp_dtype()
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    return "float32"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def frozen_file_hashes() -> dict[str, str]:
    return {
        "PANEL_PROTOCOL.md": sha256_file(PROTOCOL_DIR / "PANEL_PROTOCOL.md"),
        "panel_config.json": sha256_file(PROTOCOL_DIR / "panel_config.json"),
        "seeds.json": sha256_file(PROTOCOL_DIR / "seeds.json"),
        "attention_t8_config.json": sha256_file(CONFIG_PATHS["t8"]),
        "attention_t32_control_config.json": sha256_file(CONFIG_PATHS["t32"]),
    }


def verify_frozen_protocol() -> None:
    expected = load_json(FROZEN_HASH_PATH)["files"]
    actual = frozen_file_hashes()
    if actual != expected:
        raise RuntimeError(
            "Frozen E17 protocol/config hash mismatch. Refusing to continue.\n"
            f"Expected: {expected}\nActual: {actual}"
        )


def verify_locked_e16() -> None:
    audit = load_json(PROTOCOL_DIR / "locked_artifact_audit.json")["e16"]
    paths_and_hashes = {
        ROOT
        / "results"
        / "rebuttal_recursive_tinyimagenet_paired_horizon"
        / "config_frozen.json": audit["panel_config_sha256"],
        ROOT
        / "results"
        / "rebuttal_recursive_tinyimagenet_paired_horizon"
        / "aggregate_summary.json": audit["aggregate_summary_sha256"],
        ROOT
        / "results"
        / "rebuttal_recursive_tinyimagenet_paired_horizon"
        / "RUN_REPORT.md": audit["report_sha256"],
        ROOT
        / "results"
        / "rebuttal_recursive_tinyimagenet_paired_horizon"
        / "paired_horizon_result.png": audit["figure_sha256"],
        ROOT
        / "results"
        / "rebuttal_recursive_tinyimagenet_e16_ttrain8"
        / "config_frozen.json": audit["ttrain8_config_sha256"],
        ROOT
        / "results"
        / "rebuttal_recursive_tinyimagenet_e16_ttrain32"
        / "config_frozen.json": audit["ttrain32_config_sha256"],
        ROOT
        / "Rebuttal"
        / "Terminal_Failure_E16_Paired_Horizon_Protocol.md": audit[
            "protocol_sha256"
        ],
    }
    for path, expected in paths_and_hashes.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Locked E16 artifact changed: {path}: {actual} != {expected}")


def load_config(variant: str) -> tuple[dict[str, Any], str]:
    verify_frozen_protocol()
    verify_locked_e16()
    path = CONFIG_PATHS[variant]
    return load_json(path), sha256_file(path)


class TinyImageNetSplit(Dataset):
    def __init__(self, root: Path, split: str, transform=None):
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")
        self.root = root
        self.split = split
        self.transform = transform
        self.wnids = [
            line.strip()
            for line in (root / "wnids.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.class_to_idx = {wnid: index for index, wnid in enumerate(self.wnids)}
        self.samples: list[tuple[Path, int]] = []
        if split == "train":
            for wnid in self.wnids:
                for path in sorted((root / "train" / wnid / "images").glob("*.JPEG")):
                    self.samples.append((path, self.class_to_idx[wnid]))
        else:
            image_dir = root / "val" / "images"
            annotations = root / "val" / "val_annotations.txt"
            for line in annotations.read_text(encoding="utf-8").splitlines():
                fields = line.split("\t")
                if len(fields) >= 2:
                    self.samples.append(
                        (image_dir / fields[0], self.class_to_idx[fields[1]])
                    )
        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label


def validate_dataset(config: dict[str, Any]) -> dict[str, Any]:
    train = TinyImageNetSplit(DATA_ROOT, "train")
    validation = TinyImageNetSplit(DATA_ROOT, "val")
    train_counts = Counter(train.targets)
    validation_counts = Counter(validation.targets)
    missing = sum(
        not path.exists() for path, _ in (train.samples + validation.samples)
    )
    result = {
        "classes": len(train.wnids),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_class_counts": sorted(train_counts.values()),
        "validation_class_counts": sorted(validation_counts.values()),
        "missing_images": int(missing),
    }
    result["pass"] = bool(
        result["classes"] == int(config["dataset"]["classes"])
        and result["train_examples"] == int(config["dataset"]["train_examples"])
        and result["validation_examples"]
        == int(config["dataset"]["validation_examples"])
        and set(train_counts.values()) == {int(config["dataset"]["train_per_class"])}
        and set(validation_counts.values())
        == {int(config["dataset"]["validation_per_class"])}
        and not missing
    )
    if not result["pass"]:
        raise RuntimeError(f"Tiny ImageNet validation failed: {result}")
    return result


def cache_paths(config: dict[str, Any]) -> dict[str, Path]:
    cache_dir = ROOT / config["backbone"]["cache_directory"]
    return {
        "directory": cache_dir,
        "train_features": cache_dir / "train_spatial_tokens_fp16.npy",
        "train_labels": cache_dir / "train_labels_int64.npy",
        "validation_features": cache_dir / "validation_spatial_tokens_fp16.npy",
        "validation_labels": cache_dir / "validation_labels_int64.npy",
        "metadata": cache_dir / "cache_metadata.json",
    }


def build_spatial_backbone():
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    model = models.convnext_tiny(weights=weights)
    model.requires_grad_(False)
    model.eval().to(DEVICE)
    return model, weights.transforms(), weights


def extract_split(
    dataset: Dataset,
    split: str,
    model: nn.Module,
    config: dict[str, Any],
    feature_path: Path,
    label_path: Path,
) -> dict[str, Any]:
    if feature_path.exists() or label_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing cache for {split}")
    feature_partial = feature_path.with_suffix(feature_path.suffix + ".partial")
    label_partial = label_path.with_suffix(label_path.suffix + ".partial")
    if feature_partial.exists() or label_partial.exists():
        raise RuntimeError(
            f"Partial cache exists for {split}; stop and inspect rather than overwrite."
        )
    expected_shape = (
        len(dataset),
        int(config["backbone"]["tokens"]),
        int(config["backbone"]["expected_channels"]),
    )
    feature_map = np.lib.format.open_memmap(
        feature_partial, mode="w+", dtype=np.float16, shape=expected_shape
    )
    label_map = np.lib.format.open_memmap(
        label_partial, mode="w+", dtype=np.int64, shape=(len(dataset),)
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["backbone"]["extraction_batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )
    offset = 0
    nonfinite = 0
    started = time.time()
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(DEVICE, non_blocking=True)
        with torch.inference_mode(), autocast_context():
            spatial = model.features(images)
            spatial = model.classifier[0](spatial)
        expected_map_shape = (
            len(images),
            int(config["backbone"]["expected_channels"]),
            int(config["backbone"]["expected_height"]),
            int(config["backbone"]["expected_width"]),
        )
        if tuple(spatial.shape) != expected_map_shape:
            raise RuntimeError(
                f"Incompatible ConvNeXt spatial output: {tuple(spatial.shape)} "
                f"!= {expected_map_shape}"
            )
        tokens = spatial.permute(0, 2, 3, 1).reshape(len(images), 49, 768).float()
        nonfinite += int((~torch.isfinite(tokens)).sum().item())
        end = offset + len(images)
        feature_map[offset:end] = tokens.cpu().numpy().astype(np.float16)
        label_map[offset:end] = labels.numpy().astype(np.int64)
        offset = end
        if batch_index % 50 == 0 or batch_index == len(loader):
            rate = offset / max(time.time() - started, 1e-6)
            print(f"  {split}: {offset}/{len(dataset)} ({rate:.1f} images/s)")
    feature_map.flush()
    label_map.flush()
    del feature_map, label_map
    if offset != len(dataset) or nonfinite:
        raise RuntimeError(
            f"Invalid extracted {split} cache: offset={offset}, nonfinite={nonfinite}"
        )
    feature_partial.replace(feature_path)
    label_partial.replace(label_path)
    return {
        "examples": offset,
        "feature_shape": list(expected_shape),
        "feature_dtype": "float16",
        "label_shape": [offset],
        "label_dtype": "int64",
        "nonfinite_values": nonfinite,
        "elapsed_seconds": time.time() - started,
    }


def scan_cache_array(array: np.ndarray, chunk_size: int = 512) -> int:
    nonfinite = 0
    for start in range(0, len(array), chunk_size):
        nonfinite += int((~np.isfinite(array[start : start + chunk_size])).sum())
    return nonfinite


def validate_cache(config: dict[str, Any], verify_hashes: bool = False):
    paths = cache_paths(config)
    metadata_path = paths["metadata"]
    if not metadata_path.exists():
        raise FileNotFoundError("Spatial cache metadata is missing.")
    metadata = load_json(metadata_path)
    arrays = {
        "train_features": np.load(paths["train_features"], mmap_mode="r"),
        "train_labels": np.load(paths["train_labels"], mmap_mode="r"),
        "validation_features": np.load(
            paths["validation_features"], mmap_mode="r"
        ),
        "validation_labels": np.load(paths["validation_labels"], mmap_mode="r"),
    }
    expected = {
        "train_features": (100_000, 49, 768),
        "train_labels": (100_000,),
        "validation_features": (10_000, 49, 768),
        "validation_labels": (10_000,),
    }
    for name, shape in expected.items():
        if tuple(arrays[name].shape) != shape:
            raise RuntimeError(f"Cache shape mismatch for {name}: {arrays[name].shape}")
    if arrays["train_features"].dtype != np.float16:
        raise RuntimeError("Train spatial cache is not FP16.")
    if arrays["validation_features"].dtype != np.float16:
        raise RuntimeError("Validation spatial cache is not FP16.")
    for label_name in ("train_labels", "validation_labels"):
        values = arrays[label_name]
        if values.dtype != np.int64 or values.min() != 0 or values.max() != 199:
            raise RuntimeError(f"Invalid cached labels: {label_name}")
    if not metadata.get("all_values_finite"):
        raise RuntimeError("Cache metadata does not certify finite values.")
    if verify_hashes:
        for key in (
            "train_features",
            "train_labels",
            "validation_features",
            "validation_labels",
        ):
            actual = sha256_file(paths[key])
            expected_hash = metadata["files"][key]["sha256"]
            if actual != expected_hash:
                raise RuntimeError(f"Spatial cache hash mismatch: {key}")
    return arrays, metadata


def create_spatial_cache(config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    paths = cache_paths(config)
    if paths["metadata"].exists():
        _, metadata = validate_cache(config, verify_hashes=True)
        print("Using validated spatial feature cache already on disk.")
        return metadata
    for key, path in paths.items():
        if key != "directory" and path.exists():
            raise RuntimeError(
                f"Cache component exists without complete metadata: {path}"
            )
    free_gib = shutil.disk_usage(ROOT).free / (1024**3)
    if free_gib < float(config["backbone"]["minimum_free_disk_gib"]):
        raise RuntimeError(f"Insufficient disk space: {free_gib:.2f} GiB free")
    paths["directory"].mkdir(parents=True, exist_ok=True)
    dataset_audit = validate_dataset(config)
    model, transform, weights = build_spatial_backbone()
    train_dataset = TinyImageNetSplit(DATA_ROOT, "train", transform=transform)
    validation_dataset = TinyImageNetSplit(DATA_ROOT, "val", transform=transform)
    started = time.time()
    train_result = extract_split(
        train_dataset,
        "train",
        model,
        config,
        paths["train_features"],
        paths["train_labels"],
    )
    validation_result = extract_split(
        validation_dataset,
        "validation",
        model,
        config,
        paths["validation_features"],
        paths["validation_labels"],
    )
    arrays = {
        "train_features": np.load(paths["train_features"], mmap_mode="r"),
        "train_labels": np.load(paths["train_labels"], mmap_mode="r"),
        "validation_features": np.load(
            paths["validation_features"], mmap_mode="r"
        ),
        "validation_labels": np.load(paths["validation_labels"], mmap_mode="r"),
    }
    finite_counts = {
        "train_features": scan_cache_array(arrays["train_features"]),
        "validation_features": scan_cache_array(arrays["validation_features"]),
    }
    if any(finite_counts.values()):
        raise RuntimeError(f"Non-finite values in completed cache: {finite_counts}")
    files = {}
    for key in (
        "train_features",
        "train_labels",
        "validation_features",
        "validation_labels",
    ):
        path = paths[key]
        print(f"Hashing {path.name}...")
        files[key] = {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    metadata = {
        "experiment_config_sha256": config_sha256,
        "backbone": "torchvision.models.convnext_tiny",
        "weights": str(weights),
        "transform": repr(transform),
        "spatial_source": config["backbone"]["spatial_source"],
        "extraction_precision": precision_name(),
        "cache_dtype": "float16",
        "dataset_standardisation": False,
        "dataset_audit": dataset_audit,
        "train": train_result,
        "validation": validation_result,
        "finite_scan_nonfinite_counts": finite_counts,
        "all_values_finite": not any(finite_counts.values()),
        "class_counts": {
            "train": np.bincount(arrays["train_labels"], minlength=200).tolist(),
            "validation": np.bincount(
                arrays["validation_labels"], minlength=200
            ).tolist(),
        },
        "files": files,
        "free_disk_gib_before": free_gib,
        "elapsed_seconds_total": time.time() - started,
    }
    write_json_new(paths["metadata"], metadata)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    validate_cache(config, verify_hashes=True)
    print("Spatial cache extraction and full validation passed.")
    return metadata


def fixed_2d_sincos(height: int, width: int, dimension: int) -> torch.Tensor:
    if dimension % 4:
        raise ValueError("2D sine/cosine dimension must be divisible by four.")
    quarter = dimension // 4
    frequency = 1.0 / (
        10000.0 ** (torch.arange(quarter, dtype=torch.float32) / quarter)
    )
    rows = torch.arange(height, dtype=torch.float32)[:, None] * frequency[None, :]
    cols = torch.arange(width, dtype=torch.float32)[:, None] * frequency[None, :]
    row_encoding = torch.cat([rows.sin(), rows.cos()], dim=1)
    col_encoding = torch.cat([cols.sin(), cols.cos()], dim=1)
    grid = torch.cat(
        [
            row_encoding[:, None, :].expand(height, width, -1),
            col_encoding[None, :, :].expand(height, width, -1),
        ],
        dim=-1,
    )
    return grid.reshape(height * width, dimension)


class RecursiveAttentionClassifier(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        architecture = config["architecture"]
        token = config["token_preparation"]
        self.training_horizon = int(architecture["training_horizon"])
        model_dimension = int(architecture["model_dimension"])
        width = int(architecture["swiglu_width"])
        dropout = float(architecture["attention_output_dropout"])
        epsilon = float(architecture["layer_norm_epsilon"])
        self.input_projection = nn.Linear(
            int(token["input_dimension"]),
            model_dimension,
            bias=bool(token["projection_bias"]),
        )
        self.class_token = nn.Parameter(torch.empty(1, 1, model_dimension))
        nn.init.trunc_normal_(self.class_token, std=0.02)
        self.register_buffer(
            "spatial_position",
            fixed_2d_sincos(7, 7, model_dimension)[None],
            persistent=True,
        )
        self.attention_norm = nn.LayerNorm(model_dimension, eps=epsilon)
        self.attention = nn.MultiheadAttention(
            model_dimension,
            int(architecture["attention_heads"]),
            dropout=float(architecture["attention_probability_dropout"]),
            batch_first=True,
        )
        self.attention_output_dropout = nn.Dropout(dropout)
        self.swiglu_norm = nn.LayerNorm(model_dimension, eps=epsilon)
        self.gate_projection = nn.Linear(model_dimension, width)
        self.value_projection = nn.Linear(model_dimension, width)
        self.output_projection = nn.Linear(width, model_dimension)
        self.swiglu_output_dropout = nn.Dropout(
            float(architecture["swiglu_output_dropout"])
        )
        self.head_norm = nn.LayerNorm(model_dimension, eps=epsilon)
        self.classifier = nn.Linear(model_dimension, int(config["dataset"]["classes"]))

    def prepare_tokens(self, spatial_features: torch.Tensor) -> torch.Tensor:
        spatial = self.input_projection(spatial_features)
        spatial = spatial + self.spatial_position.to(dtype=spatial.dtype)
        class_token = self.class_token.to(dtype=spatial.dtype).expand(
            len(spatial), -1, -1
        )
        return torch.cat([class_token, spatial], dim=1)

    def step(self, state: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        normalised = self.attention_norm(state)
        attention_update, _ = self.attention(
            normalised, normalised, normalised, need_weights=False
        )
        scale = float(gain) / self.training_horizon
        half = state + scale * self.attention_output_dropout(attention_update)
        normalised_half = self.swiglu_norm(half)
        hidden = F.silu(self.gate_projection(normalised_half)) * self.value_projection(
            normalised_half
        )
        update = self.swiglu_output_dropout(self.output_projection(hidden))
        return half + scale * update

    def classify(self, class_state: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.head_norm(class_state))

    def forward_terminal(
        self, spatial_features: torch.Tensor, steps: int, gain: float = 1.0
    ) -> torch.Tensor:
        state = self.prepare_tokens(spatial_features)
        for _ in range(int(steps)):
            state = self.step(state, gain)
        return self.classify(state[:, 0])


def subset_indices(labels: np.ndarray, size: int, seed: int) -> np.ndarray:
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=seed)
    indices, _ = next(splitter.split(np.zeros(len(labels)), np.asarray(labels)))
    return indices.astype(np.int64)


def numpy_batch(
    features: np.ndarray, labels: np.ndarray, indices: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_array = np.ascontiguousarray(features[indices])
    label_array = np.ascontiguousarray(labels[indices])
    return torch.from_numpy(feature_array), torch.from_numpy(label_array)


def sequential_numpy_batch(
    features: np.ndarray, labels: np.ndarray, start: int, end: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(np.array(features[start:end], copy=True)),
        torch.from_numpy(np.array(labels[start:end], copy=True)),
    )


def lr_multiplier(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max((step + 1) / max(warmup_steps, 1), 1e-8)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.inference_mode()
def terminal_accuracy(
    model: RecursiveAttentionClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
    indices: np.ndarray | None = None,
) -> tuple[float, float]:
    model.eval()
    if indices is None:
        indices = np.arange(len(labels), dtype=np.int64)
    batch_size = int(config["training"]["physical_batch_size"])
    correct = 0
    loss_sum = 0.0
    seen = 0
    for start in range(0, len(indices), batch_size):
        chosen = indices[start : start + batch_size]
        xb, yb = numpy_batch(features, labels, chosen)
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        with autocast_context():
            logits = model.forward_terminal(
                xb,
                int(config["architecture"]["training_horizon"]),
                float(config["architecture"]["training_gain"]),
            )
        logits = logits.float()
        correct += int((logits.argmax(dim=-1) == yb).sum().item())
        loss_sum += float(F.cross_entropy(logits, yb, reduction="sum").item())
        seen += len(yb)
    return correct / seen, loss_sum / seen


class RunLogger:
    def __init__(self, path: Path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite training log: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("x", encoding="utf-8", buffering=1)

    def log(self, message: str) -> None:
        print(message)
        self.handle.write(message + "\n")

    def close(self) -> None:
        self.handle.close()


def train_model(
    config: dict[str, Any],
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    seed: int,
    epochs: int,
    logger: RunLogger,
    checkpoint_directory: Path | None,
) -> tuple[RecursiveAttentionClassifier, dict[str, Any]]:
    seed_everything(seed)
    model = RecursiveAttentionClassifier(config).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    physical_batch = int(config["training"]["physical_batch_size"])
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    group_size = physical_batch * accumulation
    steps_per_epoch = math.ceil(len(train_indices) / group_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(config["training"]["warmup_epochs"]) * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_multiplier(step, warmup_steps, total_steps),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype() == torch.float16)
    generator = torch.Generator().manual_seed(seed)
    best_accuracy = -math.inf
    best_epoch = -1
    best_checkpoint: Path | None = None
    history: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(train_indices), generator=generator).numpy()
        shuffled = train_indices[permutation]
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_seen = 0
        for group_start in range(0, len(shuffled), group_size):
            group_indices = shuffled[group_start : group_start + group_size]
            group_examples = len(group_indices)
            optimizer.zero_grad(set_to_none=True)
            for micro_start in range(0, group_examples, physical_batch):
                chosen = group_indices[micro_start : micro_start + physical_batch]
                xb, yb = numpy_batch(train_features, train_labels, chosen)
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                with autocast_context():
                    logits = model.forward_terminal(
                        xb,
                        int(config["architecture"]["training_horizon"]),
                        float(config["architecture"]["training_gain"]),
                    )
                    loss = F.cross_entropy(
                        logits,
                        yb,
                        label_smoothing=float(config["training"]["label_smoothing"]),
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss at seed={seed}, epoch={epoch + 1}"
                    )
                weighted_loss = loss * (len(chosen) / group_examples)
                scaler.scale(weighted_loss).backward()
                epoch_loss_sum += float(loss.detach().item()) * len(chosen)
                epoch_correct += int(
                    (logits.detach().argmax(dim=-1) == yb).sum().item()
                )
                epoch_seen += len(chosen)
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(
                    f"Non-finite gradient at seed={seed}, epoch={epoch + 1}"
                )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        validation_accuracy, validation_loss = terminal_accuracy(
            model,
            validation_features,
            validation_labels,
            config,
            indices=validation_indices,
        )
        row = {
            "epoch": epoch + 1,
            "training_loss": epoch_loss_sum / epoch_seen,
            "training_accuracy": epoch_correct / epoch_seen,
            "validation_terminal_accuracy": validation_accuracy,
            "validation_loss": validation_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        logger.log(
            f"seed={seed} epoch={epoch + 1:02d}/{epochs} "
            f"train_loss={row['training_loss']:.4f} "
            f"train_acc={row['training_accuracy']:.4f} "
            f"val_acc={validation_accuracy:.4f} val_loss={validation_loss:.4f} "
            f"lr={row['learning_rate']:.8f}"
        )
        if checkpoint_directory is not None:
            write_json_new(
                checkpoint_directory / f"epoch_{epoch + 1:02d}_summary.json", row
            )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch + 1
            if checkpoint_directory is not None:
                best_checkpoint = (
                    checkpoint_directory / f"checkpoint_epoch_{epoch + 1:02d}.pt"
                )
                torch.save(
                    {
                        "seed": seed,
                        "epoch": best_epoch,
                        "validation_terminal_accuracy": best_accuracy,
                        "config_sha256": sha256_file(
                            CONFIG_PATHS[
                                "t8"
                                if int(config["architecture"]["training_horizon"]) == 8
                                else "t32"
                            ]
                        ),
                        "model": model.state_dict(),
                    },
                    best_checkpoint,
                )

    if best_checkpoint is not None:
        selected = torch.load(best_checkpoint, map_location=DEVICE, weights_only=False)
        model.load_state_dict(selected["model"])
    metrics = {
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_validation_terminal_accuracy": best_accuracy,
        "history": history,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
        "peak_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
        ),
        "selected_checkpoint_source": (
            best_checkpoint.name if best_checkpoint is not None else None
        ),
    }
    return model, metrics


def aggregate_representations(
    class_trajectory: torch.Tensor, config: dict[str, Any], horizon: int
) -> dict[str, torch.Tensor]:
    prefix = class_trajectory[:, :horizon]
    early_window = int(config["evaluation"]["early_window"])
    norms = prefix.float().norm(dim=-1)
    ratios = norms / norms[:, :1].clamp_min(1e-12)
    crossed = ratios > float(config["readouts"]["adaptive"]["norm_ratio_threshold"])
    first = crossed.long().argmax(dim=1)
    never = ~crossed.any(dim=1)
    first[never] = horizon - 1
    adaptive = prefix[
        torch.arange(len(prefix), device=prefix.device),
        first,
    ]
    grace = config["readouts"]["grace"]
    raw = torch.exp(
        -float(grace["alpha"])
        * torch.clamp(
            ratios - float(grace["norm_ratio_threshold"]),
            min=0.0,
        )
    )
    weights = raw / raw.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return {
        "terminal": prefix[:, -1],
        "early": prefix[:, :early_window].mean(dim=1),
        "uniform": prefix.mean(dim=1),
        "adaptive": adaptive,
        "grace": torch.einsum(
            "bt,btd->bd", weights.to(dtype=prefix.dtype), prefix
        ),
    }


@torch.inference_mode()
def class_trajectory(
    model: RecursiveAttentionClassifier,
    spatial_features: torch.Tensor,
    steps: int,
    gain: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    state = model.prepare_tokens(spatial_features)
    initial_class = state[:, 0]
    classes = []
    nonfinite_token_values = 0
    nonfinite_class_values = 0
    for _ in range(steps):
        state = model.step(state, gain)
        class_state = state[:, 0]
        nonfinite_token_values += int((~torch.isfinite(state)).sum().item())
        nonfinite_class_values += int((~torch.isfinite(class_state)).sum().item())
        classes.append(class_state)
    trajectory = torch.stack(classes, dim=1)
    return initial_class, trajectory, {
        "nonfinite_token_state_values": nonfinite_token_values,
        "nonfinite_class_state_values": nonfinite_class_values,
    }


def prepare_variant_directory(
    variant: str, config: dict[str, Any], config_sha256: str
) -> Path:
    output = OUTPUT_DIRS[variant]
    output.mkdir(parents=True, exist_ok=True)
    config_copy = output / "config.json"
    hash_copy = output / "config_sha256.txt"
    if config_copy.exists():
        if load_json(config_copy) != config:
            raise RuntimeError(f"Variant config copy mismatch: {config_copy}")
    else:
        write_bytes_new(config_copy, CONFIG_PATHS[variant].read_bytes())
    if hash_copy.exists():
        if hash_copy.read_text(encoding="utf-8").strip() != config_sha256:
            raise RuntimeError(f"Variant config hash copy mismatch: {hash_copy}")
    else:
        write_text_new(hash_copy, config_sha256 + "\n")
    return output


def prepare_seed_directory(
    variant: str, seed: int, config: dict[str, Any], config_sha256: str
) -> Path:
    output = prepare_variant_directory(variant, config, config_sha256)
    seed_dir = output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    config_copy = seed_dir / "config.json"
    hash_copy = seed_dir / "config_sha256.txt"
    if config_copy.exists():
        if load_json(config_copy) != config:
            raise RuntimeError(f"Seed config mismatch: {config_copy}")
    else:
        write_bytes_new(config_copy, CONFIG_PATHS[variant].read_bytes())
    if hash_copy.exists():
        if hash_copy.read_text(encoding="utf-8").strip() != config_sha256:
            raise RuntimeError(f"Seed hash mismatch: {hash_copy}")
    else:
        write_text_new(hash_copy, config_sha256 + "\n")
    return seed_dir


def environment_payload(config_sha256: str) -> dict[str, Any]:
    gpu_name = None
    gpu_total = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu_name = properties.name
        gpu_total = int(properties.total_memory)
    return {
        "config_sha256": config_sha256,
        "source_sha256": sha256_file(SOURCE_PATH),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": gpu_name,
        "gpu_total_bytes": gpu_total,
        "training_precision": precision_name(),
    }


def run_smoke(config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    output = prepare_variant_directory("t8", config, config_sha256)
    smoke_dir = output / "smoke"
    gate_path = smoke_dir / "engineering_gate.json"
    if gate_path.exists():
        existing = load_json(gate_path)
        if existing.get("config_sha256") != config_sha256:
            raise RuntimeError("Existing smoke gate has a different config.")
        return existing
    smoke_dir.mkdir(parents=True, exist_ok=True)
    write_text_new(
        smoke_dir / "config.json", CONFIG_PATHS["t8"].read_text(encoding="utf-8")
    )
    write_text_new(smoke_dir / "config_sha256.txt", config_sha256 + "\n")
    arrays, metadata = validate_cache(config, verify_hashes=False)
    smoke = config["smoke_test"]
    train_indices = subset_indices(
        arrays["train_labels"],
        int(smoke["training_examples"]),
        int(smoke["subset_seed"]),
    )
    validation_indices = subset_indices(
        arrays["validation_labels"],
        int(smoke["validation_examples"]),
        int(smoke["subset_seed"]),
    )
    logger = RunLogger(smoke_dir / "training_log.txt")
    try:
        logger.log(
            f"E17 engineering smoke: {len(train_indices)} train, "
            f"{len(validation_indices)} validation, seed={smoke['training_seed']}"
        )
        model, training = train_model(
            config,
            arrays["train_features"],
            arrays["train_labels"],
            arrays["validation_features"],
            arrays["validation_labels"],
            train_indices,
            validation_indices,
            int(smoke["training_seed"]),
            int(smoke["epochs"]),
            logger,
            checkpoint_directory=None,
        )
        model.eval()
        trajectory_examples = 0
        nonfinite = {
            "nonfinite_token_state_values": 0,
            "nonfinite_class_state_values": 0,
            "nonfinite_logit_values": 0,
            "nonfinite_readout_logit_values": 0,
        }
        methods_seen: set[str] = set()
        for start in range(
            0,
            len(validation_indices),
            int(config["training"]["physical_batch_size"]),
        ):
            chosen = validation_indices[
                start : start + int(config["training"]["physical_batch_size"])
            ]
            xb, _ = numpy_batch(
                arrays["validation_features"], arrays["validation_labels"], chosen
            )
            xb = xb.to(DEVICE, non_blocking=True)
            with autocast_context():
                _, trajectory, counts = class_trajectory(
                    model,
                    xb,
                    int(config["evaluation"]["maximum_horizon"]),
                    float(config["evaluation"]["gain"]),
                )
                logits = model.classify(trajectory)
                representations = aggregate_representations(
                    trajectory,
                    config,
                    int(config["evaluation"]["maximum_horizon"]),
                )
                readout_logits = {
                    method: model.classify(representation)
                    for method, representation in representations.items()
                }
            trajectory_examples += len(xb)
            for key, value in counts.items():
                nonfinite[key] += value
            nonfinite["nonfinite_logit_values"] += int(
                (~torch.isfinite(logits)).sum().item()
            )
            nonfinite["nonfinite_readout_logit_values"] += sum(
                int((~torch.isfinite(value)).sum().item())
                for value in readout_logits.values()
            )
            methods_seen.update(readout_logits)
        maximum_bytes = float(smoke["maximum_peak_cuda_allocated_gib"]) * 1024**3
        loss_decreased = (
            training["history"][-1]["training_loss"]
            < training["history"][0]["training_loss"]
        )
        accuracy_pass = (
            training["best_validation_terminal_accuracy"]
            > float(smoke["minimum_validation_accuracy_exclusive"])
        )
        finite_pass = not any(nonfinite.values())
        trajectory_pass = (
            trajectory_examples == len(validation_indices)
            and int(config["evaluation"]["maximum_horizon"]) == 32
            and int(config["architecture"]["model_dimension"]) == 256
        )
        readout_pass = methods_seen == {
            "terminal",
            "early",
            "uniform",
            "adaptive",
            "grace",
        }
        memory_pass = training["peak_cuda_allocated_bytes"] <= maximum_bytes
        result = {
            "experiment": config["experiment"],
            "config_sha256": config_sha256,
            "source_sha256": sha256_file(SOURCE_PATH),
            "cache_metadata_sha256": sha256_file(cache_paths(config)["metadata"]),
            "training": training,
            "trajectory_examples": trajectory_examples,
            "trajectory_shape_per_example": [
                int(config["evaluation"]["maximum_horizon"]),
                int(config["architecture"]["model_dimension"]),
            ],
            "nonfinite_counts": nonfinite,
            "readouts_executed": sorted(methods_seen),
            "checks": {
                "loss_decreased": bool(loss_decreased),
                "validation_accuracy_strictly_above_10_percent": bool(
                    accuracy_pass
                ),
                "finite_through_32_steps": bool(finite_pass),
                "full_trajectory_logging": bool(trajectory_pass),
                "all_readouts_execute": bool(readout_pass),
                "peak_cuda_allocated_at_most_14_gib": bool(memory_pass),
                "no_scientific_trajectory_used_for_tuning": True,
            },
        }
        result["pass"] = all(result["checks"].values())
        write_json_new(gate_path, result)
        logger.log(
            f"Engineering gate: {'PASS' if result['pass'] else 'FAIL'} | "
            f"best_val={training['best_validation_terminal_accuracy']:.4f} | "
            f"peak_allocated={training['peak_cuda_allocated_bytes'] / 1024**3:.2f} GiB"
        )
        if not result["pass"]:
            raise RuntimeError(f"E17 engineering gate failed: {result['checks']}")
        return result
    finally:
        logger.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def require_smoke_pass(config_sha256: str) -> dict[str, Any]:
    path = OUTPUT_DIRS["t8"] / "smoke" / "engineering_gate.json"
    if not path.exists():
        raise RuntimeError("Run the E17 engineering smoke gate first.")
    result = load_json(path)
    if result.get("config_sha256") != config_sha256 or not result.get("pass"):
        raise RuntimeError("E17 smoke gate is absent, mismatched, or failed.")
    return result


def require_control_trigger() -> dict[str, Any]:
    path = OUTPUT_DIRS["t8"] / "matched_control_trigger.json"
    if not path.exists():
        raise RuntimeError("Aggregate E17 before considering the matched control.")
    trigger = load_json(path)
    if not trigger.get("run_t32_control"):
        raise RuntimeError("Frozen trigger did not authorize the T_train=32 control.")
    return trigger


def train_scientific_seed(
    variant: str, seed: int, config: dict[str, Any], config_sha256: str
) -> dict[str, Any]:
    if seed not in [int(value) for value in config["training"]["seeds"]]:
        raise ValueError(f"Seed {seed} is not frozen for {variant}.")
    t8_config_sha = sha256_file(CONFIG_PATHS["t8"])
    require_smoke_pass(t8_config_sha)
    if variant == "t32":
        require_control_trigger()
    seed_dir = prepare_seed_directory(variant, seed, config, config_sha256)
    summary_path = seed_dir / "training_history.json"
    if summary_path.exists():
        existing = load_json(summary_path)
        if existing.get("config_sha256") != config_sha256:
            raise RuntimeError("Existing training summary config mismatch.")
        print(f"Using complete existing training for {variant} seed {seed}.")
        return existing
    if (seed_dir / "training_log.txt").exists():
        raise RuntimeError(
            f"Incomplete prior seed run exists at {seed_dir}; stop rather than overwrite."
        )
    arrays, metadata = validate_cache(config, verify_hashes=False)
    logger = RunLogger(seed_dir / "training_log.txt")
    try:
        logger.log(
            f"Training {config['experiment']} seed={seed} on {DEVICE}, "
            f"precision={precision_name()}, config={config_sha256}"
        )
        model, training = train_model(
            config,
            arrays["train_features"],
            arrays["train_labels"],
            arrays["validation_features"],
            arrays["validation_labels"],
            np.arange(len(arrays["train_labels"]), dtype=np.int64),
            np.arange(len(arrays["validation_labels"]), dtype=np.int64),
            seed,
            int(config["training"]["epochs"]),
            logger,
            checkpoint_directory=seed_dir,
        )
        selected_source = seed_dir / training["selected_checkpoint_source"]
        selected_payload = torch.load(
            selected_source, map_location="cpu", weights_only=False
        )
        checkpoint_path = seed_dir / "checkpoint.pt"
        if checkpoint_path.exists():
            raise FileExistsError(f"Refusing to overwrite {checkpoint_path}")
        torch.save(selected_payload, checkpoint_path)
        selection = {
            "seed": seed,
            "config_sha256": config_sha256,
            "selection_rule": config["training"]["checkpoint_selection"],
            "best_epoch": training["best_epoch"],
            "best_validation_terminal_accuracy": training[
                "best_validation_terminal_accuracy"
            ],
            "source_checkpoint": selected_source.name,
            "final_checkpoint": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }
        write_json_new(seed_dir / "selected_checkpoint.json", selection)
        payload = {
            "experiment": config["experiment"],
            "config_sha256": config_sha256,
            "source_sha256": sha256_file(SOURCE_PATH),
            "cache_metadata_sha256": sha256_file(cache_paths(config)["metadata"]),
            "cache_train_features_sha256": metadata["files"]["train_features"][
                "sha256"
            ],
            **training,
        }
        write_json_new(summary_path, payload)
        logger.log(
            f"Selected epoch {training['best_epoch']} with "
            f"validation accuracy {training['best_validation_terminal_accuracy']:.4f}"
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return payload
    finally:
        logger.close()


def load_selected_model(
    variant: str, seed: int, config: dict[str, Any], config_sha256: str
) -> RecursiveAttentionClassifier:
    checkpoint = OUTPUT_DIRS[variant] / f"seed_{seed}" / "checkpoint.pt"
    payload = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    if payload.get("config_sha256") != config_sha256 or payload.get("seed") != seed:
        raise RuntimeError(f"Checkpoint provenance mismatch: {checkpoint}")
    model = RecursiveAttentionClassifier(config).to(DEVICE)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def evaluate_seed(
    variant: str, seed: int, config: dict[str, Any], config_sha256: str
) -> dict[str, Any]:
    seed_dir = prepare_seed_directory(variant, seed, config, config_sha256)
    required = [
        seed_dir / "trajectory_metrics.json",
        seed_dir / "readout_metrics.json",
        seed_dir / "diagnostics.json",
        seed_dir / "runtime_memory.json",
    ]
    if all(path.exists() for path in required):
        existing = load_json(seed_dir / "diagnostics.json")
        if existing.get("config_sha256") != config_sha256:
            raise RuntimeError("Existing evaluation config mismatch.")
        print(f"Using existing evaluation for {variant} seed {seed}.")
        return existing
    if any(path.exists() for path in required):
        raise RuntimeError("Partial evaluation exists; stop rather than overwrite.")
    arrays, metadata = validate_cache(config, verify_hashes=False)
    model = load_selected_model(variant, seed, config, config_sha256)
    horizons = [int(value) for value in config["evaluation"]["test_horizons"]]
    maximum_horizon = max(horizons)
    timestep_correct = torch.zeros(maximum_horizon, dtype=torch.float64)
    timestep_ce = torch.zeros(maximum_horizon, dtype=torch.float64)
    norm_sum = torch.zeros(maximum_horizon, dtype=torch.float64)
    residual_sum = torch.zeros(maximum_horizon, dtype=torch.float64)
    entropy_sum = torch.zeros(maximum_horizon, dtype=torch.float64)
    change_sum = torch.zeros(maximum_horizon, dtype=torch.float64)
    methods = ("terminal", "early", "uniform", "adaptive", "grace")
    readout_correct = {
        str(horizon): {method: 0 for method in methods} for horizon in horizons
    }
    readout_ce = {
        str(horizon): {method: 0.0 for method in methods} for horizon in horizons
    }
    nonfinite = {
        "nonfinite_token_state_values": 0,
        "nonfinite_class_state_values": 0,
        "nonfinite_timestep_logit_values": 0,
        "nonfinite_readout_logit_values": 0,
    }
    seen = 0
    batch_size = int(config["training"]["physical_batch_size"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    labels = arrays["validation_labels"]
    for start in range(0, len(labels), batch_size):
        end = min(start + batch_size, len(labels))
        xb, yb = sequential_numpy_batch(
            arrays["validation_features"], labels, start, end
        )
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        with torch.inference_mode(), autocast_context():
            initial_class, trajectory, counts = class_trajectory(
                model,
                xb,
                maximum_horizon,
                float(config["evaluation"]["gain"]),
            )
            logits = model.classify(trajectory)
        for key, value in counts.items():
            nonfinite[key] += value
        logits_float = logits.float()
        nonfinite["nonfinite_timestep_logit_values"] += int(
            (~torch.isfinite(logits_float)).sum().item()
        )
        predictions = logits_float.argmax(dim=-1)
        timestep_correct += (
            (predictions == yb[:, None]).sum(dim=0).double().cpu()
        )
        ce = F.cross_entropy(
            logits_float.reshape(-1, logits_float.shape[-1]),
            yb[:, None].expand(-1, maximum_horizon).reshape(-1),
            reduction="none",
        ).reshape(len(yb), maximum_horizon)
        timestep_ce += ce.double().sum(dim=0).cpu()
        norms = trajectory.float().norm(dim=-1)
        norm_sum += norms.double().sum(dim=0).cpu()
        previous = torch.cat([initial_class[:, None], trajectory[:, :-1]], dim=1)
        residual = (
            (trajectory.float() - previous.float()).norm(dim=-1)
            / previous.float().norm(dim=-1).clamp_min(1e-12)
        )
        residual_sum += residual.double().sum(dim=0).cpu()
        probabilities = torch.softmax(logits_float, dim=-1)
        entropy = -(
            probabilities * probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1)
        entropy_sum += entropy.double().sum(dim=0).cpu()
        changes = torch.zeros_like(predictions, dtype=torch.float32)
        changes[:, 1:] = (predictions[:, 1:] != predictions[:, :-1]).float()
        change_sum += changes.double().sum(dim=0).cpu()

        for horizon in horizons:
            with torch.inference_mode(), autocast_context():
                representations = aggregate_representations(
                    trajectory, config, horizon
                )
                method_logits = {
                    method: model.classify(representation)
                    for method, representation in representations.items()
                }
            for method, value in method_logits.items():
                value_float = value.float()
                nonfinite["nonfinite_readout_logit_values"] += int(
                    (~torch.isfinite(value_float)).sum().item()
                )
                readout_correct[str(horizon)][method] += int(
                    (value_float.argmax(dim=-1) == yb).sum().item()
                )
                readout_ce[str(horizon)][method] += float(
                    F.cross_entropy(value_float, yb, reduction="sum").item()
                )
        seen += len(yb)
        if seen % 2560 == 0 or seen == len(labels):
            print(f"  evaluate {variant} seed={seed}: {seen}/{len(labels)}")

    timestep_accuracy = (timestep_correct / seen).tolist()
    trajectory_metrics = {
        "experiment": config["experiment"],
        "config_sha256": config_sha256,
        "seed": seed,
        "gain": float(config["evaluation"]["gain"]),
        "examples": seen,
        "timestep_accuracy": timestep_accuracy,
        "timestep_cross_entropy": (timestep_ce / seen).tolist(),
        "mean_class_token_norm": (norm_sum / seen).tolist(),
        "mean_relative_update_residual": (residual_sum / seen).tolist(),
        "mean_prediction_entropy": (entropy_sum / seen).tolist(),
        "class_change_fraction": (change_sum / seen).tolist(),
    }
    readout_metrics = {
        "experiment": config["experiment"],
        "config_sha256": config_sha256,
        "seed": seed,
        "by_horizon": {
            str(horizon): {
                "accuracy": {
                    method: readout_correct[str(horizon)][method] / seen
                    for method in methods
                },
                "cross_entropy": {
                    method: readout_ce[str(horizon)][method] / seen
                    for method in methods
                },
            }
            for horizon in horizons
        },
    }
    by_horizon: dict[str, Any] = {}
    for horizon in horizons:
        prefix = timestep_accuracy[:horizon]
        best_index = max(range(horizon), key=lambda index: prefix[index])
        terminal_accuracy_value = float(prefix[-1])
        by_horizon[str(horizon)] = {
            "terminal_accuracy": terminal_accuracy_value,
            "early_accuracy": readout_metrics["by_horizon"][str(horizon)]["accuracy"][
                "early"
            ],
            "best_fixed_timestep": best_index + 1,
            "best_fixed_timestep_accuracy": float(prefix[best_index]),
            "peak_minus_terminal_pp": 100.0
            * (float(prefix[best_index]) - terminal_accuracy_value),
            "terminal_minus_early_pp": 100.0
            * (
                terminal_accuracy_value
                - readout_metrics["by_horizon"][str(horizon)]["accuracy"]["early"]
            ),
            "endpoint_mean_class_token_norm": float((norm_sum / seen)[horizon - 1]),
            "endpoint_mean_relative_update_residual": float(
                (residual_sum / seen)[horizon - 1]
            ),
        }
    all_finite = not any(nonfinite.values())
    minimum_accuracy = min(
        [
            by_horizon[str(horizon)]["terminal_accuracy"]
            for horizon in horizons
        ]
        + [
            readout_metrics["by_horizon"][str(horizon)]["accuracy"]["early"]
            for horizon in horizons
        ]
    )
    diagnostics = {
        "experiment": config["experiment"],
        "config_sha256": config_sha256,
        "source_sha256": sha256_file(SOURCE_PATH),
        "cache_metadata_sha256": sha256_file(cache_paths(config)["metadata"]),
        "cache_validation_features_sha256": metadata["files"][
            "validation_features"
        ]["sha256"],
        "seed": seed,
        "by_horizon": by_horizon,
        "nonfinite_counts": nonfinite,
        "all_finite": all_finite,
        "minimum_terminal_or_early_accuracy": minimum_accuracy,
        "materially_above_chance": minimum_accuracy
        > float(config["evaluation"]["minimum_material_accuracy"]),
    }
    training = load_json(seed_dir / "training_history.json")
    runtime = {
        "experiment": config["experiment"],
        "config_sha256": config_sha256,
        "seed": seed,
        "training_wall_seconds": training["elapsed_seconds"],
        "training_peak_cuda_allocated_bytes": training[
            "peak_cuda_allocated_bytes"
        ],
        "training_peak_cuda_reserved_bytes": training["peak_cuda_reserved_bytes"],
        "evaluation_wall_seconds": time.time() - started,
        "evaluation_peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
        "evaluation_peak_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
        ),
    }
    write_json_new(seed_dir / "trajectory_metrics.json", trajectory_metrics)
    write_json_new(seed_dir / "readout_metrics.json", readout_metrics)
    write_json_new(seed_dir / "diagnostics.json", diagnostics)
    write_json_new(seed_dir / "runtime_memory.json", runtime)
    print(
        f"Evaluated {variant} seed={seed}: "
        + ", ".join(
            f"T{h}={by_horizon[str(h)]['terminal_accuracy']:.4f}"
            for h in horizons
        )
        + f", early={by_horizon['32']['early_accuracy']:.4f}"
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return diagnostics


@torch.inference_mode()
def collect_probe_representations(
    model: RecursiveAttentionClassifier,
    features: np.ndarray,
    indices: np.ndarray,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {
        "terminal_step8": [],
        "early_mean_steps1_to_8": [],
        "terminal_step32": [],
    }
    batch_size = int(config["training"]["physical_batch_size"])
    for start in range(0, len(indices), batch_size):
        chosen = indices[start : start + batch_size]
        feature_array = np.ascontiguousarray(features[chosen])
        xb = torch.from_numpy(feature_array).to(DEVICE, non_blocking=True)
        with autocast_context():
            _, trajectory, counts = class_trajectory(
                model,
                xb,
                int(config["evaluation"]["maximum_horizon"]),
                float(config["evaluation"]["gain"]),
            )
        if counts["nonfinite_token_state_values"] or counts[
            "nonfinite_class_state_values"
        ]:
            raise RuntimeError("Non-finite probe trajectory.")
        chunks["terminal_step8"].append(trajectory[:, 7].float().cpu().numpy())
        chunks["early_mean_steps1_to_8"].append(
            trajectory[:, :8].mean(dim=1).float().cpu().numpy()
        )
        chunks["terminal_step32"].append(trajectory[:, 31].float().cpu().numpy())
    return {name: np.concatenate(values, axis=0) for name, values in chunks.items()}


def run_probe(
    variant: str, seed: int, config: dict[str, Any], config_sha256: str
) -> dict[str, Any]:
    seed_dir = prepare_seed_directory(variant, seed, config, config_sha256)
    path = seed_dir / "probe_metrics.json"
    if path.exists():
        existing = load_json(path)
        if existing.get("config_sha256") != config_sha256:
            raise RuntimeError("Existing probe config mismatch.")
        print(f"Using existing probe for {variant} seed {seed}.")
        return existing
    arrays, _ = validate_cache(config, verify_hashes=False)
    probe_path = ROOT / config["probes"]["training_subset_indices"]
    if sha256_file(probe_path) != config["probes"]["training_subset_sha256"]:
        raise RuntimeError("Frozen probe subset hash mismatch.")
    train_indices = np.load(probe_path).astype(np.int64)
    validation_indices = np.arange(
        len(arrays["validation_labels"]), dtype=np.int64
    )
    model = load_selected_model(variant, seed, config, config_sha256)
    started = time.time()
    print(f"Collecting probe train representations for {variant} seed={seed}...")
    train_representations = collect_probe_representations(
        model, arrays["train_features"], train_indices, config
    )
    print(f"Collecting probe validation representations for {variant} seed={seed}...")
    validation_representations = collect_probe_representations(
        model, arrays["validation_features"], validation_indices, config
    )
    scores = {}
    train_labels = np.asarray(arrays["train_labels"][train_indices])
    validation_labels = np.asarray(arrays["validation_labels"])
    for name in (
        "terminal_step8",
        "early_mean_steps1_to_8",
        "terminal_step32",
    ):
        classifier = RidgeClassifier(alpha=float(config["probes"]["alpha"]))
        classifier.fit(train_representations[name], train_labels)
        scores[name] = float(
            classifier.score(validation_representations[name], validation_labels)
        )
        print(f"  probe {name}: {scores[name]:.4f}")
    result = {
        "experiment": config["experiment"],
        "config_sha256": config_sha256,
        "seed": seed,
        "training_subset_path": config["probes"]["training_subset_indices"],
        "training_subset_sha256": config["probes"]["training_subset_sha256"],
        "training_subset_size": len(train_indices),
        "validation_examples": len(validation_indices),
        "classifier": config["probes"]["classifier"],
        "alpha": float(config["probes"]["alpha"]),
        "standardisation": False,
        "accuracy": scores,
        "terminal_step8_minus_step32_pp": 100.0
        * (scores["terminal_step8"] - scores["terminal_step32"]),
        "elapsed_seconds": time.time() - started,
    }
    write_json_new(path, result)
    del model, train_representations, validation_representations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def mean_std(values: Iterable[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    return {
        "values": numbers,
        "mean": statistics.mean(numbers),
        "sample_std": statistics.stdev(numbers),
    }


def aggregate_attention(
    variant: str, config: dict[str, Any], config_sha256: str
) -> dict[str, Any]:
    output = prepare_variant_directory(variant, config, config_sha256)
    summary_path = output / "attention_summary.json"
    if summary_path.exists():
        existing = load_json(summary_path)
        if existing.get("config_sha256") != config_sha256:
            raise RuntimeError("Existing attention summary config mismatch.")
        print(f"Using existing aggregate for {variant}.")
        return existing
    seeds = [int(value) for value in config["training"]["seeds"]]
    metrics: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        seed_dir = output / f"seed_{seed}"
        required = (
            "trajectory_metrics.json",
            "readout_metrics.json",
            "diagnostics.json",
            "probe_metrics.json",
            "training_history.json",
            "runtime_memory.json",
        )
        missing = [name for name in required if not (seed_dir / name).exists()]
        if missing:
            raise RuntimeError(f"Seed {seed} is incomplete: {missing}")
        metrics[seed] = {
            "trajectory": load_json(seed_dir / "trajectory_metrics.json"),
            "readout": load_json(seed_dir / "readout_metrics.json"),
            "diagnostics": load_json(seed_dir / "diagnostics.json"),
            "probe": load_json(seed_dir / "probe_metrics.json"),
            "training": load_json(seed_dir / "training_history.json"),
            "runtime": load_json(seed_dir / "runtime_memory.json"),
        }
        for payload in metrics[seed].values():
            if payload.get("config_sha256") != config_sha256:
                raise RuntimeError(f"Seed {seed} contains a config mismatch.")

    horizons = [int(value) for value in config["evaluation"]["test_horizons"]]
    seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        diagnostic = metrics[seed]["diagnostics"]["by_horizon"]
        readout = metrics[seed]["readout"]["by_horizon"]
        probe = metrics[seed]["probe"]
        terminal_8 = float(diagnostic["8"]["terminal_accuracy"])
        terminal_32 = float(diagnostic["32"]["terminal_accuracy"])
        early_32 = float(readout["32"]["accuracy"]["early"])
        seed_rows.append(
            {
                "seed": seed,
                "terminal_t8": terminal_8,
                "terminal_t32": terminal_32,
                "terminal_drop_pp": 100.0 * (terminal_8 - terminal_32),
                "early_t32": early_32,
                "absolute_early_t32_to_terminal_t8_pp": 100.0
                * abs(early_32 - terminal_8),
                "best_step_t32": int(diagnostic["32"]["best_fixed_timestep"]),
                "best_step_accuracy_t32": float(
                    diagnostic["32"]["best_fixed_timestep_accuracy"]
                ),
                "peak_minus_terminal_t32_pp": float(
                    diagnostic["32"]["peak_minus_terminal_pp"]
                ),
                "probe_terminal_t8": float(
                    probe["accuracy"]["terminal_step8"]
                ),
                "probe_terminal_t32": float(
                    probe["accuracy"]["terminal_step32"]
                ),
                "probe_drop_pp": float(probe["terminal_step8_minus_step32_pp"]),
                "best_epoch": int(metrics[seed]["training"]["best_epoch"]),
                "all_finite": bool(metrics[seed]["diagnostics"]["all_finite"]),
                "materially_above_chance": bool(
                    metrics[seed]["diagnostics"]["materially_above_chance"]
                ),
            }
        )

    trajectory_summary: dict[str, Any] = {}
    for horizon in horizons:
        horizon_key = str(horizon)
        trajectory_summary[horizon_key] = {}
        for method in ("terminal", "early", "uniform", "adaptive", "grace"):
            trajectory_summary[horizon_key][method] = mean_std(
                metrics[seed]["readout"]["by_horizon"][horizon_key]["accuracy"][
                    method
                ]
                for seed in seeds
            )
        trajectory_summary[horizon_key]["best_fixed_timestep_values"] = [
            int(
                metrics[seed]["diagnostics"]["by_horizon"][horizon_key][
                    "best_fixed_timestep"
                ]
            )
            for seed in seeds
        ]
        trajectory_summary[horizon_key]["peak_minus_terminal_pp"] = mean_std(
            metrics[seed]["diagnostics"]["by_horizon"][horizon_key][
                "peak_minus_terminal_pp"
            ]
            for seed in seeds
        )
        trajectory_summary[horizon_key]["endpoint_relative_residual"] = mean_std(
            metrics[seed]["diagnostics"]["by_horizon"][horizon_key][
                "endpoint_mean_relative_update_residual"
            ]
            for seed in seeds
        )
        trajectory_summary[horizon_key]["endpoint_class_token_norm"] = mean_std(
            metrics[seed]["diagnostics"]["by_horizon"][horizon_key][
                "endpoint_mean_class_token_norm"
            ]
            for seed in seeds
        )

    drops = [row["terminal_drop_pp"] for row in seed_rows]
    probe_drops = [row["probe_drop_pp"] for row in seed_rows]
    mean_terminal_8 = statistics.mean(row["terminal_t8"] for row in seed_rows)
    mean_early_32 = statistics.mean(row["early_t32"] for row in seed_rows)
    mean_early_difference_pp = 100.0 * abs(mean_early_32 - mean_terminal_8)
    all_finite_material = all(
        row["all_finite"] and row["materially_above_chance"] for row in seed_rows
    )
    endpoint_criteria = {
        "mean_drop_at_least_2pp": statistics.mean(drops)
        >= float(
            config["classification"]["endpoint_invalidity"][
                "minimum_mean_terminal_drop_pp"
            ]
        ),
        "drop_positive_every_seed": all(value > 0.0 for value in drops),
        "mean_early_within_1pp_of_trained_terminal": mean_early_difference_pp
        <= float(
            config["classification"]["endpoint_invalidity"][
                "maximum_absolute_mean_early_to_trained_terminal_difference_pp"
            ]
        ),
        "finite_and_material": all_finite_material,
        "probe_mean_directional_agreement": statistics.mean(probe_drops) > 0.0,
    }
    mean_peak_gap = statistics.mean(
        row["peak_minus_terminal_t32_pp"] for row in seed_rows
    )
    residual_step8 = statistics.mean(
        metrics[seed]["trajectory"]["mean_relative_update_residual"][7]
        for seed in seeds
    )
    residual_step32 = statistics.mean(
        metrics[seed]["trajectory"]["mean_relative_update_residual"][31]
        for seed in seeds
    )
    finite_norms_residuals = all(
        np.isfinite(
            metrics[seed]["trajectory"]["mean_class_token_norm"]
            + metrics[seed]["trajectory"]["mean_relative_update_residual"]
        ).all()
        for seed in seeds
    )
    stable_criteria = {
        "mean_drop_below_0_5pp": statistics.mean(drops)
        < float(
            config["classification"]["stable"][
                "maximum_mean_terminal_drop_pp_exclusive"
            ]
        ),
        "mean_peak_gap_below_0_5pp": mean_peak_gap
        < float(
            config["classification"]["stable"][
                "maximum_mean_peak_minus_terminal_pp_exclusive"
            ]
        ),
        "finite_norms_and_residuals": bool(finite_norms_residuals),
        "endpoint_residual_no_greater_than_step8": residual_step32
        <= residual_step8,
        "finite_and_material": all_finite_material,
    }
    if variant == "t32":
        classification = "matched_horizon_control"
    elif all(endpoint_criteria.values()):
        classification = "endpoint_invalidity"
    elif all(stable_criteria.values()):
        classification = "stable"
    else:
        classification = "ambiguous"

    trigger_checks = {
        "mean_terminal_drop_at_least_2pp": statistics.mean(drops)
        >= float(config["matched_control_trigger"]["minimum_mean_terminal_drop_pp"]),
        "terminal_drop_positive_every_seed": all(value > 0.0 for value in drops),
        "mean_early_within_1pp_of_trained_terminal": mean_early_difference_pp
        <= float(
            config["matched_control_trigger"][
                "maximum_absolute_mean_early_to_trained_terminal_difference_pp"
            ]
        ),
    }
    trigger = {
        "experiment": config["experiment"],
        "config_sha256": config_sha256,
        "checks": trigger_checks,
        "run_t32_control": bool(all(trigger_checks.values())) if variant == "t8" else None,
        "probe_independent": True,
        "evaluated_after_all_three_seeds": True,
    }
    summary = {
        "experiment": config["experiment"],
        "config_sha256": config_sha256,
        "source_sha256": sha256_file(SOURCE_PATH),
        "status": "complete",
        "variant": variant,
        "training_horizon": int(config["architecture"]["training_horizon"]),
        "seeds": seeds,
        "test_horizons": horizons,
        "gain": float(config["evaluation"]["gain"]),
        "seed_level": seed_rows,
        "terminal_drop_pp": mean_std(drops),
        "probe_terminal_drop_pp": mean_std(probe_drops),
        "absolute_mean_early_t32_to_terminal_t8_pp": mean_early_difference_pp,
        "trajectory_summary": trajectory_summary,
        "mean_residual_step8": residual_step8,
        "mean_residual_step32": residual_step32,
        "classification": classification,
        "endpoint_invalidity_criteria": endpoint_criteria,
        "stable_criteria": stable_criteria,
        "all_finite_and_material": all_finite_material,
    }
    write_json_new(summary_path, summary)
    write_json_new(output / "attention_classification.json", {
        "experiment": config["experiment"],
        "config_sha256": config_sha256,
        "classification": classification,
        "endpoint_invalidity_criteria": endpoint_criteria,
        "stable_criteria": stable_criteria,
    })
    if variant == "t8":
        write_json_new(output / "matched_control_trigger.json", trigger)

    table_lines = [
        "seed\tterminal_T8\tterminal_T32\tdrop_pp\tearly_T32\tbest_step"
        "\tprobe_drop_pp\tclassification"
    ]
    for row in seed_rows:
        table_lines.append(
            f"{row['seed']}\t{row['terminal_t8']:.4f}\t"
            f"{row['terminal_t32']:.4f}\t{row['terminal_drop_pp']:+.2f}\t"
            f"{row['early_t32']:.4f}\t{row['best_step_t32']}\t"
            f"{row['probe_drop_pp']:+.2f}\t{classification}"
        )
    table_lines.append(
        f"mean+-sd\t{statistics.mean(row['terminal_t8'] for row in seed_rows):.4f}"
        f"+-{statistics.stdev(row['terminal_t8'] for row in seed_rows):.4f}\t"
        f"{statistics.mean(row['terminal_t32'] for row in seed_rows):.4f}"
        f"+-{statistics.stdev(row['terminal_t32'] for row in seed_rows):.4f}\t"
        f"{statistics.mean(drops):+.2f}+-{statistics.stdev(drops):.2f}\t"
        f"{mean_early_32:.4f}+-"
        f"{statistics.stdev(row['early_t32'] for row in seed_rows):.4f}\t"
        f"{'/'.join(str(row['best_step_t32']) for row in seed_rows)}\t"
        f"{statistics.mean(probe_drops):+.2f}+-{statistics.stdev(probe_drops):.2f}"
        f"\t{classification}"
    )
    write_text_new(output / "attention_seed_table.txt", "\n".join(table_lines) + "\n")

    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    terminal_means = [
        trajectory_summary[str(horizon)]["terminal"]["mean"] * 100.0
        for horizon in horizons
    ]
    terminal_stds = [
        trajectory_summary[str(horizon)]["terminal"]["sample_std"] * 100.0
        for horizon in horizons
    ]
    early_mean = trajectory_summary["32"]["early"]["mean"] * 100.0
    early_std = trajectory_summary["32"]["early"]["sample_std"] * 100.0
    for seed in seeds:
        axis.plot(
            horizons,
            [
                metrics[seed]["readout"]["by_horizon"][str(horizon)]["accuracy"][
                    "terminal"
                ]
                * 100.0
                for horizon in horizons
            ],
            color="#6A51A3",
            alpha=0.24,
            linewidth=1.1,
        )
    axis.errorbar(
        horizons,
        terminal_means,
        yerr=terminal_stds,
        color="#54278F",
        marker="o",
        linewidth=2.2,
        capsize=3,
        label="Terminal",
    )
    axis.axhline(
        early_mean,
        color="#E6550D",
        linestyle="--",
        linewidth=1.8,
        label=f"Early-8 ({early_mean:.2f}% +/- {early_std:.2f})",
    )
    axis.set_xlabel(r"Inference horizon $T_{test}$")
    axis.set_ylabel("Validation accuracy (%)")
    axis.set_xticks(horizons)
    axis.set_title(
        rf"Recursive attention, $T_{{train}}={config['architecture']['training_horizon']}$"
    )
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "attention_horizon.png", dpi=220)
    plt.close(figure)

    report = [
        f"# {config['experiment']} run report",
        "",
        f"**Classification:** {classification}.",
        "",
        "All three frozen seeds completed. Evaluation used gain 1.0 and only "
        "test horizons 8, 16, 24, and 32.",
        "",
        "## Seed-level headline results",
        "",
        "| Seed | Terminal T=8 | Terminal T=32 | Drop | Early T=32 | "
        "Best step | Probe drop |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in seed_rows:
        report.append(
            f"| {row['seed']} | {100*row['terminal_t8']:.2f}% | "
            f"{100*row['terminal_t32']:.2f}% | "
            f"{row['terminal_drop_pp']:+.2f} pp | "
            f"{100*row['early_t32']:.2f}% | {row['best_step_t32']} | "
            f"{row['probe_drop_pp']:+.2f} pp |"
        )
    report.extend(
        [
            "",
            f"Mean terminal drop: {statistics.mean(drops):+.2f} +/- "
            f"{statistics.stdev(drops):.2f} pp.",
            "",
            f"Mean fixed-probe terminal drop: {statistics.mean(probe_drops):+.2f} "
            f"+/- {statistics.stdev(probe_drops):.2f} pp.",
            "",
            f"Mean endpoint residual changed from {residual_step8:.6f} at step 8 "
            f"to {residual_step32:.6f} at step 32.",
            "",
            "## Frozen category audit",
            "",
        ]
    )
    for name, value in endpoint_criteria.items():
        report.append(f"- Endpoint criterion `{name}`: {value}.")
    for name, value in stable_criteria.items():
        report.append(f"- Stable criterion `{name}`: {value}.")
    if variant == "t8":
        report.extend(
            [
                "",
                f"**Matched-control trigger:** "
                f"{'PASS' if trigger['run_t32_control'] else 'FAIL'}.",
            ]
        )
        for name, value in trigger_checks.items():
            report.append(f"- Trigger `{name}`: {value}.")
    report.extend(
        [
            "",
            "All state/logit finite checks, per-timestep metrics, readout "
            "metrics, probes, runtimes, and individual checkpoints are retained.",
            "",
            "No gain sweep or architecture-specific readout tuning was run.",
            "",
        ]
    )
    write_text_new(output / "ATTENTION_RUN_REPORT.md", "\n".join(report))
    print(
        f"Aggregate {variant}: classification={classification}, "
        f"mean_drop={statistics.mean(drops):+.2f} pp"
    )
    return summary


def panel_aggregate() -> dict[str, Any]:
    panel_summary_path = PANEL_DIR / "final_summary.json"
    if panel_summary_path.exists():
        existing = load_json(panel_summary_path)
        print("Using existing final panel summary.")
        return existing
    e16 = load_json(
        ROOT
        / "results"
        / "rebuttal_recursive_tinyimagenet_paired_horizon"
        / "aggregate_summary.json"
    )
    attention = load_json(OUTPUT_DIRS["t8"] / "attention_summary.json")
    trigger = load_json(OUTPUT_DIRS["t8"] / "matched_control_trigger.json")
    control = None
    if trigger["run_t32_control"]:
        control_path = OUTPUT_DIRS["t32"] / "attention_summary.json"
        if not control_path.exists():
            raise RuntimeError("Matched control was triggered but is incomplete.")
        control = load_json(control_path)

    models: list[dict[str, Any]] = []
    for train_horizon, classification in (
        (8, "endpoint_invalidity"),
        (32, "matched_horizon_control"),
    ):
        summary = e16["trajectory_summary"][str(train_horizon)]
        models.append(
            {
                "model_condition": f"SwiGLU T_train={train_horizon}",
                "family": "residual_recursive",
                "training_horizon": train_horizon,
                "terminal_t8": summary["8"]["terminal_accuracy"],
                "terminal_t32": summary["32"]["terminal_accuracy"],
                "early_t32": summary["32"]["early_eight_accuracy"],
                "best_step_t32_values": summary["32"][
                    "best_fixed_timestep_values"
                ],
                "classification": classification,
            }
        )

    def attention_model_row(
        summary: dict[str, Any], label: str, classification: str
    ) -> dict[str, Any]:
        return {
            "model_condition": label,
            "family": "recursive_attention",
            "training_horizon": summary["training_horizon"],
            "terminal_t8": summary["trajectory_summary"]["8"]["terminal"],
            "terminal_t32": summary["trajectory_summary"]["32"]["terminal"],
            "early_t32": summary["trajectory_summary"]["32"]["early"],
            "best_step_t32_values": summary["trajectory_summary"]["32"][
                "best_fixed_timestep_values"
            ],
            "classification": classification,
        }

    models.append(
        attention_model_row(
            attention, "Attention T_train=8", attention["classification"]
        )
    )
    if control is not None:
        models.append(
            attention_model_row(
                control, "Attention T_train=32", "matched_horizon_control"
            )
        )

    deviations = [
        {
            "type": "provenance",
            "description": "Git commit unavailable because the workspace .git "
            "directory is empty and no Git executable is available; exact "
            "protocol, config, cache, checkpoint, and source SHA-256 hashes "
            "were recorded instead.",
            "scientific_effect": "none",
        }
    ]
    phases = {
        "locked_e14_e15_e16_audit": "complete_pass",
        "attention_protocol_and_configs": "complete_frozen_before_smoke",
        "spatial_feature_cache": "complete_validated",
        "attention_t8_engineering_gate": "complete_pass",
        "attention_t8_scientific_seeds": "complete_101_102_103",
        "attention_t8_probe_and_classification": "complete",
        "attention_t32_trigger": (
            "passed_and_control_complete"
            if trigger["run_t32_control"]
            else "failed_control_skipped"
        ),
        "optional_deq": "skipped_optional_conditions_not_verified",
        "panel_aggregation": "complete",
    }
    final = {
        "panel": "modern_iterative_panel",
        "status": "complete",
        "locked_e16_unchanged": True,
        "attention_classification": attention["classification"],
        "attention_t32_trigger": trigger,
        "models": models,
        "phases": phases,
        "protocol_deviations": deviations,
        "frozen_hashes": load_json(FROZEN_HASH_PATH),
        "source_sha256": sha256_file(SOURCE_PATH),
    }
    write_json_new(panel_summary_path, final)
    write_json_new(PANEL_DIR / "protocol_deviations.json", {
        "status": "complete",
        "deviations": deviations,
    })

    figures_dir = PANEL_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["model_condition"] for row in models]
    drops = [
        100.0
        * (
            float(row["terminal_t8"]["mean"])
            - float(row["terminal_t32"]["mean"])
        )
        for row in models
    ]
    drop_std = [
        math.sqrt(
            (100.0 * float(row["terminal_t8"]["sample_std"])) ** 2
            + (100.0 * float(row["terminal_t32"]["sample_std"])) ** 2
        )
        for row in models
    ]
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    colors = ["#C84C4C", "#2474A6", "#6A51A3", "#4C956C"][: len(models)]
    x_values = np.arange(len(models))
    axis.bar(x_values, drops, yerr=drop_std, color=colors, capsize=3)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x_values, labels, rotation=18, ha="right")
    axis.set_ylabel(r"Terminal $A(8)-A(32)$ (pp)")
    axis.set_title("Endpoint degradation across tested iterative systems")
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    figure.savefig(figures_dir / "architecture_terminal_drop.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    terminal_32 = [100.0 * row["terminal_t32"]["mean"] for row in models]
    early_32 = [100.0 * row["early_t32"]["mean"] for row in models]
    width = 0.36
    axis.bar(x_values - width / 2, terminal_32, width, label="Terminal")
    axis.bar(x_values + width / 2, early_32, width, label="Early-8")
    axis.set_xticks(x_values, labels, rotation=18, ha="right")
    axis.set_ylabel("Validation accuracy at T=32 (%)")
    axis.set_title("Terminal versus prefix readout")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    figure.savefig(figures_dir / "terminal_vs_early_t32.png", dpi=220)
    plt.close(figure)

    report = [
        "# Modern Iterative Architecture Panel",
        "",
        f"**Attention classification:** {attention['classification']}.",
        "",
        "E16 remains the locked headline causal horizon result. E17 adds a "
        "Universal-Transformer-style shared-attention system as architectural "
        "breadth; the spatial-token and pooled-feature systems are compared "
        "as tested systems rather than as a controlled isolation of attention.",
        "",
        "## Final architecture panel",
        "",
        "| Model | Family | T_train | Terminal T=8 | Terminal T=32 | "
        "Early T=32 | Best steps | Classification |",
        "|---|---|---:|---:|---:|---:|:---:|---|",
    ]
    for row in models:
        report.append(
            f"| {row['model_condition']} | {row['family']} | "
            f"{row['training_horizon']} | "
            f"{100*row['terminal_t8']['mean']:.2f}% +/- "
            f"{100*row['terminal_t8']['sample_std']:.2f}% | "
            f"{100*row['terminal_t32']['mean']:.2f}% +/- "
            f"{100*row['terminal_t32']['sample_std']:.2f}% | "
            f"{100*row['early_t32']['mean']:.2f}% +/- "
            f"{100*row['early_t32']['sample_std']:.2f}% | "
            f"{'/'.join(str(value) for value in row['best_step_t32_values'])} | "
            f"{row['classification']} |"
        )
    if attention["classification"] == "endpoint_invalidity":
        narrative = (
            "Endpoint degradation under excess test-time recurrence appears "
            "in both the residual SwiGLU and shared-attention weight-tied "
            "recursive systems tested here."
        )
    elif attention["classification"] == "stable":
        narrative = (
            "The strong horizon-mismatch effect in recursive SwiGLU is "
            "architecture-dependent in this panel; the tested shared-attention "
            "system remains more endpoint-stable."
        )
    else:
        narrative = (
            "The recursive SwiGLU effect is strong and replicated, while the "
            "tested shared-attention system shows a weaker or mixed horizon "
            "response."
        )
    report.extend(
        [
            "",
            "## Rebuttal-ready summary",
            "",
            narrative,
            "",
            "All values are individual-seed-preserving means +/- sample "
            "standard deviations across seeds 101--103. No significance test "
            "is used. GRACE remains secondary and was not tuned.",
            "",
            "## Phase status",
            "",
        ]
    )
    for phase, status in phases.items():
        report.append(f"- `{phase}`: {status}.")
    report.extend(
        [
            "",
            "## Protocol deviations",
            "",
            "- Git commit unavailable; exact source and artifact hashes are "
            "recorded. No scientific protocol deviation occurred.",
            "",
        ]
    )
    write_text_new(PANEL_DIR / "final_report.md", "\n".join(report))
    print("Final modern iterative panel aggregation complete.")
    return final


def print_status() -> None:
    paths = {
        "frozen_protocol": PROTOCOL_DIR / "PANEL_PROTOCOL.md",
        "spatial_cache": PANEL_DIR / "spatial_cache" / "cache_metadata.json",
        "smoke_gate": OUTPUT_DIRS["t8"] / "smoke" / "engineering_gate.json",
        "attention_t8_summary": OUTPUT_DIRS["t8"] / "attention_summary.json",
        "control_trigger": OUTPUT_DIRS["t8"] / "matched_control_trigger.json",
        "attention_t32_summary": OUTPUT_DIRS["t32"] / "attention_summary.json",
        "final_panel": PANEL_DIR / "final_summary.json",
    }
    for name, path in paths.items():
        print(f"{name}: {'present' if path.exists() else 'missing'} | {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "verify",
            "features",
            "smoke",
            "train",
            "evaluate",
            "probe",
            "aggregate",
            "panel",
            "status",
        ),
    )
    parser.add_argument("--variant", choices=("t8", "t32"), default="t8")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "status":
        print_status()
        return
    config, config_sha256 = load_config(args.variant)
    output = prepare_variant_directory(args.variant, config, config_sha256)
    environment_path = output / "environment.json"
    if not environment_path.exists():
        write_json_new(environment_path, environment_payload(config_sha256))
    if args.stage == "verify":
        print(
            f"Frozen verification PASS | variant={args.variant} | "
            f"config={config_sha256} | source={sha256_file(SOURCE_PATH)}"
        )
        return
    if args.stage == "features":
        create_spatial_cache(config, config_sha256)
        return
    if args.stage == "smoke":
        if args.variant != "t8":
            raise SystemExit("The frozen engineering smoke gate is defined for t8.")
        create_spatial_cache(config, config_sha256)
        run_smoke(config, config_sha256)
        return
    if args.stage in {"train", "evaluate", "probe"}:
        if args.seed is None:
            raise SystemExit(f"--seed is required for {args.stage}.")
        if args.stage == "train":
            train_scientific_seed(
                args.variant, args.seed, config, config_sha256
            )
        elif args.stage == "evaluate":
            evaluate_seed(args.variant, args.seed, config, config_sha256)
        else:
            run_probe(args.variant, args.seed, config, config_sha256)
        return
    if args.stage == "aggregate":
        aggregate_attention(args.variant, config, config_sha256)
        return
    if args.stage == "panel":
        if args.variant != "t8":
            raise SystemExit("Panel aggregation is variant-independent; use t8.")
        panel_aggregate()
        return


if __name__ == "__main__":
    main()
