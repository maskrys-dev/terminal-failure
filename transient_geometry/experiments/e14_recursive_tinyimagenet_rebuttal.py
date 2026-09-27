"""
E14: Tiny ImageNet-200 + frozen ConvNeXt-Tiny + a trained weight-tied
recursive SwiGLU classifier.

The scientific configuration is loaded from:
  results/rebuttal_recursive_tinyimagenet/config_frozen.json

Run the complete pre-registered workflow:
  .venv/Scripts/python.exe -u \
    transient_geometry/experiments/e14_recursive_tinyimagenet_rebuttal.py all

The workflow stops after seed 0 when the predeclared continuation gate fails.
It never changes the gain grid, architecture, readout thresholds, or probe
regularisation in response to scientific results.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

_MPL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "tmp" / "matplotlib"
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
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import models


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_VARIANT = os.environ.get("TERMINAL_FAILURE_RECURSIVE_VARIANT", "e14")
VARIANT_OUTPUT_DIRS = {
    "e14": "rebuttal_recursive_tinyimagenet",
    "e15": "rebuttal_recursive_tinyimagenet_aligned_t32",
    "e16_t8": "rebuttal_recursive_tinyimagenet_e16_ttrain8",
    "e16_t32": "rebuttal_recursive_tinyimagenet_e16_ttrain32",
}
if EXPERIMENT_VARIANT not in VARIANT_OUTPUT_DIRS:
    raise ValueError(
        f"Unknown TERMINAL_FAILURE_RECURSIVE_VARIANT={EXPERIMENT_VARIANT!r}; "
        f"expected one of {sorted(VARIANT_OUTPUT_DIRS)}"
    )
OUT_DIR = ROOT / "results" / VARIANT_OUTPUT_DIRS[EXPERIMENT_VARIANT]
CONFIG_PATH = OUT_DIR / "config_frozen.json"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Frozen configuration is missing: {CONFIG_PATH}")

CONFIG: dict[str, Any] = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
CONFIG_SHA256 = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
DATA_ROOT = ROOT / "data" / "tiny-imagenet-200"
feature_cache_path = CONFIG["backbone"].get("feature_cache_path")
FEATURE_CACHE = (
    ROOT / feature_cache_path if feature_cache_path else OUT_DIR / "feature_cache.pt"
)
ENV_PATH = OUT_DIR / "environment.json"
SMOKE_PATH = OUT_DIR / "smoke_metrics.json"
GATE_PATH = OUT_DIR / "gate_decision.json"

ARCH = CONFIG["architecture"]
TRAIN = CONFIG["training"]
EVAL = CONFIG["evaluation"]
READOUT = CONFIG["readouts"]
PROBES = CONFIG["probes"]
BENCH = CONFIG["benchmark"]
SMOKE = CONFIG["smoke_test"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAIN_GRID = [float(x) for x in EVAL["gain_grid"]]
HEADLINE_GAINS = [float(x) for x in EVAL["headline_gains"]]
DENSE_PROBE_GAINS = [float(x) for x in PROBES["dense_timestep_gains"]]
METHODS = ("terminal", "early", "uniform", "fixed_profile", "adaptive", "grace")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

torch.set_float32_matmul_precision("high")


def gain_key(gain: float) -> str:
    return f"{float(gain):.2f}"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def environment_payload() -> dict[str, Any]:
    archive = ROOT / "data" / "tiny-imagenet-200.zip"
    gpu_name = None
    gpu_memory = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_memory = int(props.total_memory)
    return {
        "config_sha256": CONFIG_SHA256,
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
        "gpu_total_bytes": gpu_memory,
        "training_precision": precision_name(),
        "tiny_imagenet_archive_sha256": sha256_file(archive) if archive.exists() else None,
    }


class TinyImageNetSplit(Dataset):
    """Read the standard Tiny ImageNet train or labelled validation split."""

    def __init__(self, root: Path, split: str, transform=None):
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")
        self.root = root
        self.split = split
        self.transform = transform
        wnid_path = root / "wnids.txt"
        if not wnid_path.exists():
            raise FileNotFoundError(f"Missing {wnid_path}")
        self.wnids = [line.strip() for line in wnid_path.read_text().splitlines() if line.strip()]
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
        self.samples: list[tuple[Path, int]] = []

        if split == "train":
            for wnid in self.wnids:
                image_dir = root / "train" / wnid / "images"
                for path in sorted(image_dir.glob("*.JPEG")):
                    self.samples.append((path, self.class_to_idx[wnid]))
        else:
            annotations = root / "val" / "val_annotations.txt"
            image_dir = root / "val" / "images"
            if not annotations.exists():
                raise FileNotFoundError(f"Missing {annotations}")
            for line in annotations.read_text().splitlines():
                fields = line.split("\t")
                if len(fields) < 2:
                    continue
                image_name, wnid = fields[:2]
                self.samples.append((image_dir / image_name, self.class_to_idx[wnid]))

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


def validate_dataset() -> dict[str, Any]:
    train_ds = TinyImageNetSplit(DATA_ROOT, "train")
    val_ds = TinyImageNetSplit(DATA_ROOT, "val")
    train_counts = Counter(train_ds.targets)
    val_counts = Counter(val_ds.targets)
    missing = [str(path) for path, _ in train_ds.samples + val_ds.samples if not path.exists()]
    result = {
        "root": str(DATA_ROOT.relative_to(ROOT)),
        "classes": len(train_ds.wnids),
        "train_examples": len(train_ds),
        "validation_examples": len(val_ds),
        "train_class_counts": sorted(train_counts.values()),
        "validation_class_counts": sorted(val_counts.values()),
        "missing_images": len(missing),
        "pass": (
            len(train_ds.wnids) == 200
            and len(train_ds) == 100_000
            and len(val_ds) == 10_000
            and set(train_counts.values()) == {500}
            and set(val_counts.values()) == {50}
            and not missing
        ),
    }
    atomic_json(OUT_DIR / "dataset_validation.json", result)
    if not result["pass"]:
        raise RuntimeError(f"Tiny ImageNet validation failed: {result}")
    print(
        "Dataset valid: "
        f"{result['classes']} classes, {result['train_examples']} train, "
        f"{result['validation_examples']} validation"
    )
    return result


def build_feature_extractor() -> tuple[nn.Module, Any]:
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    model = models.convnext_tiny(weights=weights)
    model.classifier[-1] = nn.Identity()
    model.requires_grad_(False)
    model.eval().to(DEVICE)
    return model, weights.transforms()


@torch.inference_mode()
def extract_one_split(
    dataset: Dataset,
    model: nn.Module,
    split_name: str,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    started = time.time()
    for batch_index, (images, target) in enumerate(loader, start=1):
        images = images.to(DEVICE, non_blocking=True)
        with autocast_context():
            batch_features = model(images)
        features.append(batch_features.float().cpu())
        labels.append(target.long().cpu())
        if batch_index % 50 == 0 or batch_index == len(loader):
            completed = min(batch_index * batch_size, len(dataset))
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"  {split_name}: {completed}/{len(dataset)} "
                f"({completed / elapsed:.1f} images/s)"
            )
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def load_or_extract_features(force: bool = False) -> dict[str, Any]:
    if FEATURE_CACHE.exists() and not force:
        print(f"Loading feature cache: {FEATURE_CACHE}")
        cache = torch.load(FEATURE_CACHE, map_location="cpu", weights_only=False)
        validate_feature_cache(cache)
        return cache

    validate_dataset()
    print("Building ImageNet-pretrained frozen ConvNeXt-Tiny...")
    model, transform = build_feature_extractor()
    train_ds = TinyImageNetSplit(DATA_ROOT, "train", transform=transform)
    val_ds = TinyImageNetSplit(DATA_ROOT, "val", transform=transform)
    batch_size = int(CONFIG["backbone"]["feature_extraction_batch_size"])
    print(f"Extracting features with batch size {batch_size} and {precision_name()} autocast...")
    train_features, train_labels = extract_one_split(train_ds, model, "train", batch_size)
    val_features, val_labels = extract_one_split(val_ds, model, "validation", batch_size)

    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    train_features = ((train_features - mean) / std).contiguous()
    val_features = ((val_features - mean) / std).contiguous()
    cache = {
        "config_sha256": CONFIG_SHA256,
        "backbone": "torchvision.models.convnext_tiny",
        "weights": str(models.ConvNeXt_Tiny_Weights.DEFAULT),
        "transform": repr(transform),
        "precision": precision_name(),
        "train_features": train_features,
        "train_labels": train_labels,
        "validation_features": val_features,
        "validation_labels": val_labels,
        "feature_mean": mean,
        "feature_std": std,
    }
    validate_feature_cache(cache)
    torch.save(cache, FEATURE_CACHE)
    print(
        f"Saved {FEATURE_CACHE} "
        f"({FEATURE_CACHE.stat().st_size / (1024 ** 2):.1f} MiB)"
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache


def validate_feature_cache(cache: dict[str, Any]) -> None:
    expected = {
        "train_features": (100_000, int(ARCH["input_dimension"])),
        "validation_features": (10_000, int(ARCH["input_dimension"])),
        "train_labels": (100_000,),
        "validation_labels": (10_000,),
    }
    for key, shape in expected.items():
        if key not in cache or tuple(cache[key].shape) != shape:
            raise RuntimeError(f"Feature cache {key} has shape {cache.get(key, None)}; expected {shape}")
    for key in ("train_features", "validation_features"):
        if not torch.isfinite(cache[key]).all():
            raise RuntimeError(f"Feature cache contains non-finite values in {key}")
    for key in ("train_labels", "validation_labels"):
        labels = cache[key]
        if labels.min().item() != 0 or labels.max().item() != 199:
            raise RuntimeError(f"Unexpected label range in {key}")


class RecursiveSwiGLUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = int(ARCH["input_dimension"])
        state_dim = int(ARCH["state_dimension"])
        intermediate_dim = int(ARCH["intermediate_dimension"])
        self.training_horizon = int(ARCH["training_horizon"])
        self.input_projection = nn.Linear(input_dim, state_dim)
        self.block_norm = nn.LayerNorm(state_dim)
        self.gate_projection = nn.Linear(state_dim, intermediate_dim)
        self.value_projection = nn.Linear(state_dim, intermediate_dim)
        self.output_projection = nn.Linear(intermediate_dim, state_dim)
        self.dropout = nn.Dropout(float(ARCH["dropout"]))
        self.head_norm = nn.LayerNorm(state_dim)
        self.classifier = nn.Linear(state_dim, int(CONFIG["dataset"]["classes"]))

    def step(self, state: torch.Tensor, gain: float) -> torch.Tensor:
        normalised = self.block_norm(state)
        hidden = F.silu(self.gate_projection(normalised)) * self.value_projection(normalised)
        update = self.dropout(self.output_projection(hidden))
        return state + (float(gain) / self.training_horizon) * update

    def classify(self, state: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.head_norm(state))

    def trajectory(
        self,
        features: torch.Tensor,
        steps: int,
        gain: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state0 = self.input_projection(features)
        state = state0
        states = []
        for _ in range(int(steps)):
            state = self.step(state, gain)
            states.append(state)
        return state0, torch.stack(states, dim=1)

    def forward(self, features: torch.Tensor, steps: int, gain: float) -> torch.Tensor:
        state = self.input_projection(features)
        for _ in range(int(steps)):
            state = self.step(state, gain)
        return self.classify(state)


def stratified_subset(labels: torch.Tensor, size: int, seed: int) -> np.ndarray:
    y = labels.cpu().numpy()
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=seed)
    indices, _ = next(splitter.split(np.zeros(len(y)), y))
    return indices.astype(np.int64)


def tensor_loader(
    features: torch.Tensor,
    labels: torch.Tensor | None,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(features) if labels is None else TensorDataset(features, labels)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )


@torch.inference_mode()
def terminal_accuracy(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    steps: int = 8,
    gain: float = 1.0,
) -> tuple[float, float]:
    model.eval()
    loader = tensor_loader(features, labels, batch_size, False, 0)
    correct = 0
    loss_sum = 0.0
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        with autocast_context():
            logits = model(xb, steps, gain)
        loss_sum += float(F.cross_entropy(logits.float(), yb, reduction="sum").item())
        correct += int((logits.argmax(dim=-1) == yb).sum().item())
    return correct / len(labels), loss_sum / len(labels)


def lr_multiplier(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max((step + 1) / max(warmup_steps, 1), 1e-8)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_model(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    seed: int,
    epochs: int,
    checkpoint_path: Path | None,
    last_path: Path | None,
    resume: bool,
) -> tuple[RecursiveSwiGLUClassifier, dict[str, Any]]:
    seed_everything(seed)
    model = RecursiveSwiGLUClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(TRAIN["learning_rate"]),
        weight_decay=float(TRAIN["weight_decay"]),
    )
    batch_size = int(TRAIN["batch_size"])
    train_loader = tensor_loader(train_features, train_labels, batch_size, True, seed)
    total_steps = epochs * len(train_loader)
    warmup_steps = int(TRAIN["warmup_epochs"]) * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_multiplier(step, warmup_steps, total_steps),
    )
    fp16_scaling = amp_dtype() == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=fp16_scaling)
    start_epoch = 0
    best_accuracy = -math.inf
    best_epoch = -1
    history: list[dict[str, Any]] = []

    if resume and last_path is not None and last_path.exists():
        payload = torch.load(last_path, map_location=DEVICE, weights_only=False)
        if payload.get("config_sha256") != CONFIG_SHA256 or int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"Resume checkpoint does not match frozen config or seed: {last_path}")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        if fp16_scaling and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        start_epoch = int(payload["epoch"]) + 1
        best_accuracy = float(payload["best_accuracy"])
        best_epoch = int(payload["best_epoch"])
        history = list(payload["history"])
        print(f"Resuming seed {seed} at epoch {start_epoch + 1}/{epochs}")

    started = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        seen = 0
        nonfinite = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                logits = model(
                    xb,
                    int(ARCH["training_horizon"]),
                    float(ARCH["training_gain"]),
                )
                loss = F.cross_entropy(
                    logits,
                    yb,
                    label_smoothing=float(TRAIN["label_smoothing"]),
                )
            if not torch.isfinite(loss):
                nonfinite += 1
                raise RuntimeError(f"Non-finite training loss at seed {seed}, epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(TRAIN["gradient_clip_norm"]),
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient norm at seed {seed}, epoch {epoch + 1}")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            batch_n = len(yb)
            epoch_loss += float(loss.detach().item()) * batch_n
            epoch_correct += int((logits.detach().argmax(dim=-1) == yb).sum().item())
            seen += batch_n

        val_accuracy, val_loss = terminal_accuracy(
            model,
            val_features,
            val_labels,
            batch_size,
            int(ARCH["training_horizon"]),
            float(ARCH["training_gain"]),
        )
        row = {
            "epoch": epoch + 1,
            "training_loss": epoch_loss / seen,
            "training_accuracy": epoch_correct / seen,
            "validation_loss": val_loss,
            "validation_terminal_accuracy": val_accuracy,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "nonfinite_batches": nonfinite,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"  seed={seed} epoch={epoch + 1:02d}/{epochs} "
            f"train_loss={row['training_loss']:.4f} "
            f"train_acc={row['training_accuracy']:.4f} "
            f"val_acc={val_accuracy:.4f} val_loss={val_loss:.4f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch + 1
            if checkpoint_path is not None:
                torch.save(
                    {
                        "config_sha256": CONFIG_SHA256,
                        "seed": seed,
                        "epoch": best_epoch,
                        "validation_terminal_accuracy": best_accuracy,
                        "model": model.state_dict(),
                        "history": history,
                    },
                    checkpoint_path,
                )

        if last_path is not None:
            torch.save(
                {
                    "config_sha256": CONFIG_SHA256,
                    "seed": seed,
                    "epoch": epoch,
                    "best_accuracy": best_accuracy,
                    "best_epoch": best_epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if fp16_scaling else None,
                    "history": history,
                },
                last_path,
            )

    if checkpoint_path is not None:
        best = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(best["model"])
    metrics = {
        "config_sha256": CONFIG_SHA256,
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_validation_terminal_accuracy": best_accuracy,
        "history": history,
        "elapsed_seconds": time.time() - started,
    }
    return model, metrics


def run_smoke(cache: dict[str, Any], force: bool = False) -> dict[str, Any]:
    if SMOKE_PATH.exists() and not force:
        result = json.loads(SMOKE_PATH.read_text(encoding="utf-8"))
        if result.get("config_sha256") == CONFIG_SHA256 and result.get("pass"):
            print("Using passing smoke test already on disk.")
            return result

    train_idx = stratified_subset(
        cache["train_labels"],
        int(SMOKE["training_examples"]),
        int(PROBES["training_subset_seed"]),
    )
    val_idx = stratified_subset(
        cache["validation_labels"],
        int(SMOKE["validation_examples"]),
        int(PROBES["training_subset_seed"]),
    )
    print(
        f"Engineering smoke test: {len(train_idx)} train, {len(val_idx)} validation, "
        f"{SMOKE['epochs']} epochs"
    )
    model, metrics = train_model(
        cache["train_features"][train_idx],
        cache["train_labels"][train_idx],
        cache["validation_features"][val_idx],
        cache["validation_labels"][val_idx],
        seed=0,
        epochs=int(SMOKE["epochs"]),
        checkpoint_path=None,
        last_path=None,
        resume=False,
    )
    model.eval()
    sample = cache["validation_features"][val_idx[: min(512, len(val_idx))]].to(DEVICE)
    with torch.inference_mode(), autocast_context():
        state0, trajectory = model.trajectory(
            sample,
            int(EVAL["trajectory_horizon"]),
            max(GAIN_GRID),
        )
        logits = model.classify(trajectory)
        smoke_loss_decreased = (
            metrics["history"][-1]["training_loss"] < metrics["history"][0]["training_loss"]
        )
        finite = bool(torch.isfinite(state0).all() and torch.isfinite(trajectory).all())
        exposed_shape = list(trajectory.shape)
        logits_shape = list(logits.shape)

    result = {
        **metrics,
        "validation_threshold": float(SMOKE["minimum_validation_accuracy"]),
        "loss_decreased": bool(smoke_loss_decreased),
        "finite_states_at_max_gain": finite,
        "trajectory_shape": exposed_shape,
        "logits_shape": logits_shape,
        "pass": bool(
            metrics["best_validation_terminal_accuracy"]
            > float(SMOKE["minimum_validation_accuracy"])
            and smoke_loss_decreased
            and finite
            and exposed_shape[1:] == [
                int(EVAL["trajectory_horizon"]),
                int(ARCH["state_dimension"]),
            ]
            and logits_shape[1:] == [
                int(EVAL["trajectory_horizon"]),
                int(CONFIG["dataset"]["classes"]),
            ]
        ),
    }
    atomic_json(SMOKE_PATH, result)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not result["pass"]:
        raise RuntimeError(f"Recursive engineering smoke test failed: {result}")
    print(
        "Smoke test passed: "
        f"best validation accuracy={result['best_validation_terminal_accuracy']:.4f}"
    )
    return result


def load_model_checkpoint(seed: int) -> RecursiveSwiGLUClassifier:
    path = OUT_DIR / f"seed_{seed}_checkpoint.pt"
    payload = torch.load(path, map_location=DEVICE, weights_only=False)
    if payload.get("config_sha256") != CONFIG_SHA256 or int(payload.get("seed", -1)) != seed:
        raise RuntimeError(f"Checkpoint does not match frozen config: {path}")
    model = RecursiveSwiGLUClassifier().to(DEVICE)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def train_seed(cache: dict[str, Any], seed: int, resume: bool = True) -> dict[str, Any]:
    checkpoint = OUT_DIR / f"seed_{seed}_checkpoint.pt"
    training_path = OUT_DIR / f"seed_{seed}_training.json"
    if checkpoint.exists() and training_path.exists():
        existing = json.loads(training_path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") == CONFIG_SHA256:
            print(f"Using trained seed {seed} checkpoint already on disk.")
            return existing
    last = OUT_DIR / f"seed_{seed}_last.pt"
    print(f"Training scientific seed {seed}...")
    _, metrics = train_model(
        cache["train_features"],
        cache["train_labels"],
        cache["validation_features"],
        cache["validation_labels"],
        seed=seed,
        epochs=int(TRAIN["epochs"]),
        checkpoint_path=checkpoint,
        last_path=last,
        resume=resume,
    )
    atomic_json(training_path, metrics)
    return metrics


def grace_weights(trajectory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    norms = trajectory.float().norm(dim=-1)
    ratio = norms / norms[:, :1].clamp_min(1e-12)
    raw = torch.exp(
        -float(READOUT["grace"]["alpha"])
        * torch.clamp(ratio - float(READOUT["grace"]["norm_ratio_threshold"]), min=0.0)
    )
    weights = raw / raw.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return weights, ratio


def aggregate_states(
    trajectory: torch.Tensor,
    fixed_profile: torch.Tensor,
) -> dict[str, torch.Tensor]:
    weights, ratio = grace_weights(trajectory)
    crossed = ratio > float(READOUT["adaptive_stopping"]["norm_ratio_threshold"])
    first = crossed.long().argmax(dim=1)
    never = ~crossed.any(dim=1)
    first[never] = trajectory.shape[1] - 1
    adaptive = trajectory[torch.arange(len(trajectory), device=trajectory.device), first]
    early_k = int(EVAL["early_window"])
    profile = fixed_profile.to(device=trajectory.device, dtype=trajectory.dtype)
    return {
        "terminal": trajectory[:, -1],
        "early": trajectory[:, :early_k].mean(dim=1),
        "uniform": trajectory.mean(dim=1),
        "fixed_profile": torch.einsum("t,btd->bd", profile, trajectory),
        "adaptive": adaptive,
        "grace": torch.einsum("bt,btd->bd", weights.to(trajectory.dtype), trajectory),
    }


@torch.inference_mode()
def compute_fixed_profile(
    model: RecursiveSwiGLUClassifier,
    train_features: torch.Tensor,
) -> torch.Tensor:
    loader = tensor_loader(
        train_features,
        labels=None,
        batch_size=int(TRAIN["batch_size"]),
        shuffle=False,
        seed=0,
    )
    weight_sum = torch.zeros(int(EVAL["trajectory_horizon"]), dtype=torch.float64)
    count = 0
    reference_gain = float(READOUT["fixed_grace_profile"]["reference_gain"])
    print(f"Computing frozen GRACE profile on full training split at gain={reference_gain:.2f}...")
    for batch_index, (xb,) in enumerate(loader, start=1):
        xb = xb.to(DEVICE, non_blocking=True)
        with autocast_context():
            _, trajectory = model.trajectory(xb, int(EVAL["trajectory_horizon"]), reference_gain)
        weights, _ = grace_weights(trajectory)
        weight_sum += weights.double().sum(dim=0).cpu()
        count += len(xb)
        if batch_index % 50 == 0 or batch_index == len(loader):
            print(f"  fixed profile: {count}/{len(train_features)}")
    profile = (weight_sum / count).float()
    profile /= profile.sum()
    if not torch.isfinite(profile).all() or abs(float(profile.sum()) - 1.0) > 1e-5:
        raise RuntimeError("Invalid fixed GRACE profile")
    return profile


@torch.inference_mode()
def evaluate_shared_head(
    model: RecursiveSwiGLUClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    gain: float,
    fixed_profile: torch.Tensor,
) -> dict[str, Any]:
    loader = tensor_loader(
        features,
        labels,
        batch_size=int(TRAIN["batch_size"]),
        shuffle=False,
        seed=0,
    )
    horizon = int(EVAL["trajectory_horizon"])
    correct = torch.zeros(horizon, dtype=torch.float64)
    ce_sum = torch.zeros(horizon, dtype=torch.float64)
    norm_sum = torch.zeros(horizon, dtype=torch.float64)
    residual_sum = torch.zeros(horizon, dtype=torch.float64)
    entropy_sum = torch.zeros(horizon, dtype=torch.float64)
    change_sum = torch.zeros(horizon, dtype=torch.float64)
    readout_correct = {method: 0 for method in METHODS}
    readout_ce_sum = {method: 0.0 for method in METHODS}
    nonfinite_states = 0
    nonfinite_logits = 0
    seen = 0

    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        with autocast_context():
            state0, trajectory = model.trajectory(xb, horizon, gain)
            logits = model.classify(trajectory)
            representations = aggregate_states(trajectory, fixed_profile)
            readout_logits = {
                method: model.classify(representations[method]) for method in METHODS
            }

        logits_float = logits.float()
        predictions = logits_float.argmax(dim=-1)
        correct += (predictions == yb[:, None]).sum(dim=0).double().cpu()
        per_example_ce = F.cross_entropy(
            logits_float.reshape(-1, logits_float.shape[-1]),
            yb[:, None].expand(-1, horizon).reshape(-1),
            reduction="none",
        ).reshape(len(yb), horizon)
        ce_sum += per_example_ce.double().sum(dim=0).cpu()
        state_norms = trajectory.float().norm(dim=-1)
        norm_sum += state_norms.double().sum(dim=0).cpu()
        previous = torch.cat([state0[:, None], trajectory[:, :-1]], dim=1).float()
        residuals = (trajectory.float() - previous).norm(dim=-1) / previous.norm(
            dim=-1
        ).clamp_min(1e-12)
        residual_sum += residuals.double().sum(dim=0).cpu()
        probabilities = torch.softmax(logits_float, dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        entropy_sum += entropy.double().sum(dim=0).cpu()
        changes = torch.zeros_like(predictions, dtype=torch.float32)
        changes[:, 1:] = (predictions[:, 1:] != predictions[:, :-1]).float()
        change_sum += changes.double().sum(dim=0).cpu()
        nonfinite_states += int((~torch.isfinite(trajectory)).sum().item())
        nonfinite_logits += int((~torch.isfinite(logits_float)).sum().item())

        for method, method_logits in readout_logits.items():
            method_logits = method_logits.float()
            readout_correct[method] += int((method_logits.argmax(dim=-1) == yb).sum().item())
            readout_ce_sum[method] += float(
                F.cross_entropy(method_logits, yb, reduction="sum").item()
            )
        seen += len(yb)

    return {
        "gain": float(gain),
        "examples": seen,
        "timestep": {
            "accuracy": (correct / seen).tolist(),
            "cross_entropy": (ce_sum / seen).tolist(),
            "mean_state_norm": (norm_sum / seen).tolist(),
            "mean_relative_update_residual": (residual_sum / seen).tolist(),
            "mean_prediction_entropy": (entropy_sum / seen).tolist(),
            "class_change_fraction": (change_sum / seen).tolist(),
        },
        "readout_accuracy": {
            method: readout_correct[method] / seen for method in METHODS
        },
        "readout_cross_entropy": {
            method: readout_ce_sum[method] / seen for method in METHODS
        },
        "nonfinite_state_values": nonfinite_states,
        "nonfinite_logit_values": nonfinite_logits,
    }


@torch.inference_mode()
def extract_probe_representations(
    model: RecursiveSwiGLUClassifier,
    features: torch.Tensor,
    gain: float,
    fixed_profile: torch.Tensor,
    include_trajectory: bool,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    loader = tensor_loader(
        features,
        labels=None,
        batch_size=int(TRAIN["batch_size"]),
        shuffle=False,
        seed=0,
    )
    representation_chunks: dict[str, list[torch.Tensor]] = {method: [] for method in METHODS}
    trajectory_chunks: list[torch.Tensor] = []
    for xb_tuple in loader:
        xb = xb_tuple[0].to(DEVICE, non_blocking=True)
        with autocast_context():
            _, trajectory = model.trajectory(
                xb,
                int(EVAL["trajectory_horizon"]),
                gain,
            )
            representations = aggregate_states(trajectory, fixed_profile)
        for method in METHODS:
            representation_chunks[method].append(representations[method].float().cpu())
        if include_trajectory:
            trajectory_chunks.append(trajectory.to(torch.float16).cpu())
    representations_out = {
        method: torch.cat(chunks, dim=0) for method, chunks in representation_chunks.items()
    }
    trajectory_out = torch.cat(trajectory_chunks, dim=0) if include_trajectory else None
    return representations_out, trajectory_out


def fit_ridge(
    train_representation: torch.Tensor,
    train_labels: torch.Tensor,
    validation_representation: torch.Tensor,
    validation_labels: torch.Tensor,
) -> float:
    classifier = RidgeClassifier(alpha=float(PROBES["alpha"]))
    classifier.fit(train_representation.numpy(), train_labels.numpy())
    return float(classifier.score(validation_representation.numpy(), validation_labels.numpy()))


def grace_period(curve: Iterable[float], threshold: float) -> int:
    length = 0
    for step, value in enumerate(curve, start=1):
        if float(value) >= float(threshold):
            length = step
        else:
            break
    return length


def run_probe_audit(
    model: RecursiveSwiGLUClassifier,
    cache: dict[str, Any],
    fixed_profile: torch.Tensor,
) -> dict[str, Any]:
    subset_path = OUT_DIR / "probe_subset_indices.npy"
    if subset_path.exists():
        probe_indices = np.load(subset_path)
    else:
        probe_indices = stratified_subset(
            cache["train_labels"],
            int(PROBES["training_subset_size"]),
            int(PROBES["training_subset_seed"]),
        )
        np.save(subset_path, probe_indices)

    probe_train_features = cache["train_features"][probe_indices]
    probe_train_labels = cache["train_labels"][probe_indices]
    validation_features = cache["validation_features"]
    validation_labels = cache["validation_labels"]
    readout_scores: dict[str, Any] = {}
    timestep_scores: dict[str, Any] = {}

    for gain in GAIN_GRID:
        key = gain_key(gain)
        dense = any(abs(gain - x) < 1e-9 for x in DENSE_PROBE_GAINS)
        print(f"  probe gain={gain:.2f} dense_timestep={dense}")
        train_reps, train_traj = extract_probe_representations(
            model,
            probe_train_features,
            gain,
            fixed_profile,
            include_trajectory=dense,
        )
        validation_reps, validation_traj = extract_probe_representations(
            model,
            validation_features,
            gain,
            fixed_profile,
            include_trajectory=dense,
        )
        readout_scores[key] = {}
        for method in METHODS:
            score = fit_ridge(
                train_reps[method],
                probe_train_labels,
                validation_reps[method],
                validation_labels,
            )
            readout_scores[key][method] = score
            print(f"    {method}: {score:.4f}")

        if dense:
            if train_traj is None or validation_traj is None:
                raise AssertionError("Dense probe trajectory was not retained")
            curve = []
            for timestep in range(int(EVAL["trajectory_horizon"])):
                score = fit_ridge(
                    train_traj[:, timestep].float(),
                    probe_train_labels,
                    validation_traj[:, timestep].float(),
                    validation_labels,
                )
                curve.append(score)
                print(f"    timestep={timestep + 1:02d}: {score:.4f}")
            timestep_scores[key] = {
                "accuracy": curve,
                "grace_period": {
                    f"{float(threshold):.2f}": grace_period(curve, float(threshold))
                    for threshold in PROBES["grace_period_accuracy_thresholds"]
                },
            }

        del train_reps, validation_reps, train_traj, validation_traj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "training_subset_size": len(probe_indices),
        "training_subset_seed": int(PROBES["training_subset_seed"]),
        "ridge_alpha": float(PROBES["alpha"]),
        "readout_accuracy": readout_scores,
        "timestep": timestep_scores,
    }


def direct_feature_probe(cache: dict[str, Any]) -> dict[str, Any]:
    path = OUT_DIR / "direct_feature_probe.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    subset_path = OUT_DIR / "probe_subset_indices.npy"
    if subset_path.exists():
        indices = np.load(subset_path)
    else:
        indices = stratified_subset(
            cache["train_labels"],
            int(PROBES["training_subset_size"]),
            int(PROBES["training_subset_seed"]),
        )
        np.save(subset_path, indices)
    print("Fitting direct linear probe on frozen standardised ConvNeXt features...")
    score = fit_ridge(
        cache["train_features"][indices],
        cache["train_labels"][indices],
        cache["validation_features"],
        cache["validation_labels"],
    )
    result = {
        "training_subset_size": len(indices),
        "ridge_alpha": float(PROBES["alpha"]),
        "validation_accuracy": score,
    }
    atomic_json(path, result)
    print(f"Direct frozen-feature probe accuracy: {score:.4f}")
    return result


def evaluate_seed(
    cache: dict[str, Any],
    seed: int,
    force: bool = False,
) -> dict[str, Any]:
    metrics_path = OUT_DIR / f"seed_{seed}_metrics.json"
    if metrics_path.exists() and not force:
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") == CONFIG_SHA256:
            print(f"Using evaluated seed {seed} metrics already on disk.")
            return existing

    model = load_model_checkpoint(seed)
    fixed_profile = compute_fixed_profile(model, cache["train_features"])
    shared: dict[str, Any] = {}
    for gain in GAIN_GRID:
        print(f"Evaluating shared head at gain={gain:.2f}...")
        result = evaluate_shared_head(
            model,
            cache["validation_features"],
            cache["validation_labels"],
            gain,
            fixed_profile,
        )
        shared[gain_key(gain)] = result
        row = result["readout_accuracy"]
        print(
            f"  terminal={row['terminal']:.4f} early={row['early']:.4f} "
            f"uniform={row['uniform']:.4f} fixed={row['fixed_profile']:.4f} "
            f"adaptive={row['adaptive']:.4f} grace={row['grace']:.4f}"
        )

    print("Running fixed-regularisation representation probes...")
    probes = run_probe_audit(model, cache, fixed_profile)
    metrics = {
        "experiment": CONFIG["experiment"],
        "config_sha256": CONFIG_SHA256,
        "seed": seed,
        "checkpoint": f"seed_{seed}_checkpoint.pt",
        "fixed_grace_profile": fixed_profile.tolist(),
        "shared_head": shared,
        "probes": probes,
        "direct_frozen_feature_probe": direct_feature_probe(cache),
    }
    atomic_json(metrics_path, metrics)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def consecutive_passing_gains(
    gains: list[float],
    gaps: list[float],
    threshold: float,
    count: int,
) -> list[float]:
    for start in range(0, len(gaps) - count + 1):
        if all(gap >= threshold for gap in gaps[start : start + count]):
            return gains[start : start + count]
    return []


def comparison_gate(
    metrics: dict[str, Any],
    comparison: str,
) -> dict[str, Any]:
    if comparison == "shared_head":
        rows = {
            key: metrics["shared_head"][key]["readout_accuracy"]
            for key in [gain_key(x) for x in GAIN_GRID]
        }
    elif comparison == "probes":
        rows = metrics["probes"]["readout_accuracy"]
    else:
        raise ValueError(comparison)

    gaps = [
        rows[gain_key(gain)]["early"] - rows[gain_key(gain)]["terminal"]
        for gain in GAIN_GRID
    ]
    minimum_sequence_gain = float(
        CONFIG["continuation_gate"].get(
            "minimum_gain_for_sequence",
            min(GAIN_GRID),
        )
    )
    eligible_pairs = [
        (gain, gap)
        for gain, gap in zip(GAIN_GRID, gaps)
        if gain >= minimum_sequence_gain
    ]
    eligible_gains = [pair[0] for pair in eligible_pairs]
    eligible_gaps = [pair[1] for pair in eligible_pairs]
    consecutive = consecutive_passing_gains(
        eligible_gains,
        eligible_gaps,
        float(CONFIG["continuation_gate"]["minimum_early_minus_terminal_pp"]) / 100.0,
        int(CONFIG["continuation_gate"]["required_consecutive_gains"]),
    )
    stable_key = gain_key(float(CONFIG["continuation_gate"]["stable_gain"]))
    stable_row = rows[stable_key]
    stable_deficit = max(stable_row.values()) - stable_row["terminal"]
    stable_pass = (
        stable_deficit
        <= float(CONFIG["continuation_gate"]["maximum_stable_terminal_deficit_pp"]) / 100.0
    )
    return {
        "comparison": comparison,
        "early_minus_terminal": {
            gain_key(gain): gaps[index] for index, gain in enumerate(GAIN_GRID)
        },
        "minimum_gain_for_sequence": minimum_sequence_gain,
        "passing_consecutive_gains": consecutive,
        "consecutive_gain_pass": bool(consecutive),
        "stable_terminal_deficit": stable_deficit,
        "stable_pass": stable_pass,
        "pass": bool(consecutive and stable_pass),
    }


def trajectory_structure_pass(metrics: dict[str, Any]) -> dict[str, Any]:
    shared_nonmonotonic = {}
    for gain in GAIN_GRID:
        key = gain_key(gain)
        curve = metrics["shared_head"][key]["timestep"]["accuracy"]
        peak_index = int(np.argmax(curve))
        peak_minus_terminal = float(max(curve) - curve[-1])
        shared_nonmonotonic[key] = {
            "peak_timestep": peak_index + 1,
            "peak_minus_terminal": peak_minus_terminal,
            "pass": bool(peak_index < len(curve) - 1 and peak_minus_terminal >= 0.01),
        }

    probe_timestep = metrics["probes"]["timestep"]
    grace_shortening = {}
    low_key = gain_key(min(DENSE_PROBE_GAINS))
    high_key = gain_key(max(DENSE_PROBE_GAINS))
    for threshold in PROBES["grace_period_accuracy_thresholds"]:
        threshold_key = f"{float(threshold):.2f}"
        low = probe_timestep.get(low_key, {}).get("grace_period", {}).get(threshold_key)
        high = probe_timestep.get(high_key, {}).get("grace_period", {}).get(threshold_key)
        grace_shortening[threshold_key] = {
            "stable": low,
            "stressed": high,
            "pass": bool(low is not None and high is not None and high < low),
        }

    result = {
        "shared_nonmonotonic": shared_nonmonotonic,
        "grace_period_shortening": grace_shortening,
        "pass": bool(
            any(row["pass"] for row in shared_nonmonotonic.values())
            or any(row["pass"] for row in grace_shortening.values())
        ),
    }
    structure_rule = CONFIG["continuation_gate"].get("structure_rule")
    if structure_rule == "gap_increases_or_peak_shifts_earlier":
        minimum_gain = float(
            CONFIG["continuation_gate"].get("structure_minimum_gain", 1.2)
        )
        stress_gains = [gain for gain in GAIN_GRID if gain >= minimum_gain]
        first_gain = stress_gains[0]
        last_gain = stress_gains[-1]
        first_row = metrics["shared_head"][gain_key(first_gain)]
        last_row = metrics["shared_head"][gain_key(last_gain)]
        first_gap = (
            first_row["readout_accuracy"]["early"]
            - first_row["readout_accuracy"]["terminal"]
        )
        last_gap = (
            last_row["readout_accuracy"]["early"]
            - last_row["readout_accuracy"]["terminal"]
        )
        first_peak = (
            int(np.argmax(first_row["timestep"]["accuracy"])) + 1
        )
        last_peak = (
            int(np.argmax(last_row["timestep"]["accuracy"])) + 1
        )
        gap_increases = bool(last_gap > first_gap)
        peak_shifts_earlier = bool(last_peak < first_peak)
        result["aligned_horizon_stress_rule"] = {
            "minimum_gain": minimum_gain,
            "first_gain": first_gain,
            "last_gain": last_gain,
            "first_early_minus_terminal": first_gap,
            "last_early_minus_terminal": last_gap,
            "gap_increases": gap_increases,
            "first_best_timestep": first_peak,
            "last_best_timestep": last_peak,
            "best_timestep_shifts_earlier": peak_shifts_earlier,
        }
        result["pass"] = bool(gap_increases or peak_shifts_earlier)
    return result


def apply_continuation_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    shared_gate = comparison_gate(metrics, "shared_head")
    probe_gate = comparison_gate(metrics, "probes")
    comparison_policy = CONFIG["continuation_gate"].get(
        "comparison_policy",
        "shared_or_probe",
    )
    if comparison_policy == "shared_head_only":
        selected = shared_gate
        primary_gate_pass = shared_gate["pass"]
    elif comparison_policy == "shared_or_probe":
        selected = shared_gate if shared_gate["pass"] else probe_gate
        primary_gate_pass = shared_gate["pass"] or probe_gate["pass"]
    else:
        raise ValueError(f"Unknown continuation-gate comparison policy: {comparison_policy}")
    structure = trajectory_structure_pass(metrics)
    finite = all(
        metrics["shared_head"][gain_key(gain)]["nonfinite_state_values"] == 0
        and metrics["shared_head"][gain_key(gain)]["nonfinite_logit_values"] == 0
        for gain in GAIN_GRID
    )
    stressed = metrics["shared_head"][gain_key(max(GAIN_GRID))]["readout_accuracy"]
    minimum_material_accuracy = float(
        CONFIG["continuation_gate"].get(
            "minimum_material_accuracy",
            EVAL["random_chance_accuracy"],
        )
    )
    above_chance = max(stressed["terminal"], stressed["early"]) > minimum_material_accuracy
    decision = {
        "experiment": CONFIG["experiment"],
        "config_sha256": CONFIG_SHA256,
        "seed": 0,
        "shared_head_gate": shared_gate,
        "probe_gate": probe_gate,
        "comparison_policy": comparison_policy,
        "selected_comparison": selected["comparison"] if selected["pass"] else None,
        "trajectory_structure": structure,
        "all_states_and_logits_finite": finite,
        "minimum_material_accuracy": minimum_material_accuracy,
        "stressed_classification_above_random": above_chance,
        "continue_to_seeds_1_and_2": bool(
            primary_gate_pass
            and structure["pass"]
            and finite
            and above_chance
        ),
    }
    atomic_json(GATE_PATH, decision)
    print(
        "Continuation gate: "
        f"{'PASS' if decision['continue_to_seeds_1_and_2'] else 'STOP'}"
    )
    return decision


def cuda_measure(fn, warmup: int, repetitions: int) -> dict[str, float]:
    if DEVICE.type != "cuda":
        timings = []
        for _ in range(warmup):
            fn()
        for _ in range(repetitions):
            start = time.perf_counter()
            fn()
            timings.append((time.perf_counter() - start) * 1000.0)
        return {"median_ms": statistics.median(timings), "peak_additional_bytes": 0}

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    timings = []
    peaks = []
    for _ in range(repetitions):
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(float(start.elapsed_time(end)))
        peaks.append(max(0, int(torch.cuda.max_memory_allocated() - baseline)))
        del output
    return {
        "median_ms": statistics.median(timings),
        "peak_additional_bytes": int(statistics.median(peaks)),
    }


@torch.inference_mode()
def benchmark_seed0(
    cache: dict[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    path = OUT_DIR / "runtime_memory.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    model = load_model_checkpoint(0)
    batch_size = int(BENCH["batch_size"])
    horizon = int(BENCH["trajectory_horizon"])
    gain = 1.3
    xb = cache["validation_features"][:batch_size].to(DEVICE)
    fixed_profile = torch.tensor(
        json.loads((OUT_DIR / "seed_0_metrics.json").read_text())["fixed_grace_profile"],
        device=DEVICE,
    )
    warmup = int(BENCH["warmup_repetitions"])
    repetitions = int(BENCH["timed_repetitions"])

    with autocast_context():
        _, precomputed = model.trajectory(xb, horizon, gain)
    _, precomputed_ratio = grace_weights(precomputed)

    def terminal_full():
        with autocast_context():
            return model(xb, horizon, gain)

    def early_full():
        with autocast_context():
            _, trajectory = model.trajectory(xb, int(EVAL["early_window"]), gain)
            return model.classify(trajectory.mean(dim=1))

    def uniform_full():
        with autocast_context():
            state = model.input_projection(xb)
            total = torch.zeros_like(state)
            for _ in range(horizon):
                state = model.step(state, gain)
                total = total + state
            return model.classify(total / horizon)

    def grace_streaming_full():
        with autocast_context():
            state = model.input_projection(xb)
            numerator = torch.zeros_like(state)
            denominator = torch.zeros(len(state), 1, device=state.device, dtype=torch.float32)
            base_norm = None
            for _ in range(horizon):
                state = model.step(state, gain)
                norm = state.float().norm(dim=-1, keepdim=True)
                if base_norm is None:
                    base_norm = norm.clamp_min(1e-12)
                ratio = norm / base_norm
                weight = torch.exp(
                    -float(READOUT["grace"]["alpha"])
                    * torch.clamp(
                        ratio - float(READOUT["grace"]["norm_ratio_threshold"]),
                        min=0.0,
                    )
                )
                numerator = numerator + weight.to(state.dtype) * state
                denominator = denominator + weight
            return model.classify(numerator / denominator.clamp_min(1e-12).to(state.dtype))

    def grace_naive_full():
        with autocast_context():
            _, trajectory = model.trajectory(xb, horizon, gain)
            weights, _ = grace_weights(trajectory)
            state = torch.einsum("bt,btd->bd", weights.to(trajectory.dtype), trajectory)
            return model.classify(state)

    def adaptive_full():
        with autocast_context():
            state = model.input_projection(xb)
            active_indices = torch.arange(len(xb), device=DEVICE)
            active_state = state
            output_state = torch.empty_like(state)
            base_norm = None
            for timestep in range(horizon):
                active_state = model.step(active_state, gain)
                norm = active_state.float().norm(dim=-1)
                if base_norm is None:
                    base_norm = norm.clamp_min(1e-12)
                ratio = norm / base_norm
                crossing = ratio > float(
                    READOUT["adaptive_stopping"]["norm_ratio_threshold"]
                )
                if timestep == horizon - 1:
                    crossing = torch.ones_like(crossing)
                if crossing.any():
                    output_state[active_indices[crossing]] = active_state[crossing]
                keep = ~crossing
                if not keep.any():
                    break
                active_indices = active_indices[keep]
                active_state = active_state[keep]
                base_norm = base_norm[keep]
            return model.classify(output_state)

    def terminal_readout():
        with autocast_context():
            return model.classify(precomputed[:, -1])

    def early_readout():
        with autocast_context():
            return model.classify(
                precomputed[:, : int(EVAL["early_window"])].mean(dim=1)
            )

    def uniform_readout():
        with autocast_context():
            return model.classify(precomputed.mean(dim=1))

    def adaptive_readout():
        with autocast_context():
            crossed = precomputed_ratio > float(
                READOUT["adaptive_stopping"]["norm_ratio_threshold"]
            )
            first = crossed.long().argmax(dim=1)
            first[~crossed.any(dim=1)] = horizon - 1
            state = precomputed[torch.arange(batch_size, device=DEVICE), first]
            return model.classify(state)

    def grace_naive_readout():
        with autocast_context():
            weights, _ = grace_weights(precomputed)
            state = torch.einsum(
                "bt,btd->bd",
                weights.to(precomputed.dtype),
                precomputed,
            )
            return model.classify(state)

    def grace_streaming_readout():
        with autocast_context():
            numerator = torch.zeros_like(precomputed[:, 0])
            denominator = torch.zeros(
                batch_size,
                1,
                device=DEVICE,
                dtype=torch.float32,
            )
            base_norm = (
                precomputed[:, 0].float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
            )
            for timestep in range(horizon):
                state = precomputed[:, timestep]
                ratio = state.float().norm(dim=-1, keepdim=True) / base_norm
                weight = torch.exp(
                    -float(READOUT["grace"]["alpha"])
                    * torch.clamp(
                        ratio - float(READOUT["grace"]["norm_ratio_threshold"]),
                        min=0.0,
                    )
                )
                numerator = numerator + weight.to(state.dtype) * state
                denominator = denominator + weight
            return model.classify(
                numerator / denominator.clamp_min(1e-12).to(numerator.dtype)
            )

    specs = {
        "terminal": (terminal_full, terminal_readout, False, False),
        "early": (early_full, early_readout, False, True),
        "adaptive": (adaptive_full, adaptive_readout, False, True),
        "uniform": (uniform_full, uniform_readout, False, False),
        "grace_naive": (grace_naive_full, grace_naive_readout, True, False),
        "grace_streaming": (
            grace_streaming_full,
            grace_streaming_readout,
            False,
            False,
        ),
    }
    rows = {}
    for name, (full_fn, readout_fn, stores_all, can_halt) in specs.items():
        print(f"Benchmarking {name}...")
        rows[name] = {
            "full_inference": cuda_measure(full_fn, warmup, repetitions),
            "readout_only": cuda_measure(readout_fn, warmup, repetitions),
            "stores_all_states": stores_all,
            "can_halt_recurrence_early": can_halt,
        }
    result = {
        "config_sha256": CONFIG_SHA256,
        "device": str(DEVICE),
        "precision": precision_name(),
        "batch_size": batch_size,
        "horizon": horizon,
        "state_dimension": int(ARCH["state_dimension"]),
        "gain": gain,
        "warmup_repetitions": warmup,
        "timed_repetitions": repetitions,
        "rows": rows,
    }
    atomic_json(path, result)
    table_lines = [
        "Readout\tFull ms\tReadout ms\tPeak full MiB\tPeak readout MiB\tStores all\tCan halt"
    ]
    for name, row in rows.items():
        table_lines.append(
            f"{name}\t{row['full_inference']['median_ms']:.3f}\t"
            f"{row['readout_only']['median_ms']:.3f}\t"
            f"{row['full_inference']['peak_additional_bytes'] / 1048576:.2f}\t"
            f"{row['readout_only']['peak_additional_bytes'] / 1048576:.2f}\t"
            f"{row['stores_all_states']}\t{row['can_halt_recurrence_early']}"
        )
    (OUT_DIR / "runtime_memory_table.txt").write_text(
        "\n".join(table_lines) + "\n",
        encoding="utf-8",
    )
    del model, xb, precomputed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def load_seed_metrics() -> list[dict[str, Any]]:
    metrics = []
    for seed in TRAIN["seeds"]:
        path = OUT_DIR / f"seed_{seed}_metrics.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("config_sha256") == CONFIG_SHA256:
                metrics.append(payload)
    return metrics


def summarise(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "experiment": CONFIG["experiment"],
        "config_sha256": CONFIG_SHA256,
        "seeds": [int(item["seed"]) for item in metrics],
        "shared_head": {},
        "probes": {},
    }
    for gain in GAIN_GRID:
        key = gain_key(gain)
        summary["shared_head"][key] = {}
        summary["probes"][key] = {}
        for method in METHODS:
            shared_values = [
                item["shared_head"][key]["readout_accuracy"][method] for item in metrics
            ]
            probe_values = [
                item["probes"]["readout_accuracy"][key][method] for item in metrics
            ]
            summary["shared_head"][key][method] = {
                "values": shared_values,
                "mean": float(np.mean(shared_values)),
                "std": float(np.std(shared_values)),
            }
            summary["probes"][key][method] = {
                "values": probe_values,
                "mean": float(np.mean(probe_values)),
                "std": float(np.std(probe_values)),
            }
    atomic_json(OUT_DIR / "summary.json", summary)
    return summary


def plot_outputs(metrics: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    primary = metrics[0]
    colors = {
        "terminal": "#d84a4a",
        "early": "#2f8f46",
        "uniform": "#7f7f7f",
        "fixed_profile": "#b7791f",
        "adaptive": "#8162b4",
        "grace": "#2468b2",
    }
    labels = {
        "terminal": "Terminal",
        "early": "Early window",
        "uniform": "Uniform",
        "fixed_profile": "Fixed GRACE profile",
        "adaptive": "Adaptive stop",
        "grace": "GRACE",
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for gain in HEADLINE_GAINS:
        curve = primary["shared_head"][gain_key(gain)]["timestep"]["accuracy"]
        ax.plot(
            np.arange(1, len(curve) + 1),
            curve,
            marker="o",
            markersize=3,
            linewidth=2,
            label=rf"$\lambda={gain:.1f}$",
        )
    ax.set_xlabel("Recursive step")
    ax.set_ylabel("Shared-head validation accuracy")
    ax.set_title("Tiny ImageNet recursive classifier: timestep accuracy")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trajectory_accuracy.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for method in METHODS:
        means = np.array(
            [summary["shared_head"][gain_key(gain)][method]["mean"] for gain in GAIN_GRID]
        )
        stds = np.array(
            [summary["shared_head"][gain_key(gain)][method]["std"] for gain in GAIN_GRID]
        )
        ax.plot(
            GAIN_GRID,
            means,
            marker="o",
            linewidth=2,
            color=colors[method],
            label=labels[method],
        )
        if len(metrics) > 1:
            ax.fill_between(
                GAIN_GRID,
                means - stds,
                means + stds,
                color=colors[method],
                alpha=0.10,
            )
    ax.set_xlabel("Test-time recurrent gain")
    ax.set_ylabel("Shared-head validation accuracy")
    ax.set_title("Trajectory readouts under controlled recurrent stress")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "readout_baselines.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    dense_keys = [gain_key(gain) for gain in DENSE_PROBE_GAINS]
    probe_timestep = primary["probes"]["timestep"]
    for threshold in PROBES["grace_period_accuracy_thresholds"]:
        threshold_key = f"{float(threshold):.2f}"
        values = [
            probe_timestep[key]["grace_period"][threshold_key] for key in dense_keys
        ]
        axes[0].plot(
            DENSE_PROBE_GAINS,
            values,
            marker="o",
            linewidth=2,
            label=rf"$\alpha={float(threshold):.2f}$",
        )
    axes[0].set_title("Accuracy-defined grace period")
    axes[0].set_xlabel("Gain")
    axes[0].set_ylabel("Prefix length")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    residuals = [
        primary["shared_head"][gain_key(gain)]["timestep"][
            "mean_relative_update_residual"
        ][-1]
        for gain in GAIN_GRID
    ]
    axes[1].plot(GAIN_GRID, residuals, marker="o", linewidth=2, color="#b7791f")
    axes[1].set_title("Terminal relative update")
    axes[1].set_xlabel("Gain")
    axes[1].set_ylabel(r"$\|h_{32}-h_{31}\|/\|h_{31}\|$")
    axes[1].grid(alpha=0.25)

    gaps = [
        100.0
        * (
            summary["shared_head"][gain_key(gain)]["early"]["mean"]
            - summary["shared_head"][gain_key(gain)]["terminal"]["mean"]
        )
        for gain in GAIN_GRID
    ]
    axes[2].axhline(0.0, color="#666666", linewidth=1)
    axes[2].plot(GAIN_GRID, gaps, marker="o", linewidth=2, color="#2f8f46")
    axes[2].set_title("Prefix advantage")
    axes[2].set_xlabel("Gain")
    axes[2].set_ylabel("Early - terminal (pp)")
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stress_grid.png", dpi=220)
    plt.close(fig)


def write_rebuttal_table(metrics: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    if len(metrics) < 3:
        return
    lines = ["Gain\tSeed\tTerminal\tEarly\tGRACE\tEarly-Terminal"]
    for gain in HEADLINE_GAINS:
        key = gain_key(gain)
        for item in metrics:
            row = item["shared_head"][key]["readout_accuracy"]
            lines.append(
                f"{gain:.2f}\t{item['seed']}\t{row['terminal']:.4f}\t"
                f"{row['early']:.4f}\t{row['grace']:.4f}\t"
                f"{row['early'] - row['terminal']:+.4f}"
            )
        terminal = summary["shared_head"][key]["terminal"]
        early = summary["shared_head"][key]["early"]
        grace = summary["shared_head"][key]["grace"]
        differences = [
            item["shared_head"][key]["readout_accuracy"]["early"]
            - item["shared_head"][key]["readout_accuracy"]["terminal"]
            for item in metrics
        ]
        direction = sum(value > 0 for value in differences)
        lines.append(
            f"{gain:.2f}\tmean+-std\t{terminal['mean']:.4f}+-{terminal['std']:.4f}\t"
            f"{early['mean']:.4f}+-{early['std']:.4f}\t"
            f"{grace['mean']:.4f}+-{grace['std']:.4f}\t"
            f"{np.mean(differences):+.4f} ({direction}/3 positive)"
        )
    (OUT_DIR / "rebuttal_table.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def finalise_outputs() -> None:
    metrics = load_seed_metrics()
    if not metrics:
        return
    summary = summarise(metrics)
    plot_outputs(metrics, summary)
    write_rebuttal_table(metrics, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=(
            "validate",
            "features",
            "smoke",
            "seed0",
            "gate",
            "remaining",
            "benchmark",
            "finalise",
            "all",
        ),
    )
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-smoke", action="store_true")
    parser.add_argument("--force-evaluation", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json(ENV_PATH, environment_payload())
    print(
        f"{CONFIG['experiment']} | stage={args.stage} | "
        f"device={DEVICE} | precision={precision_name()} | config={CONFIG_SHA256[:12]}"
    )

    if args.stage == "validate":
        validate_dataset()
        return

    cache = load_or_extract_features(force=args.force_features)
    if args.stage == "features":
        return

    smoke = run_smoke(cache, force=args.force_smoke)
    if not smoke["pass"]:
        raise RuntimeError("Engineering smoke test did not pass.")
    if args.stage == "smoke":
        return

    if args.stage in {"seed0", "gate", "all"}:
        train_seed(cache, 0, resume=not args.no_resume)
        seed0_metrics = evaluate_seed(cache, 0, force=args.force_evaluation)
        decision = apply_continuation_gate(seed0_metrics)
        finalise_outputs()
        if args.stage == "seed0":
            return
        if args.stage == "gate":
            return
    else:
        decision = (
            json.loads(GATE_PATH.read_text(encoding="utf-8"))
            if GATE_PATH.exists()
            else None
        )

    if args.stage in {"remaining", "all"}:
        if decision is None:
            raise RuntimeError("Run the seed-0 continuation gate first.")
        if not decision["continue_to_seeds_1_and_2"]:
            print("Predeclared continuation gate failed; stopping after seed 0.")
            finalise_outputs()
            return
        for seed in (1, 2):
            train_seed(cache, seed, resume=not args.no_resume)
            evaluate_seed(cache, seed, force=args.force_evaluation)
        finalise_outputs()
        if args.stage == "remaining":
            return

    if args.stage in {"benchmark", "all"}:
        if not (OUT_DIR / "seed_0_metrics.json").exists():
            raise RuntimeError("Seed 0 must be trained and evaluated before benchmarking.")
        benchmark_seed0(cache)
        finalise_outputs()
        return

    if args.stage == "finalise":
        finalise_outputs()


if __name__ == "__main__":
    main()
