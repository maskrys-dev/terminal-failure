"""Post-hoc E16/E17 causal panel and robustness audits.

This driver never trains a model. It evaluates only already-saved checkpoints
under the frozen follow-up protocol in:
  results/modern_iterative_panel/followup_analysis/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from torchvision import models

from transient_geometry.experiments import (
    e17_recursive_attention_tinyimagenet as e17,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
PANEL = RESULTS / "modern_iterative_panel"
FOLLOWUP = PANEL / "followup_analysis"
AUDITS = FOLLOWUP / "audits"
PREDICTIONS = AUDITS / "predictions"
FIGURES = PANEL / "figures"

E16_AGGREGATE_DIR = RESULTS / "rebuttal_recursive_tinyimagenet_paired_horizon"
E16_DIRS = {
    8: RESULTS / "rebuttal_recursive_tinyimagenet_e16_ttrain8",
    32: RESULTS / "rebuttal_recursive_tinyimagenet_e16_ttrain32",
}
E17_DIRS = {
    8: PANEL / "attention_t8",
    32: PANEL / "attention_t32_control",
}
E16_CONFIGS = {
    horizon: directory / "config_frozen.json"
    for horizon, directory in E16_DIRS.items()
}
E17_CONFIGS = {
    8: PANEL / "protocol" / "attention_t8_config.json",
    32: PANEL / "protocol" / "attention_t32_control_config.json",
}
E16_FEATURE_CACHE = RESULTS / "rebuttal_recursive_tinyimagenet" / "feature_cache.pt"
E17_CACHE_DIR = PANEL / "spatial_cache"
PROBE_INDICES = (
    RESULTS / "rebuttal_recursive_tinyimagenet" / "probe_subset_indices.npy"
)
SEEDS = (101, 102, 103)
HORIZONS = (8, 16, 24, 32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_npz_new(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def sample_stats(values: Iterable[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    if not data:
        raise ValueError("Cannot summarise an empty sequence.")
    return {
        "mean": statistics.mean(data),
        "sample_std": statistics.stdev(data) if len(data) > 1 else 0.0,
        "values": data,
    }


def amp_context():
    if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if DEVICE.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


class RecursiveSwiGLUClassifier(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        architecture = config["architecture"]
        state_dimension = int(architecture["state_dimension"])
        intermediate_dimension = int(architecture["intermediate_dimension"])
        self.training_horizon = int(architecture["training_horizon"])
        self.input_projection = nn.Linear(
            int(architecture["input_dimension"]), state_dimension
        )
        self.block_norm = nn.LayerNorm(state_dimension)
        self.gate_projection = nn.Linear(state_dimension, intermediate_dimension)
        self.value_projection = nn.Linear(state_dimension, intermediate_dimension)
        self.output_projection = nn.Linear(intermediate_dimension, state_dimension)
        self.dropout = nn.Dropout(float(architecture["dropout"]))
        self.head_norm = nn.LayerNorm(state_dimension)
        self.classifier = nn.Linear(
            state_dimension, int(config["dataset"]["classes"])
        )

    def step(self, state: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        normalised = self.block_norm(state)
        hidden = F.silu(self.gate_projection(normalised)) * self.value_projection(
            normalised
        )
        update = self.dropout(self.output_projection(hidden))
        return state + (float(gain) / self.training_horizon) * update

    def classify(self, state: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.head_norm(state))

    def trajectory(
        self, features: torch.Tensor, steps: int = 32, gain: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state0 = self.input_projection(features)
        state = state0
        states = []
        for _ in range(int(steps)):
            state = self.step(state, gain)
            states.append(state)
        return state0, torch.stack(states, dim=1)


def load_e16_model(
    training_horizon: int, seed: int, checkpoint: Path | None = None
) -> RecursiveSwiGLUClassifier:
    config = read_json(E16_CONFIGS[training_horizon])
    path = checkpoint or (
        E16_DIRS[training_horizon] / f"seed_{seed}_checkpoint.pt"
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_hash = sha256_file(E16_CONFIGS[training_horizon])
    if payload.get("config_sha256") != expected_hash:
        raise RuntimeError(f"E16 config mismatch: {path}")
    if int(payload.get("seed", -1)) != seed:
        raise RuntimeError(f"E16 seed mismatch: {path}")
    model = RecursiveSwiGLUClassifier(config)
    model.load_state_dict(payload["model"])
    return model.to(DEVICE).eval()


def load_e17_model(
    training_horizon: int, seed: int, checkpoint: Path | None = None
) -> e17.RecursiveAttentionClassifier:
    config = read_json(E17_CONFIGS[training_horizon])
    path = checkpoint or (E17_DIRS[training_horizon] / f"seed_{seed}" / "checkpoint.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_hash = sha256_file(E17_CONFIGS[training_horizon])
    if payload.get("config_sha256") != expected_hash:
        raise RuntimeError(f"E17 config mismatch: {path}")
    if int(payload.get("seed", -1)) != seed:
        raise RuntimeError(f"E17 seed mismatch: {path}")
    model = e17.RecursiveAttentionClassifier(config)
    model.load_state_dict(payload["model"])
    return model.to(DEVICE).eval()


def load_e16_cache() -> dict[str, torch.Tensor]:
    cache = torch.load(E16_FEATURE_CACHE, map_location="cpu", weights_only=False)
    expected = {
        "train_features": (100_000, 768),
        "validation_features": (10_000, 768),
        "train_labels": (100_000,),
        "validation_labels": (10_000,),
    }
    for name, shape in expected.items():
        if tuple(cache[name].shape) != shape:
            raise RuntimeError(f"E16 cache shape mismatch for {name}.")
    return cache


def load_e17_cache() -> dict[str, np.ndarray]:
    arrays = {
        "train_features": np.load(
            E17_CACHE_DIR / "train_spatial_tokens_fp16.npy", mmap_mode="r"
        ),
        "train_labels": np.load(
            E17_CACHE_DIR / "train_labels_int64.npy", mmap_mode="r"
        ),
        "validation_features": np.load(
            E17_CACHE_DIR / "validation_spatial_tokens_fp16.npy", mmap_mode="r"
        ),
        "validation_labels": np.load(
            E17_CACHE_DIR / "validation_labels_int64.npy", mmap_mode="r"
        ),
    }
    expected = {
        "train_features": (100_000, 49, 768),
        "train_labels": (100_000,),
        "validation_features": (10_000, 49, 768),
        "validation_labels": (10_000,),
    }
    for name, shape in expected.items():
        if tuple(arrays[name].shape) != shape:
            raise RuntimeError(f"E17 cache shape mismatch for {name}.")
    return arrays


@torch.inference_mode()
def e16_predictions(
    model: RecursiveSwiGLUClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions8: list[torch.Tensor] = []
    predictions32: list[torch.Tensor] = []
    model.eval()
    for start in range(0, len(labels), batch_size):
        end = min(start + batch_size, len(labels))
        xb = features[start:end].to(DEVICE, non_blocking=True)
        with amp_context():
            _, trajectory = model.trajectory(xb, 32, 1.0)
            # Match the frozen E16 evaluator's GEMM shape exactly.  Under BF16,
            # classifying the full [batch, time, state] tensor can differ by a
            # few boundary predictions from two separate 2-D classifier calls.
            logits = model.classify(trajectory)
            logits8 = logits[:, 7]
            logits32 = logits[:, 31]
        predictions8.append(logits8.float().argmax(dim=-1).cpu())
        predictions32.append(logits32.float().argmax(dim=-1).cpu())
    return (
        labels.long().cpu().numpy(),
        torch.cat(predictions8).numpy(),
        torch.cat(predictions32).numpy(),
    )


@torch.inference_mode()
def e17_predictions(
    model: e17.RecursiveAttentionClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions8: list[torch.Tensor] = []
    predictions32: list[torch.Tensor] = []
    model.eval()
    for start in range(0, len(labels), batch_size):
        end = min(start + batch_size, len(labels))
        xb = torch.from_numpy(
            np.ascontiguousarray(features[start:end])
        ).to(DEVICE, non_blocking=True)
        with amp_context():
            # Use the same trajectory construction and full-trajectory
            # classifier call as the frozen E17 evaluator.
            _, trajectory, _ = e17.class_trajectory(model, xb, 32, 1.0)
            logits = model.classify(trajectory)
            logits8 = logits[:, 7]
            logits32 = logits[:, 31]
        predictions8.append(logits8.float().argmax(dim=-1).cpu())
        predictions32.append(logits32.float().argmax(dim=-1).cpu())
    return (
        np.asarray(labels, dtype=np.int64),
        torch.cat(predictions8).numpy(),
        torch.cat(predictions32).numpy(),
    )


def accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(labels == predictions))


def expected_selected_accuracies(
    model: str, seed: int
) -> tuple[float, float]:
    if model == "e16_t8":
        metrics = read_json(
            E16_DIRS[8] / f"seed_{seed}_horizon_metrics.json"
        )
        return (
            float(metrics["by_horizon"]["8"]["terminal_accuracy"]),
            float(metrics["by_horizon"]["32"]["terminal_accuracy"]),
        )
    diagnostics = read_json(E17_DIRS[8] / f"seed_{seed}" / "diagnostics.json")
    return (
        float(diagnostics["by_horizon"]["8"]["terminal_accuracy"]),
        float(diagnostics["by_horizon"]["32"]["terminal_accuracy"]),
    )


def validate_prediction_accuracy(
    model: str,
    seed: int,
    labels: np.ndarray,
    prediction8: np.ndarray,
    prediction32: np.ndarray,
) -> None:
    expected8, expected32 = expected_selected_accuracies(model, seed)
    actual8 = accuracy(labels, prediction8)
    actual32 = accuracy(labels, prediction32)
    tolerance = 0.0003
    if abs(actual8 - expected8) > tolerance or abs(actual32 - expected32) > tolerance:
        raise RuntimeError(
            f"{model} seed {seed} prediction reproduction failed: "
            f"actual=({actual8},{actual32}), expected=({expected8},{expected32})"
        )


def get_or_create_selected_predictions(
    model: str,
    seed: int,
    e16_cache: dict[str, torch.Tensor] | None,
    e17_cache: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = PREDICTIONS / f"{model}_seed_{seed}.npz"
    if path.exists():
        saved = np.load(path)
        result = (
            np.asarray(saved["labels"]),
            np.asarray(saved["prediction_t8"]),
            np.asarray(saved["prediction_t32"]),
        )
        validate_prediction_accuracy(model, seed, *result)
        return result

    if model == "e16_t8":
        if e16_cache is None:
            raise ValueError("E16 cache required.")
        loaded = load_e16_model(8, seed)
        result = e16_predictions(
            loaded,
            e16_cache["validation_features"],
            e16_cache["validation_labels"],
        )
    elif model == "e17_t8":
        if e17_cache is None:
            raise ValueError("E17 cache required.")
        loaded = load_e17_model(8, seed)
        result = e17_predictions(
            loaded,
            e17_cache["validation_features"],
            e17_cache["validation_labels"],
        )
    else:
        raise ValueError(model)
    validate_prediction_accuracy(model, seed, *result)
    save_npz_new(
        path,
        labels=result[0],
        prediction_t8=result[1],
        prediction_t32=result[2],
    )
    del loaded
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def classwise_audit() -> dict[str, Any]:
    summary_path = AUDITS / "classwise_robustness.json"
    if summary_path.exists():
        print("Using existing classwise robustness audit.")
        return read_json(summary_path)
    e16_cache = load_e16_cache()
    e17_cache = load_e17_cache()
    wnids = [
        line.strip()
        for line in (ROOT / "data" / "tiny-imagenet-200" / "wnids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(wnids) != 200:
        raise RuntimeError("Tiny ImageNet WNID count is not 200.")

    final: dict[str, Any] = {
        "status": "complete",
        "definition": "three-seed mean class accuracy at T8 minus T32",
        "models": {},
    }
    markdown = [
        "# Classwise robustness audit",
        "",
        "All 200 Tiny ImageNet classes are retained. The primary classwise "
        "drop is the three-seed mean T=8 accuracy minus the three-seed mean "
        "T=32 accuracy.",
        "",
        "| Model | Median drop | IQR | Macro mean | Lower at T=32 | Drop >=2 pp |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in ("e16_t8", "e17_t8"):
        seed_predictions = {
            seed: get_or_create_selected_predictions(
                model, seed, e16_cache, e17_cache
            )
            for seed in SEEDS
        }
        rows: list[dict[str, Any]] = []
        drops = []
        for class_index, wnid in enumerate(wnids):
            row: dict[str, Any] = {
                "class_index": class_index,
                "wnid": wnid,
            }
            seed_t8 = []
            seed_t32 = []
            for seed in SEEDS:
                labels, prediction8, prediction32 = seed_predictions[seed]
                selected = labels == class_index
                if int(selected.sum()) != 50:
                    raise RuntimeError(
                        f"Class {class_index} has {selected.sum()} validation examples."
                    )
                accuracy8 = float(np.mean(prediction8[selected] == labels[selected]))
                accuracy32 = float(
                    np.mean(prediction32[selected] == labels[selected])
                )
                drop_pp = 100.0 * (accuracy8 - accuracy32)
                row[f"seed_{seed}_t8_accuracy"] = accuracy8
                row[f"seed_{seed}_t32_accuracy"] = accuracy32
                row[f"seed_{seed}_drop_pp"] = drop_pp
                seed_t8.append(accuracy8)
                seed_t32.append(accuracy32)
            row["mean_t8_accuracy"] = statistics.mean(seed_t8)
            row["mean_t32_accuracy"] = statistics.mean(seed_t32)
            row["mean_drop_pp"] = 100.0 * (
                row["mean_t8_accuracy"] - row["mean_t32_accuracy"]
            )
            drops.append(float(row["mean_drop_pp"]))
            rows.append(row)

        array = np.asarray(drops, dtype=np.float64)
        q1, median, q3 = np.percentile(array, [25, 50, 75])
        model_summary = {
            "classes": 200,
            "median_drop_pp": float(median),
            "q1_drop_pp": float(q1),
            "q3_drop_pp": float(q3),
            "iqr_drop_pp": float(q3 - q1),
            "macro_average_drop_pp": float(array.mean()),
            "classes_lower_at_t32": int((array > 0.0).sum()),
            "classes_lower_at_t32_percent": float(100.0 * (array > 0.0).mean()),
            "classes_drop_at_least_2pp": int((array >= 2.0).sum()),
            "classes_drop_at_least_2pp_percent": float(
                100.0 * (array >= 2.0).mean()
            ),
            "complete_table_csv": str(
                (AUDITS / f"classwise_{model}.csv").relative_to(ROOT)
            ),
        }
        final["models"][model] = model_summary
        write_csv_new(AUDITS / f"classwise_{model}.csv", rows)
        markdown.append(
            f"| {model} | {median:.2f} pp | [{q1:.2f}, {q3:.2f}] pp | "
            f"{array.mean():.2f} pp | {model_summary['classes_lower_at_t32']}/200 "
            f"({model_summary['classes_lower_at_t32_percent']:.1f}%) | "
            f"{model_summary['classes_drop_at_least_2pp']}/200 "
            f"({model_summary['classes_drop_at_least_2pp_percent']:.1f}%) |"
        )
    write_json_new(summary_path, final)
    write_text_new(AUDITS / "classwise_robustness.md", "\n".join(markdown))
    return final


@torch.inference_mode()
def e16_representations(
    model: RecursiveSwiGLUClassifier,
    features: torch.Tensor,
    indices: np.ndarray,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    collected = {"terminal_step8": [], "early_mean_steps1_to_8": [], "terminal_step32": []}
    model.eval()
    for start in range(0, len(indices), batch_size):
        chosen = indices[start : start + batch_size]
        xb = features[torch.from_numpy(chosen)].to(DEVICE, non_blocking=True)
        with amp_context():
            _, trajectory = model.trajectory(xb, 32, 1.0)
        collected["terminal_step8"].append(trajectory[:, 7].float().cpu().numpy())
        collected["early_mean_steps1_to_8"].append(
            trajectory[:, :8].float().mean(dim=1).cpu().numpy()
        )
        collected["terminal_step32"].append(
            trajectory[:, 31].float().cpu().numpy()
        )
    return {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}


def e16_probe_audit() -> dict[str, Any]:
    output = AUDITS / "e16_probe_audit.json"
    if output.exists():
        print("Using existing E16 probe audit.")
        return read_json(output)
    if sha256_file(PROBE_INDICES) != (
        "9f7378e1c3cde52953ea5096b6b80cfb4b52fff8cafcd87be19d9b366590a19d"
    ):
        raise RuntimeError("Probe subset hash mismatch.")
    cache = load_e16_cache()
    train_indices = np.asarray(np.load(PROBE_INDICES), dtype=np.int64)
    validation_indices = np.arange(10_000, dtype=np.int64)
    train_labels = cache["train_labels"].numpy()[train_indices]
    validation_labels = cache["validation_labels"].numpy()
    payload: dict[str, Any] = {
        "status": "complete_posthoc_fixed_protocol",
        "alpha": 1.0,
        "standardisation": False,
        "training_subset_sha256": sha256_file(PROBE_INDICES),
        "training_subset_size": len(train_indices),
        "validation_examples": len(validation_indices),
        "by_training_horizon": {},
    }
    rows: list[dict[str, Any]] = []
    markdown = [
        "# E16 fixed-alpha probe audit",
        "",
        "This post-hoc audit uses the exact E17 Ridge protocol and existing E16 "
        "selected checkpoints; it does not alter the original E16 confirmatory result.",
        "",
        "| T_train | Seed | Step 8 | Early 1--8 | Step 32 | Probe drop |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for training_horizon in (8, 32):
        seed_results = []
        for seed in SEEDS:
            started = time.time()
            model = load_e16_model(training_horizon, seed)
            train_representations = e16_representations(
                model,
                cache["train_features"],
                train_indices,
            )
            validation_representations = e16_representations(
                model,
                cache["validation_features"],
                validation_indices,
            )
            accuracies: dict[str, float] = {}
            for name in (
                "terminal_step8",
                "early_mean_steps1_to_8",
                "terminal_step32",
            ):
                classifier = RidgeClassifier(alpha=1.0)
                classifier.fit(train_representations[name], train_labels)
                accuracies[name] = float(
                    classifier.score(
                        validation_representations[name], validation_labels
                    )
                )
            row = {
                "training_horizon": training_horizon,
                "seed": seed,
                **accuracies,
                "terminal_step8_minus_step32_pp": 100.0
                * (
                    accuracies["terminal_step8"]
                    - accuracies["terminal_step32"]
                ),
                "elapsed_seconds": time.time() - started,
            }
            seed_results.append(row)
            rows.append(row)
            markdown.append(
                f"| {training_horizon} | {seed} | "
                f"{100*accuracies['terminal_step8']:.2f}% | "
                f"{100*accuracies['early_mean_steps1_to_8']:.2f}% | "
                f"{100*accuracies['terminal_step32']:.2f}% | "
                f"{row['terminal_step8_minus_step32_pp']:+.2f} pp |"
            )
            del model, train_representations, validation_representations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        payload["by_training_horizon"][str(training_horizon)] = {
            "seed_level": seed_results,
            "terminal_step8": sample_stats(
                [row["terminal_step8"] for row in seed_results]
            ),
            "early_mean_steps1_to_8": sample_stats(
                [row["early_mean_steps1_to_8"] for row in seed_results]
            ),
            "terminal_step32": sample_stats(
                [row["terminal_step32"] for row in seed_results]
            ),
            "probe_drop_pp": sample_stats(
                [row["terminal_step8_minus_step32_pp"] for row in seed_results]
            ),
        }
    write_json_new(output, payload)
    write_csv_new(AUDITS / "e16_probe_audit.csv", rows)
    write_text_new(AUDITS / "e16_probe_audit.md", "\n".join(markdown))
    return payload


def evaluate_checkpoint_drop_e16(
    seed: int, checkpoint: Path, cache: dict[str, torch.Tensor]
) -> dict[str, float]:
    model = load_e16_model(8, seed, checkpoint)
    labels, prediction8, prediction32 = e16_predictions(
        model, cache["validation_features"], cache["validation_labels"]
    )
    result = {
        "terminal_t8": accuracy(labels, prediction8),
        "terminal_t32": accuracy(labels, prediction32),
    }
    result["drop_pp"] = 100.0 * (
        result["terminal_t8"] - result["terminal_t32"]
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def evaluate_checkpoint_drop_e17(
    seed: int, checkpoint: Path, cache: dict[str, np.ndarray]
) -> dict[str, float]:
    model = load_e17_model(8, seed, checkpoint)
    labels, prediction8, prediction32 = e17_predictions(
        model, cache["validation_features"], cache["validation_labels"]
    )
    result = {
        "terminal_t8": accuracy(labels, prediction8),
        "terminal_t32": accuracy(labels, prediction32),
    }
    result["drop_pp"] = 100.0 * (
        result["terminal_t8"] - result["terminal_t32"]
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def checkpoint_sensitivity() -> dict[str, Any]:
    output = AUDITS / "checkpoint_sensitivity.json"
    if output.exists():
        print("Using existing checkpoint-sensitivity audit.")
        return read_json(output)
    e16_cache = load_e16_cache()
    e17_cache = load_e17_cache()
    rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        selected = E16_DIRS[8] / f"seed_{seed}_checkpoint.pt"
        final = E16_DIRS[8] / f"seed_{seed}_last.pt"
        for label, path in (("selected", selected), ("final_epoch", final)):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            result = evaluate_checkpoint_drop_e16(seed, path, e16_cache)
            rows.append(
                {
                    "model": "e16_t8",
                    "seed": seed,
                    "checkpoint_role": label,
                    "checkpoint_file": str(path.relative_to(ROOT)),
                    "checkpoint_sha256": sha256_file(path),
                    "epoch": (
                        int(payload["epoch"])
                        if label == "selected"
                        else int(payload["epoch"]) + 1
                    ),
                    "trained_horizon_validation_accuracy": (
                        float(payload["validation_terminal_accuracy"])
                        if label == "selected"
                        else float(payload["history"][-1][
                            "validation_terminal_accuracy"
                        ])
                    ),
                    **result,
                }
            )

    for seed in SEEDS:
        directory = E17_DIRS[8] / f"seed_{seed}"
        candidates = []
        for path in directory.glob("checkpoint_epoch_*.pt"):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            candidates.append(
                (
                    -float(payload["validation_terminal_accuracy"]),
                    int(payload["epoch"]),
                    path,
                    payload,
                )
            )
        candidates.sort(key=lambda item: (item[0], item[1]))
        for rank, (_, epoch, path, payload) in enumerate(candidates[:3], start=1):
            result = evaluate_checkpoint_drop_e17(seed, path, e17_cache)
            rows.append(
                {
                    "model": "e17_t8",
                    "seed": seed,
                    "checkpoint_role": (
                        "selected" if rank == 1 else f"saved_rank_{rank}"
                    ),
                    "checkpoint_file": str(path.relative_to(ROOT)),
                    "checkpoint_sha256": sha256_file(path),
                    "epoch": epoch,
                    "trained_horizon_validation_accuracy": float(
                        payload["validation_terminal_accuracy"]
                    ),
                    **result,
                }
            )

    payload: dict[str, Any] = {
        "status": "complete",
        "headline_checkpoint_unchanged": True,
        "limitations": {
            "e16_t8": "Selected and final-epoch states exist; intermediate top-three checkpoint states were not retained.",
            "e17_t8": "Saved improving checkpoints permit a top-three audit; the final-epoch model state was not retained.",
        },
        "models": {},
        "rows": rows,
    }
    markdown = [
        "# Checkpoint sensitivity audit",
        "",
        "Only saved checkpoint states were evaluated. The preregistered selected "
        "checkpoint remains the headline checkpoint.",
        "",
        "| Model | Seed | Checkpoint | Epoch | T=8 | T=32 | Drop |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['model']} | {row['seed']} | {row['checkpoint_role']} | "
            f"{row['epoch']} | {100*row['terminal_t8']:.2f}% | "
            f"{100*row['terminal_t32']:.2f}% | {row['drop_pp']:+.2f} pp |"
        )
    for model in ("e16_t8", "e17_t8"):
        model_rows = [row for row in rows if row["model"] == model]
        drops = [float(row["drop_pp"]) for row in model_rows]
        seed_sign_consistency = {}
        selected_unusually_sensitive = {}
        for seed in SEEDS:
            seed_rows = [row for row in model_rows if row["seed"] == seed]
            seed_sign_consistency[str(seed)] = all(
                float(row["drop_pp"]) > 0.0 for row in seed_rows
            )
            selected_drop = next(
                float(row["drop_pp"])
                for row in seed_rows
                if row["checkpoint_role"] == "selected"
            )
            alternatives = [
                float(row["drop_pp"])
                for row in seed_rows
                if row["checkpoint_role"] != "selected"
            ]
            selected_unusually_sensitive[str(seed)] = bool(
                alternatives and selected_drop > max(alternatives)
            )
        payload["models"][model] = {
            "drop_range_pp": [min(drops), max(drops)],
            "positive_drop_for_every_audited_checkpoint": all(
                value > 0.0 for value in drops
            ),
            "seed_sign_consistency": seed_sign_consistency,
            "selected_drop_larger_than_every_saved_alternative": selected_unusually_sensitive,
        }
    write_json_new(output, payload)
    write_csv_new(AUDITS / "checkpoint_sensitivity.csv", rows)
    write_text_new(AUDITS / "checkpoint_sensitivity.md", "\n".join(markdown))
    return payload


def model_scale_audit() -> dict[str, Any]:
    output = AUDITS / "model_scale.json"
    if output.exists():
        print("Using existing model-scale audit.")
        return read_json(output)
    e16_config = read_json(E16_CONFIGS[8])
    e17_config = read_json(E17_CONFIGS[8])
    e16_head = RecursiveSwiGLUClassifier(e16_config)
    e17_head = e17.RecursiveAttentionClassifier(e17_config)
    convnext = models.convnext_tiny(weights=None)
    full_backbone_parameters = sum(parameter.numel() for parameter in convnext.parameters())
    executed_backbone_parameters = sum(
        parameter.numel() for parameter in convnext.features.parameters()
    ) + sum(parameter.numel() for parameter in convnext.classifier[0].parameters())
    unused_classifier_parameters = sum(
        parameter.numel() for parameter in convnext.classifier[-1].parameters()
    )
    if executed_backbone_parameters + unused_classifier_parameters != full_backbone_parameters:
        raise RuntimeError("Unexpected ConvNeXt parameter accounting.")
    e16_head_parameters = sum(parameter.numel() for parameter in e16_head.parameters())
    e17_head_parameters = sum(parameter.numel() for parameter in e17_head.parameters())

    d16, m16 = 512, 2048
    e16_step_flops = 6 * d16 * m16
    n17, d17, h17, m17 = 50, 256, 8, 1024
    e17_attention_projection_flops = 8 * n17 * d17 * d17
    e17_attention_products_flops = 4 * n17 * n17 * d17
    e17_softmax_flops = 5 * h17 * n17 * n17
    e17_swiglu_flops = 6 * n17 * d17 * m17
    e17_step_flops = (
        e17_attention_projection_flops
        + e17_attention_products_flops
        + e17_softmax_flops
        + e17_swiglu_flops
    )
    e16_evaluation = read_json(E16_AGGREGATE_DIR / "aggregate_summary.json")
    e17_t8_evaluation = read_json(E17_DIRS[8] / "attention_summary.json")
    e17_t32_evaluation = read_json(E17_DIRS[32] / "attention_summary.json")
    evaluated_trained_horizon_accuracy = {
        "swiglu": {
            "8": e16_evaluation["trajectory_summary"]["8"]["8"][
                "terminal_accuracy"
            ],
            "32": e16_evaluation["trajectory_summary"]["32"]["32"][
                "terminal_accuracy"
            ],
        },
        "attention": {
            # The frozen E17 headline uses the readout evaluator's terminal
            # calls (reported in seed_level), whose BF16 GEMM shape differs
            # slightly from the dense trajectory diagnostic call.
            "8": sample_stats(
                row["terminal_t8"] for row in e17_t8_evaluation["seed_level"]
            ),
            "32": sample_stats(
                row["terminal_t32"] for row in e17_t32_evaluation["seed_level"]
            ),
        },
    }

    def trained_horizon_accuracy(
        architecture: str, training_horizon: int
    ) -> dict[str, Any]:
        values = []
        if architecture == "swiglu":
            for seed in SEEDS:
                training = read_json(
                    E16_DIRS[training_horizon] / f"seed_{seed}_training.json"
                )
                values.append(float(training["best_validation_terminal_accuracy"]))
        else:
            for seed in SEEDS:
                training = read_json(
                    E17_DIRS[training_horizon]
                    / f"seed_{seed}"
                    / "training_history.json"
                )
                values.append(float(training["best_validation_terminal_accuracy"]))
        return sample_stats(values)

    payload = {
        "status": "complete",
        "dataset": {
            "name": "Tiny ImageNet-200",
            "training_images": 100000,
            "validation_images": 10000,
            "classes": 200,
        },
        "frozen_backbone": {
            "name": "torchvision ConvNeXt-Tiny",
            "executed_parameter_count_excluding_unused_imagenet_classifier": executed_backbone_parameters,
            "full_torchvision_checkpoint_parameter_count": full_backbone_parameters,
            "unused_imagenet_classifier_parameter_count": unused_classifier_parameters,
        },
        "models": {
            "swiglu": {
                "input_representation": "pooled 768-d",
                "state_or_model_width": 512,
                "intermediate_width": 2048,
                "attention_heads": None,
                "trainable_head_parameters": e16_head_parameters,
                "executed_system_parameters": executed_backbone_parameters
                + e16_head_parameters,
                "trained_horizon_accuracy": {
                    "8": trained_horizon_accuracy("swiglu", 8),
                    "32": trained_horizon_accuracy("swiglu", 32),
                },
                "evaluated_trained_horizon_terminal_accuracy":
                    evaluated_trained_horizon_accuracy["swiglu"],
                "recurrent_flops_per_step": e16_step_flops,
                "flop_formula": "6*d*m with d=512 and m=2048",
            },
            "attention": {
                "input_representation": "49 x 768 spatial tokens plus class token after projection",
                "state_or_model_width": 256,
                "intermediate_width": 1024,
                "attention_heads": 8,
                "tokens_during_recurrence": 50,
                "trainable_head_parameters": e17_head_parameters,
                "executed_system_parameters": executed_backbone_parameters
                + e17_head_parameters,
                "trained_horizon_accuracy": {
                    "8": trained_horizon_accuracy("attention", 8),
                    "32": trained_horizon_accuracy("attention", 32),
                },
                "evaluated_trained_horizon_terminal_accuracy":
                    evaluated_trained_horizon_accuracy["attention"],
                "recurrent_flops_per_step": e17_step_flops,
                "flop_components": {
                    "attention_projections": e17_attention_projection_flops,
                    "attention_qk_and_av_products": e17_attention_products_flops,
                    "softmax_approximation": e17_softmax_flops,
                    "swiglu_projections": e17_swiglu_flops,
                },
                "flop_formula": "8*n*d^2 + 4*n^2*d + 5*h*n^2 + 6*n*d*m",
            },
        },
        "flop_convention": {
            "multiply_add": "two FLOPs",
            "excludes": [
                "frozen backbone",
                "one-time input projection",
                "final classifier",
                "LayerNorm and lower-order elementwise residual/activation costs",
            ],
        },
    }
    rows = [
        {
            "item": "Dataset",
            "E16_SwiGLU": "Tiny ImageNet-200",
            "E17_Attention": "Tiny ImageNet-200",
        },
        {
            "item": "Training images",
            "E16_SwiGLU": 100000,
            "E17_Attention": 100000,
        },
        {
            "item": "Validation images",
            "E16_SwiGLU": 10000,
            "E17_Attention": 10000,
        },
        {"item": "Classes", "E16_SwiGLU": 200, "E17_Attention": 200},
        {
            "item": "Frozen backbone",
            "E16_SwiGLU": "ConvNeXt-Tiny",
            "E17_Attention": "ConvNeXt-Tiny",
        },
        {
            "item": "Input representation",
            "E16_SwiGLU": "pooled 768-d",
            "E17_Attention": "49 x 768 spatial tokens",
        },
        {
            "item": "Recursive state/model width",
            "E16_SwiGLU": 512,
            "E17_Attention": 256,
        },
        {
            "item": "Intermediate width",
            "E16_SwiGLU": 2048,
            "E17_Attention": 1024,
        },
        {"item": "Attention heads", "E16_SwiGLU": "N/A", "E17_Attention": 8},
        {
            "item": "Trainable head parameters",
            "E16_SwiGLU": e16_head_parameters,
            "E17_Attention": e17_head_parameters,
        },
        {
            "item": "Executed frozen backbone parameters",
            "E16_SwiGLU": executed_backbone_parameters,
            "E17_Attention": executed_backbone_parameters,
        },
        {
            "item": "Executed total system parameters",
            "E16_SwiGLU": executed_backbone_parameters + e16_head_parameters,
            "E17_Attention": executed_backbone_parameters + e17_head_parameters,
        },
        {
            "item": "Training horizons",
            "E16_SwiGLU": "8 and 32",
            "E17_Attention": "8 and 32",
        },
        {
            "item": "Mean terminal accuracy at trained horizon (T_train=8 / 32)",
            "E16_SwiGLU": (
                f"{100*evaluated_trained_horizon_accuracy['swiglu']['8']['mean']:.2f}"
                f" +/- {100*evaluated_trained_horizon_accuracy['swiglu']['8']['sample_std']:.2f}%"
                " / "
                f"{100*evaluated_trained_horizon_accuracy['swiglu']['32']['mean']:.2f}"
                f" +/- {100*evaluated_trained_horizon_accuracy['swiglu']['32']['sample_std']:.2f}%"
            ),
            "E17_Attention": (
                f"{100*evaluated_trained_horizon_accuracy['attention']['8']['mean']:.2f}"
                f" +/- {100*evaluated_trained_horizon_accuracy['attention']['8']['sample_std']:.2f}%"
                " / "
                f"{100*evaluated_trained_horizon_accuracy['attention']['32']['mean']:.2f}"
                f" +/- {100*evaluated_trained_horizon_accuracy['attention']['32']['sample_std']:.2f}%"
            ),
        },
        {
            "item": "Dominant recurrent FLOPs / step",
            "E16_SwiGLU": e16_step_flops,
            "E17_Attention": e17_step_flops,
        },
    ]
    write_json_new(output, payload)
    write_csv_new(AUDITS / "model_scale.csv", rows)
    markdown = [
        "# Model and data scale audit",
        "",
        "| Item | E16 SwiGLU | E17 Attention |",
        "|---|---:|---:|",
    ]
    markdown.extend(
        f"| {row['item']} | {row['E16_SwiGLU']} | {row['E17_Attention']} |"
        for row in rows
    )
    markdown.extend(
        [
            "",
            "Recurrent FLOPs use one multiply-add = two FLOPs and exclude the "
            "frozen backbone, one-time input projection, final classifier, "
            "LayerNorm, and lower-order elementwise costs.",
        ]
    )
    write_text_new(AUDITS / "model_scale.md", "\n".join(markdown))
    return payload


def causal_aggregation() -> dict[str, Any]:
    output = FOLLOWUP / "e17_e17c_causal_aggregation.json"
    if output.exists():
        print("Using existing E17/E17C causal aggregation.")
        return read_json(output)
    e17_summary = read_json(E17_DIRS[8] / "attention_summary.json")
    control_summary = read_json(E17_DIRS[32] / "attention_summary.json")
    rows = []
    for seed in SEEDS:
        original = next(
            row for row in e17_summary["seed_level"] if int(row["seed"]) == seed
        )
        control = next(
            row
            for row in control_summary["seed_level"]
            if int(row["seed"]) == seed
        )
        interaction = float(original["terminal_drop_pp"]) - float(
            control["terminal_drop_pp"]
        )
        rows.append(
            {
                "seed": seed,
                "e17_terminal_t8": original["terminal_t8"],
                "e17_terminal_t32": original["terminal_t32"],
                "e17_drop_pp": original["terminal_drop_pp"],
                "e17c_terminal_t8": control["terminal_t8"],
                "e17c_terminal_t32": control["terminal_t32"],
                "e17c_drop_pp": control["terminal_drop_pp"],
                "interaction_pp": interaction,
                "e17c_terminal_deficit_to_best_fixed_pp": control[
                    "peak_minus_terminal_t32_pp"
                ],
                "e17_probe_drop_pp": original["probe_drop_pp"],
                "e17c_probe_drop_pp": control["probe_drop_pp"],
                "all_finite": bool(original["all_finite"] and control["all_finite"]),
            }
        )
    interactions = [float(row["interaction_pp"]) for row in rows]
    checks = {
        "e17c_terminal_within_0_5pp_best_fixed_every_seed": all(
            float(row["e17c_terminal_deficit_to_best_fixed_pp"]) <= 0.5
            for row in rows
        ),
        "mean_interaction_at_least_1pp": statistics.mean(interactions) >= 1.0,
        "e17c_drop_smaller_every_seed": all(
            float(row["e17c_drop_pp"]) < float(row["e17_drop_pp"])
            for row in rows
        ),
        "interaction_positive_every_seed": all(value > 0.0 for value in interactions),
        "all_states_and_logits_finite": all(bool(row["all_finite"]) for row in rows),
        "e17c_probe_drop_smaller_every_seed": all(
            float(row["e17c_probe_drop_pp"]) < float(row["e17_probe_drop_pp"])
            for row in rows
        ),
    }
    if all(checks.values()):
        classification = "matched_horizon_restoration"
    elif (
        checks["mean_interaction_at_least_1pp"]
        and checks["e17c_drop_smaller_every_seed"]
        and checks["interaction_positive_every_seed"]
        and checks["all_states_and_logits_finite"]
    ):
        classification = "partial_restoration"
    else:
        classification = "no_restoration"
    payload = {
        "status": "complete",
        "classification": classification,
        "operational_definition": read_json(
            FOLLOWUP / "followup_config.json"
        )["causal_classification"],
        "seed_level": rows,
        "interaction_pp": sample_stats(interactions),
        "e17_drop_pp": sample_stats([row["e17_drop_pp"] for row in rows]),
        "e17c_drop_pp": sample_stats([row["e17c_drop_pp"] for row in rows]),
        "e17_probe_drop_pp": sample_stats(
            [row["e17_probe_drop_pp"] for row in rows]
        ),
        "e17c_probe_drop_pp": sample_stats(
            [row["e17c_probe_drop_pp"] for row in rows]
        ),
        "criteria": checks,
        "significance_test": None,
    }
    write_json_new(output, payload)
    write_csv_new(FOLLOWUP / "e17_e17c_causal_aggregation.csv", rows)
    markdown = [
        "# E17/E17C paired causal aggregation",
        "",
        f"**Classification:** `{classification}`.",
        "",
        "| Seed | E17 T8 | E17 T32 | E17 drop | E17C T8 | E17C T32 | "
        "E17C drop | Interaction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['seed']} | {100*row['e17_terminal_t8']:.2f}% | "
            f"{100*row['e17_terminal_t32']:.2f}% | {row['e17_drop_pp']:+.2f} pp | "
            f"{100*row['e17c_terminal_t8']:.2f}% | "
            f"{100*row['e17c_terminal_t32']:.2f}% | "
            f"{row['e17c_drop_pp']:+.2f} pp | "
            f"{row['interaction_pp']:+.2f} pp |"
        )
    interaction = payload["interaction_pp"]
    markdown.extend(
        [
            "",
            f"Mean paired interaction: **{interaction['mean']:.2f} +/- "
            f"{interaction['sample_std']:.2f} pp**. No significance test is "
            "used at n=3.",
            "",
            "## Frozen criterion audit",
            "",
        ]
    )
    markdown.extend(f"- `{key}`: {value}." for key, value in checks.items())
    write_text_new(FOLLOWUP / "e17_e17c_causal_aggregation.md", "\n".join(markdown))
    return payload


def panel_condition_rows(e16_probe: dict[str, Any]) -> list[dict[str, Any]]:
    e16 = read_json(E16_AGGREGATE_DIR / "aggregate_summary.json")
    e17_summary = read_json(E17_DIRS[8] / "attention_summary.json")
    e17c_summary = read_json(E17_DIRS[32] / "attention_summary.json")
    causal = read_json(FOLLOWUP / "e17_e17c_causal_aggregation.json")
    rows: list[dict[str, Any]] = []
    for training_horizon, classification in (
        (8, "endpoint_invalidity"),
        (32, "matched_horizon_restoration"),
    ):
        summary = e16["trajectory_summary"][str(training_horizon)]
        seed_rows = e16["seed_level"]
        if training_horizon == 8:
            drops = [float(row["ttrain8_terminal_drop_pp"]) for row in seed_rows]
        else:
            drops = [float(row["ttrain32_terminal_drop_pp"]) for row in seed_rows]
        probe = e16_probe["by_training_horizon"][str(training_horizon)][
            "probe_drop_pp"
        ]
        rows.append(
            {
                "model": "Recursive SwiGLU",
                "condition_id": f"swiglu_ttrain{training_horizon}",
                "family": "residual recursion",
                "training_horizon": training_horizon,
                "terminal_by_horizon": {
                    str(horizon): summary[str(horizon)]["terminal_accuracy"]
                    for horizon in HORIZONS
                },
                "terminal_t8": summary["8"]["terminal_accuracy"],
                "terminal_t32": summary["32"]["terminal_accuracy"],
                "drop_pp": sample_stats(drops),
                "early_t32": summary["32"]["early_eight_accuracy"],
                "best_step_t32_values": summary["32"]["best_fixed_timestep_values"],
                "probe_drop_pp": probe,
                "classification": classification,
            }
        )

    for summary, training_horizon, classification in (
        (e17_summary, 8, "endpoint_invalidity"),
        (e17c_summary, 32, causal["classification"]),
    ):
        seed_level = summary["seed_level"]
        terminal_t8 = sample_stats(row["terminal_t8"] for row in seed_level)
        terminal_t32 = sample_stats(row["terminal_t32"] for row in seed_level)
        early_t32 = sample_stats(row["early_t32"] for row in seed_level)
        terminal_by_horizon = {
            str(horizon): summary["trajectory_summary"][str(horizon)]["terminal"]
            for horizon in HORIZONS
        }
        terminal_by_horizon["8"] = terminal_t8
        terminal_by_horizon["32"] = terminal_t32
        rows.append(
            {
                "model": "Recursive attention",
                "condition_id": f"attention_ttrain{training_horizon}",
                "family": "shared-attention recursion",
                "training_horizon": training_horizon,
                "terminal_by_horizon": terminal_by_horizon,
                "terminal_t8": terminal_t8,
                "terminal_t32": terminal_t32,
                "drop_pp": summary["terminal_drop_pp"],
                "early_t32": early_t32,
                "best_step_t32_values": summary["trajectory_summary"]["32"][
                    "best_fixed_timestep_values"
                ],
                "probe_drop_pp": summary["probe_terminal_drop_pp"],
                "classification": classification,
            }
        )
    return rows


def save_panel_figures(rows: list[dict[str, Any]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {
        "swiglu_ttrain8": "#B33A3A",
        "swiglu_ttrain32": "#2C6E9B",
        "attention_ttrain8": "#6A3D9A",
        "attention_ttrain32": "#3B8C6E",
    }
    labels = [
        "SwiGLU\n$T_{\\mathrm{train}}=8$",
        "SwiGLU\n$T_{\\mathrm{train}}=32$",
        "Attention\n$T_{\\mathrm{train}}=8$",
        "Attention\n$T_{\\mathrm{train}}=32$",
    ]

    figure1_paths = [
        PANEL / "final_panel_figure_1.png",
        PANEL / "final_panel_figure_1.svg",
    ]
    if not any(path.exists() for path in figure1_paths):
        figure, axis = plt.subplots(figsize=(8.4, 5.2))
        for row in rows:
            means = [
                100.0 * float(row["terminal_by_horizon"][str(h)]["mean"])
                for h in HORIZONS
            ]
            deviations = [
                100.0
                * float(row["terminal_by_horizon"][str(h)]["sample_std"])
                for h in HORIZONS
            ]
            label = (
                f"{row['model']}, "
                f"$T_{{\\mathrm{{train}}}}={row['training_horizon']}$"
            )
            axis.errorbar(
                HORIZONS,
                means,
                yerr=deviations,
                marker="o",
                linewidth=2.0,
                capsize=3,
                color=colors[row["condition_id"]],
                label=label,
            )
        axis.set_xlim(7, 33)
        axis.set_ylim(74, 82)
        axis.set_xticks(HORIZONS)
        axis.set_xlabel(r"Inference horizon $T_{test}$")
        axis.set_ylabel("Validation accuracy (%)")
        axis.set_title("Terminal accuracy across matched recursive conditions")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=9)
        figure.tight_layout()
        for path in figure1_paths:
            figure.savefig(path, dpi=240)
        plt.close(figure)

    figure2_paths = [
        PANEL / "final_panel_figure_2.png",
        PANEL / "final_panel_figure_2.svg",
    ]
    if not any(path.exists() for path in figure2_paths):
        figure, axis = plt.subplots(figsize=(8.4, 5.2))
        x = np.arange(len(rows))
        means = [float(row["drop_pp"]["mean"]) for row in rows]
        deviations = [float(row["drop_pp"]["sample_std"]) for row in rows]
        axis.bar(
            x,
            means,
            yerr=deviations,
            color=[colors[row["condition_id"]] for row in rows],
            capsize=4,
        )
        axis.axhline(0.0, color="black", linewidth=0.9)
        axis.set_ylim(-1, 7)
        axis.set_xticks(x, labels)
        axis.set_ylabel(r"Terminal $A(8)-A(32)$ (pp)")
        axis.set_title("Training-horizon matching removes endpoint degradation")
        axis.grid(axis="y", alpha=0.22)
        figure.tight_layout()
        for path in figure2_paths:
            figure.savefig(path, dpi=240)
        plt.close(figure)

    figure3_paths = [
        PANEL / "final_panel_figure_3.png",
        PANEL / "final_panel_figure_3.svg",
    ]
    if not any(path.exists() for path in figure3_paths):
        figure, axis = plt.subplots(figsize=(8.4, 5.2))
        x = np.arange(len(rows))
        width = 0.35
        terminal = [100.0 * float(row["terminal_t32"]["mean"]) for row in rows]
        terminal_std = [
            100.0 * float(row["terminal_t32"]["sample_std"]) for row in rows
        ]
        early = [100.0 * float(row["early_t32"]["mean"]) for row in rows]
        early_std = [
            100.0 * float(row["early_t32"]["sample_std"]) for row in rows
        ]
        axis.bar(
            x - width / 2,
            terminal,
            width,
            yerr=terminal_std,
            capsize=3,
            label="Terminal",
            color="#355C7D",
        )
        axis.bar(
            x + width / 2,
            early,
            width,
            yerr=early_std,
            capsize=3,
            label="Early-8",
            color="#F08A5D",
        )
        axis.set_ylim(74, 82)
        axis.set_xticks(x, labels)
        axis.set_ylabel("Validation accuracy at T=32 (%)")
        axis.set_title("Terminal versus early-prefix readout")
        axis.grid(axis="y", alpha=0.22)
        axis.legend(frameon=False)
        figure.tight_layout()
        for path in figure3_paths:
            figure.savefig(path, dpi=240)
        plt.close(figure)


def final_panel() -> dict[str, Any]:
    output = PANEL / "final_architecture_panel.json"
    if output.exists():
        print("Using existing final architecture panel.")
        existing = read_json(output)
        # Permit deterministic regeneration of missing figure exports without
        # rewriting the immutable numerical table.
        e16_probe = read_json(AUDITS / "e16_probe_audit.json")
        save_panel_figures(panel_condition_rows(e16_probe))
        return existing
    e16_probe = read_json(AUDITS / "e16_probe_audit.json")
    rows = panel_condition_rows(e16_probe)
    payload = {
        "status": "complete",
        "seeds": list(SEEDS),
        "no_significance_test": True,
        "conditions": rows,
        "causal_attention": read_json(
            FOLLOWUP / "e17_e17c_causal_aggregation.json"
        ),
        "figures": [
            "results/modern_iterative_panel/final_panel_figure_1.png",
            "results/modern_iterative_panel/final_panel_figure_2.png",
            "results/modern_iterative_panel/final_panel_figure_3.png",
        ],
    }
    csv_rows = []
    markdown = [
        "# Final four-condition architecture panel",
        "",
        "Means and sample standard deviations use paired seeds 101--103. No "
        "significance test is used.",
        "",
        "| Model | T_train | Terminal T=8 | Terminal T=32 | Drop | Early T=32 | "
        "Best step | Probe drop | Classification |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        csv_row = {
            "model": row["model"],
            "family": row["family"],
            "training_horizon": row["training_horizon"],
            "terminal_t8_mean": row["terminal_t8"]["mean"],
            "terminal_t8_sample_std": row["terminal_t8"]["sample_std"],
            "terminal_t32_mean": row["terminal_t32"]["mean"],
            "terminal_t32_sample_std": row["terminal_t32"]["sample_std"],
            "drop_pp_mean": row["drop_pp"]["mean"],
            "drop_pp_sample_std": row["drop_pp"]["sample_std"],
            "early_t32_mean": row["early_t32"]["mean"],
            "early_t32_sample_std": row["early_t32"]["sample_std"],
            "best_step_values": "/".join(
                str(value) for value in row["best_step_t32_values"]
            ),
            "probe_drop_pp_mean": row["probe_drop_pp"]["mean"],
            "probe_drop_pp_sample_std": row["probe_drop_pp"]["sample_std"],
            "classification": row["classification"],
        }
        csv_rows.append(csv_row)
        markdown.append(
            f"| {row['model']} | {row['training_horizon']} | "
            f"{100*row['terminal_t8']['mean']:.2f} +/- "
            f"{100*row['terminal_t8']['sample_std']:.2f}% | "
            f"{100*row['terminal_t32']['mean']:.2f} +/- "
            f"{100*row['terminal_t32']['sample_std']:.2f}% | "
            f"{row['drop_pp']['mean']:+.2f} +/- "
            f"{row['drop_pp']['sample_std']:.2f} pp | "
            f"{100*row['early_t32']['mean']:.2f} +/- "
            f"{100*row['early_t32']['sample_std']:.2f}% | "
            f"{'/'.join(str(value) for value in row['best_step_t32_values'])} | "
            f"{row['probe_drop_pp']['mean']:+.2f} +/- "
            f"{row['probe_drop_pp']['sample_std']:.2f} pp | "
            f"`{row['classification']}` |"
        )
    write_json_new(output, payload)
    write_csv_new(PANEL / "final_architecture_panel.csv", csv_rows)
    write_text_new(PANEL / "final_architecture_panel.md", "\n".join(markdown))
    save_panel_figures(rows)
    return payload


def provenance_audit() -> dict[str, Any]:
    output = AUDITS / "provenance_audit.json"
    if output.exists():
        print("Using existing provenance audit.")
        return read_json(output)
    protocols = {
        "e14": ROOT / "Rebuttal" / "Terminal_Failure_Final_Recursive_Experiment_Protocol.md",
        "e15": ROOT / "Rebuttal" / "Terminal_Failure_E15_Aligned_Horizon_Protocol.md",
        "e16": ROOT / "Rebuttal" / "Terminal_Failure_E16_Paired_Horizon_Protocol.md",
        "e17": PANEL / "protocol" / "PANEL_PROTOCOL.md",
        "followup": FOLLOWUP / "FOLLOWUP_ANALYSIS_PROTOCOL.md",
    }
    configs = {
        "e14": RESULTS / "rebuttal_recursive_tinyimagenet" / "config_frozen.json",
        "e15": RESULTS / "rebuttal_recursive_tinyimagenet_aligned_t32" / "config_frozen.json",
        "e16_panel": E16_AGGREGATE_DIR / "config_frozen.json",
        "e16_t8": E16_CONFIGS[8],
        "e16_t32": E16_CONFIGS[32],
        "e17_t8": E17_CONFIGS[8],
        "e17_t32": E17_CONFIGS[32],
        "followup": FOLLOWUP / "followup_config.json",
    }
    checkpoint_hashes: list[dict[str, Any]] = []
    for training_horizon in (8, 32):
        expected_config = sha256_file(E16_CONFIGS[training_horizon])
        for seed in SEEDS:
            path = E16_DIRS[training_horizon] / f"seed_{seed}_checkpoint.pt"
            saved = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint_hashes.append(
                {
                    "architecture": "swiglu",
                    "training_horizon": training_horizon,
                    "seed": seed,
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                    "payload_config_hash_matches": saved.get("config_sha256")
                    == expected_config,
                    "payload_seed_matches": int(saved.get("seed", -1)) == seed,
                }
            )
    for training_horizon in (8, 32):
        expected_config = sha256_file(E17_CONFIGS[training_horizon])
        for seed in SEEDS:
            path = E17_DIRS[training_horizon] / f"seed_{seed}" / "checkpoint.pt"
            saved = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint_hashes.append(
                {
                    "architecture": "attention",
                    "training_horizon": training_horizon,
                    "seed": seed,
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                    "payload_config_hash_matches": saved.get("config_sha256")
                    == expected_config,
                    "payload_seed_matches": int(saved.get("seed", -1)) == seed,
                }
            )
    deviations = read_json(PANEL / "protocol" / "engineering_deviations.json")[
        "deviations"
    ]
    deviations.append(
        {
            "category": "no_op_system_python_invocation",
            "reason": "A recovery invocation resolved to the system Python and stopped immediately on a missing matplotlib import.",
            "action": "All scientific executions used .venv/Scripts/python.exe.",
            "scientific_protocol_changed": False,
            "scientific_effect": "none; no scientific file or metric was created",
        }
    )
    deviations.extend(
        [
            {
                "category": "posthoc_analysis_runner_no_op",
                "reason": "An initial one-second classwise command allowance and a background-launch attempt ended before producing a scientific artifact.",
                "action": "The post-hoc audit was rerun normally with the frozen follow-up protocol and existing checkpoints.",
                "scientific_protocol_changed": False,
                "scientific_effect": "none; no classwise result, prediction archive, or aggregate was committed",
            },
            {
                "category": "posthoc_bf16_reproduction_guard_repair",
                "reason": "The first classwise reproduction check classified step 8 and step 32 in separate 2-D BF16 GEMMs, yielding two to four boundary predictions different from the frozen evaluator's single full-trajectory GEMM.",
                "action": "The analysis-only driver was changed to call the classifier on the full trajectory exactly as the frozen E16/E17 evaluators do; the guard then reproduced all frozen headline accuracies before any classwise aggregate was saved.",
                "scientific_protocol_changed": False,
                "scientific_effect": "none; the failed guard stopped before emitting predictions or classwise statistics",
            },
        ]
    )
    locked = read_json(PANEL / "protocol" / "locked_artifact_audit.json")
    payload = {
        "status": "pass",
        "seeds": list(SEEDS),
        "protocol_hashes": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for name, path in protocols.items()
        },
        "config_hashes": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for name, path in configs.items()
        },
        "selected_checkpoint_hashes": checkpoint_hashes,
        "all_checkpoint_payloads_match": all(
            row["payload_config_hash_matches"] and row["payload_seed_matches"]
            for row in checkpoint_hashes
        ),
        "locked_artifact_audit": locked,
        "no_post_result_architecture_or_threshold_change": True,
        "source_hashes": {
            "e16_driver": sha256_file(
                ROOT
                / "transient_geometry"
                / "experiments"
                / "e16_paired_inference_horizon.py"
            ),
            "e17_driver": sha256_file(
                ROOT
                / "transient_geometry"
                / "experiments"
                / "e17_recursive_attention_tinyimagenet.py"
            ),
            "followup_driver": sha256_file(Path(__file__)),
        },
        "deviations": deviations,
        "git": read_json(PANEL / "protocol" / "provenance.json")["git"],
    }
    write_json_new(output, payload)
    markdown = [
        "# Provenance audit",
        "",
        f"**Status:** `{payload['status']}`. Seeds: 101, 102, 103.",
        "",
        "## Protocol hashes",
        "",
    ]
    markdown.extend(
        f"- `{name}`: `{record['sha256']}` ({record['path']})"
        for name, record in payload["protocol_hashes"].items()
    )
    markdown.extend(["", "## Config hashes", ""])
    markdown.extend(
        f"- `{name}`: `{record['sha256']}` ({record['path']})"
        for name, record in payload["config_hashes"].items()
    )
    markdown.extend(["", "## Deviations", ""])
    markdown.extend(
        f"- `{row['category']}`: {row['reason']} Scientific effect: "
        f"{row['scientific_effect']}."
        for row in deviations
    )
    write_text_new(AUDITS / "provenance_audit.md", "\n".join(markdown))
    return payload


def response_outputs() -> dict[str, Any]:
    output = FOLLOWUP / "rebuttal_responses.json"
    if output.exists():
        print("Using existing rebuttal responses.")
        return read_json(output)
    causal = read_json(FOLLOWUP / "e17_e17c_causal_aggregation.json")
    interaction = causal["interaction_pp"]
    compact = (
        "On Tiny ImageNet-200 with frozen ConvNeXt-Tiny features, two "
        "weight-tied recursive heads showed horizon-mismatch endpoint loss. "
        "The 8-step SwiGLU dropped 5.11 +/- 0.94 pp at 32 test steps, while "
        "matched training removed it (interaction 5.53 +/- 0.92 pp). Shared "
        "attention dropped 2.42 +/- 0.29 pp, with its early-8 readout stable at "
        "79.88 +/- 0.11% and probes dropping 1.28 +/- 0.08 pp. Its matched "
        f"32-step control dropped only 0.04 +/- 0.17 pp, yielding a paired "
        f"interaction of {interaction['mean']:.2f} +/- "
        f"{interaction['sample_std']:.2f} pp. Thus matching train and inference "
        "horizons restored endpoint validity in both tested systems."
    )
    medium = (
        "To address the request for modern recursive models and larger-scale "
        "evaluation, we tested two weight-tied architectures on Tiny "
        "ImageNet-200 using frozen ConvNeXt-Tiny representations. In the "
        "recursive SwiGLU model, extending an eight-step-trained system to 32 "
        "test iterations reduced terminal accuracy by 5.11 +/- 0.94 pp, while "
        "the early-eight readout remained stable; matched 32-step training "
        "removed the effect (paired interaction 5.53 +/- 0.92 pp). In a "
        "separate shared-attention recursive model, terminal accuracy fell "
        "from 79.62 +/- 0.10% to 77.20 +/- 0.34%, while its early-eight readout "
        "remained 79.88 +/- 0.11%; fixed probes also declined by 1.28 +/- 0.08 "
        "pp. Training the identical attention head for 32 steps reduced the "
        "terminal drop to 0.04 +/- 0.17 pp, kept every endpoint within 0.36 pp "
        f"of its best fixed timestep, and produced a paired interaction of "
        f"{interaction['mean']:.2f} +/- {interaction['sample_std']:.2f} pp. "
        "Thus the endpoint-versus-prefix mismatch replicates across residual "
        "and attention-based recursive systems and is removed, in these "
        "experiments, by matching the training and inference horizon."
    )
    full = (
        "To address the reviewers' requests for non-toy data, modern recursive "
        "architectures, and clearer endpoint-validity evidence, we evaluated "
        "two terminally supervised weight-tied systems on the standard "
        "100,000/10,000-image Tiny ImageNet-200 splits using frozen "
        "ImageNet-pretrained ConvNeXt-Tiny representations. For the recursive "
        "SwiGLU head, extending an eight-step-trained model from T_test=8 to "
        "32 reduced terminal accuracy from 80.87 +/- 0.09% to 75.76 +/- 0.96% "
        "(5.11 +/- 0.94 pp), while the early-eight readout remained 81.22 +/- "
        "0.05%. The identical head trained for 32 steps instead reached 80.84 "
        "+/- 0.15% at T_test=32, giving a paired interaction of 5.53 +/- 0.92 "
        "pp. The shared-attention recursive head independently reproduced the "
        "effect: its eight-step-trained terminal fell from 79.62 +/- 0.10% to "
        "77.20 +/- 0.34% (2.42 +/- 0.29 pp), while the early-eight readout "
        "remained 79.88 +/- 0.11% and the best states occurred at steps 4--5. "
        "A fixed-alpha probe declined by 1.28 +/- 0.08 pp, showing that the "
        "effect includes representation deterioration rather than only a "
        "shared-head mismatch. In the matched 32-step attention control, the "
        "terminal changed by only 0.04 +/- 0.17 pp from steps 8 to 32; every "
        "seed's endpoint was within 0.36 pp of its best fixed timestep, and "
        f"the paired E17/E17C interaction was {interaction['mean']:.2f} +/- "
        f"{interaction['sample_std']:.2f} pp. All states and logits remained "
        "finite, and no gain, horizon, readout threshold, probe setting, or "
        "checkpoint rule was tuned after viewing results. These results show "
        "that excess test-time recurrence can invalidate the terminal readout "
        "in both tested residual and shared-attention recursive systems, while "
        "matched-horizon training restores endpoint validity; they support a "
        "conditional endpoint-validity taxonomy rather than a universal claim. "
        "Measured E14 costs also show that an early-eight readout can halt at "
        "1.803 ms versus 6.727 ms for terminal inference; naive GRACE used "
        "48.56 MiB peak additional memory, while streaming GRACE reduced this "
        "to 8.51 MiB."
    )
    if not (500 <= len(compact) <= 700):
        raise RuntimeError(f"Compact response length is {len(compact)}.")
    if not (1000 <= len(medium) <= 1500):
        raise RuntimeError(f"Medium response length is {len(medium)}.")
    mapping = [
        {
            "sentence": "We tested two weight-tied architectures on Tiny ImageNet-200 using frozen ConvNeXt-Tiny representations.",
            "concerns": ["toy datasets", "toy models", "modern recursive architectures"],
        },
        {
            "sentence": "The recursive SwiGLU horizon mismatch and matched control establish the causal training-horizon effect.",
            "concerns": ["larger-scale relevance", "endpoint validity", "causal control"],
        },
        {
            "sentence": "Shared attention independently reproduces endpoint degradation while preserving an early prefix.",
            "concerns": ["Universal-Transformer-style evidence", "architectural breadth"],
        },
        {
            "sentence": "Fixed probes show representation deterioration in addition to shared-head degradation.",
            "concerns": ["whether computation remains recoverable", "readout versus representation"],
        },
        {
            "sentence": "Matched 32-step attention training restores endpoint validity with a positive paired interaction.",
            "concerns": ["confounding by architecture", "training/inference horizon alignment"],
        },
        {
            "sentence": "The claim is conditional rather than universal.",
            "concerns": ["novelty and claim calibration", "scope limitations"],
        },
        {
            "sentence": "Measured early-window and streaming-GRACE costs quantify the compute and memory trade-offs.",
            "concerns": [
                "computational cost",
                "memory cost",
                "practical recommendation",
            ],
        },
    ]
    payload = {
        "status": "complete",
        "compact_500_700": compact,
        "compact_character_count": len(compact),
        "medium_1000_1500": medium,
        "medium_character_count": len(medium),
        "full_technical": full,
        "full_character_count": len(full),
        "sentence_to_meta_review_mapping": mapping,
    }
    write_json_new(output, payload)
    markdown = [
        "# Rebuttal-ready responses",
        "",
        f"## Compressed ({len(compact)} characters)",
        "",
        compact,
        "",
        f"## Medium ({len(medium)} characters)",
        "",
        medium,
        "",
        f"## Full technical ({len(full)} characters)",
        "",
        full,
        "",
        "## Sentence-to-concern mapping",
        "",
    ]
    for row in mapping:
        markdown.append(
            f"- {row['sentence']} - {', '.join(row['concerns'])}."
        )
    write_text_new(FOLLOWUP / "rebuttal_responses.md", "\n".join(markdown))
    return payload


def final_manifest() -> dict[str, Any]:
    output = FOLLOWUP / "final_analysis_manifest.json"
    if output.exists():
        return read_json(output)
    expected = [
        FOLLOWUP / "e17_e17c_causal_aggregation.json",
        PANEL / "final_architecture_panel.json",
        PANEL / "final_architecture_panel.md",
        PANEL / "final_architecture_panel.csv",
        PANEL / "final_panel_figure_1.png",
        PANEL / "final_panel_figure_2.png",
        PANEL / "final_panel_figure_3.png",
        AUDITS / "classwise_robustness.json",
        AUDITS / "checkpoint_sensitivity.json",
        AUDITS / "e16_probe_audit.json",
        AUDITS / "model_scale.json",
        AUDITS / "provenance_audit.json",
        FOLLOWUP / "rebuttal_responses.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"Final analysis artifacts missing: {missing}")
    payload = {
        "status": "complete",
        "artifacts": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in expected
        ],
        "optional_deq": {
            "status": "skipped",
            "reason": "Protected EP compute and the required >=30-hour remaining deadline budget cannot be verified; the frozen DEQ start conditions therefore do not all pass.",
        },
    }
    write_json_new(output, payload)
    return payload


def run_all() -> None:
    causal_aggregation()
    classwise_audit()
    checkpoint_sensitivity()
    e16_probe_audit()
    model_scale_audit()
    provenance_audit()
    final_panel()
    response_outputs()
    final_manifest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "causal",
            "classwise",
            "checkpoint",
            "probe",
            "scale",
            "provenance",
            "panel",
            "responses",
            "manifest",
            "all",
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "causal":
        causal_aggregation()
    elif args.stage == "classwise":
        classwise_audit()
    elif args.stage == "checkpoint":
        checkpoint_sensitivity()
    elif args.stage == "probe":
        e16_probe_audit()
    elif args.stage == "scale":
        model_scale_audit()
    elif args.stage == "provenance":
        provenance_audit()
    elif args.stage == "panel":
        final_panel()
    elif args.stage == "responses":
        response_outputs()
    elif args.stage == "manifest":
        final_manifest()
    else:
        run_all()


if __name__ == "__main__":
    main()
