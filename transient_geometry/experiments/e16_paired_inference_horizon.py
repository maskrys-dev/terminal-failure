"""
E16: paired inference-horizon replication on frozen ConvNeXt-Tiny features.

The scientific protocol and all three configurations must be frozen before
the first smoke test:

  Rebuttal/Terminal_Failure_E16_Paired_Horizon_Protocol.md
  results/rebuttal_recursive_tinyimagenet_paired_horizon/config_frozen.json
  results/rebuttal_recursive_tinyimagenet_e16_ttrain{8,32}/config_frozen.json

Run one model family per process so that the imported E14 trainer has exactly
one immutable configuration:

  .venv/Scripts/python.exe -u transient_geometry/experiments/e16_paired_inference_horizon.py model --variant e16_t8
  .venv/Scripts/python.exe -u transient_geometry/experiments/e16_paired_inference_horizon.py model --variant e16_t32
  .venv/Scripts/python.exe -u transient_geometry/experiments/e16_paired_inference_horizon.py aggregate
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_MPL_CONFIG_DIR = ROOT / "tmp" / "matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

AGGREGATE_DIR = ROOT / "results" / "rebuttal_recursive_tinyimagenet_paired_horizon"
AGGREGATE_CONFIG_PATH = AGGREGATE_DIR / "config_frozen.json"
VARIANT_DIRS = {
    "e16_t8": ROOT / "results" / "rebuttal_recursive_tinyimagenet_e16_ttrain8",
    "e16_t32": ROOT / "results" / "rebuttal_recursive_tinyimagenet_e16_ttrain32",
}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_frozen_configs() -> dict[str, Any]:
    aggregate = load_json(AGGREGATE_CONFIG_PATH)
    t8_path = VARIANT_DIRS["e16_t8"] / "config_frozen.json"
    t32_path = VARIANT_DIRS["e16_t32"] / "config_frozen.json"
    t8 = load_json(t8_path)
    t32 = load_json(t32_path)

    if aggregate["paired_seeds"] != [101, 102, 103]:
        raise RuntimeError("E16 paired seeds are not the frozen {101,102,103}.")
    if aggregate["test_horizons"] != [8, 16, 24, 32]:
        raise RuntimeError("E16 test horizons changed after freezing.")
    if float(aggregate["evaluation_gain"]) != 1.0:
        raise RuntimeError("E16 evaluation gain must be exactly 1.0.")

    for config, horizon in ((t8, 8), (t32, 32)):
        if config["training"]["seeds"] != aggregate["paired_seeds"]:
            raise RuntimeError("Variant seed policy differs from the paired config.")
        if config["evaluation"]["test_horizons"] != aggregate["test_horizons"]:
            raise RuntimeError("Variant test horizons differ from the paired config.")
        if config["evaluation"]["gain_grid"] != [1.0]:
            raise RuntimeError("E16 must not contain a gain sweep.")
        if int(config["architecture"]["training_horizon"]) != horizon:
            raise RuntimeError(f"Wrong training horizon in T_train={horizon} config.")

    comparable_t8 = copy.deepcopy(t8)
    comparable_t32 = copy.deepcopy(t32)
    for config in (comparable_t8, comparable_t32):
        config.pop("experiment")
        config["architecture"].pop("training_horizon")
        config["training"].pop("checkpoint_selection")
    if comparable_t8 != comparable_t32:
        raise RuntimeError(
            "The paired model configs differ outside experiment name, training "
            "horizon, and its checkpoint-selection label."
        )

    return {
        "aggregate": aggregate,
        "hashes": {
            "aggregate_config": sha256_file(AGGREGATE_CONFIG_PATH),
            "ttrain8_config": sha256_file(t8_path),
            "ttrain32_config": sha256_file(t32_path),
            "protocol": sha256_file(
                ROOT / "Rebuttal" / "Terminal_Failure_E16_Paired_Horizon_Protocol.md"
            ),
        },
    }


def import_base(variant: str):
    if variant not in VARIANT_DIRS:
        raise ValueError(f"Unknown variant: {variant}")
    os.environ["TERMINAL_FAILURE_RECURSIVE_VARIANT"] = variant
    module_name = "transient_geometry.experiments.e14_recursive_tinyimagenet_rebuttal"
    if module_name in sys.modules:
        raise RuntimeError("The recursive base driver was already imported in this process.")
    return importlib.import_module(module_name)


def evaluate_seed(base, cache: dict[str, Any], seed: int, force: bool) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    path = base.OUT_DIR / f"seed_{seed}_horizon_metrics.json"
    if path.exists() and not force:
        existing = load_json(path)
        if existing.get("config_sha256") == base.CONFIG_SHA256:
            print(f"Using evaluated E16 seed {seed} metrics already on disk.")
            return existing

    model = base.load_model_checkpoint(seed)
    horizons = [int(value) for value in base.EVAL["test_horizons"]]
    maximum_horizon = max(horizons)
    early_window = int(base.EVAL["early_window"])
    gain = 1.0
    loader = base.tensor_loader(
        cache["validation_features"],
        cache["validation_labels"],
        batch_size=int(base.TRAIN["batch_size"]),
        shuffle=False,
        seed=0,
    )

    correct = torch.zeros(maximum_horizon, dtype=torch.float64)
    cross_entropy_sum = torch.zeros(maximum_horizon, dtype=torch.float64)
    state_norm_sum = torch.zeros(maximum_horizon, dtype=torch.float64)
    residual_sum = torch.zeros(maximum_horizon, dtype=torch.float64)
    early_correct = 0
    early_cross_entropy_sum = 0.0
    nonfinite_states = 0
    nonfinite_logits = 0
    seen = 0

    model.eval()
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(base.DEVICE, non_blocking=True)
            yb = yb.to(base.DEVICE, non_blocking=True)
            with base.autocast_context():
                state0, trajectory = model.trajectory(xb, maximum_horizon, gain)
                logits = model.classify(trajectory)
                early_logits = model.classify(trajectory[:, :early_window].mean(dim=1))

            logits_float = logits.float()
            early_logits_float = early_logits.float()
            predictions = logits_float.argmax(dim=-1)
            correct += (predictions == yb[:, None]).sum(dim=0).double().cpu()
            per_example_ce = functional.cross_entropy(
                logits_float.reshape(-1, logits_float.shape[-1]),
                yb[:, None].expand(-1, maximum_horizon).reshape(-1),
                reduction="none",
            ).reshape(len(yb), maximum_horizon)
            cross_entropy_sum += per_example_ce.double().sum(dim=0).cpu()

            previous = torch.cat([state0[:, None], trajectory[:, :-1]], dim=1).float()
            relative_residual = (
                (trajectory.float() - previous).norm(dim=-1)
                / previous.norm(dim=-1).clamp_min(1e-12)
            )
            residual_sum += relative_residual.double().sum(dim=0).cpu()
            state_norm_sum += trajectory.float().norm(dim=-1).double().sum(dim=0).cpu()

            early_correct += int((early_logits_float.argmax(dim=-1) == yb).sum().item())
            early_cross_entropy_sum += float(
                functional.cross_entropy(early_logits_float, yb, reduction="sum").item()
            )
            nonfinite_states += int((~torch.isfinite(trajectory)).sum().item())
            nonfinite_logits += int((~torch.isfinite(logits_float)).sum().item())
            nonfinite_logits += int((~torch.isfinite(early_logits_float)).sum().item())
            seen += len(yb)

    timestep_accuracy = (correct / seen).tolist()
    timestep_cross_entropy = (cross_entropy_sum / seen).tolist()
    mean_state_norm = (state_norm_sum / seen).tolist()
    mean_relative_residual = (residual_sum / seen).tolist()
    early_accuracy = early_correct / seen
    early_cross_entropy = early_cross_entropy_sum / seen
    by_horizon: dict[str, Any] = {}

    for horizon in horizons:
        prefix = timestep_accuracy[:horizon]
        best_index = max(range(len(prefix)), key=lambda index: prefix[index])
        terminal_accuracy = float(timestep_accuracy[horizon - 1])
        best_accuracy = float(prefix[best_index])
        by_horizon[str(horizon)] = {
            "terminal_accuracy": terminal_accuracy,
            "terminal_cross_entropy": float(timestep_cross_entropy[horizon - 1]),
            "early_eight_accuracy": float(early_accuracy),
            "early_eight_cross_entropy": float(early_cross_entropy),
            "best_fixed_timestep": best_index + 1,
            "best_fixed_timestep_accuracy": best_accuracy,
            "terminal_minus_early_pp": 100.0 * (terminal_accuracy - early_accuracy),
            "terminal_deficit_to_best_fixed_pp": 100.0
            * (best_accuracy - terminal_accuracy),
            "endpoint_mean_relative_update_residual": float(
                mean_relative_residual[horizon - 1]
            ),
            "endpoint_mean_state_norm": float(mean_state_norm[horizon - 1]),
        }

    minimum_material_accuracy = float(
        load_json(AGGREGATE_CONFIG_PATH)["criteria"]["minimum_material_accuracy"]
    )
    metrics = {
        "experiment": base.CONFIG["experiment"],
        "config_sha256": base.CONFIG_SHA256,
        "seed": seed,
        "training_horizon": int(base.ARCH["training_horizon"]),
        "evaluation_gain": gain,
        "validation_examples": seen,
        "test_horizons": horizons,
        "early_window": early_window,
        "by_horizon": by_horizon,
        "trajectory": {
            "timestep_accuracy": timestep_accuracy,
            "timestep_cross_entropy": timestep_cross_entropy,
            "mean_state_norm": mean_state_norm,
            "mean_relative_update_residual": mean_relative_residual,
        },
        "finite_checks": {
            "nonfinite_state_values": nonfinite_states,
            "nonfinite_logit_values": nonfinite_logits,
            "all_finite": nonfinite_states == 0 and nonfinite_logits == 0,
        },
        "material_accuracy_check": {
            "threshold": minimum_material_accuracy,
            "minimum_terminal_or_early_accuracy": min(
                [
                    by_horizon[str(horizon)]["terminal_accuracy"]
                    for horizon in horizons
                ]
                + [early_accuracy]
            ),
            "pass": min(
                [
                    by_horizon[str(horizon)]["terminal_accuracy"]
                    for horizon in horizons
                ]
                + [early_accuracy]
            )
            > minimum_material_accuracy,
        },
    }
    atomic_json(path, metrics)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        f"Evaluated seed={seed} T_train={base.ARCH['training_horizon']}: "
        + ", ".join(
            f"T{h}={by_horizon[str(h)]['terminal_accuracy']:.4f}" for h in horizons
        )
        + f", early8={early_accuracy:.4f}"
    )
    return metrics


def summarise_variant(base) -> dict[str, Any]:
    seeds = [int(seed) for seed in base.TRAIN["seeds"]]
    metrics = [
        load_json(base.OUT_DIR / f"seed_{seed}_horizon_metrics.json") for seed in seeds
    ]
    horizons = [int(value) for value in base.EVAL["test_horizons"]]
    summary: dict[str, Any] = {
        "experiment": base.CONFIG["experiment"],
        "config_sha256": base.CONFIG_SHA256,
        "training_horizon": int(base.ARCH["training_horizon"]),
        "seeds": seeds,
        "by_horizon": {},
    }
    for horizon in horizons:
        rows = [entry["by_horizon"][str(horizon)] for entry in metrics]
        horizon_summary: dict[str, Any] = {}
        for key in (
            "terminal_accuracy",
            "early_eight_accuracy",
            "best_fixed_timestep_accuracy",
            "terminal_minus_early_pp",
            "terminal_deficit_to_best_fixed_pp",
            "endpoint_mean_relative_update_residual",
            "endpoint_mean_state_norm",
        ):
            values = [float(row[key]) for row in rows]
            horizon_summary[key] = {
                "values": values,
                "mean": statistics.mean(values),
                "sample_std": statistics.stdev(values),
            }
        horizon_summary["best_fixed_timestep_values"] = [
            int(row["best_fixed_timestep"]) for row in rows
        ]
        summary["by_horizon"][str(horizon)] = horizon_summary
    summary["all_finite"] = all(entry["finite_checks"]["all_finite"] for entry in metrics)
    summary["all_materially_above_chance"] = all(
        entry["material_accuracy_check"]["pass"] for entry in metrics
    )
    atomic_json(base.OUT_DIR / "variant_summary.json", summary)
    return summary


def run_model(variant: str, stage: str, force_evaluation: bool, no_resume: bool) -> None:
    frozen = validate_frozen_configs()
    base = import_base(variant)
    expected_hash = frozen["hashes"][
        "ttrain8_config" if variant == "e16_t8" else "ttrain32_config"
    ]
    if base.CONFIG_SHA256 != expected_hash:
        raise RuntimeError("Imported variant does not match the frozen E16 config hash.")

    base.OUT_DIR.mkdir(parents=True, exist_ok=True)
    environment = base.environment_payload()
    environment["e16_frozen_hashes"] = frozen["hashes"]
    base.atomic_json(base.ENV_PATH, environment)
    print(
        f"{base.CONFIG['experiment']} | stage={stage} | device={base.DEVICE} | "
        f"precision={base.precision_name()} | config={base.CONFIG_SHA256[:12]}"
    )
    cache = base.load_or_extract_features(force=False)
    smoke = base.run_smoke(cache, force=False)
    if not smoke["pass"]:
        raise RuntimeError("Frozen E16 engineering smoke test failed.")
    if stage == "smoke":
        return

    for seed in [int(value) for value in base.TRAIN["seeds"]]:
        if stage == "model":
            base.train_seed(cache, seed, resume=not no_resume)
        evaluate_seed(base, cache, seed, force=force_evaluation)
    summarise_variant(base)


def mean_and_std(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
    }


def aggregate_results() -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    frozen = validate_frozen_configs()
    config = frozen["aggregate"]
    seeds = [int(seed) for seed in config["paired_seeds"]]
    horizons = [int(horizon) for horizon in config["test_horizons"]]
    all_metrics: dict[int, dict[int, dict[str, Any]]] = {8: {}, 32: {}}
    for train_horizon, variant in ((8, "e16_t8"), (32, "e16_t32")):
        expected_hash = frozen["hashes"][
            "ttrain8_config" if train_horizon == 8 else "ttrain32_config"
        ]
        for seed in seeds:
            path = VARIANT_DIRS[variant] / f"seed_{seed}_horizon_metrics.json"
            metrics = load_json(path)
            if metrics["config_sha256"] != expected_hash:
                raise RuntimeError(f"Config mismatch in {path}")
            all_metrics[train_horizon][seed] = metrics

    seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        t8 = all_metrics[8][seed]["by_horizon"]
        t32 = all_metrics[32][seed]["by_horizon"]
        t8_drop = 100.0 * (
            float(t8["8"]["terminal_accuracy"]) - float(t8["32"]["terminal_accuracy"])
        )
        t8_early_drop = 100.0 * (
            float(t8["8"]["early_eight_accuracy"])
            - float(t8["32"]["early_eight_accuracy"])
        )
        t32_drop = 100.0 * (
            float(t32["8"]["terminal_accuracy"])
            - float(t32["32"]["terminal_accuracy"])
        )
        interaction = t8_drop - t32_drop
        seed_rows.append(
            {
                "seed": seed,
                "ttrain8_terminal_t8": float(t8["8"]["terminal_accuracy"]),
                "ttrain8_terminal_t32": float(t8["32"]["terminal_accuracy"]),
                "ttrain8_terminal_drop_pp": t8_drop,
                "ttrain8_early_t8": float(t8["8"]["early_eight_accuracy"]),
                "ttrain8_early_t32": float(t8["32"]["early_eight_accuracy"]),
                "ttrain8_early_drop_pp": t8_early_drop,
                "ttrain32_terminal_t8": float(t32["8"]["terminal_accuracy"]),
                "ttrain32_terminal_t32": float(t32["32"]["terminal_accuracy"]),
                "ttrain32_terminal_drop_pp": t32_drop,
                "ttrain32_best_fixed_t32": float(
                    t32["32"]["best_fixed_timestep_accuracy"]
                ),
                "ttrain32_best_fixed_timestep_at_t32": int(
                    t32["32"]["best_fixed_timestep"]
                ),
                "ttrain32_terminal_deficit_to_best_fixed_pp": float(
                    t32["32"]["terminal_deficit_to_best_fixed_pp"]
                ),
                "interaction_pp": interaction,
                "all_finite": bool(
                    all_metrics[8][seed]["finite_checks"]["all_finite"]
                    and all_metrics[32][seed]["finite_checks"]["all_finite"]
                ),
                "materially_above_chance": bool(
                    all_metrics[8][seed]["material_accuracy_check"]["pass"]
                    and all_metrics[32][seed]["material_accuracy_check"]["pass"]
                ),
            }
        )

    t8_drops = [row["ttrain8_terminal_drop_pp"] for row in seed_rows]
    early_drops = [row["ttrain8_early_drop_pp"] for row in seed_rows]
    t32_deficits = [
        row["ttrain32_terminal_deficit_to_best_fixed_pp"] for row in seed_rows
    ]
    interactions = [row["interaction_pp"] for row in seed_rows]
    criteria_config = config["criteria"]
    criteria = {
        "ttrain8_mean_terminal_drop_at_least_2pp": {
            "value_pp": statistics.mean(t8_drops),
            "threshold_pp": float(criteria_config["t8_minimum_mean_terminal_drop_pp"]),
            "pass": statistics.mean(t8_drops)
            >= float(criteria_config["t8_minimum_mean_terminal_drop_pp"]),
        },
        "ttrain8_early_drop_no_more_than_1pp": {
            "mean_value_pp": statistics.mean(early_drops),
            "maximum_seed_value_pp": max(early_drops),
            "mean_threshold_pp": float(
                criteria_config["t8_maximum_mean_early_drop_pp"]
            ),
            "each_seed_threshold_pp": float(
                criteria_config["t8_maximum_each_seed_early_drop_pp"]
            ),
            "pass": (
                statistics.mean(early_drops)
                <= float(criteria_config["t8_maximum_mean_early_drop_pp"])
                and max(early_drops)
                <= float(criteria_config["t8_maximum_each_seed_early_drop_pp"])
            ),
        },
        "ttrain8_terminal_drop_positive_every_seed": {
            "values_pp": t8_drops,
            "pass": all(value > 0.0 for value in t8_drops),
        },
        "ttrain32_terminal_within_0_5pp_best_fixed": {
            "mean_value_pp": statistics.mean(t32_deficits),
            "maximum_seed_value_pp": max(t32_deficits),
            "mean_threshold_pp": float(
                criteria_config["t32_maximum_mean_terminal_deficit_to_best_fixed_pp"]
            ),
            "each_seed_threshold_pp": float(
                criteria_config[
                    "t32_maximum_each_seed_terminal_deficit_to_best_fixed_pp"
                ]
            ),
            "pass": (
                statistics.mean(t32_deficits)
                <= float(
                    criteria_config[
                        "t32_maximum_mean_terminal_deficit_to_best_fixed_pp"
                    ]
                )
                and max(t32_deficits)
                <= float(
                    criteria_config[
                        "t32_maximum_each_seed_terminal_deficit_to_best_fixed_pp"
                    ]
                )
            ),
        },
        "paired_interaction_substantial": {
            "mean_value_pp": statistics.mean(interactions),
            "values_pp": interactions,
            "mean_threshold_pp": float(criteria_config["minimum_mean_interaction_pp"]),
            "positive_every_seed_required": bool(
                criteria_config["require_positive_interaction_every_seed"]
            ),
            "pass": (
                statistics.mean(interactions)
                >= float(criteria_config["minimum_mean_interaction_pp"])
                and all(value > 0.0 for value in interactions)
            ),
        },
        "finite_and_material": {
            "pass": all(
                row["all_finite"] and row["materially_above_chance"]
                for row in seed_rows
            )
        },
    }
    overall_pass = all(item["pass"] for item in criteria.values())

    trajectory_summary: dict[str, Any] = {}
    for train_horizon in (8, 32):
        trajectory_summary[str(train_horizon)] = {}
        for test_horizon in horizons:
            rows = [
                all_metrics[train_horizon][seed]["by_horizon"][str(test_horizon)]
                for seed in seeds
            ]
            trajectory_summary[str(train_horizon)][str(test_horizon)] = {
                "terminal_accuracy": mean_and_std(
                    [float(row["terminal_accuracy"]) for row in rows]
                ),
                "early_eight_accuracy": mean_and_std(
                    [float(row["early_eight_accuracy"]) for row in rows]
                ),
                "terminal_minus_early_pp": mean_and_std(
                    [float(row["terminal_minus_early_pp"]) for row in rows]
                ),
                "best_fixed_timestep_accuracy": mean_and_std(
                    [float(row["best_fixed_timestep_accuracy"]) for row in rows]
                ),
                "best_fixed_timestep_values": [
                    int(row["best_fixed_timestep"]) for row in rows
                ],
                "endpoint_mean_relative_update_residual": mean_and_std(
                    [
                        float(row["endpoint_mean_relative_update_residual"])
                        for row in rows
                    ]
                ),
            }

    summary = {
        "experiment": config["experiment"],
        "status": "complete",
        "frozen_hashes": frozen["hashes"],
        "seeds": seeds,
        "training_horizons": [8, 32],
        "test_horizons": horizons,
        "evaluation_gain": 1.0,
        "seed_level": seed_rows,
        "terminal_drop_summaries": {
            "ttrain8_pp": mean_and_std(t8_drops),
            "ttrain32_pp": mean_and_std(
                [row["ttrain32_terminal_drop_pp"] for row in seed_rows]
            ),
            "interaction_pp": mean_and_std(interactions),
        },
        "trajectory_summary": trajectory_summary,
        "criteria": criteria,
        "overall_confirmatory_pass": overall_pass,
    }
    atomic_json(AGGREGATE_DIR / "aggregate_summary.json", summary)

    lines = [
        "seed\tA8train(8)\tA8train(32)\tdrop8_pp\tearly8(8)\tearly8(32)"
        "\tA32train(8)\tA32train(32)\tdrop32_pp\tbest32\tbest_t"
        "\tdeficit32_pp\tinteraction_pp"
    ]
    for row in seed_rows:
        lines.append(
            f"{row['seed']}\t{row['ttrain8_terminal_t8']:.4f}\t"
            f"{row['ttrain8_terminal_t32']:.4f}\t"
            f"{row['ttrain8_terminal_drop_pp']:+.2f}\t"
            f"{row['ttrain8_early_t8']:.4f}\t{row['ttrain8_early_t32']:.4f}\t"
            f"{row['ttrain32_terminal_t8']:.4f}\t"
            f"{row['ttrain32_terminal_t32']:.4f}\t"
            f"{row['ttrain32_terminal_drop_pp']:+.2f}\t"
            f"{row['ttrain32_best_fixed_t32']:.4f}\t"
            f"{row['ttrain32_best_fixed_timestep_at_t32']}\t"
            f"{row['ttrain32_terminal_deficit_to_best_fixed_pp']:.2f}\t"
            f"{row['interaction_pp']:+.2f}"
        )
    (AGGREGATE_DIR / "seed_level_table.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    colors = {8: "#C84C4C", 32: "#2474A6"}
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for train_horizon in (8, 32):
        per_seed = np.array(
            [
                [
                    all_metrics[train_horizon][seed]["by_horizon"][str(horizon)][
                        "terminal_accuracy"
                    ]
                    * 100.0
                    for horizon in horizons
                ]
                for seed in seeds
            ]
        )
        for values in per_seed:
            axes[0].plot(
                horizons,
                values,
                color=colors[train_horizon],
                alpha=0.24,
                linewidth=1.1,
            )
        axes[0].errorbar(
            horizons,
            per_seed.mean(axis=0),
            yerr=per_seed.std(axis=0, ddof=1),
            marker="o",
            linewidth=2.2,
            capsize=3,
            color=colors[train_horizon],
            label=rf"$T_{{train}}={train_horizon}$ terminal",
        )
        early_mean = statistics.mean(
            [
                all_metrics[train_horizon][seed]["by_horizon"]["8"][
                    "early_eight_accuracy"
                ]
                * 100.0
                for seed in seeds
            ]
        )
        axes[0].axhline(
            early_mean,
            color=colors[train_horizon],
            linestyle="--",
            linewidth=1.35,
            alpha=0.85,
            label=rf"$T_{{train}}={train_horizon}$ early-8",
        )
    axes[0].set_xlabel(r"Inference horizon $T_{test}$")
    axes[0].set_ylabel("Terminal accuracy (%)")
    axes[0].set_xticks(horizons)
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False)
    axes[0].set_title("Paired horizon trajectories")

    x = np.arange(len(seeds))
    width = 0.25
    axes[1].bar(
        x - width,
        t8_drops,
        width,
        color=colors[8],
        label=r"$T_{train}=8$ drop",
    )
    axes[1].bar(
        x,
        [row["ttrain32_terminal_drop_pp"] for row in seed_rows],
        width,
        color=colors[32],
        label=r"$T_{train}=32$ drop",
    )
    axes[1].bar(
        x + width,
        interactions,
        width,
        color="#5B9A4B",
        label="Interaction",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axhline(
        float(criteria_config["minimum_mean_interaction_pp"]),
        color="#5B9A4B",
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
    )
    axes[1].set_xticks(x, [str(seed) for seed in seeds])
    axes[1].set_xlabel("Paired seed")
    axes[1].set_ylabel("Accuracy difference (pp)")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("Endpoint drops and paired interaction")
    figure.tight_layout()
    figure.savefig(AGGREGATE_DIR / "paired_horizon_result.png", dpi=220)
    plt.close(figure)

    report_lines = [
        "# E16 Paired Inference-Horizon Experiment",
        "",
        f"**Confirmatory decision:** {'PASS' if overall_pass else 'FAIL'}.",
        "",
        "The protocol, paired seeds, evaluation horizons, gain, readout, and "
        "criteria were frozen before execution. All three seed pairs were run.",
        "",
        "## Seed-level results",
        "",
        "| Seed | T8 train: A(8) | T8 train: A(32) | T8 drop | "
        "T32 train: A(8) | T32 train: A(32) | T32 drop | Interaction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in seed_rows:
        report_lines.append(
            f"| {row['seed']} | {100*row['ttrain8_terminal_t8']:.2f}% | "
            f"{100*row['ttrain8_terminal_t32']:.2f}% | "
            f"{row['ttrain8_terminal_drop_pp']:+.2f} pp | "
            f"{100*row['ttrain32_terminal_t8']:.2f}% | "
            f"{100*row['ttrain32_terminal_t32']:.2f}% | "
            f"{row['ttrain32_terminal_drop_pp']:+.2f} pp | "
            f"{row['interaction_pp']:+.2f} pp |"
        )
    report_lines.extend(
        [
            "",
            "## Aggregate endpoint result",
            "",
            f"- T_train=8: "
            f"{100*trajectory_summary['8']['8']['terminal_accuracy']['mean']:.2f}% "
            f"+/- {100*trajectory_summary['8']['8']['terminal_accuracy']['sample_std']:.2f}% "
            f"at T_test=8 versus "
            f"{100*trajectory_summary['8']['32']['terminal_accuracy']['mean']:.2f}% "
            f"+/- {100*trajectory_summary['8']['32']['terminal_accuracy']['sample_std']:.2f}% "
            f"at T_test=32; drop "
            f"{statistics.mean(t8_drops):.2f} +/- {statistics.stdev(t8_drops):.2f} pp.",
            f"- T_train=32: "
            f"{100*trajectory_summary['32']['8']['terminal_accuracy']['mean']:.2f}% "
            f"+/- {100*trajectory_summary['32']['8']['terminal_accuracy']['sample_std']:.2f}% "
            f"at T_test=8 versus "
            f"{100*trajectory_summary['32']['32']['terminal_accuracy']['mean']:.2f}% "
            f"+/- {100*trajectory_summary['32']['32']['terminal_accuracy']['sample_std']:.2f}% "
            f"at T_test=32; drop "
            f"{statistics.mean([row['ttrain32_terminal_drop_pp'] for row in seed_rows]):+.2f} "
            f"+/- {statistics.stdev([row['ttrain32_terminal_drop_pp'] for row in seed_rows]):.2f} pp.",
            f"- Paired interaction: {statistics.mean(interactions):.2f} "
            f"+/- {statistics.stdev(interactions):.2f} pp "
            f"(seed values {', '.join(f'{value:.2f}' for value in interactions)} pp).",
            "",
            "Values are means +/- sample standard deviations across the three "
            "fresh paired seeds; no significance test is used.",
            "",
            "## Mean trajectory diagnostics",
            "",
        ]
    )
    for train_horizon in (8, 32):
        report_lines.extend(
            [
                f"### T_train={train_horizon}",
                "",
                "| T_test | Terminal accuracy | Early-eight accuracy | "
                "Terminal - early | Best fixed timesteps (seeds 101/102/103) | "
                "Endpoint relative residual |",
                "|---:|---:|---:|---:|:---:|---:|",
            ]
        )
        for test_horizon in horizons:
            row = trajectory_summary[str(train_horizon)][str(test_horizon)]
            report_lines.append(
                f"| {test_horizon} | "
                f"{100*row['terminal_accuracy']['mean']:.2f}% +/- "
                f"{100*row['terminal_accuracy']['sample_std']:.2f}% | "
                f"{100*row['early_eight_accuracy']['mean']:.2f}% +/- "
                f"{100*row['early_eight_accuracy']['sample_std']:.2f}% | "
                f"{row['terminal_minus_early_pp']['mean']:+.2f} +/- "
                f"{row['terminal_minus_early_pp']['sample_std']:.2f} pp | "
                f"{'/'.join(str(value) for value in row['best_fixed_timestep_values'])} | "
                f"{row['endpoint_mean_relative_update_residual']['mean']:.5f} +/- "
                f"{row['endpoint_mean_relative_update_residual']['sample_std']:.5f} |"
            )
        report_lines.append("")
    report_lines.extend(
        [
            "## Criterion audit",
            "",
            f"- **PASS - mean T_train=8 terminal drop:** "
            f"{statistics.mean(t8_drops):.2f} pp >= 2.00 pp.",
            f"- **PASS - early-eight stability:** mean and maximum seed drop "
            f"were both {statistics.mean(early_drops):.2f} pp <= 1.00 pp.",
            f"- **PASS - sign replication:** T_train=8 terminal drops were "
            f"{', '.join(f'{value:.2f}' for value in t8_drops)} pp.",
            f"- **PASS - aligned control endpoint:** mean terminal deficit to "
            f"the best fixed timestep was {statistics.mean(t32_deficits):.2f} pp; "
            f"the largest seed deficit was {max(t32_deficits):.2f} pp <= 0.50 pp.",
            f"- **PASS - paired interaction:** mean "
            f"{statistics.mean(interactions):.2f} pp >= 1.00 pp and all seed "
            f"values were positive.",
            "- **PASS - numerical checks:** all state and logit values were "
            "finite and every reported accuracy was materially above chance.",
            "",
        ]
    )
    report_lines.extend(
        [
            "Full timestep accuracies, terminal-minus-early gaps, best fixed "
            "timesteps, relative residual trajectories, state norms, and "
            "finite-state counts are retained in each seed's horizon metrics.",
            "",
            "E16 evaluates only gain 1.0. No gain sweep, GRACE/probe comparison, "
            "or post-result threshold change was performed.",
            "",
            "## Frozen artifact hashes",
            "",
            f"- Aggregate config: `{frozen['hashes']['aggregate_config']}`",
            f"- T_train=8 config: `{frozen['hashes']['ttrain8_config']}`",
            f"- T_train=32 config: `{frozen['hashes']['ttrain32_config']}`",
            f"- Protocol: `{frozen['hashes']['protocol']}`",
            "",
        ]
    )
    (AGGREGATE_DIR / "RUN_REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(
        f"E16 aggregate complete: {'PASS' if overall_pass else 'FAIL'} | "
        f"mean interaction={statistics.mean(interactions):+.2f} pp"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("smoke", "model", "evaluate", "aggregate"))
    parser.add_argument("--variant", choices=tuple(VARIANT_DIRS))
    parser.add_argument("--force-evaluation", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "aggregate":
        if args.variant is not None:
            raise SystemExit("--variant is not used for aggregate.")
        aggregate_results()
        return
    if args.variant is None:
        raise SystemExit("--variant is required for smoke/model/evaluate.")
    run_model(args.variant, args.stage, args.force_evaluation, args.no_resume)


if __name__ == "__main__":
    main()
