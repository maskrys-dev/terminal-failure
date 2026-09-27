"""Derive the predeclared E14 horizon-extension audit from saved seed-0 curves."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MPL_DIR = ROOT / "tmp" / "matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = ROOT / "results" / "rebuttal_recursive_tinyimagenet"
HORIZONS = [8, 12, 16, 24, 32]
GAIN_KEY = "1.00"


def main() -> None:
    metrics = json.loads((OUT_DIR / "seed_0_metrics.json").read_text(encoding="utf-8"))
    training = json.loads((OUT_DIR / "seed_0_training.json").read_text(encoding="utf-8"))
    shared = metrics["shared_head"][GAIN_KEY]["timestep"]
    probe = metrics["probes"]["timestep"][GAIN_KEY]["accuracy"]

    rows = []
    shared_reference = float(shared["accuracy"][HORIZONS[0] - 1])
    probe_reference = float(probe[HORIZONS[0] - 1])
    for horizon in HORIZONS:
        shared_accuracy = float(shared["accuracy"][horizon - 1])
        probe_accuracy = float(probe[horizon - 1])
        rows.append(
            {
                "test_horizon": horizon,
                "shared_head_terminal_accuracy": shared_accuracy,
                "shared_head_change_from_T8_pp": 100.0
                * (shared_accuracy - shared_reference),
                "fixed_probe_terminal_accuracy": probe_accuracy,
                "fixed_probe_change_from_T8_pp": 100.0
                * (probe_accuracy - probe_reference),
                "cross_entropy": float(shared["cross_entropy"][horizon - 1]),
                "mean_state_norm": float(shared["mean_state_norm"][horizon - 1]),
                "mean_relative_update_residual": float(
                    shared["mean_relative_update_residual"][horizon - 1]
                ),
            }
        )

    result = {
        "experiment": "e14_horizon_extension_seed0",
        "source_config_sha256": metrics["config_sha256"],
        "seed": 0,
        "training_horizon": 8,
        "training_gain": 1.0,
        "test_gain": 1.0,
        "test_horizons": HORIZONS,
        "checkpoint_selected_terminal_accuracy_T8": float(
            training["best_validation_terminal_accuracy"]
        ),
        "rows": rows,
        "interpretation": (
            "At fixed gain 1.0, extending recurrence beyond the trained horizon "
            "degrades both shared-head terminal accuracy and fixed-probe accuracy."
        ),
    }
    (OUT_DIR / "horizon_extension_audit.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    table = [
        "T_test\tShared terminal\tChange from T8 (pp)\tProbe terminal\tProbe change (pp)"
    ]
    for row in rows:
        table.append(
            f"{row['test_horizon']}\t"
            f"{row['shared_head_terminal_accuracy']:.4f}\t"
            f"{row['shared_head_change_from_T8_pp']:+.2f}\t"
            f"{row['fixed_probe_terminal_accuracy']:.4f}\t"
            f"{row['fixed_probe_change_from_T8_pp']:+.2f}"
        )
    (OUT_DIR / "horizon_extension_table.txt").write_text(
        "\n".join(table) + "\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    ax.plot(
        HORIZONS,
        [row["shared_head_terminal_accuracy"] for row in rows],
        marker="o",
        linewidth=2.2,
        label="Shared trained head",
        color="#d84a4a",
    )
    ax.plot(
        HORIZONS,
        [row["fixed_probe_terminal_accuracy"] for row in rows],
        marker="s",
        linewidth=2.2,
        label="Fixed-alpha linear probe",
        color="#2468b2",
    )
    ax.axvline(8, color="#666666", linestyle="--", linewidth=1.2, label="Training horizon")
    ax.set_xlabel("Test-time recursive horizon")
    ax.set_ylabel("Terminal validation accuracy")
    ax.set_title(r"E14 horizon overcomputation at fixed $\lambda=1$")
    ax.set_xticks(HORIZONS)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "horizon_extension_audit.png", dpi=220)
    plt.close(fig)
    print(f"Saved horizon-extension audit to {OUT_DIR}")


if __name__ == "__main__":
    main()
