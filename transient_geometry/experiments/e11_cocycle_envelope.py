"""
E11: Numerical cocycle-envelope estimates for the tanh ESN branch.

This script evaluates the finite-horizon cocycle envelope from Appendix P.4
on the real-valued tanh ESN cross-system sweep. For the block norm used in the
paper, the envelope can be evaluated exactly as the maximum weighted Jacobian-
product norm across samples and timesteps, so no lifted resolvent inversion is
required.

Outputs:
  - results/e11_cocycle_envelope/summary.json
  - results/e11_cocycle_envelope/real_esn_per_run.json
  - results/e11_cocycle_envelope/e11_cocycle_envelope.png
  - results/e11_cocycle_envelope/e11_cocycle_envelope.pdf
"""

import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from transient_geometry.system import RealESN  # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(ROOT, "results", "e11_cocycle_envelope")
A5_RESULTS = os.path.join(ROOT, "results", "a5_cross_system", "results.json")

INPUT_DIM = 784
HIDDEN_DIM = 128
ROLLOUT_STEPS = 40
LEAK_RATE = 1.0
NUM_EVAL_SAMPLES = 128
SUBSET_SEED = 0
DATA_BATCH_SIZE = 128


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def pearson(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = np.asarray(list(xs), dtype=np.float64)
    y = np.asarray(list(ys), dtype=np.float64)
    if x.size < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def pooled_within_group_corr(
    xs: Iterable[float], ys: Iterable[float], groups: Iterable[float]
) -> float:
    x = np.asarray(list(xs), dtype=np.float64)
    y = np.asarray(list(ys), dtype=np.float64)
    g = np.asarray(list(groups), dtype=np.float64)
    x_residuals = []
    y_residuals = []
    for group in np.unique(g):
        mask = g == group
        if mask.sum() < 2:
            continue
        x_centered = x[mask] - x[mask].mean()
        y_centered = y[mask] - y[mask].mean()
        x_residuals.append(x_centered)
        y_residuals.append(y_centered)
    if not x_residuals:
        return float("nan")
    x_res = np.concatenate(x_residuals)
    y_res = np.concatenate(y_residuals)
    return pearson(x_res, y_res)


def early_final_gap(record: Dict[str, object]) -> float:
    if "gap_early_final" in record:
        return float(record["gap_early_final"])
    return float(record["acc_early_window"]) - float(record["acc_final"])


def get_eval_batch(device: torch.device) -> torch.Tensor:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
    )
    root = os.path.join(ROOT, "data")
    test_ds = datasets.MNIST(root=root, train=False, download=False, transform=transform)
    rng = np.random.RandomState(SUBSET_SEED)
    indices = rng.permutation(len(test_ds))[:NUM_EVAL_SAMPLES]
    subset = torch.utils.data.Subset(test_ds, indices.tolist())
    loader = torch.utils.data.DataLoader(
        subset, batch_size=DATA_BATCH_SIZE, shuffle=False, num_workers=0
    )
    xb, _ = next(iter(loader))
    return xb.to(device)


def build_real_esn(spectral_radius: float, seed: int, device: torch.device) -> RealESN:
    return RealESN(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        spectral_radius=float(spectral_radius),
        noise_std=0.0,
        leak_rate=LEAK_RATE,
        seed=int(seed),
    ).to(device)


@torch.no_grad()
def estimate_cocycle_envelope(
    system: RealESN, xb: torch.Tensor, rollout_steps: int
) -> Dict[str, object]:
    """
    Evaluate the finite-horizon envelope on actual trajectories.

    For the block norm used in the paper, the induced norm of the block-row
    operator equals the maximum block norm, so

      R_t(lambda) = max_{i,s} lambda^{-(t-1-s)} ||Phi_{t:s+1}^{(i)}||_2.

    We therefore compute the exact finite-horizon estimate directly from the
    Jacobian cocycle along each held-out trajectory.
    """

    lam = float(system.nominal_spectral_radius)
    batch = xb.shape[0]
    hidden_dim = system.hidden_dim
    device = xb.device

    W = system.W
    Ux = torch.matmul(xb, system.U.T)
    h = torch.zeros(batch, hidden_dim, device=device)
    derivs: List[torch.Tensor] = []

    for _ in range(rollout_steps):
        pre = torch.matmul(h, W.T) + Ux + system.b
        h = torch.tanh(pre)
        derivs.append(1.0 - h * h)

    identity = torch.eye(hidden_dim, device=device).expand(batch, -1, -1).clone()
    cocycle = identity

    raw_max = 0.0
    raw_arg_s = rollout_steps - 1
    raw_arg_i = 0
    scaled_max = 0.0
    scaled_arg_s = rollout_steps - 1
    scaled_arg_i = 0

    raw_all: List[np.ndarray] = []
    scaled_all: List[np.ndarray] = []
    raw_per_s: List[float] = []
    scaled_per_s: List[float] = []

    for s in range(rollout_steps - 1, -1, -1):
        norms = torch.linalg.matrix_norm(cocycle, ord=2)
        raw_vals = norms
        scaled_vals = norms / (lam ** (rollout_steps - 1 - s))

        raw_all.append(raw_vals.detach().cpu().numpy())
        scaled_all.append(scaled_vals.detach().cpu().numpy())
        raw_per_s.append(float(raw_vals.max().item()))
        scaled_per_s.append(float(scaled_vals.max().item()))

        raw_run_max, raw_idx = torch.max(raw_vals, dim=0)
        if raw_run_max.item() > raw_max:
            raw_max = float(raw_run_max.item())
            raw_arg_s = s
            raw_arg_i = int(raw_idx.item())

        scaled_run_max, scaled_idx = torch.max(scaled_vals, dim=0)
        if scaled_run_max.item() > scaled_max:
            scaled_max = float(scaled_run_max.item())
            scaled_arg_s = s
            scaled_arg_i = int(scaled_idx.item())

        J_s = derivs[s].unsqueeze(-1) * W.unsqueeze(0)
        cocycle = torch.bmm(cocycle, J_s)

    raw_flat = np.concatenate(raw_all, axis=0)
    scaled_flat = np.concatenate(scaled_all, axis=0)

    return {
        "cocycle_envelope_raw_t40_est": raw_max,
        "cocycle_envelope_scaled_t40_est": scaled_max,
        "cocycle_envelope_raw_q90": float(np.quantile(raw_flat, 0.90)),
        "cocycle_envelope_scaled_q90": float(np.quantile(scaled_flat, 0.90)),
        "cocycle_envelope_raw_mean": float(raw_flat.mean()),
        "cocycle_envelope_scaled_mean": float(scaled_flat.mean()),
        "cocycle_envelope_raw_arg_s": int(raw_arg_s),
        "cocycle_envelope_scaled_arg_s": int(scaled_arg_s),
        "cocycle_envelope_raw_arg_sample": int(raw_arg_i),
        "cocycle_envelope_scaled_arg_sample": int(scaled_arg_i),
        "cocycle_envelope_raw_per_s_max": list(reversed(raw_per_s)),
        "cocycle_envelope_scaled_per_s_max": list(reversed(scaled_per_s)),
    }


def aggregate_by_rho(records: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        rho_key = f"{float(record['spectral_radius']):.2f}"
        grouped[rho_key]["raw_max"].append(float(record["cocycle_envelope_raw_t40_est"]))
        grouped[rho_key]["scaled_max"].append(float(record["cocycle_envelope_scaled_t40_est"]))
        grouped[rho_key]["raw_q90"].append(float(record["cocycle_envelope_raw_q90"]))
        grouped[rho_key]["scaled_q90"].append(float(record["cocycle_envelope_scaled_q90"]))
        grouped[rho_key]["gap_early_final"].append(early_final_gap(record))
        grouped[rho_key]["acc_final"].append(float(record["acc_final"]))

    summary: Dict[str, Dict[str, float]] = {}
    for rho_key in sorted(grouped.keys(), key=lambda item: float(item)):
        bucket = grouped[rho_key]
        summary[rho_key] = {
            "n": int(len(bucket["raw_max"])),
            "raw_max_mean": float(np.mean(bucket["raw_max"])),
            "raw_max_std": float(np.std(bucket["raw_max"])),
            "scaled_max_mean": float(np.mean(bucket["scaled_max"])),
            "scaled_max_std": float(np.std(bucket["scaled_max"])),
            "raw_q90_mean": float(np.mean(bucket["raw_q90"])),
            "raw_q90_std": float(np.std(bucket["raw_q90"])),
            "scaled_q90_mean": float(np.mean(bucket["scaled_q90"])),
            "scaled_q90_std": float(np.std(bucket["scaled_q90"])),
            "gap_early_final_mean": float(np.mean(bucket["gap_early_final"])),
            "gap_early_final_std": float(np.std(bucket["gap_early_final"])),
            "acc_final_mean": float(np.mean(bucket["acc_final"])),
            "acc_final_std": float(np.std(bucket["acc_final"])),
        }
    return summary


def plot_summary(records: List[Dict[str, object]], summary: Dict[str, Dict[str, float]]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rho_values = np.array(sorted(float(key) for key in summary.keys()), dtype=np.float64)
    raw_mean = np.array([summary[f"{rho:.2f}"]["raw_max_mean"] for rho in rho_values])
    raw_std = np.array([summary[f"{rho:.2f}"]["raw_max_std"] for rho in rho_values])
    scaled_mean = np.array([summary[f"{rho:.2f}"]["scaled_max_mean"] for rho in rho_values])
    scaled_std = np.array([summary[f"{rho:.2f}"]["scaled_max_std"] for rho in rho_values])

    gap = np.array([early_final_gap(r) for r in records], dtype=np.float64)
    log_raw = np.log10(
        np.array([float(r["cocycle_envelope_raw_t40_est"]) for r in records], dtype=np.float64)
    )
    rho_scatter = np.array([float(r["spectral_radius"]) for r in records], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax = axes[0]
    ax.plot(
        rho_values,
        raw_mean,
        "o-",
        lw=2.5,
        ms=6,
        color="#1f77b4",
        label=r"$\widehat{\mathcal{R}}_{40}(1)$",
    )
    ax.fill_between(rho_values, raw_mean - raw_std, raw_mean + raw_std, color="#1f77b4", alpha=0.16)
    ax.plot(
        rho_values,
        scaled_mean,
        "s--",
        lw=2.2,
        ms=5,
        color="#d95f02",
        label=r"$\widehat{\mathcal{R}}_{40}(\rho(W))$",
    )
    ax.fill_between(
        rho_values,
        np.maximum(scaled_mean - scaled_std, 1e-6),
        scaled_mean + scaled_std,
        color="#d95f02",
        alpha=0.16,
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"Spectral Radius $\rho(W)$")
    ax.set_ylabel(r"Finite-horizon cocycle envelope")
    ax.set_title("Exact Numerical Envelope on Tanh ESN Trajectories", fontweight="bold")
    ax.axvline(1.0, color="gray", ls="--", lw=1.0, alpha=0.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")

    ax = axes[1]
    scatter = ax.scatter(
        log_raw,
        gap,
        c=rho_scatter,
        cmap="viridis",
        s=48,
        edgecolors="none",
        alpha=0.9,
    )
    if len(log_raw) >= 2:
        coeffs = np.polyfit(log_raw, gap, deg=1)
        xs = np.linspace(log_raw.min(), log_raw.max(), 100)
        ax.plot(xs, coeffs[0] * xs + coeffs[1], color="black", lw=1.6, ls="--")
    corr = pearson(log_raw, gap)
    ax.text(
        0.04,
        0.96,
        rf"$r = {corr:.2f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor="0.8"),
    )
    ax.set_xlabel(r"$\log_{10}\widehat{\mathcal{R}}_{40}(1)$")
    ax.set_ylabel("Early - Final accuracy")
    ax.set_title("Raw Cocycle Amplification Tracks the Bounded-Activation Gap", fontweight="bold")
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(r"$\rho(W)$")

    fig.tight_layout()

    png_path = os.path.join(RESULTS_DIR, "e11_cocycle_envelope.png")
    pdf_path = os.path.join(RESULTS_DIR, "e11_cocycle_envelope.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results = load_json(A5_RESULTS)
    real_esn_runs = [record for record in all_results if record["system"] == "real_esn"]
    real_esn_runs.sort(key=lambda item: (float(item["spectral_radius"]), int(item["seed"])))

    xb = get_eval_batch(device)
    print(f"Eval subset: {xb.shape[0]} held-out MNIST digits")

    merged: List[Dict[str, object]] = []
    t0 = time.time()
    total = len(real_esn_runs)

    for idx, run in enumerate(real_esn_runs, start=1):
        rho = float(run["spectral_radius"])
        seed = int(run["seed"])
        print(f"[{idx:02d}/{total}] rho={rho:.2f} seed={seed} ... ", end="", flush=True)
        run_t0 = time.time()
        system = build_real_esn(rho, seed, device)
        env_metrics = estimate_cocycle_envelope(system, xb, ROLLOUT_STEPS)

        merged_record = {
            **run,
            "gap_early_final": early_final_gap(run),
            **env_metrics,
            "eval_samples": int(xb.shape[0]),
            "rollout_steps": int(ROLLOUT_STEPS),
            "lambda_raw": 1.0,
            "lambda_scaled": rho,
            "elapsed_envelope_sec": float(time.time() - run_t0),
        }
        merged.append(merged_record)
        print(
            f"R1={merged_record['cocycle_envelope_raw_t40_est']:.3f} "
            f"Rrho={merged_record['cocycle_envelope_scaled_t40_est']:.3f} "
            f"({merged_record['elapsed_envelope_sec']:.1f}s)"
        )

    by_rho = aggregate_by_rho(merged)
    correlations = {
        "log_raw_max_vs_gap_early_final": pearson(
            np.log10([record["cocycle_envelope_raw_t40_est"] for record in merged]),
            [early_final_gap(record) for record in merged],
        ),
        "log_raw_max_vs_acc_final": pearson(
            np.log10([record["cocycle_envelope_raw_t40_est"] for record in merged]),
            [record["acc_final"] for record in merged],
        ),
        "scaled_max_vs_gap_early_final": pearson(
            [record["cocycle_envelope_scaled_t40_est"] for record in merged],
            [early_final_gap(record) for record in merged],
        ),
        "scaled_max_vs_acc_final": pearson(
            [record["cocycle_envelope_scaled_t40_est"] for record in merged],
            [record["acc_final"] for record in merged],
        ),
        "within_rho_log_raw_max_vs_gap_early_final": pooled_within_group_corr(
            np.log10([record["cocycle_envelope_raw_t40_est"] for record in merged]),
            [early_final_gap(record) for record in merged],
            [record["spectral_radius"] for record in merged],
        ),
        "within_rho_log_raw_max_vs_acc_final": pooled_within_group_corr(
            np.log10([record["cocycle_envelope_raw_t40_est"] for record in merged]),
            [record["acc_final"] for record in merged],
            [record["spectral_radius"] for record in merged],
        ),
    }

    summary = {
        "config": {
            "source_results": os.path.relpath(A5_RESULTS, ROOT),
            "system": "real_esn",
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "rollout_steps": ROLLOUT_STEPS,
            "num_eval_samples": NUM_EVAL_SAMPLES,
            "subset_seed": SUBSET_SEED,
            "lambda_values": [1.0, "rho(W)"],
            "device": str(device),
        },
        "correlations": correlations,
        "by_rho": by_rho,
        "elapsed_total_sec": float(time.time() - t0),
    }

    save_json(os.path.join(RESULTS_DIR, "real_esn_per_run.json"), merged)
    save_json(os.path.join(RESULTS_DIR, "summary.json"), summary)
    plot_summary(merged, by_rho)

    print("\nSaved:")
    print(f"  {os.path.relpath(os.path.join(RESULTS_DIR, 'summary.json'), ROOT)}")
    print(f"  {os.path.relpath(os.path.join(RESULTS_DIR, 'real_esn_per_run.json'), ROOT)}")
    print(f"  {os.path.relpath(os.path.join(RESULTS_DIR, 'e11_cocycle_envelope.png'), ROOT)}")
    print(f"  {os.path.relpath(os.path.join(RESULTS_DIR, 'e11_cocycle_envelope.pdf'), ROOT)}")


if __name__ == "__main__":
    main()
