"""
E10: Operator-level diagnostics for the pre-collapse regime.

This script does not rerun MNIST experiments. Instead, it reconstructs the
fixed operators used in existing sweeps and merges operator-level diagnostics
with the saved readout results.

Outputs:
  - results/e10_operator_diagnostics/summary.json
  - results/e10_operator_diagnostics/modrelu_per_run.json
  - results/e10_operator_diagnostics/cross_system_per_run.json
  - results/e10_operator_diagnostics/e10_operator_diagnostics.png
  - results/e10_operator_diagnostics/e10_operator_diagnostics.pdf
"""

import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from transient_geometry.system import (  # noqa: E402
    ComplexDynamicalSystem,
    LinearDynamicalSystem,
    RealESN,
)


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(ROOT, "results", "e10_operator_diagnostics")

OVERNIGHT_RESULTS = os.path.join(ROOT, "results", "overnight_10seed", "results.json")
GRACE_RESULTS = os.path.join(
    ROOT, "results", "grace_period", "accuracy_grace_period.json"
)
CROSS_SYSTEM_RESULTS = os.path.join(ROOT, "results", "a5_cross_system", "results.json")

INPUT_DIM = 784
HIDDEN_DIM = 128
ROLLOUT_STEPS = 40
MODRELU_BIAS = -0.5
LEAK_RATE = 1.0
KREISS_RADII = 1.0 + np.array(
    [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 2e-1, 4e-1, 7e-1],
    dtype=np.float64,
)
KREISS_ANGLES = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
KREISS_REFINE_RADII = np.array([-0.06, -0.03, -0.01, 0.0, 0.01, 0.03, 0.06], dtype=np.float64)
KREISS_REFINE_ANGLES = np.deg2rad(np.array([-18, -9, -4.5, 0.0, 4.5, 9, 18], dtype=np.float64))

SYSTEM_SPECS = {
    "complex_modrelu": {
        "cls": ComplexDynamicalSystem,
        "kwargs": {"modrelu_bias": MODRELU_BIAS},
        "label": "Complex modReLU",
        "color": "#1f77b4",
    },
    "real_esn": {
        "cls": RealESN,
        "kwargs": {"leak_rate": LEAK_RATE},
        "label": "Real ESN (tanh)",
        "color": "#d95f02",
    },
    "linear": {
        "cls": LinearDynamicalSystem,
        "kwargs": {},
        "label": "Linear null",
        "color": "#2ca25f",
    },
}


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


def build_base_operator(system_name: str, seed: int) -> np.ndarray:
    spec = SYSTEM_SPECS[system_name]
    system = spec["cls"](
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        spectral_radius=1.0,
        noise_std=0.0,
        seed=seed,
        **spec["kwargs"],
    )
    W = system.W.detach().cpu().numpy()
    if np.iscomplexobj(W):
        return np.asarray(W, dtype=np.complex128)
    return np.asarray(W, dtype=np.float64)


def commutator_ratio(W: np.ndarray) -> float:
    W_star = W.conj().T if np.iscomplexobj(W) else W.T
    numer = np.linalg.norm(W_star @ W - W @ W_star, ord="fro")
    denom = max(np.linalg.norm(W, ord="fro") ** 2, 1e-12)
    return float(numer / denom)


def smallest_singular_value(M: np.ndarray) -> float:
    svals = np.linalg.svd(M, compute_uv=False)
    return float(svals[-1])


def resolvent_objective(B: np.ndarray, identity: np.ndarray, radius: float, theta: float) -> Tuple[float, float]:
    z = radius * np.exp(1j * theta)
    sigma_min = smallest_singular_value(z * identity - B)
    value = (radius - 1.0) / max(sigma_min, 1e-15)
    return float(value), float(sigma_min)


def approx_scaled_kreiss_constant(B: np.ndarray) -> Dict[str, float]:
    """
    Approximate K(B) = sup_{|z|>1} (|z|-1) ||(zI-B)^{-1}|| for a base operator B
    whose spectral radius has been normalized to 1.
    """
    n = B.shape[0]
    identity = np.eye(n, dtype=np.complex128)
    B = np.asarray(B, dtype=np.complex128)

    coarse_candidates: List[Tuple[float, float, float, float]] = []
    best_value = -np.inf
    best_radius = float(KREISS_RADII[0])
    best_theta = float(KREISS_ANGLES[0])
    best_sigma = float("nan")

    for radius in KREISS_RADII:
        for theta in KREISS_ANGLES:
            value, sigma_min = resolvent_objective(B, identity, float(radius), float(theta))
            coarse_candidates.append((value, float(radius), float(theta), sigma_min))
            if value > best_value:
                best_value = value
                best_radius = float(radius)
                best_theta = float(theta)
                best_sigma = sigma_min

    coarse_candidates.sort(key=lambda item: item[0], reverse=True)
    top_candidates = coarse_candidates[:3]

    for _, radius_center, theta_center, _ in top_candidates:
        local_radius = np.clip(radius_center + KREISS_REFINE_RADII, 1.0 + 1e-6, None)
        local_angles = (theta_center + KREISS_REFINE_ANGLES) % (2.0 * np.pi)
        for radius in local_radius:
            for theta in local_angles:
                value, sigma_min = resolvent_objective(B, identity, float(radius), float(theta))
                if value > best_value:
                    best_value = value
                    best_radius = float(radius)
                    best_theta = float(theta)
                    best_sigma = sigma_min

    return {
        "kreiss_constant_scaled_est": float(best_value),
        "kreiss_arg_radius": float(best_radius),
        "kreiss_arg_theta": float(best_theta),
        "kreiss_sigma_min": float(best_sigma),
        "resolvent_peak_est": float(1.0 / max(best_sigma, 1e-15)),
        "resolvent_eval_count": int(len(KREISS_RADII) * len(KREISS_ANGLES) + len(top_candidates) * len(KREISS_REFINE_RADII) * len(KREISS_REFINE_ANGLES)),
    }


def base_operator_metrics(W: np.ndarray, steps: int) -> Dict[str, object]:
    eigvals = np.linalg.eigvals(W)
    rho_emp = float(np.max(np.abs(eigvals)))
    spectral_norm = float(np.linalg.norm(W, ord=2))

    power_curve: List[float] = []
    excess_curve: List[float] = []
    W_t = np.array(W, copy=True)

    for t in range(1, steps + 1):
        norm_t = float(np.linalg.norm(W_t, ord=2))
        power_curve.append(norm_t)
        excess_curve.append(norm_t / max(rho_emp**t, 1e-12))
        if t < steps:
            W_t = W_t @ W

    power_arr = np.asarray(power_curve, dtype=np.float64)
    excess_arr = np.asarray(excess_curve, dtype=np.float64)
    B = W / max(rho_emp, 1e-12)
    kreiss_metrics = approx_scaled_kreiss_constant(B)

    return {
        "rho_emp_base": rho_emp,
        "spectral_norm_base": spectral_norm,
        "commutator_ratio": commutator_ratio(W),
        "power_curve_base": power_curve,
        "excess_curve_base": excess_curve,
        "peak_power_norm_base": float(power_arr.max()),
        "peak_power_t_base": int(power_arr.argmax() + 1),
        "peak_excess_ratio": float(excess_arr.max()),
        "peak_excess_t": int(excess_arr.argmax() + 1),
        "kreiss_excess_upper_bound": float(math.e * W.shape[0] * kreiss_metrics["kreiss_constant_scaled_est"]),
        **kreiss_metrics,
    }


def scaled_operator_metrics(
    base_metrics: Dict[str, object], spectral_radius: float
) -> Dict[str, float]:
    base_curve = np.asarray(base_metrics["power_curve_base"], dtype=np.float64)
    timesteps = np.arange(1, len(base_curve) + 1, dtype=np.float64)
    scaled_curve = base_curve * (float(spectral_radius) ** timesteps)
    peak_idx = int(np.argmax(scaled_curve))
    rho_emp = float(base_metrics["rho_emp_base"]) * float(spectral_radius)

    return {
        "rho_emp": rho_emp,
        "spectral_norm": float(base_metrics["spectral_norm_base"]) * float(spectral_radius),
        "peak_power_norm": float(scaled_curve[peak_idx]),
        "peak_power_t": peak_idx + 1,
        "peak_excess_ratio": float(base_metrics["peak_excess_ratio"]),
        "peak_excess_t": int(base_metrics["peak_excess_t"]),
        "commutator_ratio": float(base_metrics["commutator_ratio"]),
        "kreiss_constant_scaled_est": float(base_metrics["kreiss_constant_scaled_est"]),
        "kreiss_excess_upper_bound": float(base_metrics["kreiss_excess_upper_bound"]),
        "kreiss_arg_radius": float(base_metrics["kreiss_arg_radius"]),
        "kreiss_arg_theta": float(base_metrics["kreiss_arg_theta"]),
        "resolvent_peak_est": float(base_metrics["resolvent_peak_est"]),
    }


def get_base_metrics_cache(records: Iterable[Dict[str, object]]) -> Dict[Tuple[str, int], Dict[str, object]]:
    cache: Dict[Tuple[str, int], Dict[str, object]] = {}
    keys = sorted({(str(r["system"]), int(r["seed"])) for r in records})
    for system_name, seed in keys:
        W = build_base_operator(system_name, seed)
        cache[(system_name, seed)] = base_operator_metrics(W, steps=ROLLOUT_STEPS)
    return cache


def merge_modrelu_runs(
    overnight_runs: List[Dict[str, object]],
    grace_payload: Dict[str, object],
    base_cache: Dict[Tuple[str, int], Dict[str, object]],
) -> List[Dict[str, object]]:
    g70_map = {
        (float(run["spectral_radius"]), int(run["seed"])): int(run["G"]["0.70"])
        for run in grace_payload["per_run"]
    }

    merged = []
    for run in overnight_runs:
        sr = float(run["spectral_radius"])
        seed = int(run["seed"])
        op = scaled_operator_metrics(base_cache[("complex_modrelu", seed)], sr)
        merged.append(
            {
                "system": "complex_modrelu",
                "spectral_radius": sr,
                "seed": seed,
                "acc_final": float(run["acc_final"]),
                "acc_early_window": float(run["acc_early_window"]),
                "acc_adaptive": float(run["acc_adaptive"]),
                "acc_grace": float(run["acc_d1_best"]),
                "gap_early_final": float(run["acc_early_window"] - run["acc_final"]),
                "gap_adaptive_final": float(run["acc_adaptive"] - run["acc_final"]),
                "gap_grace_final": float(run["acc_d1_best"] - run["acc_final"]),
                "adaptive_stop_t": int(run["adaptive_stop_t"]),
                "G_070": g70_map[(sr, seed)],
                **op,
            }
        )
    return merged


def merge_cross_system_runs(
    cross_runs: List[Dict[str, object]],
    base_cache: Dict[Tuple[str, int], Dict[str, object]],
) -> List[Dict[str, object]]:
    merged = []
    for run in cross_runs:
        system_name = str(run["system"])
        sr = float(run["spectral_radius"])
        seed = int(run["seed"])
        op = scaled_operator_metrics(base_cache[(system_name, seed)], sr)
        norms = [float(v) for v in run.get("norms", [])]
        merged.append(
            {
                "system": system_name,
                "spectral_radius": sr,
                "seed": seed,
                "acc_final": float(run["acc_final"]),
                "acc_early_window": float(run["acc_early_window"]),
                "gap_early_final": float(run["acc_early_window"] - run["acc_final"]),
                "mean_norm_t1": norms[0] if norms else float("nan"),
                "mean_norm_t40": norms[-1] if norms else float("nan"),
                **op,
            }
        )
    return merged


def aggregate_by_rho(records: List[Dict[str, object]], metrics: Iterable[str]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[float, List[Dict[str, object]]] = defaultdict(list)
    for rec in records:
        grouped[float(rec["spectral_radius"])].append(rec)

    summary: Dict[str, Dict[str, float]] = {}
    for rho in sorted(grouped):
        runs = grouped[rho]
        entry = {"n": len(runs)}
        for metric in metrics:
            vals = np.asarray([float(r[metric]) for r in runs], dtype=np.float64)
            entry[f"{metric}_mean"] = float(vals.mean())
            entry[f"{metric}_std"] = float(vals.std())
        summary[f"{rho:.2f}"] = entry
    return summary


def aggregate_cross_system(
    records: List[Dict[str, object]], metrics: Iterable[str]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    grouped: Dict[str, Dict[float, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        grouped[str(rec["system"])][float(rec["spectral_radius"])].append(rec)

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for system_name, by_rho in grouped.items():
        system_summary: Dict[str, Dict[str, float]] = {}
        for rho in sorted(by_rho):
            runs = by_rho[rho]
            entry = {"n": len(runs)}
            for metric in metrics:
                vals = np.asarray([float(r[metric]) for r in runs], dtype=np.float64)
                entry[f"{metric}_mean"] = float(vals.mean())
                entry[f"{metric}_std"] = float(vals.std())
            system_summary[f"{rho:.2f}"] = entry
        summary[system_name] = system_summary
    return summary


def demeaned_correlation(
    records: List[Dict[str, object]], x_key: str, y_key: str, group_key: str
) -> float:
    grouped: Dict[float, List[Dict[str, object]]] = defaultdict(list)
    for rec in records:
        grouped[float(rec[group_key])].append(rec)

    x_res, y_res = [], []
    for _, runs in grouped.items():
        mean_x = float(np.mean([float(r[x_key]) for r in runs]))
        mean_y = float(np.mean([float(r[y_key]) for r in runs]))
        for run in runs:
            x_res.append(float(run[x_key]) - mean_x)
            y_res.append(float(run[y_key]) - mean_y)
    return pearson(x_res, y_res)


def build_summary(
    modrelu_runs: List[Dict[str, object]], cross_runs: List[Dict[str, object]]
) -> Dict[str, object]:
    modrelu_by_rho = aggregate_by_rho(
        modrelu_runs,
        metrics=[
            "peak_power_norm",
            "kreiss_constant_scaled_est",
            "gap_grace_final",
            "gap_early_final",
            "G_070",
            "adaptive_stop_t",
            "acc_final",
        ],
    )
    cross_by_system = aggregate_cross_system(
        cross_runs,
        metrics=["peak_power_norm", "kreiss_constant_scaled_est", "gap_early_final", "acc_final"],
    )

    modrelu_summary = {
        "correlations": {
            "log_peak_power_vs_gap_grace_final": pearson(
                [math.log10(r["peak_power_norm"]) for r in modrelu_runs],
                [r["gap_grace_final"] for r in modrelu_runs],
            ),
            "log_peak_power_vs_gap_early_final": pearson(
                [math.log10(r["peak_power_norm"]) for r in modrelu_runs],
                [r["gap_early_final"] for r in modrelu_runs],
            ),
            "log_peak_power_vs_G_070": pearson(
                [math.log10(r["peak_power_norm"]) for r in modrelu_runs],
                [r["G_070"] for r in modrelu_runs],
            ),
            "log_peak_power_vs_acc_final": pearson(
                [math.log10(r["peak_power_norm"]) for r in modrelu_runs],
                [r["acc_final"] for r in modrelu_runs],
            ),
            "kreiss_scaled_vs_peak_excess_ratio": pearson(
                [r["kreiss_constant_scaled_est"] for r in modrelu_runs],
                [r["peak_excess_ratio"] for r in modrelu_runs],
            ),
        },
        "within_rho_correlations": {
            "peak_excess_vs_gap_grace_final": demeaned_correlation(
                modrelu_runs, "peak_excess_ratio", "gap_grace_final", "spectral_radius"
            ),
            "peak_excess_vs_gap_early_final": demeaned_correlation(
                modrelu_runs, "peak_excess_ratio", "gap_early_final", "spectral_radius"
            ),
            "peak_excess_vs_G_070": demeaned_correlation(
                modrelu_runs, "peak_excess_ratio", "G_070", "spectral_radius"
            ),
            "kreiss_scaled_vs_gap_grace_final": demeaned_correlation(
                modrelu_runs,
                "kreiss_constant_scaled_est",
                "gap_grace_final",
                "spectral_radius",
            ),
            "kreiss_scaled_vs_gap_early_final": demeaned_correlation(
                modrelu_runs,
                "kreiss_constant_scaled_est",
                "gap_early_final",
                "spectral_radius",
            ),
            "kreiss_scaled_vs_G_070": demeaned_correlation(
                modrelu_runs,
                "kreiss_constant_scaled_est",
                "G_070",
                "spectral_radius",
            ),
        },
        "by_rho": modrelu_by_rho,
    }

    cross_summary = {
        "correlations": {
            "pooled_log_peak_power_vs_gap_early_final": pearson(
                [math.log10(r["peak_power_norm"]) for r in cross_runs],
                [r["gap_early_final"] for r in cross_runs],
            )
        },
        "by_system": cross_by_system,
    }

    return {
        "config": {
            "rollout_steps": ROLLOUT_STEPS,
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "kreiss_grid_radii": [float(v) for v in KREISS_RADII],
            "kreiss_grid_angles": int(len(KREISS_ANGLES)),
            "kreiss_refine_radii": [float(v) for v in KREISS_REFINE_RADII],
            "kreiss_refine_angles_deg": [float(np.rad2deg(v)) for v in KREISS_REFINE_ANGLES],
            "overnight_results": os.path.relpath(OVERNIGHT_RESULTS, ROOT),
            "grace_results": os.path.relpath(GRACE_RESULTS, ROOT),
            "cross_system_results": os.path.relpath(CROSS_SYSTEM_RESULTS, ROOT),
        },
        "modrelu_summary": modrelu_summary,
        "cross_system_summary": cross_summary,
    }


def fit_line(ax, x_vals: List[float], y_vals: List[float], color: str) -> None:
    x = np.asarray(x_vals, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)
    if x.size < 2:
        return
    coeffs = np.polyfit(x, y, deg=1)
    x_line = np.linspace(float(x.min()), float(x.max()), 100)
    y_line = coeffs[0] * x_line + coeffs[1]
    ax.plot(x_line, y_line, color=color, lw=2.0, alpha=0.9)


def plot_figure(
    modrelu_runs: List[Dict[str, object]],
    cross_runs: List[Dict[str, object]],
    summary: Dict[str, object],
) -> None:
    fig = plt.figure(figsize=(13.5, 9.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], hspace=0.32, wspace=0.24)

    rho_vals = np.asarray([r["spectral_radius"] for r in modrelu_runs], dtype=np.float64)
    log_peak = np.asarray([math.log10(r["peak_power_norm"]) for r in modrelu_runs], dtype=np.float64)

    # Panel A: operator amplification vs grace period
    ax1 = fig.add_subplot(gs[0, 0])
    sc1 = ax1.scatter(
        log_peak,
        [r["G_070"] for r in modrelu_runs],
        c=rho_vals,
        cmap="viridis",
        s=48,
        edgecolors="white",
        linewidths=0.5,
        alpha=0.9,
    )
    fit_line(ax1, list(log_peak), [r["G_070"] for r in modrelu_runs], color="#1f77b4")
    ax1.set_xlabel(r"$\log_{10}\max_{t \leq T}\|W^t\|_2$")
    ax1.set_ylabel(r"Grace Period $G(\rho, 0.70)$")
    ax1.set_title("Operator Amplification Tracks Prefix Length", fontweight="bold")
    corr_g = summary["modrelu_summary"]["correlations"]["log_peak_power_vs_G_070"]
    ax1.text(
        0.04,
        0.93,
        f"Pearson r = {corr_g:.2f}",
        transform=ax1.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.85"),
    )
    ax1.grid(alpha=0.18)
    cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.01)
    cbar1.set_label(r"Spectral radius $\rho$")

    # Panel B: operator amplification vs terminal-readout failure
    ax2 = fig.add_subplot(gs[0, 1])
    gap_pp = [100.0 * r["gap_grace_final"] for r in modrelu_runs]
    sc2 = ax2.scatter(
        log_peak,
        gap_pp,
        c=rho_vals,
        cmap="viridis",
        s=48,
        edgecolors="white",
        linewidths=0.5,
        alpha=0.9,
    )
    fit_line(ax2, list(log_peak), gap_pp, color="#2a6fbb")
    ax2.set_xlabel(r"$\log_{10}\max_{t \leq T}\|W^t\|_2$")
    ax2.set_ylabel(r"$\Delta$(GRACE $-$ Final)  [pp]")
    ax2.set_title("Amplification Predicts Terminal Readout Failure", fontweight="bold")
    corr_gap = summary["modrelu_summary"]["correlations"]["log_peak_power_vs_gap_grace_final"]
    corr_seed = summary["modrelu_summary"]["within_rho_correlations"][
        "peak_excess_vs_gap_grace_final"
    ]
    corr_kreiss = summary["modrelu_summary"]["within_rho_correlations"][
        "kreiss_scaled_vs_gap_grace_final"
    ]
    ax2.text(
        0.04,
        0.93,
        "Pearson r = "
        f"{corr_gap:.2f}\n"
        f"Within-$\\rho$ excess r = {corr_seed:.2f}\n"
        f"Within-$\\rho$ $K_\\rho$ r = {corr_kreiss:.2f}",
        transform=ax2.transAxes,
        fontsize=11,
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.85"),
    )
    ax2.grid(alpha=0.18)
    cbar2 = fig.colorbar(sc2, ax=ax2, pad=0.01)
    cbar2.set_label(r"Spectral radius $\rho$")

    # Panel C: cross-system comparison
    ax3 = fig.add_subplot(gs[1, :])
    by_system = summary["cross_system_summary"]["by_system"]
    for system_name, rho_table in by_system.items():
        spec = SYSTEM_SPECS[system_name]
        xs, ys = [], []
        for rho_key in sorted(rho_table.keys(), key=float):
            row = rho_table[rho_key]
            xs.append(math.log10(row["peak_power_norm_mean"]))
            ys.append(100.0 * row["gap_early_final_mean"])
        ax3.plot(
            xs,
            ys,
            marker="o",
            ms=7,
            lw=2.4,
            color=spec["color"],
            label=spec["label"],
        )

    ax3.set_xlabel(r"$\log_{10}\max_{t \leq T}\|W^t\|_2$  (mean across seeds)")
    ax3.set_ylabel(r"$\Delta$(Early $-$ Final)  [pp]")
    ax3.set_title("Comparable Operator Amplification, Different Regime Severity", fontweight="bold")
    ax3.grid(alpha=0.18)
    ax3.legend(frameon=False, ncol=3, loc="upper left")
    ax3.text(
        0.98,
        0.08,
        "Linear and tanh systems share nearly the same operator envelope;\n"
        "the large gap emerges only when nonlinearity converts amplification\n"
        "into endpoint distortion.",
        transform=ax3.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.85"),
    )

    inset = ax3.inset_axes([0.13, 0.53, 0.28, 0.32])
    for system_name in ["real_esn", "linear"]:
        spec = SYSTEM_SPECS[system_name]
        xs, ys = [], []
        for rho_key in sorted(by_system[system_name].keys(), key=float):
            row = by_system[system_name][rho_key]
            xs.append(math.log10(row["peak_power_norm_mean"]))
            ys.append(100.0 * row["gap_early_final_mean"])
        inset.plot(
            xs,
            ys,
            marker="o",
            ms=4.5,
            lw=1.8,
            color=spec["color"],
            label=spec["label"],
        )
    inset.set_title("Zoom: tanh vs. linear", fontsize=9, fontweight="bold")
    inset.set_ylim(-0.5, 13.0)
    inset.grid(alpha=0.15)
    inset.tick_params(labelsize=8)
    inset.legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle(
        "Operator Diagnostics: Finite-Time Amplification Explains When the Regime Appears",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])

    png_path = os.path.join(RESULTS_DIR, "e10_operator_diagnostics.png")
    pdf_path = os.path.join(RESULTS_DIR, "e10_operator_diagnostics.pdf")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    overnight_runs = load_json(OVERNIGHT_RESULTS)
    grace_payload = load_json(GRACE_RESULTS)
    cross_runs = load_json(CROSS_SYSTEM_RESULTS)

    overnight_runs = [dict(r, system="complex_modrelu") for r in overnight_runs]
    all_runs_for_cache = overnight_runs + cross_runs
    base_cache = get_base_metrics_cache(all_runs_for_cache)

    modrelu_merged = merge_modrelu_runs(overnight_runs, grace_payload, base_cache)
    cross_merged = merge_cross_system_runs(cross_runs, base_cache)
    summary = build_summary(modrelu_merged, cross_merged)

    save_json(os.path.join(RESULTS_DIR, "modrelu_per_run.json"), modrelu_merged)
    save_json(os.path.join(RESULTS_DIR, "cross_system_per_run.json"), cross_merged)
    save_json(os.path.join(RESULTS_DIR, "summary.json"), summary)

    plot_figure(modrelu_merged, cross_merged, summary)

    print("Saved:")
    print(f"  {os.path.relpath(os.path.join(RESULTS_DIR, 'summary.json'), ROOT)}")
    print(f"  {os.path.relpath(os.path.join(RESULTS_DIR, 'modrelu_per_run.json'), ROOT)}")
    print(f"  {os.path.relpath(os.path.join(RESULTS_DIR, 'cross_system_per_run.json'), ROOT)}")
    print(f"  {os.path.relpath(os.path.join(RESULTS_DIR, 'e10_operator_diagnostics.png'), ROOT)}")
    print(f"  {os.path.relpath(os.path.join(RESULTS_DIR, 'e10_operator_diagnostics.pdf'), ROOT)}")


if __name__ == "__main__":
    main()
