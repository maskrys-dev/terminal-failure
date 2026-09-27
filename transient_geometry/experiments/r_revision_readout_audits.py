"""
Revision readout audits for the NeurIPS submission.

This script provides two appendix-facing analyses:

R2. Early-window sensitivity:
    K in {1, 3, 5, 8, 10, 15, 20}

R3. Lightweight reference-radius selection:
    - tau_stop in {1.0, 1.2, 1.5, 2.0}
    - tau      in {1.0, 1.2, 1.5}
    - alpha    in {5, 10, 15}
    Selection happens only at rho = 1.3, then the chosen settings are frozen
    across the full spectral sweep.

The setup mirrors the publication MNIST reservoir protocol.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transient_geometry.probes import LinearProbe
from transient_geometry.system import ComplexDynamicalSystem


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(ROOT, "results", "revision_readout_audits")

CONFIG = {
    "input_dim": 784,
    "hidden_dim": 128,
    "rollout_steps": 40,
    "modrelu_bias": -0.5,
    "noise_std": 0.0,
    "spectral_radii": [0.5, 1.0, 1.20, 1.30, 1.35, 1.40, 1.45, 1.50, 1.60],
    "num_seeds": 10,
    "batch_size": 512,
    "train_n": 5000,
    "test_n": 5000,
    "early_K_values": [1, 3, 5, 8, 10, 15, 20],
    "tau_stop_values": [1.0, 1.2, 1.5, 2.0],
    "tau_values": [1.0, 1.2, 1.5],
    "alpha_values": [5, 10, 15],
    "reference_rho": 1.30,
    "default_tau_stop": 1.5,
    "default_tau": 1.2,
    "default_alpha": 10,
}


def get_data(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    root = os.path.join(ROOT, "data")
    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    rng = np.random.RandomState(0)

    if config["train_n"] < len(train_ds):
        idx = rng.permutation(len(train_ds))[:config["train_n"]]
        train_ds = torch.utils.data.Subset(train_ds, idx)
    if config["test_n"] < len(test_ds):
        idx = rng.permutation(len(test_ds))[:config["test_n"]]
        test_ds = torch.utils.data.Subset(test_ds, idx)

    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=config["batch_size"], shuffle=False)
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)
    return train_ld, test_ld


@torch.no_grad()
def collect_trajectory_and_norms(system, loader, T, device):
    all_z_real = []
    all_norms = []
    all_labels = []

    for xb, yb in loader:
        xb = xb.to(device)
        batch = xb.shape[0]
        d = system.hidden_dim

        x_complex = xb.to(torch.complex64)
        Ux = torch.matmul(x_complex, system.U.T)

        z = torch.zeros(batch, d, dtype=torch.complex64, device=device)
        batch_z_real = np.zeros((T, batch, 2 * d), dtype=np.float32)
        batch_norms = np.zeros((T, batch), dtype=np.float32)

        for t in range(T):
            pre = torch.matmul(z, system.W.T) + Ux + system.b
            if system.noise_std > 0:
                noise = torch.complex(
                    torch.randn_like(pre.real) * system.noise_std,
                    torch.randn_like(pre.imag) * system.noise_std,
                )
                pre = pre + noise
            z = system.modrelu(pre)

            mag = torch.abs(z)
            mask = mag > 50.0
            if mask.any():
                z = torch.where(mask, z * (50.0 / mag.clamp(min=1e-8)), z)

            batch_z_real[t] = torch.cat([z.real, z.imag], dim=-1).cpu().numpy()
            batch_norms[t] = torch.abs(z).mean(dim=-1).cpu().numpy()

        all_z_real.append(batch_z_real)
        all_norms.append(batch_norms)
        all_labels.append(yb.numpy())

    return {
        "z_real": np.concatenate(all_z_real, axis=1),
        "norms": np.concatenate(all_norms, axis=1),
        "labels": np.concatenate(all_labels),
    }


def early_window_features(z_real, K):
    return z_real[:K].mean(axis=0)


def adaptive_stop_time(norms, threshold):
    mean_norms = norms.mean(axis=1)
    ratio = mean_norms / max(mean_norms[0], 1e-10)
    exceeded = np.where(ratio > threshold)[0]
    if len(exceeded) == 0:
        return len(mean_norms) - 1
    return max(0, int(exceeded[0]) - 1)


def compute_grace_features(z_real, norms, tau, alpha):
    g = norms / np.maximum(norms[0:1, :], 1e-10)
    w_raw = np.exp(-alpha * np.maximum(0.0, g - tau))
    w = w_raw / np.maximum(w_raw.sum(axis=0, keepdims=True), 1e-10)
    return np.einsum("tn,tnf->nf", w, z_real)


def score_probe(train_features, train_labels, test_features, test_labels):
    probe = LinearProbe()
    probe.fit(train_features, train_labels)
    return probe.score(test_features, test_labels)


def aggregate_metric(per_run, metric_name):
    grouped = defaultdict(list)
    for rec in per_run:
        grouped[rec["spectral_radius"]].append(rec[metric_name])
    out = {}
    for sr, vals in sorted(grouped.items()):
        out[f"{sr:.2f}"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def build_selection_summary(per_run, config):
    ref_key = f"{config['reference_rho']:.2f}"
    ref_runs = [r for r in per_run if abs(r["spectral_radius"] - config["reference_rho"]) < 1e-9]

    # Deterministic candidate order keeps the published defaults first in tie cases.
    tau_stop_order = [config["default_tau_stop"]] + [
        t for t in config["tau_stop_values"] if t != config["default_tau_stop"]
    ]
    grace_order = [(config["default_tau"], config["default_alpha"])] + [
        (tau, alpha)
        for tau in config["tau_values"]
        for alpha in config["alpha_values"]
        if not (tau == config["default_tau"] and alpha == config["default_alpha"])
    ]

    best_tau_stop = None
    best_tau_stop_mean = -np.inf
    for tau_stop in tau_stop_order:
        vals = [r["adaptive_acc"][f"{tau_stop:.1f}"]["accuracy"] for r in ref_runs]
        mean_val = float(np.mean(vals))
        if mean_val > best_tau_stop_mean + 1e-12:
            best_tau_stop = tau_stop
            best_tau_stop_mean = mean_val

    best_grace = None
    best_grace_mean = -np.inf
    for tau, alpha in grace_order:
        vals = [r["grace_acc"][f"{tau:.1f}|{alpha}"] for r in ref_runs]
        mean_val = float(np.mean(vals))
        if mean_val > best_grace_mean + 1e-12:
            best_grace = (tau, alpha)
            best_grace_mean = mean_val

    summary = {
        "reference_rho": config["reference_rho"],
        "adaptive_selected": {
            "tau_stop": best_tau_stop,
            "mean_ref_accuracy": best_tau_stop_mean,
        },
        "grace_selected": {
            "tau": best_grace[0],
            "alpha": best_grace[1],
            "mean_ref_accuracy": best_grace_mean,
        },
        "frozen_by_radius": {},
    }

    grouped = defaultdict(list)
    for rec in per_run:
        grouped[rec["spectral_radius"]].append(rec)

    for sr, runs in sorted(grouped.items()):
        sr_key = f"{sr:.2f}"
        early_k10 = [r["early_k_acc"]["10"] for r in runs]
        adaptive_vals = [r["adaptive_acc"][f"{best_tau_stop:.1f}"]["accuracy"] for r in runs]
        grace_vals = [r["grace_acc"][f"{best_grace[0]:.1f}|{best_grace[1]}"] for r in runs]
        summary["frozen_by_radius"][sr_key] = {
            "early_K10_mean": float(np.mean(early_k10)),
            "early_K10_std": float(np.std(early_k10)),
            "adaptive_mean": float(np.mean(adaptive_vals)),
            "adaptive_std": float(np.std(adaptive_vals)),
            "grace_mean": float(np.mean(grace_vals)),
            "grace_std": float(np.std(grace_vals)),
        }

    return summary


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ld, test_ld = get_data(CONFIG)
    print(f"Train: {len(train_ld.dataset)} | Test: {len(test_ld.dataset)}")

    T = CONFIG["rollout_steps"]
    total = len(CONFIG["spectral_radii"]) * CONFIG["num_seeds"]
    per_run = []
    start = time.time()

    for idx, sr in enumerate(CONFIG["spectral_radii"]):
        for seed in range(CONFIG["num_seeds"]):
            run_id = idx * CONFIG["num_seeds"] + seed + 1
            t0 = time.time()
            print(f"[{run_id}/{total}] rho={sr:.2f} seed={seed} ... ", end="", flush=True)

            system = ComplexDynamicalSystem(
                input_dim=CONFIG["input_dim"],
                hidden_dim=CONFIG["hidden_dim"],
                spectral_radius=sr,
                noise_std=CONFIG["noise_std"],
                modrelu_bias=CONFIG["modrelu_bias"],
                seed=seed,
            ).to(device)

            train_data = collect_trajectory_and_norms(system, train_ld, T, device)
            test_data = collect_trajectory_and_norms(system, test_ld, T, device)

            early_acc = {}
            for K in CONFIG["early_K_values"]:
                train_features = early_window_features(train_data["z_real"], K)
                test_features = early_window_features(test_data["z_real"], K)
                early_acc[str(K)] = score_probe(
                    train_features, train_data["labels"], test_features, test_data["labels"]
                )

            adaptive_acc = {}
            for tau_stop in CONFIG["tau_stop_values"]:
                stop_t = adaptive_stop_time(train_data["norms"], tau_stop)
                acc = score_probe(
                    train_data["z_real"][stop_t],
                    train_data["labels"],
                    test_data["z_real"][stop_t],
                    test_data["labels"],
                )
                adaptive_acc[f"{tau_stop:.1f}"] = {
                    "accuracy": acc,
                    "stop_t": int(stop_t + 1),
                }

            grace_acc = {}
            for tau in CONFIG["tau_values"]:
                for alpha in CONFIG["alpha_values"]:
                    key = f"{tau:.1f}|{alpha}"
                    train_features = compute_grace_features(train_data["z_real"], train_data["norms"], tau, alpha)
                    test_features = compute_grace_features(test_data["z_real"], test_data["norms"], tau, alpha)
                    grace_acc[key] = score_probe(
                        train_features, train_data["labels"], test_features, test_data["labels"]
                    )

            elapsed = time.time() - t0
            per_run.append({
                "spectral_radius": sr,
                "seed": seed,
                "early_k_acc": early_acc,
                "adaptive_acc": adaptive_acc,
                "grace_acc": grace_acc,
                "elapsed": elapsed,
            })

            print(
                f"K10={early_acc['10']:.4f} "
                f"tau_stop=1.5->{adaptive_acc['1.5']['accuracy']:.4f}@t={adaptive_acc['1.5']['stop_t']} "
                f"grace(1.2,10)={grace_acc['1.2|10']:.4f} ({elapsed:.0f}s)"
            )

    k_summary = {}
    for K in CONFIG["early_K_values"]:
        metric_name = f"early_K_{K}"
        for rec in per_run:
            rec[metric_name] = rec["early_k_acc"][str(K)]
        k_summary[str(K)] = aggregate_metric(per_run, metric_name)

    selection = build_selection_summary(per_run, CONFIG)

    payload = {
        "config": CONFIG,
        "per_run": per_run,
        "k_summary": k_summary,
        "selection": selection,
        "total_runtime_sec": time.time() - start,
    }

    out_path = os.path.join(RESULTS_DIR, "readout_audits.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved JSON to {out_path}")
    print(f"Done in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
