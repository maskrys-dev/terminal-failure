"""
Targeted FashionMNIST GRACE audit.

This script reuses the exact FashionMNIST setup from a3_fashion.py, but only
computes the GRACE readout. It is intentionally narrower than a full rerun of
the appendix experiment, so the manuscript table can include the named method
without recomputing best-timestep and grace-period diagnostics.

Usage:
    python -m transient_geometry.experiments.a3_fashion_grace
"""

import json
import os
import sys
import time

import numpy as np
import torch
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transient_geometry.probes import LinearProbe
from transient_geometry.system import ComplexDynamicalSystem


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(ROOT, "results", "a3_fashion")

CONFIG = {
    "input_dim": 784,
    "hidden_dim": 128,
    "rollout_steps": 40,
    "modrelu_bias": -0.5,
    "noise_std": 0.0,
    "spectral_radii": [0.5, 1.0, 1.20, 1.30, 1.35, 1.40, 1.45, 1.50, 1.60],
    "num_seeds": 3,
    "batch_size": 512,
    "train_n": 5000,
    "test_n": 5000,
    "grace_tau": 1.2,
    "grace_alpha": 10.0,
}


def get_fashion_data(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    root = os.path.join(ROOT, "data")
    train_ds = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    rng = np.random.RandomState(42)
    if config["train_n"] < len(train_ds):
        idx = rng.permutation(len(train_ds))[:config["train_n"]]
        train_ds = torch.utils.data.Subset(train_ds, idx)
    if config["test_n"] < len(test_ds):
        idx = rng.permutation(len(test_ds))[:config["test_n"]]
        test_ds = torch.utils.data.Subset(test_ds, idx)
    train_ld = torch.utils.data.DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=False
    )
    test_ld = torch.utils.data.DataLoader(
        test_ds, batch_size=config["batch_size"], shuffle=False
    )
    return train_ld, test_ld


@torch.no_grad()
def extract_grace_features(system, loader, config, device):
    T = config["rollout_steps"]
    tau = config["grace_tau"]
    alpha = config["grace_alpha"]
    features, labels = [], []

    for xb, yb in loader:
        xb = xb.to(device)
        traj = system(xb, T)  # [T, batch, d] complex
        norms = torch.abs(traj).mean(dim=-1)  # [T, batch], matching existing sweeps
        growth = norms / norms[0:1].clamp(min=1e-10)
        w_raw = torch.exp(-alpha * torch.clamp(growth - tau, min=0.0))
        weights = w_raw / w_raw.sum(dim=0, keepdim=True).clamp(min=1e-10)
        z_grace = (weights[..., None] * traj).sum(dim=0)
        feat = torch.cat([z_grace.real, z_grace.imag], dim=-1).cpu().numpy()
        features.append(feat)
        labels.append(yb.numpy())

    return np.concatenate(features), np.concatenate(labels)


def aggregate(per_run):
    summary = {}
    for sr in sorted({r["spectral_radius"] for r in per_run}):
        vals = np.array([r["acc_grace"] for r in per_run if r["spectral_radius"] == sr])
        summary[f"{sr:.2f}"] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
        }
    return summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    train_ld, test_ld = get_fashion_data(CONFIG)
    per_run = []

    total = len(CONFIG["spectral_radii"]) * CONFIG["num_seeds"]
    run_idx = 0
    for sr in CONFIG["spectral_radii"]:
        for seed in range(CONFIG["num_seeds"]):
            run_idx += 1
            t0 = time.time()
            system = ComplexDynamicalSystem(
                input_dim=CONFIG["input_dim"],
                hidden_dim=CONFIG["hidden_dim"],
                spectral_radius=sr,
                noise_std=CONFIG["noise_std"],
                modrelu_bias=CONFIG["modrelu_bias"],
                seed=seed,
            ).to(device)

            x_train, y_train = extract_grace_features(system, train_ld, CONFIG, device)
            x_test, y_test = extract_grace_features(system, test_ld, CONFIG, device)
            probe = LinearProbe()
            probe.fit(x_train, y_train)
            acc = probe.score(x_test, y_test)
            elapsed = time.time() - t0

            rec = {
                "spectral_radius": sr,
                "seed": seed,
                "acc_grace": float(acc),
                "tau": CONFIG["grace_tau"],
                "alpha": CONFIG["grace_alpha"],
                "elapsed": elapsed,
            }
            per_run.append(rec)
            print(f"[{run_idx}/{total}] rho={sr:.2f} seed={seed} GRACE={acc:.4f} ({elapsed:.0f}s)")

    out = {
        "config": CONFIG,
        "per_run": per_run,
        "summary": aggregate(per_run),
    }
    out_path = os.path.join(RESULTS_DIR, "grace_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
