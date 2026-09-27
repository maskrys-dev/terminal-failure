"""
Magnitude-clipping audit for the fixed complex-modReLU reservoir.

This script does not train probes. It reruns the published MNIST reservoir
rollouts on the held-out split and counts how often the numerical safeguard
|z| > 50 is activated after modReLU and before phase-preserving clipping.

Usage:
    python -m transient_geometry.experiments.r_clipping_audit
"""

import json
import os
import sys
import time

import numpy as np
import torch
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transient_geometry.system import ComplexDynamicalSystem


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(ROOT, "results", "clipping_audit")

CONFIG = {
    "input_dim": 784,
    "hidden_dim": 128,
    "rollout_steps": 40,
    "modrelu_bias": -0.5,
    "noise_std": 0.0,
    "clip_threshold": 50.0,
    "spectral_radii": [0.5, 1.0, 1.20, 1.30, 1.35, 1.40, 1.45, 1.50, 1.60],
    "num_seeds": 10,
    "batch_size": 512,
    "test_n": 5000,
}


def get_test_loader(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    test_ds = datasets.MNIST(
        root=os.path.join(ROOT, "data"),
        train=False,
        download=True,
        transform=transform,
    )
    rng = np.random.RandomState(0)
    if config["test_n"] < len(test_ds):
        idx = rng.permutation(len(test_ds))[:config["test_n"]]
        test_ds = torch.utils.data.Subset(test_ds, idx)
    return torch.utils.data.DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
    )


@torch.no_grad()
def audit_system(system, loader, config, device):
    T = config["rollout_steps"]
    threshold = config["clip_threshold"]
    entry_hits = np.zeros(T, dtype=np.int64)
    entry_totals = np.zeros(T, dtype=np.int64)
    sample_step_hits = np.zeros(T, dtype=np.int64)
    sample_totals = np.zeros(T, dtype=np.int64)
    max_preclip_mag = np.zeros(T, dtype=np.float64)

    for xb, _ in loader:
        xb = xb.to(device)
        batch = xb.shape[0]
        d = system.hidden_dim

        x_complex = xb.to(torch.complex64)
        Ux = torch.matmul(x_complex, system.U.T)
        z = torch.zeros(batch, d, dtype=torch.complex64, device=device)

        for t in range(T):
            pre = torch.matmul(z, system.W.T) + Ux + system.b
            z = system.modrelu(pre)
            mag = torch.abs(z)
            mask = mag > threshold

            entry_hits[t] += int(mask.sum().item())
            entry_totals[t] += int(mask.numel())
            sample_step_hits[t] += int(mask.any(dim=1).sum().item())
            sample_totals[t] += int(batch)
            max_preclip_mag[t] = max(max_preclip_mag[t], float(mag.max().item()))

            if mask.any():
                z = torch.where(mask, z * (threshold / mag.clamp(min=1e-8)), z)

    total_entries = int(entry_totals.sum())
    total_sample_steps = int(sample_totals.sum())
    first_timestep = None
    hit_ts = np.where(entry_hits > 0)[0]
    if len(hit_ts):
        first_timestep = int(hit_ts[0] + 1)

    return {
        "entry_clip_fraction": float(entry_hits.sum() / max(total_entries, 1)),
        "sample_step_clip_fraction": float(sample_step_hits.sum() / max(total_sample_steps, 1)),
        "max_entry_clip_fraction_at_one_timestep": float(
            np.max(entry_hits / np.maximum(entry_totals, 1))
        ),
        "max_sample_step_clip_fraction_at_one_timestep": float(
            np.max(sample_step_hits / np.maximum(sample_totals, 1))
        ),
        "first_timestep_with_clipping": first_timestep,
        "max_preclip_magnitude": float(max_preclip_mag.max()),
        "entry_hits_by_timestep": entry_hits.tolist(),
        "entry_totals_by_timestep": entry_totals.tolist(),
        "sample_step_hits_by_timestep": sample_step_hits.tolist(),
        "sample_totals_by_timestep": sample_totals.tolist(),
        "max_preclip_magnitude_by_timestep": max_preclip_mag.tolist(),
    }


def summarise(records):
    by_rho = {}
    for rho in CONFIG["spectral_radii"]:
        runs = [r for r in records if abs(r["spectral_radius"] - rho) < 1e-12]
        entry = np.array([r["entry_clip_fraction"] for r in runs], dtype=float)
        sample = np.array([r["sample_step_clip_fraction"] for r in runs], dtype=float)
        first_ts = [r["first_timestep_with_clipping"] for r in runs]
        by_rho[f"{rho:.2f}"] = {
            "entry_clip_fraction_mean": float(entry.mean()),
            "entry_clip_fraction_max": float(entry.max()),
            "sample_step_clip_fraction_mean": float(sample.mean()),
            "sample_step_clip_fraction_max": float(sample.max()),
            "first_timestep_min": min([t for t in first_ts if t is not None], default=None),
            "runs_with_any_clipping": int(sum(t is not None for t in first_ts)),
            "max_preclip_magnitude": float(max(r["max_preclip_magnitude"] for r in runs)),
        }
    return by_rho


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_test_loader(CONFIG)
    records = []
    t0 = time.time()

    for rho in CONFIG["spectral_radii"]:
        for seed in range(CONFIG["num_seeds"]):
            run_t0 = time.time()
            system = ComplexDynamicalSystem(
                input_dim=CONFIG["input_dim"],
                hidden_dim=CONFIG["hidden_dim"],
                spectral_radius=rho,
                noise_std=CONFIG["noise_std"],
                modrelu_bias=CONFIG["modrelu_bias"],
                seed=seed,
            ).to(device)
            rec = audit_system(system, loader, CONFIG, device)
            rec.update({
                "spectral_radius": float(rho),
                "seed": int(seed),
                "elapsed": float(time.time() - run_t0),
            })
            records.append(rec)
            print(
                f"rho={rho:.2f} seed={seed} "
                f"entry={100*rec['entry_clip_fraction']:.6f}% "
                f"sample-step={100*rec['sample_step_clip_fraction']:.6f}% "
                f"first={rec['first_timestep_with_clipping']} "
                f"({rec['elapsed']:.1f}s)"
            )

    output = {
        "config": CONFIG,
        "device": str(device),
        "elapsed": float(time.time() - t0),
        "summary_by_rho": summarise(records),
        "runs": records,
    }
    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
