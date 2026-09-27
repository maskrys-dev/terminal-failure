"""
Accuracy-defined grace-period analysis for the publication revision.

This script measures the formal grace period

    G(rho, alpha) = max { t : acc(phi_t) >= alpha }

from per-timestep probe accuracies rather than from the norm-growth proxy.

Key runtime protections:
  - checkpoint/resume after every completed (rho, seed) run
  - optional small-slice benchmarking via CLI flags
  - direct collection of per-timestep real features, avoiding repeated tensor
    conversion work during probe fitting
  - optional threaded probe fitting across timesteps

Outputs:
  - results/grace_period/<stem>.json
  - results/grace_period/<stem>.png
  - results/grace_period/<stem>.pdf
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transient_geometry.probes import LinearProbe
from transient_geometry.system import ComplexDynamicalSystem


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results", "grace_period")

DEFAULT_CONFIG = {
    "input_dim": 784,
    "hidden_dim": 128,
    "rollout_steps": 40,
    "modrelu_bias": -0.5,
    "noise_std": 0.0,
    "spectral_radii": [0.5, 1.0, 1.20, 1.30, 1.35, 1.40, 1.45, 1.50, 1.60],
    "num_seeds": 10,
    "train_n": 5000,
    "test_n": 5000,
    "batch_size": 512,
    "alpha_thresholds": [0.65, 0.70, 0.75],
    "probe_workers": 1,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compute accuracy-defined grace periods.")
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_CONFIG["num_seeds"])
    parser.add_argument("--train-n", type=int, default=DEFAULT_CONFIG["train_n"])
    parser.add_argument("--test-n", type=int, default=DEFAULT_CONFIG["test_n"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--probe-workers", type=int, default=DEFAULT_CONFIG["probe_workers"])
    parser.add_argument("--output-stem", type=str, default="accuracy_grace_period")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing JSON file.")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="Run only the first N missing (rho, seed) jobs for benchmarking.")
    parser.add_argument("--spectral-radii", type=float, nargs="*",
                        default=DEFAULT_CONFIG["spectral_radii"])
    parser.add_argument("--alpha-thresholds", type=float, nargs="*",
                        default=DEFAULT_CONFIG["alpha_thresholds"])
    return parser.parse_args()


def build_config(args):
    return {
        **DEFAULT_CONFIG,
        "spectral_radii": [float(v) for v in args.spectral_radii],
        "num_seeds": args.num_seeds,
        "train_n": args.train_n,
        "test_n": args.test_n,
        "batch_size": args.batch_size,
        "alpha_thresholds": [float(v) for v in args.alpha_thresholds],
        "probe_workers": max(1, int(args.probe_workers)),
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
def collect_z_real(system, loader, T, device):
    all_z_real = []
    all_labels = []

    for xb, yb in loader:
        xb = xb.to(device)
        batch = xb.shape[0]
        d = system.hidden_dim

        x_complex = xb.to(torch.complex64)
        Ux = torch.matmul(x_complex, system.U.T)

        z = torch.zeros(batch, d, dtype=torch.complex64, device=device)
        batch_z_real = np.zeros((T, batch, 2 * d), dtype=np.float32)

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

        all_z_real.append(batch_z_real)
        all_labels.append(yb.numpy())

    return np.concatenate(all_z_real, axis=1), np.concatenate(all_labels)


def fit_score_one_timestep(train_features, train_labels, test_features, test_labels):
    probe = LinearProbe()
    probe.fit(train_features, train_labels)
    return probe.score(test_features, test_labels)


def per_timestep_curve(train_z_real, train_labels, test_z_real, test_labels, probe_workers):
    T = train_z_real.shape[0]

    if probe_workers <= 1:
        return np.array([
            fit_score_one_timestep(train_z_real[t], train_labels, test_z_real[t], test_labels)
            for t in range(T)
        ], dtype=np.float64)

    accs = np.zeros(T, dtype=np.float64)
    with concurrent.futures.ThreadPoolExecutor(max_workers=probe_workers) as ex:
        futures = {
            ex.submit(
                fit_score_one_timestep,
                train_z_real[t], train_labels,
                test_z_real[t], test_labels,
            ): t
            for t in range(T)
        }
        for future in concurrent.futures.as_completed(futures):
            t = futures[future]
            accs[t] = future.result()
    return accs


def true_grace_period(accs, alpha):
    below = np.where(np.asarray(accs) < alpha)[0]
    return int(below[0]) if len(below) else int(len(accs))


def aggregate_runs(per_run, thresholds):
    aggregated = {f"{alpha:.2f}": {} for alpha in thresholds}
    mean_curves = {}

    grouped = defaultdict(list)
    for rec in per_run:
        grouped[rec["spectral_radius"]].append(rec)

    for sr, runs in sorted(grouped.items()):
        sr_key = f"{sr:.2f}"
        curves = np.array([r["per_timestep_acc"] for r in runs], dtype=np.float64)
        mean_curves[sr_key] = {
            "mean": curves.mean(axis=0).tolist(),
            "std": curves.std(axis=0).tolist(),
        }
        for alpha in thresholds:
            alpha_key = f"{alpha:.2f}"
            vals = np.array([r["G"][alpha_key] for r in runs], dtype=np.float64)
            aggregated[alpha_key][sr_key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
            }

    return aggregated, mean_curves


def plot_quicklook(aggregated, thresholds, out_dir, stem, rollout_steps):
    colors = {
        "0.65": "#2166ac",
        "0.70": "#e07b39",
        "0.75": "#2d8f4e",
    }
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for alpha in thresholds:
        alpha_key = f"{alpha:.2f}"
        if not aggregated[alpha_key]:
            continue
        xs = [float(k) for k in aggregated[alpha_key].keys()]
        ys = [aggregated[alpha_key][k]["mean"] for k in aggregated[alpha_key]]
        es = [aggregated[alpha_key][k]["std"] for k in aggregated[alpha_key]]
        color = colors.get(alpha_key, None)
        ax.plot(xs, ys, "o-", color=color, lw=2.3, ms=7, label=fr"$\alpha={alpha:.2f}$")
        ax.fill_between(xs, np.array(ys) - np.array(es), np.array(ys) + np.array(es),
                        color=color, alpha=0.15)

    ax.axvline(1.0, color="0.5", ls=":", lw=1.2)
    ax.axhline(rollout_steps, color="0.5", ls=":", lw=1.2)
    ax.set_xlabel(r"Spectral radius $\rho(W)$")
    ax.set_ylabel(r"Grace period $G(\rho,\alpha)$")
    ax.set_ylim(0, rollout_steps + 2)
    ax.set_title("Accuracy-defined grace period")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_payload(payload, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_existing(out_path):
    if not os.path.exists(out_path):
        return None
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    config = build_config(args)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    out_path = os.path.join(RESULTS_DIR, f"{args.output_stem}.json")
    existing = load_existing(out_path) if args.resume else None
    per_run = existing.get("per_run", []) if existing else []

    completed = {
        (round(float(rec["spectral_radius"]), 8), int(rec["seed"]))
        for rec in per_run
    }

    pending = []
    for sr in config["spectral_radii"]:
        for seed in range(config["num_seeds"]):
            key = (round(float(sr), 8), seed)
            if key not in completed:
                pending.append((float(sr), seed))

    if args.max_runs is not None:
        pending = pending[:args.max_runs]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"Config: radii={config['spectral_radii']}, seeds={config['num_seeds']}, "
        f"train={config['train_n']}, test={config['test_n']}, "
        f"probe_workers={config['probe_workers']}"
    )

    if not pending:
        print("No pending runs. Regenerating aggregates/figure from existing JSON.")
    else:
        print(f"Pending runs: {len(pending)}")

    train_ld, test_ld = get_data(config)
    print(f"Train: {len(train_ld.dataset)} | Test: {len(test_ld.dataset)}")

    start = time.time()
    total = len(pending)

    for run_idx, (sr, seed) in enumerate(pending, start=1):
        t0 = time.time()
        print(f"[{run_idx}/{total}] rho={sr:.2f} seed={seed} ... ", end="", flush=True)

        system = ComplexDynamicalSystem(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            spectral_radius=sr,
            noise_std=config["noise_std"],
            modrelu_bias=config["modrelu_bias"],
            seed=seed,
        ).to(device)

        t_collect = time.time()
        train_z_real, train_labels = collect_z_real(system, train_ld, config["rollout_steps"], device)
        test_z_real, test_labels = collect_z_real(system, test_ld, config["rollout_steps"], device)
        collect_elapsed = time.time() - t_collect

        t_probe = time.time()
        accs = per_timestep_curve(
            train_z_real,
            train_labels,
            test_z_real,
            test_labels,
            config["probe_workers"],
        )
        probe_elapsed = time.time() - t_probe

        g_vals = {
            f"{alpha:.2f}": true_grace_period(accs, alpha)
            for alpha in config["alpha_thresholds"]
        }
        elapsed = time.time() - t0

        per_run.append({
            "spectral_radius": sr,
            "seed": seed,
            "per_timestep_acc": accs.tolist(),
            "G": g_vals,
            "best_timestep": int(np.argmax(accs) + 1),
            "best_accuracy": float(accs.max()),
            "timing": {
                "collect_sec": collect_elapsed,
                "probe_sec": probe_elapsed,
                "total_sec": elapsed,
            },
        })

        aggregated, mean_curves = aggregate_runs(per_run, config["alpha_thresholds"])
        payload = {
            "config": config,
            "per_run": per_run,
            "aggregated_G": aggregated,
            "mean_curves": mean_curves,
            "total_runtime_sec": time.time() - start,
        }
        save_payload(payload, out_path)

        print(
            " ".join([f"G@{alpha:.2f}={g_vals[f'{alpha:.2f}']}" for alpha in config["alpha_thresholds"]]) +
            f" best={accs.max():.4f}@t={np.argmax(accs) + 1} "
            f"[collect {collect_elapsed:.0f}s, probes {probe_elapsed:.0f}s, total {elapsed:.0f}s]"
        )

    final_payload = load_existing(out_path) if os.path.exists(out_path) else {
        "config": config,
        "per_run": per_run,
        "aggregated_G": {},
        "mean_curves": {},
        "total_runtime_sec": time.time() - start,
    }
    plot_quicklook(
        final_payload["aggregated_G"],
        config["alpha_thresholds"],
        RESULTS_DIR,
        args.output_stem,
        config["rollout_steps"],
    )
    print(f"Saved JSON to {out_path}")
    print(f"Done in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
