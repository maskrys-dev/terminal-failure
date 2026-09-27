"""
E12: CIFAR-10 pre-collapse test with a stronger frozen feature extractor.

Pipeline:
  CIFAR-10 images -> frozen ImageNet ResNet-18 penultimate features ->
  complex modReLU reservoir -> readout spectral sweep.

This directly tests whether the endpoint-vs-trajectory gap survives when the
input representation is much stronger than raw pixels.

Run:
  python -W ignore transient_geometry/experiments/e12_cifar10_resnet18_features.py

Results:
  results/e12_cifar10_resnet18/e12_cifar10_resnet18.json
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

warnings.filterwarnings("ignore")


SPECTRAL_RADII = [0.5, 0.9, 1.0, 1.2, 1.3, 1.4, 1.5, 1.6]
N_SEEDS = 5
N_TRAIN = 10_000
D = 512
T = 40
EARLY_K = 8
MODRELU_BIAS = -0.5
STATE_CLIP = 50.0
TAU_STOP = 1.5
TAU_GRACE = 1.2
ALPHA_GRACE = 10.0
BATCH_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("results/e12_cifar10_resnet18")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_CACHE = OUT_DIR / "resnet18_cifar10_features.pt"
torch.hub.set_dir(str(OUT_DIR / "torch_hub"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--radii", type=str, default=",".join(map(str, SPECTRAL_RADII)))
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def modrelu_torch(z: torch.Tensor, bias: float = MODRELU_BIAS) -> torch.Tensor:
    mag = z.abs()
    scale = torch.clamp(mag + bias, min=0.0) / (mag + 1e-12)
    return scale * z


def build_resnet18_feature_extractor() -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(DEVICE)
    return model


def cifar_transform():
    return transforms.Compose(
        [
            transforms.Resize(224, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


@torch.no_grad()
def extract_split_features(ds, model: nn.Module, split_name: str):
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    feats, labels = [], []
    for i, (imgs, y) in enumerate(loader):
        imgs = imgs.to(DEVICE, non_blocking=True)
        h = model(imgs).float().cpu().numpy()
        feats.append(h)
        labels.append(y.numpy())
        if (i + 1) % 50 == 0:
            print(f"  {split_name}: extracted {(i + 1) * BATCH_SIZE} examples")
    return np.concatenate(feats).astype(np.float32), np.concatenate(labels).astype(int)


def load_or_extract_features(force: bool = False):
    if FEATURE_CACHE.exists() and not force:
        print(f"Loading cached ResNet-18 features from {FEATURE_CACHE}")
        cached = torch.load(FEATURE_CACHE, map_location="cpu", weights_only=False)
        return cached["X_train"], cached["y_train"], cached["X_test"], cached["y_test"]

    print("Extracting frozen ResNet-18 features for CIFAR-10...")
    model = build_resnet18_feature_extractor()
    tf = cifar_transform()
    train_ds = datasets.CIFAR10("data", train=True, download=False, transform=tf)
    test_ds = datasets.CIFAR10("data", train=False, download=False, transform=tf)
    X_train, y_train = extract_split_features(train_ds, model, "train")
    X_test, y_test = extract_split_features(test_ds, model, "test")
    torch.save(
        {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        },
        FEATURE_CACHE,
    )
    print(f"Saved feature cache to {FEATURE_CACHE}")
    return X_train, y_train, X_test, y_test


def split_for_seed(X_train_all, y_train_all, seed):
    sss = StratifiedShuffleSplit(
        n_splits=1,
        train_size=N_TRAIN,
        random_state=seed,
    )
    (idx_train, _idx_unused) = next(sss.split(X_train_all, y_train_all))
    return X_train_all[idx_train], y_train_all[idx_train]


def estimate_spectral_radius(W: np.ndarray) -> float:
    # E12 is meant to be a clean reviewer-facing check, so use the exact
    # eigenvalue radius rather than the fast approximate scaling used in older
    # pilot scripts.
    return float(np.max(np.abs(np.linalg.eigvals(W))))


@torch.no_grad()
def run_reservoir_gpu(UX: torch.Tensor, W: torch.Tensor, b: torch.Tensor):
    n = UX.shape[0]
    traj = torch.empty(n, T, D, dtype=torch.complex64, device=DEVICE)
    z = torch.zeros(n, D, dtype=torch.complex64, device=DEVICE)

    for t in range(T):
        z = modrelu_torch(z @ W.T + UX + b)
        mag = z.abs()
        z = torch.where(mag > STATE_CLIP, z * (STATE_CLIP / (mag + 1e-12)), z)
        traj[:, t, :] = z

    norms = traj.abs().norm(dim=2)

    final_z = traj[:, -1, :]
    final = torch.cat([final_z.real, final_z.imag], dim=1)

    early_z = traj[:, :EARLY_K, :].mean(dim=1)
    early = torch.cat([early_z.real, early_z.imag], dim=1)

    ratio = norms / (norms[:, 0:1] + 1e-12)
    stop_t = (ratio > TAU_STOP).long().argmax(dim=1)
    never = ratio.max(dim=1).values <= TAU_STOP
    stop_t[never] = T - 1
    adaptive_z = traj[torch.arange(n, device=DEVICE), stop_t]
    adaptive = torch.cat([adaptive_z.real, adaptive_z.imag], dim=1)

    w_raw = torch.exp(-ALPHA_GRACE * torch.clamp(ratio - TAU_GRACE, min=0.0))
    w = w_raw / (w_raw.sum(dim=1, keepdim=True) + 1e-12)
    grace_z = torch.einsum("nt,ntd->nd", w.to(torch.complex64), traj)
    grace = torch.cat([grace_z.real, grace_z.imag], dim=1)

    def to_np(x):
        return x.float().cpu().numpy()

    return {
        "final": to_np(final),
        "early": to_np(early),
        "adaptive": to_np(adaptive),
        "grace": to_np(grace),
    }


def score_readouts(train_feats, test_feats, y_train, y_test):
    scores = {}
    for name in ("final", "early", "adaptive", "grace"):
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(train_feats[name], y_train)
        scores[name] = float(clf.score(test_feats[name], y_test))
    return scores


def plot_summary(summary, radii):
    colors = {
        "final": "#e05263",
        "early": "#4c9a2a",
        "adaptive": "#8a5fbf",
        "grace": "#2166ac",
    }
    labels = {
        "final": "Terminal",
        "early": "Early window",
        "adaptive": "Adaptive",
        "grace": "GRACE",
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for key in ("final", "early", "adaptive", "grace"):
        means = np.array([summary[str(r)][key]["mean"] for r in radii])
        stds = np.array([summary[str(r)][key]["std"] for r in radii])
        ax.plot(radii, means, marker="o", lw=2.0, color=colors[key], label=labels[key])
        ax.fill_between(radii, means - stds, means + stds, color=colors[key], alpha=0.12)
    ax.set_xlabel("Spectral radius")
    ax.set_ylabel("CIFAR-10 accuracy")
    ax.set_title("Frozen ResNet-18 features -> complex modReLU reservoir")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e12_cifar10_resnet18.pdf")
    fig.savefig(OUT_DIR / "e12_cifar10_resnet18.png", dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    radii = [float(x) for x in args.radii.split(",") if x.strip()]
    t0 = time.time()

    print(
        f"E12: CIFAR-10 ResNet-18 features | {args.seeds} seeds x {len(radii)} radii "
        f"| N_train={N_TRAIN} N_test=10000 | D={D} | {DEVICE}"
    )
    X_train_all, y_train_all, X_test_all, y_test = load_or_extract_features(args.force_features)
    print(f"Feature shapes: train={X_train_all.shape}, test={X_test_all.shape}")

    all_results = {}
    direct_scores = {}

    for seed in range(args.seeds):
        print(f"\nSeed {seed + 1}/{args.seeds}")
        seed_t0 = time.time()
        rng = np.random.RandomState(seed)

        X_train, y_train = split_for_seed(X_train_all, y_train_all, seed)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test_all).astype(np.float32)

        direct = RidgeClassifier(alpha=1.0)
        direct.fit(X_train, y_train)
        direct_scores[str(seed)] = float(direct.score(X_test, y_test))
        print(f"  direct ResNet-18 ridge={direct_scores[str(seed)]:.3f}")

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
        n_in = X_train.shape[1]
        seed_results = {}

        for rho in radii:
            rho_t0 = time.time()
            W_np = (rng.randn(D, D) + 1j * rng.randn(D, D)).astype(np.complex64) / np.sqrt(D)
            sr = estimate_spectral_radius(W_np)
            W_np *= rho / (sr + 1e-12)
            U_np = (rng.randn(D, n_in) + 1j * rng.randn(D, n_in)).astype(np.complex64) / np.sqrt(n_in)
            U_np /= np.linalg.norm(U_np, "fro") / np.sqrt(D)

            W_t = torch.tensor(W_np, dtype=torch.complex64, device=DEVICE)
            U_t = torch.tensor(U_np, dtype=torch.complex64, device=DEVICE)
            b_t = torch.full((D,), MODRELU_BIAS, dtype=torch.complex64, device=DEVICE)

            UX_train = X_train_t.to(torch.complex64) @ U_t.T
            UX_test = X_test_t.to(torch.complex64) @ U_t.T
            train_readouts = run_reservoir_gpu(UX_train, W_t, b_t)
            test_readouts = run_reservoir_gpu(UX_test, W_t, b_t)
            scores = score_readouts(train_readouts, test_readouts, y_train, y_test)
            seed_results[str(rho)] = scores

            print(
                f"  rho={rho:.2f} final={scores['final']:.3f} early={scores['early']:.3f} "
                f"adaptive={scores['adaptive']:.3f} grace={scores['grace']:.3f} "
                f"({time.time() - rho_t0:.1f}s)"
            )

            del W_t, U_t, UX_train, UX_test, train_readouts, test_readouts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_results[str(seed)] = seed_results
        print(f"  seed done in {(time.time() - seed_t0) / 60:.1f} min")

    summary = {}
    for rho in radii:
        key = str(rho)
        summary[key] = {}
        for method in ("final", "early", "adaptive", "grace"):
            vals = [all_results[str(seed)][key][method] for seed in range(args.seeds)]
            summary[key][method] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

    out = {
        "experiment": "E12_cifar10_resnet18_features",
        "description": "Frozen ImageNet ResNet-18 features -> complex modReLU reservoir",
        "device": str(DEVICE),
        "n_seeds": args.seeds,
        "n_train": N_TRAIN,
        "n_test": int(len(y_test)),
        "reservoir_dim": D,
        "rollout_steps": T,
        "early_k": EARLY_K,
        "spectral_radii": radii,
        "tau_stop": TAU_STOP,
        "tau_grace": TAU_GRACE,
        "alpha_grace": ALPHA_GRACE,
        "direct_resnet18_ridge": {
            "per_seed": direct_scores,
            "mean": float(np.mean(list(direct_scores.values()))),
            "std": float(np.std(list(direct_scores.values()))),
        },
        "per_seed": all_results,
        "summary": summary,
        "elapsed_seconds": time.time() - t0,
    }
    out_path = OUT_DIR / "e12_cifar10_resnet18.json"
    out_path.write_text(json.dumps(out, indent=2))
    plot_summary(summary, radii)

    print(f"\nSaved {out_path}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")
    print(
        "Direct ResNet-18 ridge: "
        f"{out['direct_resnet18_ridge']['mean']:.3f} +/- {out['direct_resnet18_ridge']['std']:.3f}"
    )
    for rho in radii:
        key = str(rho)
        row = summary[key]
        gap = row["early"]["mean"] - row["final"]["mean"]
        print(
            f"rho={rho:.2f}: final={row['final']['mean']:.3f}, "
            f"early={row['early']['mean']:.3f}, grace={row['grace']['mean']:.3f}, "
            f"early-final={100 * gap:+.1f} pp"
        )


if __name__ == "__main__":
    main()
