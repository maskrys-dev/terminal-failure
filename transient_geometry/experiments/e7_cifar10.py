"""
E7: Pre-collapse regime on CIFAR-10 (multi-seed) -- GPU accelerated.

Uses PyTorch complex tensors on CUDA for all reservoir operations.
Expected runtime: ~3-5 minutes total (vs >1hr on CPU).

Run:  python -W ignore transient_geometry/experiments/e7_cifar10.py
Results: results/e7_cifar10/e7_cifar10.json
"""

import json, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SPECTRAL_RADII = [0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.25,
                  1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60]
N_SEEDS      = 10
D            = 512          # back to 512 — GPU handles it easily
T            = 40
MODRELU_BIAS = -0.5
STATE_CLIP   = 50.0
TAU_STOP     = 1.5
TAU_GRACE    = 1.2
ALPHA_GRACE  = 10.0
N_TRAIN      = 5000
N_TEST       = 5000
EARLY_K      = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = Path("results/e7_cifar10")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# modReLU (PyTorch)
# ---------------------------------------------------------------------------
def modrelu_torch(z: torch.Tensor, bias: float = MODRELU_BIAS) -> torch.Tensor:
    """Complex modReLU: max(|z|+b, 0) * z/|z|"""
    mag = z.abs()
    scale = torch.clamp(mag + bias, min=0.0) / (mag + 1e-12)
    return scale * z


# ---------------------------------------------------------------------------
# Data loading -- once
# ---------------------------------------------------------------------------
def load_all_cifar10(data_root: str = "data"):
    print(f"Loading CIFAR-10 (device={DEVICE})...")
    tf = transforms.ToTensor()
    train_ds = datasets.CIFAR10(data_root, train=True,  download=True, transform=tf)
    test_ds  = datasets.CIFAR10(data_root, train=False, download=True, transform=tf)

    def _extract(ds):
        loader = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=0)
        Xs, ys = [], []
        for imgs, labels in loader:
            Xs.append(imgs.view(imgs.size(0), -1).numpy())
            ys.append(labels.numpy())
        return np.concatenate(Xs).astype(np.float32), np.concatenate(ys).astype(int)

    X_tr, y_tr = _extract(train_ds)
    X_te, y_te = _extract(test_ds)
    X_all = np.concatenate([X_tr, X_te])
    y_all = np.concatenate([y_tr, y_te])
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)
    print(f"  Loaded: {X_all.shape}")
    return X_all, y_all


def split_for_seed(X_all, y_all, seed):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=N_TRAIN,
                                 test_size=N_TEST, random_state=seed)
    tr, te = next(sss.split(X_all, y_all))
    return X_all[tr], y_all[tr], X_all[te], y_all[te]


# ---------------------------------------------------------------------------
# GPU reservoir run -- all ops on CUDA
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_reservoir_gpu(UX: torch.Tensor, W: torch.Tensor,
                      b: torch.Tensor) -> dict:
    """
    UX : (N, D) complex64 on DEVICE  -- pre-projected input
    W  : (D, D) complex64 on DEVICE
    b  : (D,)   complex64 on DEVICE
    Returns dict of (N, 2D) float32 numpy arrays.
    """
    N = UX.shape[0]
    traj = torch.empty(N, T, D, dtype=torch.complex64, device=DEVICE)
    z = torch.zeros(N, D, dtype=torch.complex64, device=DEVICE)

    for t in range(T):
        z = modrelu_torch(z @ W.T + UX + b)
        # State clipping
        mag = z.abs()
        z = torch.where(mag > STATE_CLIP, z * (STATE_CLIP / (mag + 1e-12)), z)
        traj[:, t, :] = z

    # Norms (N, T)
    norms = traj.abs().norm(dim=2)   # (N, T)

    # ── Final ──
    final_z = traj[:, -1, :]
    final = torch.cat([final_z.real, final_z.imag], dim=1)

    # ── Early window ──
    ez = traj[:, :EARLY_K, :].mean(dim=1)
    early = torch.cat([ez.real, ez.imag], dim=1)

    # ── Adaptive stopping (vectorised) ──
    ratio  = norms / (norms[:, 0:1] + 1e-12)
    stop_t = (ratio > TAU_STOP).long().argmax(dim=1)   # first crossing
    never  = ratio.max(dim=1).values <= TAU_STOP
    stop_t[never] = T - 1
    az = traj[torch.arange(N, device=DEVICE), stop_t]
    adaptive = torch.cat([az.real, az.imag], dim=1)

    # ── GRACE (vectorised einsum) ──
    growth = norms / (norms[:, 0:1] + 1e-12)
    w_raw  = torch.exp(-ALPHA_GRACE * torch.clamp(growth - TAU_GRACE, min=0.0))
    w      = w_raw / (w_raw.sum(dim=1, keepdim=True) + 1e-12)   # (N, T)
    gz     = torch.einsum("nt,ntd->nd", w.to(torch.complex64), traj)  # (N, D) complex
    grace  = torch.cat([gz.real, gz.imag], dim=1)

    def to_np(t):
        return t.float().cpu().numpy()

    return {"final": to_np(final), "early": to_np(early),
            "adaptive": to_np(adaptive), "grace": to_np(grace)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print(f"\nE7: CIFAR-10 pre-collapse  |  {N_SEEDS} seeds x {len(SPECTRAL_RADII)} radii  |  D={D}  |  {DEVICE}")
    print("=" * 70)

    X_all, y_all = load_all_cifar10()
    n_in = X_all.shape[1]   # 3072

    all_results = {}

    for seed in range(N_SEEDS):
        t_seed = time.time()
        print(f"\nSeed {seed+1}/{N_SEEDS}")
        rng = np.random.RandomState(seed)

        X_train, y_train, X_test, y_test = split_for_seed(X_all, y_all, seed)

        # Move input projections to GPU as complex
        X_tr_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
        X_te_t = torch.tensor(X_test,  dtype=torch.float32, device=DEVICE)

        seed_results = {}

        for rho in SPECTRAL_RADII:
            t_rho = time.time()

            # --- Build complex reservoir weights on GPU ---
            W_np = (rng.randn(D, D) + 1j * rng.randn(D, D)).astype(np.complex64) / np.sqrt(D)
            # Power iteration for spectral radius
            v = np.random.randn(D).astype(np.complex64)
            for _ in range(50):
                v = W_np @ v
                v /= (np.linalg.norm(v) + 1e-30)
            sr = float(abs(v @ (W_np @ v) / (v @ v + 1e-30)))
            W_np *= rho / (sr + 1e-12)

            U_np = (rng.randn(D, n_in) + 1j * rng.randn(D, n_in)).astype(np.complex64) / np.sqrt(n_in)
            U_np /= (np.linalg.norm(U_np, "fro") / np.sqrt(D))

            W_t = torch.tensor(W_np, dtype=torch.complex64, device=DEVICE)
            U_t = torch.tensor(U_np, dtype=torch.complex64, device=DEVICE)
            b_t = torch.full((D,), MODRELU_BIAS, dtype=torch.complex64, device=DEVICE)

            # Pre-project inputs: (N, n_in) real @ (n_in, D) complex = (N, D) complex
            UX_tr = X_tr_t.to(torch.complex64) @ U_t.T
            UX_te = X_te_t.to(torch.complex64) @ U_t.T

            tr = run_reservoir_gpu(UX_tr, W_t, b_t)
            te = run_reservoir_gpu(UX_te, W_t, b_t)

            rho_res = {}
            for m in ("final", "early", "adaptive", "grace"):
                ft, fte = tr[m], te[m]
                if not (np.isfinite(ft).all() and np.isfinite(fte).all()):
                    print(f"  WARNING: non-finite in {m} at rho={rho:.2f}")
                    rho_res[m] = float("nan")
                    continue
                clf = RidgeClassifier(alpha=1.0)
                clf.fit(ft, y_train)
                rho_res[m] = float(clf.score(fte, y_test))

            rs = str(round(rho, 2))
            seed_results[rs] = rho_res
            print(f"  rho={rho:.2f}  final={rho_res['final']:.3f}  "
                  f"early={rho_res['early']:.3f}  "
                  f"grace={rho_res['grace']:.3f}  "
                  f"({time.time()-t_rho:.1f}s)")

        all_results[seed] = seed_results
        print(f"  Seed {seed+1} done in {time.time()-t_seed:.1f}s")

    # Aggregate
    summary = {}
    for rho in SPECTRAL_RADII:
        rs = str(round(rho, 2))
        summary[rs] = {}
        for m in ("final", "early", "adaptive", "grace"):
            vals = [all_results[s][rs][m] for s in range(N_SEEDS)
                    if not np.isnan(all_results[s][rs][m])]
            summary[rs][m] = {"mean": float(np.mean(vals)) if vals else float("nan"),
                               "std":  float(np.std(vals))  if vals else float("nan")}

    out = {
        "experiment": "E7_cifar10_gpu", "n_seeds": N_SEEDS, "d": D, "T": T,
        "n_train": N_TRAIN, "n_test": N_TEST, "device": str(DEVICE),
        "spectral_radii": SPECTRAL_RADII,
        "tau_stop": TAU_STOP, "tau_grace": TAU_GRACE, "alpha_grace": ALPHA_GRACE,
        "per_seed": all_results, "summary": summary,
    }
    OUT_DIR.joinpath("e7_cifar10.json").write_text(json.dumps(out, indent=2))

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min  ->  results/e7_cifar10/e7_cifar10.json")
    print("\nHeadline (mean over seeds):")
    for rho in [0.9, 1.2, 1.4, 1.6]:
        rs = str(round(rho, 2))
        if rs in summary:
            r = summary[rs]
            print(f"  rho={rho}  final={r['final']['mean']:.3f}  "
                  f"early={r['early']['mean']:.3f}  "
                  f"grace={r['grace']['mean']:.3f}")


if __name__ == "__main__":
    main()
