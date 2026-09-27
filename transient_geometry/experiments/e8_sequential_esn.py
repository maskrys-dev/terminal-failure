"""
E8: Permuted Sequential MNIST ESN — Pre-Collapse Regime.

Design:
    Permuted Sequential MNIST (psMNIST): every pixel fed one at a time in a
    fixed random permuted order (T=784 steps, input_dim=1).
    The permutation removes spatial structure, so:
      - information accumulates monotonically as more pixels are seen
      - any accuracy drop BEFORE t=784 must come from recurrent distortion,
        not from "bottom of image is blank"

    For a STABLE reservoir (rho << 1), the terminal state (t=784) is optimal.
    For a SUPERCRITICAL reservoir, recurrent amplification corrupts late states:
    accuracy peaks at some t* < 784 and declines toward the terminal state.

    This is the cleanest direct analogue of the main paper's pre-collapse regime
    in a standard temporal benchmark.

Readout strategies:
    final:    z_{784}   -- standard endpoint assumption
    early:    mean(z_{t} for t in last K_window steps of trajectory that peak)
              -> implemented as mean over z_{K_start}...z_{K_end}
              K chosen adaptively: mean over t=500..700 (middle-late, before
              potential late distortion)
    adaptive: z_{t^}  where t^ = first t with ||z_t||/||z_1|| > tau_stop
              (only meaningful if norm growth is monotone — checked)
    grace:    norm-growth-weighted average over full trajectory

Sweep:
    spectral_radius in {0.6, 0.7, 0.8, 0.9, 1.0, 1.05, 1.10, 1.15, 1.20}
    leak_rate       in {0.1, 0.3, 0.5}
    n_seeds         = 5  (smaller since T=784 is heavier)

Results: results/e8_sequential_esn/e8_psmnist.json
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SPECTRAL_RADII = [0.60, 0.70, 0.80, 0.90, 1.00, 1.05, 1.10, 1.15, 1.20]
LEAK_RATES     = [0.1, 0.3, 0.5]
N_SEEDS        = 5
D              = 256          # smaller D to keep runtime manageable at T=784
N_TRAIN        = 3000
N_TEST         = 1000
T              = 784          # one pixel per step
INPUT_DIM      = 1

# Window for "middle-late" averaging: ~rows 17-25 of pixel sequence
EARLY_START    = 500
EARLY_END      = 700

TAU_STOP       = 1.5
TAU_GRACE      = 1.2
ALPHA_GRACE    = 10.0

# Fixed random permutation (same across all runs)
PERMUTATION_SEED = 42

OUT_DIR = Path("results/e8_sequential_esn")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_psmnist(n_train: int, n_test: int, perm: np.ndarray,
                 data_root: str = "data"):
    """
    Load MNIST and apply fixed permutation to pixel order.
    Returns X (N, 784, 1) and y (N,).
    """
    tf = transforms.ToTensor()
    train_ds = datasets.MNIST(data_root, train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST(data_root, train=False, download=True, transform=tf)

    rng = np.random.RandomState(0)

    def _sample(ds, n):
        idx  = rng.choice(len(ds), n, replace=False)
        imgs = np.stack([ds[i][0].numpy().flatten() for i in idx])   # (N, 784)
        imgs = imgs[:, perm]           # apply permutation
        labs = np.array([ds[i][1] for i in idx])
        return imgs[:, :, None].astype(np.float32), labs  # (N, 784, 1)

    return _sample(train_ds, n_train) + _sample(test_ds, n_test)


# ---------------------------------------------------------------------------
# Reservoir
# ---------------------------------------------------------------------------
def init_reservoir(d: int, rho: float, seed: int):
    rng = np.random.RandomState(seed)
    W   = rng.randn(d, d) / np.sqrt(d)
    sr  = np.abs(np.linalg.eigvals(W)).max()
    W   = (W * rho / sr).astype(np.float32)
    U   = (rng.randn(d, INPUT_DIM) / np.sqrt(INPUT_DIM)).astype(np.float32)
    b   = np.zeros(d, dtype=np.float32)
    return W, U, b


def run_psesn(X: np.ndarray, W, U, b, leak: float,
              subsample: int = 10) -> dict:
    """
    Run T=784-step reservoir on permuted pixel sequence.
    Collects states every `subsample` steps to save memory.

    X: (N, 784, 1)
    Returns features and per-window norm trajectory.
    """
    N  = X.shape[0]
    D  = W.shape[0]
    T  = X.shape[1]

    # We store only sampled states to manage memory
    n_samples = T // subsample
    sampled_states = np.zeros((N, n_samples, D), dtype=np.float32)
    sampled_t      = np.arange(subsample - 1, T, subsample)  # t=9,19,...,779

    z = np.zeros((N, D), dtype=np.float32)
    s_idx = 0

    for t in range(T):
        x_t  = X[:, t, :]              # (N, 1)
        pre  = z @ W.T + x_t @ U.T + b
        z_new = np.tanh(pre)
        z = (1.0 - leak) * z + leak * z_new
        if t in sampled_t:
            sampled_states[:, s_idx, :] = z
            s_idx += 1

    # sampled_states: (N, 78, D) at t=9,19,...,779
    # final state is last sample (~t=779 or exactly t=783 if we save it)
    # Re-run last steps to get exact final:
    z_final = z.copy()   # z at t=783

    # Norm trajectory over sampled states
    norms = np.linalg.norm(sampled_states, axis=2)   # (N, 78)

    # --- Final ---
    final = z_final

    # --- Early window: sample indices corresponding to EARLY_START..EARLY_END ---
    early_mask = (sampled_t >= EARLY_START) & (sampled_t <= EARLY_END)
    if early_mask.any():
        early = sampled_states[:, early_mask, :].mean(axis=1)
    else:
        early = sampled_states[:, :10, :].mean(axis=1)

    # --- Adaptive stopping (over sampled states) ---
    ref_n  = norms[:, 0:1] + 1e-9
    ratio  = norms / ref_n
    stop_s = np.argmax(ratio > TAU_STOP, axis=1)
    never  = ratio.max(axis=1) <= TAU_STOP
    stop_s[never] = n_samples - 1
    adaptive = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        adaptive[i] = sampled_states[i, stop_s[i]]

    # --- GRACE ---
    grace = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        g     = norms[i] / (norms[i, 0] + 1e-9)
        w_raw = np.exp(-ALPHA_GRACE * np.maximum(0.0, g - TAU_GRACE))
        w     = w_raw / (w_raw.sum() + 1e-9)
        grace[i] = (w[:, None] * sampled_states[i]).sum(axis=0)

    return {
        "final":    final,
        "early":    early,
        "adaptive": adaptive,
        "grace":    grace,
    }

METHODS = ("final", "early", "adaptive", "grace")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0  = time.time()
    perm = np.random.RandomState(PERMUTATION_SEED).permutation(784)

    print("E8: Permuted Sequential MNIST ESN")
    print(f"  T={T}, D={D}, {N_SEEDS} seeds x {len(SPECTRAL_RADII)} radii")
    print("=" * 70)

    print("Loading psMNIST...")
    X_tr, y_tr, X_te, y_te = load_psmnist(N_TRAIN, N_TEST, perm)
    print(f"  X_tr: {X_tr.shape}")

    all_results = {}
    summaries   = {}

    for leak in LEAK_RATES:
        ls = str(round(leak, 2))
        print(f"\n--- Leak={leak} ---")
        all_results[ls] = {}

        for seed in range(N_SEEDS):
            seed_res = {}
            for rho in SPECTRAL_RADII:
                W, U, b = init_reservoir(D, rho, seed)
                tr = run_psesn(X_tr, W, U, b, leak)
                te = run_psesn(X_te, W, U, b, leak)
                rho_res = {}
                for m in METHODS:
                    clf = RidgeClassifier(alpha=1.0)
                    clf.fit(tr[m], y_tr)
                    rho_res[m] = float(clf.score(te[m], y_te))
                seed_res[str(round(rho, 2))] = rho_res
            all_results[ls][seed] = seed_res

            r16 = seed_res.get(str(round(SPECTRAL_RADII[-1], 2)), {})
            r09 = seed_res.get(str(round(SPECTRAL_RADII[0], 2)), {})
            print(f"  s{seed}  rho={SPECTRAL_RADII[-1]}: "
                  f"final={r16.get('final',0):.3f}  "
                  f"early={r16.get('early',0):.3f}  "
                  f"grace={r16.get('grace',0):.3f}  || "
                  f"rho={SPECTRAL_RADII[0]}: "
                  f"final={r09.get('final',0):.3f}")

        # Aggregate
        lk_sum = {}
        for rho in SPECTRAL_RADII:
            rs = str(round(rho, 2))
            lk_sum[rs] = {}
            for m in METHODS:
                vals = [all_results[ls][s][rs][m] for s in range(N_SEEDS)]
                lk_sum[rs][m] = {"mean": float(np.mean(vals)),
                                  "std":  float(np.std(vals))}
        summaries[ls] = lk_sum

    # Headline
    print("\n--- Headline: leak=0.3 ---")
    for rho in SPECTRAL_RADII:
        rs  = str(round(rho, 2))
        row = summaries["0.3"][rs]
        print(f"  rho={rho:.2f}  "
              f"final={row['final']['mean']:.3f}  "
              f"early={row['early']['mean']:.3f}  "
              f"grace={row['grace']['mean']:.3f}")

    out = {
        "experiment":    "E8_psmnist_esn",
        "T":             T,
        "D":             D,
        "input_dim":     INPUT_DIM,
        "early_window":  [EARLY_START, EARLY_END],
        "n_seeds":       N_SEEDS,
        "spectral_radii": SPECTRAL_RADII,
        "leak_rates":    LEAK_RATES,
        "tau_stop":      TAU_STOP,
        "tau_grace":     TAU_GRACE,
        "alpha_grace":   ALPHA_GRACE,
        "permutation_seed": PERMUTATION_SEED,
        "per_seed":      all_results,
        "summary":       summaries,
    }
    path = OUT_DIR / "e8_psmnist.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nDone in {(time.time()-t0)/60:.1f} min -> {path}")


if __name__ == "__main__":
    main()
