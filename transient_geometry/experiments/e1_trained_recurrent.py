"""
E1: Trained Iterative Recurrent System — Pre-Collapse Regime Under Regime Shift
=================================================================================
Train a small shared-weight recurrent classifier end-to-end on MNIST.
Then at test time sweep the recurrent gain lambda, pushing the system
into the unstable/pre-collapse regime.

Architecture:
    z_{t+1} = tanh(lambda * W @ z_t + U @ x + b)
    classifier: linear(z_T)

Training:  lambda_train = 0.9 (stable regime)
Test:      lambda_test  in {0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5}

Readouts evaluated at test time:
    - final (z_T)
    - early window (mean z_1..z_K, K=5)
    - adaptive stopping (norm growth threshold)
    - GRACE

Key thesis: even in a TRAINED system, endpoint readout degrades under
regime shift while trajectory-aware readouts remain robust.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from pathlib import Path
from sklearn.linear_model import RidgeClassifierCV
from torchvision import datasets, transforms

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "e1_trained_recurrent"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(__file__).parent.parent.parent / "data"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CFG = dict(
    hidden_dim=128,
    input_dim=784,
    T_train=20,          # rollout during training
    T_test=30,           # rollout during test (longer to see degradation)
    early_K=5,           # early window size
    lambda_train=0.9,    # recurrent gain during training
    lambda_test_vals=[0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    lr=1e-3,
    epochs=30,
    batch_size=256,
    n_train=5000,        # subset of MNIST for speed
    n_test=2000,
    seeds=[0, 1, 2, 3, 4],
    grace_tau=1.2,
    grace_alpha=10.0,
    stop_tau=1.5,
)

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class TrainedRecurrentSystem(nn.Module):
    """
    Trained shared-weight recurrent classifier.

    Architecture
    ------------
    z_{t+1} = tanh(lam * W z_t + U x + b)
    logits  = V z_T + c

    W is a parameter (trained). U, b, V, c are also trained.
    lam (lambda) is a gain scalar — varied at test time for regime sweep.
    """

    def __init__(self, input_dim, hidden_dim, n_classes=10, lam=0.9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lam = lam

        # Recurrent weight (normalised init to spectral radius ≈ lam)
        W_init = torch.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        eigs = torch.linalg.eigvals(W_init)
        sr = torch.max(torch.abs(eigs)).item()
        W_init = W_init * (lam / max(sr, 1e-8))
        self.W = nn.Parameter(W_init)

        self.U = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(hidden_dim))

        self.V = nn.Parameter(torch.randn(n_classes, hidden_dim) * 0.01)
        self.c = nn.Parameter(torch.zeros(n_classes))

    def rollout(self, x, T, lam=None, return_trajectory=False):
        """
        Roll out recurrent dynamics for T steps.

        Parameters
        ----------
        x : Tensor [batch, input_dim]
        T : int
        lam : float or None — if None, use self.lam (training gain)
        return_trajectory : bool — if True, return all z_t, else just z_T

        Returns
        -------
        z_T : Tensor [batch, hidden_dim]
        trajectory (optional): list of Tensor [batch, hidden_dim]
        """
        if lam is None:
            lam = self.lam

        batch = x.shape[0]
        z = torch.zeros(batch, self.hidden_dim, device=x.device)
        Ux = F.linear(x, self.U)  # [batch, hidden_dim]

        trajectory = [] if return_trajectory else None

        for _ in range(T):
            z = torch.tanh(lam * F.linear(z, self.W) + Ux + self.b)
            if return_trajectory:
                trajectory.append(z)

        if return_trajectory:
            return z, trajectory
        return z

    def forward(self, x, T=None, lam=None):
        if T is None:
            T = 20  # default training T
        z_T = self.rollout(x, T, lam=lam)
        return F.linear(z_T, self.V, self.c)


# ─────────────────────────────────────────────────────────────────────────────
# GRACE and adaptive stopping (real-valued)
# ─────────────────────────────────────────────────────────────────────────────

def grace_readout(trajectory, tau=1.2, alpha=10.0):
    """
    GRACE readout over a trajectory of real-valued states.

    Parameters
    ----------
    trajectory : list of Tensor [batch, d], length T
    tau, alpha  : GRACE hyperparameters

    Returns
    -------
    features : ndarray [batch, d]
    """
    T = len(trajectory)
    norm_t0 = trajectory[0].norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [batch, 1]

    weighted_sum = torch.zeros_like(trajectory[0])
    weight_sum   = torch.zeros(trajectory[0].shape[0], 1, device=trajectory[0].device)

    for z_t in trajectory:
        g_t = z_t.norm(dim=-1, keepdim=True) / norm_t0          # growth ratio
        w_t = torch.exp(-alpha * torch.clamp(g_t - tau, min=0.0))
        weighted_sum += w_t * z_t
        weight_sum   += w_t

    grace_state = weighted_sum / weight_sum.clamp(min=1e-8)
    return grace_state.cpu().numpy()


def adaptive_stop_readout(trajectory, tau_stop=1.5):
    """
    Read out the state at the first timestep where norm growth exceeds tau_stop.
    Falls back to the last timestep if threshold is never crossed.
    """
    T = len(trajectory)
    batch = trajectory[0].shape[0]
    device = trajectory[0].device

    norm_t0 = trajectory[0].norm(dim=-1).clamp(min=1e-8)  # [batch]
    result = trajectory[-1].clone()  # default: last state
    stopped = torch.zeros(batch, dtype=torch.bool, device=device)

    for z_t in trajectory:
        growth = z_t.norm(dim=-1) / norm_t0
        just_stopped = (~stopped) & (growth > tau_stop)
        if just_stopped.any():
            result[just_stopped] = z_t[just_stopped]
            stopped |= just_stopped
        if stopped.all():
            break

    return result.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_mnist(n_train=5000, n_test=2000):
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tf)
    te = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)

    # Subset
    tr_idx = torch.randperm(len(tr))[:n_train]
    te_idx = torch.randperm(len(te))[:n_test]

    X_tr = torch.stack([tr[i][0] for i in tr_idx])
    y_tr = torch.tensor([tr[i][1] for i in tr_idx])
    X_te = torch.stack([te[i][0] for i in te_idx])
    y_te = torch.tensor([te[i][1] for i in te_idx])

    return X_tr, y_tr, X_te, y_te


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, X_tr, y_tr, cfg, verbose=True):
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, cfg['epochs'])

    X_tr = X_tr.to(DEVICE)
    y_tr = y_tr.to(DEVICE)

    for epoch in range(cfg['epochs']):
        perm = torch.randperm(len(X_tr), device=DEVICE)
        epoch_loss = 0.0
        n_correct = 0

        for i in range(0, len(X_tr), cfg['batch_size']):
            idx = perm[i:i + cfg['batch_size']]
            xb, yb = X_tr[idx], y_tr[idx]

            optimiser.zero_grad()
            logits = model(xb, T=cfg['T_train'], lam=cfg['lambda_train'])
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            epoch_loss += loss.item() * len(xb)
            n_correct += (logits.argmax(1) == yb).sum().item()

        scheduler.step()
        if verbose and (epoch + 1) % 10 == 0:
            acc = n_correct / len(X_tr)
            print(f"  Epoch {epoch+1:3d}/{cfg['epochs']}  "
                  f"loss={epoch_loss/len(X_tr):.4f}  acc={acc:.3f}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation at test time with regime sweep
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_regime_sweep(model, X_te, y_te, cfg):
    """
    For each test lambda, evaluate:
      - final-state accuracy (direct logit from z_T)
      - linear probe on z_T features
      - early-window probe
      - GRACE probe
      - adaptive stopping probe
      - per-timestep probe accuracy (grace period measurement)
    """
    model.eval()
    X_te = X_te.to(DEVICE)
    y_te_np = y_te.cpu().numpy()

    results = {}

    for lam in cfg['lambda_test_vals']:
        print(f"  lambda_test={lam:.2f} ...", end=" ", flush=True)

        # Collect full trajectory
        all_states = [[] for _ in range(cfg['T_test'])]
        final_logits_all = []

        for i in range(0, len(X_te), cfg['batch_size']):
            xb = X_te[i:i + cfg['batch_size']]
            z_T, traj = model.rollout(xb, cfg['T_test'], lam=lam,
                                      return_trajectory=True)
            # Direct logit from trained head (final state)
            logits = F.linear(z_T, model.V, model.c)
            final_logits_all.append(logits.cpu())

            for t, z_t in enumerate(traj):
                all_states[t].append(z_t.detach())

        # Consolidate
        trajectory = [torch.cat(states, dim=0) for states in all_states]
        final_logits = torch.cat(final_logits_all, dim=0)

        # 1. Final-state accuracy (trained head, no probe needed)
        final_acc_trained = (final_logits.argmax(1).numpy() == y_te_np).mean()

        # Feature arrays
        feat_final = trajectory[-1].cpu().numpy()
        feat_early = np.stack([trajectory[t].cpu().numpy()
                               for t in range(cfg['early_K'])]).mean(axis=0)
        feat_grace = grace_readout(trajectory, cfg['grace_tau'], cfg['grace_alpha'])
        feat_stop  = adaptive_stop_readout(trajectory, cfg['stop_tau'])

        # 2. Ridge probes (train on first half, test on second half)
        n = len(y_te_np)
        split = n // 2
        y_tr_p, y_te_p = y_te_np[:split], y_te_np[split:]

        def probe_acc(feat):
            clf = RidgeClassifierCV(alphas=[0.1, 1, 10, 100])
            clf.fit(feat[:split], y_tr_p)
            return clf.score(feat[split:], y_te_p)

        acc_final  = probe_acc(feat_final)
        acc_early  = probe_acc(feat_early)
        acc_grace  = probe_acc(feat_grace)
        acc_stop   = probe_acc(feat_stop)

        # 3. Per-timestep probe accuracy (grace period characterisation)
        per_t_acc = []
        for t in range(cfg['T_test']):
            feat_t = trajectory[t].cpu().numpy()
            per_t_acc.append(probe_acc(feat_t))

        results[lam] = {
            'final_trained_head': float(final_acc_trained),
            'final_probe':  float(acc_final),
            'early':        float(acc_early),
            'grace':        float(acc_grace),
            'adaptive':     float(acc_stop),
            'per_timestep': per_t_acc,
        }
        print(f"final(head)={final_acc_trained:.3f}  final(probe)={acc_final:.3f}  "
              f"grace={acc_grace:.3f}  adaptive={acc_stop:.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_seed(seed, cfg):
    print(f"\n{'='*60}")
    print(f"Seed {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr, y_tr, X_te, y_te = load_mnist(cfg['n_train'], cfg['n_test'])

    model = TrainedRecurrentSystem(
        input_dim=cfg['input_dim'],
        hidden_dim=cfg['hidden_dim'],
        lam=cfg['lambda_train'],
    ).to(DEVICE)

    print(f"Training with lambda={cfg['lambda_train']}...")
    model = train_model(model, X_tr, y_tr, cfg, verbose=True)

    print("\nEvaluating regime sweep...")
    results = evaluate_regime_sweep(model, X_te, y_te, cfg)

    return results


def main():
    all_results = {}

    for seed in CFG['seeds']:
        results = run_seed(seed, CFG)
        all_results[seed] = results

    # ── Aggregate across seeds ──────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("AGGREGATE RESULTS (mean ± std across seeds)")
    print("="*60)

    metrics = ['final_probe', 'early', 'grace', 'adaptive']
    summary = {}

    for lam in CFG['lambda_test_vals']:
        accs = {m: [all_results[s][lam][m] for s in CFG['seeds']]
                for m in metrics}
        summary[lam] = {m: {'mean': np.mean(v), 'std': np.std(v)}
                        for m, v in accs.items()}

        print(f"\nlambda={lam:.1f}:")
        for m in metrics:
            mn, sd = summary[lam][m]['mean'], summary[lam][m]['std']
            print(f"  {m:15s}: {mn:.3f} ± {sd:.3f}")

    # ── Save ───────────────────────────────────────────────────────────────
    save_data = {
        'config': CFG,
        'per_seed': {str(k): v for k, v in all_results.items()},
        'summary': {str(k): v for k, v in summary.items()},
    }
    out_path = RESULTS_DIR / "e1_trained_recurrent.json"
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Plot ───────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        lams = CFG['lambda_test_vals']
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Trained Recurrent System — Pre-Collapse Regime Under Gain Sweep",
                     fontsize=13, fontweight='bold')

        colors = {
            'final_probe':  '#e74c3c',
            'early':        '#f39c12',
            'adaptive':     '#3498db',
            'grace':        '#2ecc71',
        }
        labels = {
            'final_probe':  'Final state (z_T)',
            'early':        f'Early window (z_1..z_{CFG["early_K"]})',
            'adaptive':     'Adaptive stopping',
            'grace':        'GRACE',
        }

        ax = axes[0]
        for m in metrics:
            means = [summary[lam][m]['mean'] for lam in lams]
            stds  = [summary[lam][m]['std']  for lam in lams]
            ax.plot(lams, means, 'o-', color=colors[m], label=labels[m], linewidth=2)
            ax.fill_between(lams,
                            [mn - sd for mn, sd in zip(means, stds)],
                            [mn + sd for mn, sd in zip(means, stds)],
                            alpha=0.15, color=colors[m])
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='ρ=1 boundary')
        ax.set_xlabel("Test-time gain λ (recurrent spectral radius)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Regime sweep — trained recurrent system")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

        # Per-timestep accuracy at highest lambda
        ax2 = axes[1]
        lam_high = CFG['lambda_test_vals'][-1]
        for s in CFG['seeds']:
            per_t = all_results[s][lam_high]['per_timestep']
            ax2.plot(range(1, len(per_t)+1), per_t, alpha=0.4, color='#95a5a6', linewidth=1)
        mean_per_t = np.mean([[all_results[s][lam_high]['per_timestep'][t]
                               for t in range(CFG['T_test'])]
                              for s in CFG['seeds']], axis=0)
        ax2.plot(range(1, CFG['T_test']+1), mean_per_t, 'k-', linewidth=2.5,
                 label=f'Mean (λ={lam_high})')
        ax2.axhline(y=summary[lam_high]['grace']['mean'], color=colors['grace'],
                    linestyle='--', linewidth=1.5, label=f"GRACE={summary[lam_high]['grace']['mean']:.3f}")
        ax2.axhline(y=summary[lam_high]['final_probe']['mean'], color=colors['final_probe'],
                    linestyle='--', linewidth=1.5, label=f"Final={summary[lam_high]['final_probe']['mean']:.3f}")
        ax2.set_xlabel("Timestep t")
        ax2.set_ylabel("Probe accuracy")
        ax2.set_title(f"Per-timestep accuracy at λ={lam_high} (grace period)")
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "e1_trained_recurrent.png", dpi=150, bbox_inches='tight')
        print(f"Figure saved.")
        plt.close()

    except Exception as e:
        print(f"Plot failed: {e}")

    return summary


if __name__ == "__main__":
    main()
