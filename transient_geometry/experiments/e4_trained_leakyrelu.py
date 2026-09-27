"""
E4: Trained Iterative Classifier with LeakyReLU — Pre-Collapse Under Gain Sweep
==================================================================================
Train a shared-weight iterative (recurrent) classifier with LeakyReLU dynamics.

LeakyReLU is the key design choice:
  - Unbounded piecewise-linear: permits magnitude growth (unlike tanh)
  - Trainable: clean gradients, converges fast
  - Phase-like asymmetry: different behaviour for small vs large activations

This directly tests whether the pre-collapse regime appears in a TRAINED
iterative model with unbounded piecewise-linear dynamics - the decisive extension
to the fixed-substrate results.

Architecture:
    h_{t+1} = LeakyReLU(lambda * W @ h_t + U @ x + b)
    logits   = V @ h_T + c

Training:
    T_train = 10 (short horizon)
    lambda_train = 0.85 (firmly contractive)

Test:
    T_test = 40 (long horizon)
    lambda_test in {0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0}

Expected result:
    - At lambda=0.85: final ≈ early ≈ grace (all fine)
    - As lambda rises past 1.0-1.2: final degrades first
    - Early window, GRACE, adaptive stopping retain accuracy
    - This confirms the pre-collapse regime in a trained iterative model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import RidgeClassifierCV
from torchvision import datasets, transforms

# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "e4_trained_leakyrelu"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).parent.parent.parent / "data"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CFG = dict(
    hidden_dim=256,      # larger to give more expressive power
    input_dim=784,
    n_classes=10,
    # Training: short horizon + low gain = model cannot learn to be robust to
    #           long-horizon or high-gain test conditions
    T_train=10,
    lambda_train=0.85,
    leaky_slope=0.2,     # LeakyReLU negative slope
    # Test: long horizon + wide gain sweep
    T_test=40,
    early_K=8,
    lambda_test_vals=[0.7, 0.85, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0],
    # Training
    lr=1e-3,
    epochs=40,
    batch_size=256,
    n_train=10000,       # more data = better features
    n_test=2000,
    seeds=[0, 1, 2, 3, 4],
    # Readout hyperparameters (same as main paper)
    grace_tau=1.2,
    grace_alpha=10.0,
    stop_tau=1.5,
    # Numerical safety
    state_clip=100.0,
)

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class TrainedLeakyReLURNN(nn.Module):
    """
    Trained shared-weight iterative classifier with LeakyReLU.

    Recurrence:
        h_{t+1} = LeakyReLU(lambda * W @ h_t + U @ x + b)

    Classifier:
        logits = V @ h_T + c

    W, U, b, V, c are all trained. lambda is fixed during training, swept at test time.

    Initialisation:
        W: spectral radius normalised to 1 (lambda controls actual sr)
        U: standard Glorot
        V: small normal
    """

    def __init__(self, input_dim, hidden_dim, n_classes=10,
                 lam_init=0.85, leaky_slope=0.2, state_clip=100.0):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.lam         = lam_init
        self.leaky_slope = leaky_slope
        self.state_clip  = state_clip

        # ── W: normalised to spectral radius ≈ 1 ────────────────────────
        W = torch.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        eigs = torch.linalg.eigvals(W)
        sr   = torch.max(torch.abs(eigs)).item()
        W    = W / max(sr, 1e-8)          # sr(W) = 1; lam * sr(W) = lam
        self.W = nn.Parameter(W)

        # ── U: input projection ──────────────────────────────────────────
        self.U = nn.Parameter(
            torch.randn(hidden_dim, input_dim) / np.sqrt(input_dim)
        )

        # ── b: recurrent bias ────────────────────────────────────────────
        self.b = nn.Parameter(torch.zeros(hidden_dim))

        # ── V, c: output head ────────────────────────────────────────────
        self.V = nn.Parameter(
            torch.randn(n_classes, hidden_dim) * 0.01
        )
        self.c = nn.Parameter(torch.zeros(n_classes))

    def _act(self, x):
        return F.leaky_relu(x, negative_slope=self.leaky_slope)

    def rollout(self, x, T, lam=None, return_trajectory=False):
        """
        Roll out for T steps.

        Parameters
        ----------
        x : Tensor [batch, input_dim]
        T : int
        lam : float — recurrent gain (defaults to self.lam)
        return_trajectory : bool

        Returns
        -------
        h_T : Tensor [batch, hidden_dim]
        trajectory (optional): list of Tensor [batch, hidden_dim], length T
        """
        if lam is None:
            lam = self.lam

        batch = x.shape[0]
        Ux    = F.linear(x, self.U)          # [batch, hidden_dim] — constant drive

        h = torch.zeros(batch, self.hidden_dim, device=x.device)
        trajectory = [] if return_trajectory else None

        for _ in range(T):
            h = self._act(lam * F.linear(h, self.W) + Ux + self.b)
            # Clip for numerical safety (prevents gradient catastrophe)
            h = torch.clamp(h, -self.state_clip, self.state_clip)
            if return_trajectory:
                trajectory.append(h)

        if return_trajectory:
            return h, trajectory
        return h

    def forward(self, x, T=None, lam=None):
        if T is None:
            T = 10
        h_T = self.rollout(x, T, lam=lam)
        return F.linear(h_T, self.V, self.c)


# ─────────────────────────────────────────────────────────────────────────────
# GRACE and adaptive stopping (real-valued)
# ─────────────────────────────────────────────────────────────────────────────

def grace_readout(trajectory, tau=1.2, alpha=10.0):
    """GRACE over real-valued trajectory."""
    norm_0 = trajectory[0].norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w_sum  = torch.zeros_like(trajectory[0])
    wt_sum = torch.zeros(trajectory[0].shape[0], 1, device=trajectory[0].device)
    for h_t in trajectory:
        g_t = h_t.norm(dim=-1, keepdim=True) / norm_0
        w_t = torch.exp(-alpha * torch.clamp(g_t - tau, min=0.0))
        w_sum  += w_t * h_t
        wt_sum += w_t
    return (w_sum / wt_sum.clamp(min=1e-8)).cpu().numpy()


def adaptive_stop(trajectory, tau_stop=1.5):
    """Read out at first timestep where norm growth > tau_stop."""
    batch  = trajectory[0].shape[0]
    device = trajectory[0].device
    norm_0 = trajectory[0].norm(dim=-1).clamp(min=1e-8)
    result  = trajectory[-1].clone()
    stopped = torch.zeros(batch, dtype=torch.bool, device=device)
    for h_t in trajectory:
        growth = h_t.norm(dim=-1) / norm_0
        just_stopped = (~stopped) & (growth > tau_stop)
        if just_stopped.any():
            result[just_stopped] = h_t[just_stopped]
            stopped |= just_stopped
        if stopped.all():
            break
    return result.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_mnist(n_train, n_test, seed=0):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda img: img.view(-1)),
    ])
    tr = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tf)
    te = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    rng = torch.Generator().manual_seed(seed)
    tr_idx = torch.randperm(len(tr), generator=rng)[:n_train]
    te_idx = torch.randperm(len(te), generator=rng)[:n_test]
    X_tr = torch.stack([tr[i][0] for i in tr_idx])
    y_tr = torch.tensor([tr[i][1] for i in tr_idx])
    X_te = torch.stack([te[i][0] for i in te_idx])
    y_te = torch.tensor([te[i][1] for i in te_idx])
    return X_tr, y_tr, X_te, y_te


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, X_tr, y_tr, cfg, verbose=True):
    """
    Short-horizon terminal-state training.
    Deliberately narrow scope: the model only learns to use h_{T_train}
    at lambda_train. It cannot learn robustness to longer horizons or
    larger gains — that mismatch is what exposes the pre-collapse regime.
    """
    model.train()
    opt   = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'])
    X_tr  = X_tr.to(DEVICE)
    y_tr  = y_tr.to(DEVICE)

    best_acc = 0.0
    for epoch in range(cfg['epochs']):
        perm = torch.randperm(len(X_tr), device=DEVICE)
        total_loss = 0.0
        n_correct  = 0
        for i in range(0, len(X_tr), cfg['batch_size']):
            idx = perm[i:i + cfg['batch_size']]
            xb, yb = X_tr[idx], y_tr[idx]
            opt.zero_grad()
            logits = model(xb, T=cfg['T_train'], lam=cfg['lambda_train'])
            loss   = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(xb)
            n_correct  += (logits.argmax(1) == yb).sum().item()
        sched.step()
        acc = n_correct / len(X_tr)
        best_acc = max(best_acc, acc)
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{cfg['epochs']}  "
                  f"loss={total_loss/len(X_tr):.4f}  acc={acc:.3f}")
    print(f"  Best train acc: {best_acc:.3f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_regime_sweep(model, X_te, y_te, cfg):
    model.eval()
    X_te    = X_te.to(DEVICE)
    y_te_np = y_te.cpu().numpy()
    n = len(y_te_np);  split = n // 2
    y_tr_p, y_te_p = y_te_np[:split], y_te_np[split:]

    def probe(feat):
        clf = RidgeClassifierCV(alphas=[0.1, 1.0, 10.0, 100.0])
        clf.fit(feat[:split], y_tr_p)
        return clf.score(feat[split:], y_te_p)

    results = {}
    for lam in cfg['lambda_test_vals']:
        print(f"  lam={lam:.2f} ...", end=" ", flush=True)

        all_traj        = [[] for _ in range(cfg['T_test'])]
        head_logits_all = []
        norm_trajs      = []

        for i in range(0, len(X_te), cfg['batch_size']):
            xb = X_te[i:i + cfg['batch_size']]
            h_T, traj = model.rollout(xb, cfg['T_test'],
                                      lam=lam, return_trajectory=True)
            logits = F.linear(h_T, model.V, model.c)
            head_logits_all.append(logits.cpu())
            for t, h_t in enumerate(traj):
                all_traj[t].append(h_t.detach())
            norm_trajs.append([h_t.norm(dim=-1).mean().item() for h_t in traj])

        # Consolidate
        trajectory     = [torch.cat(all_traj[t], dim=0) for t in range(cfg['T_test'])]
        mean_norm_traj = np.mean(norm_trajs, axis=0).tolist()

        # Features
        feat_final = trajectory[-1].cpu().numpy()
        feat_early = np.stack([trajectory[t].cpu().numpy()
                               for t in range(cfg['early_K'])]).mean(axis=0)
        feat_grace = grace_readout(trajectory, cfg['grace_tau'], cfg['grace_alpha'])
        feat_stop  = adaptive_stop(trajectory, cfg['stop_tau'])

        # Accuracies
        acc_final    = probe(feat_final)
        acc_early    = probe(feat_early)
        acc_grace    = probe(feat_grace)
        acc_adaptive = probe(feat_stop)
        head_logits  = torch.cat(head_logits_all, dim=0)
        acc_head     = (head_logits.argmax(1).numpy() == y_te_np).mean()

        per_t_acc = [probe(trajectory[t].cpu().numpy()) for t in range(cfg['T_test'])]

        results[lam] = {
            'head':      float(acc_head),
            'final':     float(acc_final),
            'early':     float(acc_early),
            'grace':     float(acc_grace),
            'adaptive':  float(acc_adaptive),
            'per_t':     per_t_acc,
            'norm_traj': mean_norm_traj,
        }
        gain = acc_grace - acc_final
        flag = '  <-- GRACE wins' if gain > 0.005 else ''
        print(f"head={acc_head:.3f}  final={acc_final:.3f}  "
              f"early={acc_early:.3f}  grace={acc_grace:.3f}  "
              f"adap={acc_adaptive:.3f}  "
              f"norm_T={mean_norm_traj[-1]:.1f}{flag}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate and visualise
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(all_results, cfg):
    metrics = ['final', 'early', 'grace', 'adaptive']
    summary = {}
    for lam in cfg['lambda_test_vals']:
        summary[lam] = {
            m: {'mean': float(np.mean([all_results[s][lam][m] for s in cfg['seeds']])),
                'std':  float(np.std( [all_results[s][lam][m] for s in cfg['seeds']]))}
            for m in metrics
        }
    return summary


def make_plots(all_results, summary, cfg, save_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lams    = cfg['lambda_test_vals']
    metrics = ['final', 'early', 'grace', 'adaptive']
    colors  = {'final': '#e74c3c', 'early': '#f39c12',
               'adaptive': '#3498db', 'grace': '#2ecc71'}
    labels  = {
        'final':    f'Final state (h_T, T={cfg["T_test"]})',
        'early':    f'Early window (t=1..{cfg["early_K"]})',
        'adaptive': 'Adaptive stopping',
        'grace':    'GRACE',
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Trained LeakyReLU Iterative Classifier — Pre-Collapse Under Test-Time Gain Sweep\n"
        f"Trained: T={cfg['T_train']}, lam={cfg['lambda_train']} | "
        f"Tested: T={cfg['T_test']}, lam up to {max(lams)}",
        fontsize=12, fontweight='bold'
    )

    # Panel 1: accuracy sweep
    ax = axes[0]
    for m in metrics:
        mn = [summary[l][m]['mean'] for l in lams]
        sd = [summary[l][m]['std']  for l in lams]
        ax.plot(lams, mn, 'o-', color=colors[m], label=labels[m], lw=2, markersize=5)
        ax.fill_between(lams, [a-b for a,b in zip(mn,sd)],
                              [a+b for a,b in zip(mn,sd)], alpha=0.15, color=colors[m])
    ax.axvline(1.0, color='gray', ls='--', alpha=0.6, label='spectral boundary (rho=1)')
    ax.axvline(cfg['lambda_train'], color='k', ls=':', alpha=0.5,
               label=f'lambda_train={cfg["lambda_train"]}')
    ax.set_xlabel('Test-time gain lambda')
    ax.set_ylabel('Linear probe accuracy')
    ax.set_title('Readout accuracy vs. test-time gain')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Panel 2: per-timestep at first lambda where final degrades >2%
    ax2 = axes[1]
    baseline = summary[lams[0]]['final']['mean']
    tgt = next((l for l in lams if l>1.0 and summary[l]['final']['mean'] < baseline-0.02),
               lams[-1])
    for s in cfg['seeds']:
        ax2.plot(range(1, cfg['T_test']+1), all_results[s][tgt]['per_t'],
                 alpha=0.2, color='#95a5a6', lw=1)
    mean_pt = np.mean([[all_results[s][tgt]['per_t'][t]
                        for t in range(cfg['T_test'])] for s in cfg['seeds']], axis=0)
    ax2.plot(range(1, cfg['T_test']+1), mean_pt, 'k-', lw=2.5,
             label=f'Mean per-timestep (lam={tgt})')
    for m in ['grace', 'final', 'adaptive', 'early']:
        ax2.axhline(summary[tgt][m]['mean'], color=colors[m], ls='--', lw=1.5,
                    label=f'{m}={summary[tgt][m]["mean"]:.3f}')
    ax2.set_xlabel('Timestep t')
    ax2.set_ylabel('Probe accuracy')
    ax2.set_title(f'Grace period at lam={tgt}')
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    # Panel 3: GRACE advantage bar chart
    ax3 = axes[2]
    gains = [summary[l]['grace']['mean'] - summary[l]['final']['mean'] for l in lams]
    ax3.bar(range(len(lams)), gains,
            color=['#2ecc71' if g > 0 else '#e74c3c' for g in gains])
    ax3.axhline(0, color='k', lw=0.8)
    ax3.set_xticks(range(len(lams)))
    ax3.set_xticklabels([f'{l:.1f}' for l in lams], rotation=45, fontsize=8)
    ax3.set_xlabel('Test-time gain lambda')
    ax3.set_ylabel('GRACE - Final accuracy')
    ax3.set_title('GRACE advantage over endpoint readout')
    ax3.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(save_dir / 'e4_trained_leakyrelu.png', dpi=150, bbox_inches='tight')
    print(f"Figure saved.")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    all_results = {}

    for seed in CFG['seeds']:
        print(f"\n{'='*65}")
        print(f"E4 Seed {seed} | T_train={CFG['T_train']} lam_train={CFG['lambda_train']}"
              f" | T_test={CFG['T_test']} lam up to {max(CFG['lambda_test_vals'])}")
        print(f"{'='*65}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        X_tr, y_tr, X_te, y_te = load_mnist(CFG['n_train'], CFG['n_test'], seed=seed)

        model = TrainedLeakyReLURNN(
            input_dim=CFG['input_dim'],
            hidden_dim=CFG['hidden_dim'],
            n_classes=CFG['n_classes'],
            lam_init=CFG['lambda_train'],
            leaky_slope=CFG['leaky_slope'],
            state_clip=CFG['state_clip'],
        ).to(DEVICE)

        print("Training...")
        train_model(model, X_tr, y_tr, CFG)

        print("\nEvaluating regime sweep...")
        all_results[seed] = evaluate_regime_sweep(model, X_te, y_te, CFG)

    # Aggregate
    summary = aggregate(all_results, CFG)

    print("\n\n" + "="*80)
    print("AGGREGATE (5 seeds)")
    print(f"{'lam':>6}  {'final':>14}  {'early':>14}  "
          f"{'grace':>14}  {'adaptive':>14}  {'GRACE-final':>12}")
    print("-"*80)
    for lam in CFG['lambda_test_vals']:
        s    = summary[lam]
        gain = s['grace']['mean'] - s['final']['mean']
        flag = ' <<' if gain > 0.005 else ''
        print(f"{lam:>6.2f}  "
              f"{s['final']['mean']:.3f}+-{s['final']['std']:.3f}  "
              f"{s['early']['mean']:.3f}+-{s['early']['std']:.3f}  "
              f"{s['grace']['mean']:.3f}+-{s['grace']['std']:.3f}  "
              f"{s['adaptive']['mean']:.3f}+-{s['adaptive']['std']:.3f}  "
              f"{gain:+.4f}{flag}")

    # Save
    out = RESULTS_DIR / "e4_trained_leakyrelu.json"
    with open(out, 'w') as f:
        json.dump({
            'config':   CFG,
            'per_seed': {str(k): v for k, v in all_results.items()},
            'summary':  {str(k): v for k, v in summary.items()}
        }, f, indent=2)
    print(f"\nSaved: {out}")

    try:
        make_plots(all_results, summary, CFG, RESULTS_DIR)
    except Exception as e:
        print(f"Plot error: {e}")

    return summary


if __name__ == "__main__":
    main()
