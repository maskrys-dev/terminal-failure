"""
E6: Trained at Operating Point — Does Training Suppress the Pre-Collapse Regime?
==================================================================================
Reviewer 2 attack: "The trained-system result only shows distribution shift (model
trained at T=10, lam=0.85, tested at lam>>1.0). It does not show whether the regime
is intrinsic to learned dynamics."

This experiment directly answers that.

Design
------
Three training conditions, each evaluated at T_test=40, lam swept:

  Condition A (SHORT, reference):  T_train=10, lam_train=0.85  (= original e4)
  Condition B1 (AT-OP):            T_train=40, lam_train=1.20
  Condition B2 (AT-OP harder):     T_train=40, lam_train=1.30

For B1 and B2 the model is trained at the unstable operating point itself.
If the regime disappears in B1/B2 near the training lambda -> training suppresses it.
If the regime persists -> it is intrinsic to the dynamics, not an artefact of shift.
Either outcome is scientifically informative and publishable.

Architecture
------------
Identical to e4:
    h_{t+1} = LeakyReLU(lambda * W @ h_t + U @ x + b)
    logits   = V @ h_T + c

Output figure: 3-panel per condition (accuracy vs lambda, per-timestep at collapse,
GRACE advantage). Final summary figure shows all conditions on one axis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import RidgeClassifierCV
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "e6_trained_at_op"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).parent.parent.parent / "data"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# Training conditions
# ─────────────────────────────────────────────────────────────────────────────

CONDITIONS = [
    dict(
        label="Short-horizon shift (reference)",
        name="cond_A",
        T_train=10,
        lambda_train=0.85,
        epochs=40,
    ),
    dict(
        label="Trained at operating point lam=1.20",
        name="cond_B1",
        T_train=40,
        lambda_train=1.20,
        epochs=60,   # more epochs: harder to train in unstable regime
    ),
    dict(
        label="Trained at operating point lam=1.30",
        name="cond_B2",
        T_train=40,
        lambda_train=1.30,
        epochs=60,
    ),
]

BASE_CFG = dict(
    hidden_dim=256,
    input_dim=784,
    n_classes=10,
    leaky_slope=0.2,
    # Eval
    T_test=40,
    early_K=8,
    lambda_test_vals=[0.7, 0.85, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0],
    # Training
    lr=1e-3,
    batch_size=256,
    n_train=10000,
    n_test=2000,
    seeds=[0, 1, 2, 3, 4],
    # Readout hyperparams
    grace_tau=1.2,
    grace_alpha=10.0,
    stop_tau=1.5,
    state_clip=100.0,
)


# ─────────────────────────────────────────────────────────────────────────────
# Model (identical to e4)
# ─────────────────────────────────────────────────────────────────────────────

class TrainedLeakyReLURNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes=10,
                 lam_init=0.85, leaky_slope=0.2, state_clip=100.0):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.lam         = lam_init
        self.leaky_slope = leaky_slope
        self.state_clip  = state_clip

        W = torch.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        eigs = torch.linalg.eigvals(W)
        sr   = torch.max(torch.abs(eigs)).item()
        W    = W / max(sr, 1e-8)
        self.W = nn.Parameter(W)
        self.U = nn.Parameter(torch.randn(hidden_dim, input_dim) / np.sqrt(input_dim))
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        self.V = nn.Parameter(torch.randn(n_classes, hidden_dim) * 0.01)
        self.c = nn.Parameter(torch.zeros(n_classes))

    def _act(self, x):
        return F.leaky_relu(x, negative_slope=self.leaky_slope)

    def rollout(self, x, T, lam=None, return_trajectory=False):
        if lam is None:
            lam = self.lam
        batch = x.shape[0]
        Ux    = F.linear(x, self.U)
        h     = torch.zeros(batch, self.hidden_dim, device=x.device)
        trajectory = [] if return_trajectory else None
        for _ in range(T):
            h = self._act(lam * F.linear(h, self.W) + Ux + self.b)
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
# GRACE and adaptive stopping
# ─────────────────────────────────────────────────────────────────────────────

def grace_readout(trajectory, tau=1.2, alpha=10.0):
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
    batch   = trajectory[0].shape[0]
    device  = trajectory[0].device
    norm_0  = trajectory[0].norm(dim=-1).clamp(min=1e-8)
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

def train_model(model, X_tr, y_tr, cfg, cond, verbose=True):
    """
    Train with the condition-specific T_train and lambda_train.
    For at-operating-point conditions, gradient clipping is tighter to
    prevent exploding training gradients.
    """
    model.train()
    clip_norm = 0.5 if cond['lambda_train'] >= 1.2 else 1.0
    opt   = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cond['epochs'])
    X_tr  = X_tr.to(DEVICE)
    y_tr  = y_tr.to(DEVICE)

    best_acc = 0.0
    for epoch in range(cond['epochs']):
        perm = torch.randperm(len(X_tr), device=DEVICE)
        total_loss = 0.0
        n_correct  = 0
        for i in range(0, len(X_tr), cfg['batch_size']):
            idx = perm[i:i + cfg['batch_size']]
            xb, yb = X_tr[idx], y_tr[idx]
            opt.zero_grad()
            logits = model(xb, T=cond['T_train'], lam=cond['lambda_train'])
            loss   = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            opt.step()
            total_loss += loss.item() * len(xb)
            n_correct  += (logits.argmax(1) == yb).sum().item()
        sched.step()
        acc = n_correct / len(X_tr)
        best_acc = max(best_acc, acc)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{cond['epochs']}  "
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

        all_traj   = [[] for _ in range(cfg['T_test'])]
        norm_trajs = []

        for i in range(0, len(X_te), cfg['batch_size']):
            xb = X_te[i:i + cfg['batch_size']]
            _, traj = model.rollout(xb, cfg['T_test'],
                                    lam=lam, return_trajectory=True)
            for t, h_t in enumerate(traj):
                all_traj[t].append(h_t.detach())
            norm_trajs.append([h_t.norm(dim=-1).mean().item() for h_t in traj])

        trajectory     = [torch.cat(all_traj[t], dim=0) for t in range(cfg['T_test'])]
        mean_norm_traj = np.mean(norm_trajs, axis=0).tolist()

        feat_final = trajectory[-1].cpu().numpy()
        feat_early = np.stack([trajectory[t].cpu().numpy()
                               for t in range(cfg['early_K'])]).mean(axis=0)
        feat_grace = grace_readout(trajectory, cfg['grace_tau'], cfg['grace_alpha'])
        feat_stop  = adaptive_stop(trajectory, cfg['stop_tau'])
        per_t      = [probe(trajectory[t].cpu().numpy()) for t in range(cfg['T_test'])]

        acc_final    = probe(feat_final)
        acc_early    = probe(feat_early)
        acc_grace    = probe(feat_grace)
        acc_adaptive = probe(feat_stop)

        results[lam] = {
            'final':     float(acc_final),
            'early':     float(acc_early),
            'grace':     float(acc_grace),
            'adaptive':  float(acc_adaptive),
            'per_t':     per_t,
            'norm_traj': mean_norm_traj,
        }
        best_non_final = max(acc_early, acc_grace, acc_adaptive)
        gap = best_non_final - acc_final
        flag = f'  <-- gap={gap:+.3f}' if gap > 0.01 else ''
        print(f"final={acc_final:.3f}  early={acc_early:.3f}  "
              f"grace={acc_grace:.3f}  adap={acc_adaptive:.3f}"
              f"  norm_T={mean_norm_traj[-1]:.1f}{flag}")
    return results


def aggregate(all_results, cfg):
    metrics = ['final', 'early', 'grace', 'adaptive']
    return {
        lam: {
            m: {
                'mean': float(np.mean([all_results[s][lam][m] for s in cfg['seeds']])),
                'std':  float(np.std( [all_results[s][lam][m] for s in cfg['seeds']])),
            }
            for m in metrics
        }
        for lam in cfg['lambda_test_vals']
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    'final':    '#e74c3c',
    'early':    '#f39c12',
    'adaptive': '#3498db',
    'grace':    '#27ae60',
}
LABELS = {
    'final':    'Final state',
    'early':    'Early window',
    'adaptive': 'Adaptive stopping',
    'grace':    'GRACE',
}


def plot_condition(ax, summary, cfg, cond, title):
    """Single accuracy-vs-lambda panel."""
    lams    = cfg['lambda_test_vals']
    metrics = ['final', 'early', 'adaptive', 'grace']
    for m in metrics:
        mn = [summary[l][m]['mean'] for l in lams]
        sd = [summary[l][m]['std']  for l in lams]
        ax.plot(lams, mn, 'o-', color=COLORS[m], label=LABELS[m], lw=2, ms=5)
        ax.fill_between(lams,
                        [a - b for a, b in zip(mn, sd)],
                        [a + b for a, b in zip(mn, sd)],
                        alpha=0.12, color=COLORS[m])
    ax.axvline(1.0, color='gray', ls='--', alpha=0.5, lw=1)
    ax.axvline(cond['lambda_train'], color='black', ls=':', lw=1.5,
               label=f"Train $\\lambda$={cond['lambda_train']}")
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Test-time gain $\\lambda$', fontsize=9)
    ax.set_ylabel('Linear probe accuracy', fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(alpha=0.25)


def make_summary_figure(all_condition_summaries, cfg, save_dir):
    """
    Main output: 3-panel figure, one panel per condition.
    Top row: accuracy vs lambda for each condition.
    Bottom row: GRACE-advantage bar chart for each condition.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Experiment B: Does Training at the Operating Point Suppress the Pre-Collapse Regime?",
        fontsize=13, fontweight='bold', y=1.01
    )

    lams = cfg['lambda_test_vals']
    for col, (cond, summary) in enumerate(zip(CONDITIONS, all_condition_summaries)):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        # Top: accuracy sweep
        short_label = cond['label']
        plot_condition(ax_top, summary, cfg, cond,
                       title=f"{chr(65+col)}. {short_label}")

        # Bottom: advantage of best non-final over final
        gains_early  = [summary[l]['early']['mean']    - summary[l]['final']['mean'] for l in lams]
        gains_grace  = [summary[l]['grace']['mean']    - summary[l]['final']['mean'] for l in lams]
        gains_adap   = [summary[l]['adaptive']['mean'] - summary[l]['final']['mean'] for l in lams]

        x   = np.arange(len(lams))
        w   = 0.25
        ax_bot.bar(x - w,   gains_early, w, color=COLORS['early'],    label='Early window',     alpha=0.85)
        ax_bot.bar(x,       gains_adap,  w, color=COLORS['adaptive'], label='Adaptive stopping', alpha=0.85)
        ax_bot.bar(x + w,   gains_grace, w, color=COLORS['grace'],    label='GRACE',             alpha=0.85)
        ax_bot.axhline(0, color='black', lw=0.8, ls='-')
        ax_bot.axvline(lams.index(cond['lambda_train']) - 0.5 if cond['lambda_train'] in lams
                       else -1, color='black', ls=':', lw=1.2)
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels([f'{l:.2f}' for l in lams], rotation=45, fontsize=7)
        ax_bot.set_xlabel('Test-time gain $\\lambda$', fontsize=9)
        ax_bot.set_ylabel('Accuracy gain over final-state', fontsize=9)
        ax_bot.set_title(f"Non-final advantage (condition {chr(65+col)})", fontsize=9)
        ax_bot.legend(fontsize=7, loc='upper left')
        ax_bot.grid(alpha=0.25, axis='y')
        lim = max(0.25, max(abs(g) for g in gains_early + gains_grace + gains_adap) + 0.02)
        ax_bot.set_ylim(-lim * 0.5, lim)

    plt.tight_layout()
    out = save_dir / 'e6_trained_at_op_summary.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Summary figure saved: {out}")
    plt.close()


def make_overlay_figure(all_condition_summaries, cfg, save_dir):
    """
    Overlay: final-state accuracy across all three conditions on one axis,
    to show whether training at the operating point 'fixes' the endpoint.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Endpoint (Final-State) vs. Early-Window Accuracy: Effect of Training Condition",
        fontsize=12, fontweight='bold'
    )

    lams          = cfg['lambda_test_vals']
    cond_colors   = ['#e74c3c', '#3498db', '#8e44ad']
    cond_styles   = ['-', '--', ':']

    for i, (cond, summary) in enumerate(zip(CONDITIONS, all_condition_summaries)):
        label = f"{chr(65+i)}: {cond['name']} (train $\\lambda$={cond['lambda_train']}, T={cond['T_train']})"
        mn_f  = [summary[l]['final']['mean'] for l in lams]
        sd_f  = [summary[l]['final']['std']  for l in lams]
        mn_e  = [summary[l]['early']['mean'] for l in lams]
        sd_e  = [summary[l]['early']['std']  for l in lams]

        ax1.plot(lams, mn_f, ls=cond_styles[i], color=cond_colors[i],
                 lw=2.2, marker='o', ms=5, label=label)
        ax1.fill_between(lams,
                         [a - b for a, b in zip(mn_f, sd_f)],
                         [a + b for a, b in zip(mn_f, sd_f)],
                         alpha=0.10, color=cond_colors[i])

        ax2.plot(lams, mn_e, ls=cond_styles[i], color=cond_colors[i],
                 lw=2.2, marker='s', ms=5, label=label)
        ax2.fill_between(lams,
                         [a - b for a, b in zip(mn_e, sd_e)],
                         [a + b for a, b in zip(mn_e, sd_e)],
                         alpha=0.10, color=cond_colors[i])

    for ax in (ax1, ax2):
        ax.axvline(1.0, color='gray', ls='--', alpha=0.5, lw=1, label='$\\lambda=1$ boundary')
        ax.set_xlabel('Test-time gain $\\lambda$', fontsize=10)
        ax.set_ylabel('Linear probe accuracy', fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

    ax1.set_title('Final-state readout accuracy', fontweight='bold')
    ax2.set_title('Early-window readout accuracy', fontweight='bold')

    plt.tight_layout()
    out = save_dir / 'e6_trained_at_op_overlay.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Overlay figure saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_condition(cond, cfg):
    """Run one training condition across all seeds."""
    print(f"\n{'='*70}")
    print(f"CONDITION: {cond['label']}")
    print(f"  T_train={cond['T_train']}  lambda_train={cond['lambda_train']}  "
          f"epochs={cond['epochs']}")
    print(f"{'='*70}")

    all_results = {}

    for seed in cfg['seeds']:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        X_tr, y_tr, X_te, y_te = load_mnist(cfg['n_train'], cfg['n_test'], seed=seed)

        model = TrainedLeakyReLURNN(
            input_dim=cfg['input_dim'],
            hidden_dim=cfg['hidden_dim'],
            n_classes=cfg['n_classes'],
            lam_init=cond['lambda_train'],
            leaky_slope=cfg['leaky_slope'],
            state_clip=cfg['state_clip'],
        ).to(DEVICE)

        print(f"Training (T={cond['T_train']}, lam={cond['lambda_train']})...")
        train_model(model, X_tr, y_tr, cfg, cond, verbose=True)

        print(f"\nEvaluating (T_test={cfg['T_test']}) ...")
        all_results[seed] = evaluate_regime_sweep(model, X_te, y_te, cfg)

    summary = aggregate(all_results, cfg)

    # Print summary
    print(f"\n\n{'='*75}")
    print(f"SUMMARY — {cond['label']}")
    print(f"{'lam':>6}  {'final':>14}  {'early':>14}  {'grace':>14}  {'adaptive':>14}  {'best_gap':>10}")
    print("-" * 75)
    for lam in cfg['lambda_test_vals']:
        s   = summary[lam]
        gap = max(s['early']['mean'], s['grace']['mean'], s['adaptive']['mean']) - s['final']['mean']
        flag = ' <<' if gap > 0.02 else ''
        print(f"{lam:>6.2f}  "
              f"{s['final']['mean']:.3f}+-{s['final']['std']:.3f}  "
              f"{s['early']['mean']:.3f}+-{s['early']['std']:.3f}  "
              f"{s['grace']['mean']:.3f}+-{s['grace']['std']:.3f}  "
              f"{s['adaptive']['mean']:.3f}+-{s['adaptive']['std']:.3f}  "
              f"{gap:+.4f}{flag}")

    return all_results, summary


def main():
    all_condition_results   = []
    all_condition_summaries = []

    for cond in CONDITIONS:
        results, summary = run_condition(cond, BASE_CFG)
        all_condition_results.append(results)
        all_condition_summaries.append(summary)

        # Save per-condition JSON
        out = RESULTS_DIR / f"{cond['name']}.json"
        with open(out, 'w') as f:
            json.dump({
                'condition': cond,
                'config':    BASE_CFG,
                'per_seed':  {str(k): v for k, v in results.items()},
                'summary':   {str(k): v for k, v in summary.items()},
            }, f, indent=2)
        print(f"Saved: {out}")

    # Combined JSON
    combined = {
        'conditions': CONDITIONS,
        'config':     BASE_CFG,
        'summaries':  [
            {str(k): v for k, v in s.items()}
            for s in all_condition_summaries
        ],
    }
    with open(RESULTS_DIR / 'e6_combined.json', 'w') as f:
        json.dump(combined, f, indent=2)

    # Figures
    print("\nGenerating figures...")
    make_summary_figure(all_condition_summaries, BASE_CFG, RESULTS_DIR)
    make_overlay_figure(all_condition_summaries, BASE_CFG, RESULTS_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
