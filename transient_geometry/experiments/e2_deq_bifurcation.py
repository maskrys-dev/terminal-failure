"""
E2: DEQ-Style Near-Bifurcation — Pre-Collapse in Fixed-Point Learners
======================================================================
Train a small deep equilibrium model (DEQ) on MNIST.
Then sweep the recurrent gain lambda at test time, approaching and
passing the fixed-point instability boundary.

DEQ setup:
    z* = tanh(lambda * W z* + U x + b)       [implicit layer]
    logits = V z* + c

Solver: simple Picard / Anderson iteration for K steps.

At test time, we evaluate:
    - Final solver iterate z_K (standard DEQ readout)
    - Intermediate iterates z_k, k=1..K (grace period over solver iterates)
    - GRACE weighting over solver iterates
    - Adaptive stopping over solver iterates

Key thesis: in DEQ models, which are architecturally committed to the
endpoint assumption, the pre-collapse regime appears in the solver iteration
itself near bifurcation — intermediate solver iterates retain more
class information than the "converged" final iterate.
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

RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "e2_deq_bifurcation"
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
    K_train=10,              # few training iterations — widens training/test solver gap
    K_test=50,               # many test iterations to observe divergence
    early_K=5,
    lambda_train=0.80,       # firmly contractive during training
    lambda_test_vals=[0.80, 0.90, 1.00, 1.05, 1.10, 1.20, 1.30, 1.40],
    lr=1e-3,
    epochs=25,
    batch_size=256,
    n_train=5000,
    n_test=2000,
    seeds=[0, 1, 2, 3, 4],
    grace_tau=1.15,          # tighter tau for faster-diverging systems
    grace_alpha=8.0,
    stop_tau=1.4,
)


# ─────────────────────────────────────────────────────────────────────────────
# DEQ Model
# ─────────────────────────────────────────────────────────────────────────────

class SmallDEQ(nn.Module):
    """
    Small DEQ classifier.

    Implicit layer: z* = tanh(lam * W z* + U x + b)
    Solved by Picard iteration: z_{k+1} = tanh(lam * W z_k + U x + b)
    Classifier: logits = V z* + c

    At test time, lam is varied to sweep the regime. At lam >= ~1/spectral_radius(W),
    the Picard iteration may fail to converge — the per-iterate readout then
    shows the pre-collapse grace period over solver iterations.
    """

    def __init__(self, input_dim, hidden_dim, n_classes=10, lam=0.85):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lam = lam

        # Fixed-point layer weights (train-normalised to sr ≈ lam for stability)
        W_init = torch.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        eigs = torch.linalg.eigvals(W_init)
        sr = torch.max(torch.abs(eigs)).item()
        # Scale so effective spectral radius = lam * sr_W = lam
        W_init = W_init / max(sr, 1e-8)
        self.W = nn.Parameter(W_init)   # unbounded — gain controlled by lam

        self.U = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(hidden_dim))

        self.V = nn.Parameter(torch.randn(n_classes, hidden_dim) * 0.01)
        self.c = nn.Parameter(torch.zeros(n_classes))

    def picard(self, x, K, lam=None, return_iterates=False):
        """
        Run K steps of Picard iteration.

        z_{k+1} = tanh(lam * W z_k + U x + b)

        Parameters
        ----------
        x : Tensor [batch, input_dim]
        K : int
        lam : float — gain at evaluation time
        return_iterates : bool

        Returns
        -------
        z_K : Tensor [batch, hidden_dim]
        iterates (optional): list of Tensor [batch, hidden_dim]
        """
        if lam is None:
            lam = self.lam

        batch = x.shape[0]
        z = torch.zeros(batch, self.hidden_dim, device=x.device)
        Ux = F.linear(x, self.U)

        iterates = [] if return_iterates else None

        for _ in range(K):
            z = torch.tanh(lam * F.linear(z, self.W) + Ux + self.b)
            if return_iterates:
                iterates.append(z)

        if return_iterates:
            return z, iterates
        return z

    def forward(self, x, K=None, lam=None):
        if K is None:
            K = 20
        z_star = self.picard(x, K, lam=lam)
        return F.linear(z_star, self.V, self.c)


# ─────────────────────────────────────────────────────────────────────────────
# GRACE and adaptive stopping (real-valued, identical to E1)
# ─────────────────────────────────────────────────────────────────────────────

def grace_readout(iterates, tau=1.2, alpha=10.0):
    """GRACE over solver iterates."""
    norm_k0 = iterates[0].norm(dim=-1, keepdim=True).clamp(min=1e-8)
    weighted_sum = torch.zeros_like(iterates[0])
    weight_sum   = torch.zeros(iterates[0].shape[0], 1, device=iterates[0].device)
    for z_k in iterates:
        g_k = z_k.norm(dim=-1, keepdim=True) / norm_k0
        w_k = torch.exp(-alpha * torch.clamp(g_k - tau, min=0.0))
        weighted_sum += w_k * z_k
        weight_sum   += w_k
    return (weighted_sum / weight_sum.clamp(min=1e-8)).cpu().numpy()


def adaptive_stop_readout(iterates, tau_stop=1.5):
    """Read out the iterate at the first solver step where norm growth > tau_stop."""
    batch = iterates[0].shape[0]
    device = iterates[0].device
    norm_k0 = iterates[0].norm(dim=-1).clamp(min=1e-8)
    result  = iterates[-1].clone()
    stopped = torch.zeros(batch, dtype=torch.bool, device=device)
    for z_k in iterates:
        growth = z_k.norm(dim=-1) / norm_k0
        just_stopped = (~stopped) & (growth > tau_stop)
        if just_stopped.any():
            result[just_stopped] = z_k[just_stopped]
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
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'])

    X_tr = X_tr.to(DEVICE)
    y_tr = y_tr.to(DEVICE)

    for epoch in range(cfg['epochs']):
        perm = torch.randperm(len(X_tr), device=DEVICE)
        epoch_loss = 0.0
        n_correct = 0

        for i in range(0, len(X_tr), cfg['batch_size']):
            idx = perm[i:i + cfg['batch_size']]
            xb, yb = X_tr[idx], y_tr[idx]

            opt.zero_grad()
            logits = model(xb, K=cfg['K_train'], lam=cfg['lambda_train'])
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item() * len(xb)
            n_correct += (logits.argmax(1) == yb).sum().item()

        sched.step()
        if verbose and (epoch + 1) % 5 == 0:
            acc = n_correct / len(X_tr)
            print(f"  Epoch {epoch+1:3d}/{cfg['epochs']}  "
                  f"loss={epoch_loss/len(X_tr):.4f}  acc={acc:.3f}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_regime_sweep(model, X_te, y_te, cfg):
    model.eval()
    X_te = X_te.to(DEVICE)
    y_te_np = y_te.cpu().numpy()

    results = {}

    for lam in cfg['lambda_test_vals']:
        print(f"  lambda={lam:.2f} ...", end=" ", flush=True)

        all_iters = [[] for _ in range(cfg['K_test'])]
        final_logits_all = []
        convergence = []

        for i in range(0, len(X_te), cfg['batch_size']):
            xb = X_te[i:i + cfg['batch_size']]
            z_K, iters = model.picard(xb, cfg['K_test'], lam=lam,
                                      return_iterates=True)
            # Check convergence: delta between last two iterates
            delta = (iters[-1] - iters[-2]).norm(dim=-1).mean().item()
            convergence.append(delta)

            logits = F.linear(z_K, model.V, model.c)
            final_logits_all.append(logits.cpu())
            for k, z_k in enumerate(iters):
                all_iters[k].append(z_k.detach())

        # Consolidate
        iterates = [torch.cat(states, dim=0) for states in all_iters]
        final_logits = torch.cat(final_logits_all, dim=0)
        mean_convergence_delta = float(np.mean(convergence))

        # Readout features
        feat_final = iterates[-1].cpu().numpy()
        feat_early = np.stack([iterates[k].cpu().numpy()
                               for k in range(cfg['early_K'])]).mean(axis=0)
        feat_grace = grace_readout(iterates, cfg['grace_tau'], cfg['grace_alpha'])
        feat_stop  = adaptive_stop_readout(iterates, cfg['stop_tau'])

        # Probe (half-split within test set)
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

        # DEQ head accuracy (trained endpoint readout)
        acc_deq_head = (final_logits.argmax(1).numpy() == y_te_np).mean()

        # Per-iterate probe accuracy
        per_k_acc = []
        for k in range(cfg['K_test']):
            feat_k = iterates[k].cpu().numpy()
            per_k_acc.append(probe_acc(feat_k))

        results[lam] = {
            'final_probe':      float(acc_final),
            'final_deq_head':   float(acc_deq_head),
            'early':            float(acc_early),
            'grace':            float(acc_grace),
            'adaptive':         float(acc_stop),
            'per_iterate':      per_k_acc,
            'convergence_delta': mean_convergence_delta,
        }
        print(f"deq_head={acc_deq_head:.3f}  final_probe={acc_final:.3f}  "
              f"grace={acc_grace:.3f}  conv_delta={mean_convergence_delta:.4f}")

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

    model = SmallDEQ(
        input_dim=cfg['input_dim'],
        hidden_dim=cfg['hidden_dim'],
        lam=cfg['lambda_train'],
    ).to(DEVICE)

    print(f"Training DEQ with lambda_train={cfg['lambda_train']}...")
    model = train_model(model, X_tr, y_tr, cfg)

    print("\nEvaluating lambda sweep...")
    return evaluate_regime_sweep(model, X_te, y_te, cfg)


def main():
    all_results = {}
    for seed in CFG['seeds']:
        all_results[seed] = run_seed(seed, CFG)

    # Aggregate
    print("\n\n" + "="*60)
    print("AGGREGATE RESULTS (mean ± std)")
    print("="*60)

    metrics = ['final_probe', 'early', 'grace', 'adaptive']
    summary = {}

    for lam in CFG['lambda_test_vals']:
        accs = {m: [all_results[s][lam][m] for s in CFG['seeds']] for m in metrics}
        conv = [all_results[s][lam]['convergence_delta'] for s in CFG['seeds']]
        summary[lam] = {m: {'mean': np.mean(v), 'std': np.std(v)} for m, v in accs.items()}
        summary[lam]['conv_delta'] = float(np.mean(conv))

        print(f"\nlambda={lam:.2f}  (conv_delta={summary[lam]['conv_delta']:.4f}):")
        for m in metrics:
            mn, sd = summary[lam][m]['mean'], summary[lam][m]['std']
            print(f"  {m:15s}: {mn:.3f} ± {sd:.3f}")

    # Save
    save_data = {
        'config': CFG,
        'per_seed': {str(k): v for k, v in all_results.items()},
        'summary': {str(k): v for k, v in summary.items()},
    }
    out_path = RESULTS_DIR / "e2_deq_bifurcation.json"
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        fig.suptitle("DEQ-Style Model — Pre-Collapse Regime Near Fixed-Point Instability",
                     fontsize=13, fontweight='bold')

        colors = {'final_probe': '#e74c3c', 'early': '#f39c12',
                  'adaptive': '#3498db', 'grace': '#2ecc71'}
        labels = {'final_probe': 'Final iterate z_K',
                  'early': f'Early window (k=1..{CFG["early_K"]})',
                  'adaptive': 'Adaptive stopping',
                  'grace': 'GRACE'}

        lams = CFG['lambda_test_vals']

        # Panel 1: accuracy sweep
        ax1 = fig.add_subplot(gs[0, 0])
        for m in metrics:
            means = [summary[lam][m]['mean'] for lam in lams]
            stds  = [summary[lam][m]['std']  for lam in lams]
            ax1.plot(lams, means, 'o-', color=colors[m], label=labels[m], linewidth=2)
            ax1.fill_between(lams,
                             [mn - sd for mn, sd in zip(means, stds)],
                             [mn + sd for mn, sd in zip(means, stds)],
                             alpha=0.15, color=colors[m])
        ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel("Test-time gain λ")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Accuracy vs. lambda (DEQ solver iterates)")
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

        # Panel 2: convergence delta
        ax2 = fig.add_subplot(gs[0, 1])
        conv_vals = [summary[lam]['conv_delta'] for lam in lams]
        ax2.semilogy(lams, conv_vals, 'ko-', linewidth=2)
        ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel("Test-time gain λ")
        ax2.set_ylabel("‖z_K − z_{K−1}‖ (log scale)")
        ax2.set_title("Solver convergence failure near bifurcation")
        ax2.grid(alpha=0.3)

        # Panel 3: per-iterate accuracy at near-bifurcation lambda
        ax3 = fig.add_subplot(gs[1, :])
        near_bif = max([l for l in lams if l <= 1.15])
        for s in CFG['seeds']:
            per_k = all_results[s][near_bif]['per_iterate']
            ax3.plot(range(1, len(per_k)+1), per_k, alpha=0.3,
                     color='#95a5a6', linewidth=1)
        mean_per_k = np.mean(
            [[all_results[s][near_bif]['per_iterate'][k]
              for k in range(CFG['K_test'])]
             for s in CFG['seeds']], axis=0)
        ax3.plot(range(1, CFG['K_test']+1), mean_per_k, 'k-',
                 linewidth=2.5, label=f'Mean (λ={near_bif})')
        ax3.axhline(y=summary[near_bif]['grace']['mean'], color=colors['grace'],
                    linestyle='--', linewidth=1.5,
                    label=f"GRACE={summary[near_bif]['grace']['mean']:.3f}")
        ax3.axhline(y=summary[near_bif]['final_probe']['mean'],
                    color=colors['final_probe'], linestyle='--', linewidth=1.5,
                    label=f"Final iterate={summary[near_bif]['final_probe']['mean']:.3f}")
        ax3.set_xlabel("Solver iteration k")
        ax3.set_ylabel("Per-iterate probe accuracy")
        ax3.set_title(f"Pre-collapse grace period over DEQ solver iterations (λ={near_bif})")
        ax3.legend(fontsize=10)
        ax3.grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "e2_deq_bifurcation.png", dpi=150, bbox_inches='tight')
        print("Figure saved.")
        plt.close()

    except Exception as e:
        print(f"Plot failed: {e}")

    return summary


if __name__ == "__main__":
    main()
