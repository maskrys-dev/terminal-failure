"""
E5: DEQ with LeakyReLU — Pre-Collapse in Fixed-Point Solver Iterations
========================================================================
Train a DEQ-style classifier where the implicit layer uses LeakyReLU.
Then sweep the solver gain at test time, approaching and passing
the fixed-point instability boundary.

DEQ fixed-point equation:
    z* = LeakyReLU(lambda * W z* + U x + b)
Solved by Picard iteration:
    z_{k+1} = LeakyReLU(lambda * W z_k + U x + b)
Classifier:
    logits = V z* + c

Key design insight from E4:
    LeakyReLU (negative slope 0.2) permits unbounded state growth —
    unlike tanh which self-stabilises and washes out the phenomenon.

Training:
    K_train = 10 (few solver steps — model commits to early iterations)
    lambda_train = 0.80 (firmly contractive)

Test:
    K_test = 50 (many iterations to see divergence)
    lambda_test up to 1.40

Key thesis:
    In a DEQ model — architecturally committed to the endpoint assumption —
    the pre-collapse grace period appears in the SOLVER ITERATION itself
    near instability. Intermediate iterates z_k retain more class information
    than the "converged" final iterate z_K.

This directly challenges the endpoint assumption in a model class that
relies on it by design.
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
RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "e5_deq_leakyrelu"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).parent.parent.parent / "data"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CFG = dict(
    hidden_dim=256,
    input_dim=784,
    n_classes=10,
    leaky_slope=0.05,        # small slope: closer to ReLU, more contractive
    # Training at moderate K: model sees near-convergence during training
    K_train=30,
    lambda_train=0.75,       # strongly contractive at training time
    # Test: same K, sweep lambda
    # The key: as lambda rises, the Picard map becomes less contractive
    # and the per-iterate accuracy shows a grace period before divergence
    K_test=30,
    early_K=5,
    lambda_test_vals=[0.65, 0.75, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20],
    lr=1e-3,
    epochs=50,
    batch_size=256,
    n_train=10000,
    n_test=2000,
    seeds=[0, 1, 2, 3, 4],
    grace_tau=1.2,
    grace_alpha=10.0,
    stop_tau=1.5,
    state_clip=20.0,         # tight clip: forces solver to stay near fixed point
)

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class LeakyReLUDEQ(nn.Module):
    """
    DEQ-style classifier with LeakyReLU implicit layer.

    Fixed-point equation:
        z* = LeakyReLU(lam * W z* + U x + b)

    Solved by K steps of Picard iteration starting from z_0 = 0.
    Classifier reads out the final iterate z_K.

    At test time lam is swept upward, eventually crossing the instability
    boundary where the Picard map is no longer contractive. The per-iterate
    probe accuracy then reveals the grace period in solver iteration space.
    """

    def __init__(self, input_dim, hidden_dim, n_classes=10,
                 lam_init=0.80, leaky_slope=0.2, state_clip=200.0):
        super().__init__()
        self.lam         = lam_init
        self.leaky_slope = leaky_slope
        self.state_clip  = state_clip
        self.hidden_dim  = hidden_dim

        # W: spectral radius normalised to 1.0 at init
        # lam * W has spectral radius lam_init — firmly contractive
        W = torch.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        eigs = torch.linalg.eigvals(W)
        sr   = torch.max(torch.abs(eigs)).item()
        W    = W / max(sr, 1e-8)
        self.W = nn.Parameter(W)

        self.U = nn.Parameter(
            torch.randn(hidden_dim, input_dim) / np.sqrt(input_dim)
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        self.V = nn.Parameter(torch.randn(n_classes, hidden_dim) * 0.01)
        self.c = nn.Parameter(torch.zeros(n_classes))

    def _act(self, x):
        return F.leaky_relu(x, negative_slope=self.leaky_slope)

    def picard(self, x, K, lam=None, return_iterates=False):
        """
        Run K Picard iterations.

        z_{k+1} = LeakyReLU(lam * W z_k + Ux + b)

        Returns
        -------
        z_K : Tensor [batch, hidden_dim]
        iterates (optional): list of Tensor [batch, hidden_dim], length K
        """
        if lam is None:
            lam = self.lam

        batch = x.shape[0]
        Ux    = F.linear(x, self.U)            # constant input drive
        z     = torch.zeros(batch, self.hidden_dim, device=x.device)
        iterates = [] if return_iterates else None

        for _ in range(K):
            z = self._act(lam * F.linear(z, self.W) + Ux + self.b)
            z = torch.clamp(z, -self.state_clip, self.state_clip)
            if return_iterates:
                iterates.append(z)

        if return_iterates:
            return z, iterates
        return z

    def forward(self, x, K=None, lam=None):
        if K is None:
            K = 10
        z_K = self.picard(x, K, lam=lam)
        return F.linear(z_K, self.V, self.c)


# ─────────────────────────────────────────────────────────────────────────────
# GRACE and adaptive stopping over solver iterates
# ─────────────────────────────────────────────────────────────────────────────

def grace_readout(iterates, tau=1.2, alpha=10.0):
    norm_0  = iterates[0].norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w_feat  = torch.zeros_like(iterates[0])
    w_total = torch.zeros(iterates[0].shape[0], 1, device=iterates[0].device)
    for z_k in iterates:
        g_k  = z_k.norm(dim=-1, keepdim=True) / norm_0
        w_k  = torch.exp(-alpha * torch.clamp(g_k - tau, min=0.0))
        w_feat  += w_k * z_k
        w_total += w_k
    return (w_feat / w_total.clamp(min=1e-8)).cpu().numpy()


def adaptive_stop(iterates, tau_stop=1.5):
    batch   = iterates[0].shape[0]
    device  = iterates[0].device
    norm_0  = iterates[0].norm(dim=-1).clamp(min=1e-8)
    result  = iterates[-1].clone()
    stopped = torch.zeros(batch, dtype=torch.bool, device=device)
    for z_k in iterates:
        growth = z_k.norm(dim=-1) / norm_0
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

def load_mnist(n_train, n_test, seed=0):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda img: img.view(-1)),
    ])
    tr  = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tf)
    te  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    rng = torch.Generator().manual_seed(seed)
    X_tr = torch.stack([tr[i][0] for i in torch.randperm(len(tr), generator=rng)[:n_train]])
    y_tr = torch.tensor([tr[i][1] for i in range(n_train)])
    X_te = torch.stack([te[i][0] for i in torch.randperm(len(te), generator=rng)[:n_test]])
    y_te = torch.tensor([te[i][1] for i in range(n_test)])
    return X_tr, y_tr, X_te, y_te


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, X_tr, y_tr, cfg):
    model.train()
    opt   = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'])
    X_tr  = X_tr.to(DEVICE)
    y_tr  = y_tr.to(DEVICE)

    for epoch in range(cfg['epochs']):
        perm = torch.randperm(len(X_tr), device=DEVICE)
        total_loss = 0.0;  n_correct = 0
        for i in range(0, len(X_tr), cfg['batch_size']):
            idx = perm[i:i + cfg['batch_size']]
            xb, yb = X_tr[idx], y_tr[idx]
            opt.zero_grad()
            logits = model(xb, K=cfg['K_train'], lam=cfg['lambda_train'])
            loss   = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(xb)
            n_correct  += (logits.argmax(1) == yb).sum().item()
        sched.step()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{cfg['epochs']}  "
                  f"loss={total_loss/len(X_tr):.4f}  acc={n_correct/len(X_tr):.3f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_sweep(model, X_te, y_te, cfg):
    model.eval()
    X_te    = X_te.to(DEVICE)
    y_te_np = y_te.cpu().numpy()

    results = {}
    for lam in cfg['lambda_test_vals']:
        print(f"  lam={lam:.2f} ...", end=" ", flush=True)

        all_iters  = [[] for _ in range(cfg['K_test'])]
        conv_deltas = []
        norm_trajs  = []

        for i in range(0, len(X_te), cfg['batch_size']):
            xb = X_te[i:i + cfg['batch_size']]
            z_K, iters = model.picard(xb, cfg['K_test'],
                                      lam=lam, return_iterates=True)
            for k, z_k in enumerate(iters):
                all_iters[k].append(z_k.detach())
            conv_deltas.append((iters[-1] - iters[-2]).norm(dim=-1).mean().item())
            norm_trajs.append([z_k.norm(dim=-1).mean().item() for z_k in iters])

        iterates       = [torch.cat(all_iters[k], dim=0) for k in range(cfg['K_test'])]
        mean_norm_traj = np.mean(norm_trajs, axis=0).tolist()
        conv_delta     = float(np.mean(conv_deltas))

        # Evaluate accuracy of the TRAINED HEAD applied to each iterate
        # This is the right diagnostic: how well does the trained endpoint
        # classifier work when applied to earlier/later iterates?
        def head_acc(z_feat):
            logits = F.linear(z_feat, model.V, model.c)
            return (logits.argmax(1).cpu().numpy() == y_te_np).mean()

        acc_final = head_acc(iterates[-1])
        acc_early = head_acc(torch.stack(iterates[:cfg['early_K']]).mean(dim=0))

        # GRACE and adaptive over iterates, then apply head
        norm_0   = iterates[0].norm(dim=-1, keepdim=True).clamp(min=1e-8)
        w_feat   = torch.zeros_like(iterates[0])
        w_total  = torch.zeros(iterates[0].shape[0], 1, device=iterates[0].device)
        res_stop = iterates[-1].clone()
        norm_0_s = iterates[0].norm(dim=-1).clamp(min=1e-8)
        stopped  = torch.zeros(len(y_te_np), dtype=torch.bool, device=iterates[0].device)
        for z_k in iterates:
            g_k = z_k.norm(dim=-1, keepdim=True) / norm_0
            w_k = torch.exp(-cfg['grace_alpha'] * torch.clamp(g_k - cfg['grace_tau'], min=0.0))
            w_feat  += w_k * z_k
            w_total += w_k
            # adaptive stop
            growth = z_k.norm(dim=-1) / norm_0_s
            just_stopped = (~stopped) & (growth > cfg['stop_tau'])
            if just_stopped.any():
                res_stop[just_stopped] = z_k[just_stopped]
                stopped |= just_stopped

        grace_feat = w_feat / w_total.clamp(min=1e-8)
        acc_grace    = head_acc(grace_feat)
        acc_adaptive = head_acc(res_stop)

        per_k_acc = [float(head_acc(iterates[k])) for k in range(cfg['K_test'])]

        results[lam] = {
            'final':      float(acc_final),
            'early':      float(acc_early),
            'grace':      float(acc_grace),
            'adaptive':   float(acc_adaptive),
            'per_k':      per_k_acc,
            'norm_traj':  mean_norm_traj,
            'conv_delta': conv_delta,
        }
        gain = acc_grace - acc_final
        flag = '  <-- GRACE wins' if gain > 0.005 else ''
        print(f"final={acc_final:.3f}  early={acc_early:.3f}  "
              f"grace={acc_grace:.3f}  adap={acc_adaptive:.3f}  "
              f"conv_delta={conv_delta:.4f}{flag}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(all_results, summary, cfg, save_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lams    = cfg['lambda_test_vals']
    metrics = ['final', 'early', 'grace', 'adaptive']
    colors  = {'final': '#e74c3c', 'early': '#f39c12',
               'adaptive': '#3498db', 'grace': '#2ecc71'}
    labels  = {'final':    f'Final iterate (z_K, K={cfg["K_test"]})',
               'early':    f'Early iterates (k=1..{cfg["early_K"]})',
               'adaptive': 'Adaptive stopping',
               'grace':    'GRACE'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "DEQ + LeakyReLU — Pre-Collapse Grace Period in Solver Iterations\n"
        f"Trained: K={cfg['K_train']}, lam={cfg['lambda_train']}  |  "
        f"Tested: K={cfg['K_test']}, lam up to {max(lams)}",
        fontsize=12, fontweight='bold')

    # Panel 1: accuracy vs lambda
    ax = axes[0]
    for m in metrics:
        mn = [summary[l][m]['mean'] for l in lams]
        sd = [summary[l][m]['std']  for l in lams]
        ax.plot(lams, mn, 'o-', color=colors[m], label=labels[m], lw=2, markersize=5)
        ax.fill_between(lams, [a-b for a,b in zip(mn,sd)],
                              [a+b for a,b in zip(mn,sd)], alpha=0.15, color=colors[m])
    ax.axvline(1.0, color='gray', ls='--', alpha=0.6, label='solver boundary (lam=1)')
    ax.axvline(cfg['lambda_train'], color='k', ls=':', alpha=0.5,
               label=f'lam_train={cfg["lambda_train"]}')
    ax.set_xlabel('Test-time solver gain lambda')
    ax.set_ylabel('Linear probe accuracy')
    ax.set_title('Readout accuracy vs. solver gain')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Panel 2: per-iterate accuracy at near-instability lambda
    ax2 = axes[1]
    baseline = summary[lams[0]]['final']['mean']
    tgt = next((l for l in lams if l >= 1.0 and summary[l]['final']['mean'] < baseline - 0.02),
               lams[-1])
    for s in cfg['seeds']:
        ax2.plot(range(1, cfg['K_test']+1), all_results[s][tgt]['per_k'],
                 alpha=0.2, color='#95a5a6', lw=1)
    mean_pk = np.mean([[all_results[s][tgt]['per_k'][k]
                        for k in range(cfg['K_test'])] for s in cfg['seeds']], axis=0)
    ax2.plot(range(1, cfg['K_test']+1), mean_pk, 'k-', lw=2.5,
             label=f'Mean per-iterate (lam={tgt})')
    for m in ['grace', 'final', 'adaptive', 'early']:
        ax2.axhline(summary[tgt][m]['mean'], color=colors[m], ls='--', lw=1.5,
                    label=f'{m}={summary[tgt][m]["mean"]:.3f}')
    ax2.set_xlabel('Solver iteration k')
    ax2.set_ylabel('Probe accuracy')
    ax2.set_title(f'Grace period in solver iterations (lam={tgt})')
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    # Panel 3: convergence delta vs lambda (solver diagnostics)
    ax3 = axes[2]
    conv_means = [np.mean([all_results[s][l]['conv_delta'] for s in cfg['seeds']])
                  for l in lams]
    conv_stds  = [np.std([all_results[s][l]['conv_delta'] for s in cfg['seeds']])
                  for l in lams]
    ax3.semilogy(lams, conv_means, 'ko-', lw=2, markersize=5)
    ax3.fill_between(lams,
                     [max(m-s, 1e-6) for m,s in zip(conv_means, conv_stds)],
                     [m+s for m,s in zip(conv_means, conv_stds)],
                     alpha=0.2, color='gray')
    ax3.axvline(1.0, color='gray', ls='--', alpha=0.6)
    ax3.set_xlabel('Test-time solver gain lambda')
    ax3.set_ylabel('||z_K - z_{K-1}|| (log scale)')
    ax3.set_title('Solver convergence residual vs. gain')
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_dir / 'e5_deq_leakyrelu.png', dpi=150, bbox_inches='tight')
    print("Figure saved.")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    all_results = {}
    for seed in CFG['seeds']:
        print(f"\n{'='*65}")
        print(f"E5 Seed {seed} | K_train={CFG['K_train']} lam_train={CFG['lambda_train']}"
              f" | K_test={CFG['K_test']} lam up to {max(CFG['lambda_test_vals'])}")
        print(f"{'='*65}")
        torch.manual_seed(seed);  np.random.seed(seed)

        X_tr, y_tr, X_te, y_te = load_mnist(CFG['n_train'], CFG['n_test'], seed=seed)
        model = LeakyReLUDEQ(
            input_dim=CFG['input_dim'],
            hidden_dim=CFG['hidden_dim'],
            n_classes=CFG['n_classes'],
            lam_init=CFG['lambda_train'],
            leaky_slope=CFG['leaky_slope'],
            state_clip=CFG['state_clip'],
        ).to(DEVICE)

        print("Training DEQ...")
        train_model(model, X_tr, y_tr, CFG)
        print("\nEvaluating solver gain sweep...")
        all_results[seed] = evaluate_sweep(model, X_te, y_te, CFG)

    # Aggregate
    metrics = ['final', 'early', 'grace', 'adaptive']
    summary = {
        lam: {
            m: {'mean': float(np.mean([all_results[s][lam][m] for s in CFG['seeds']])),
                'std':  float(np.std( [all_results[s][lam][m] for s in CFG['seeds']]))}
            for m in metrics
        }
        for lam in CFG['lambda_test_vals']
    }

    print("\n\n" + "="*85)
    print("E5 AGGREGATE — DEQ + LeakyReLU (5 seeds)")
    print(f"{'lam':>6}  {'final':>14}  {'early':>14}  "
          f"{'grace':>14}  {'adaptive':>14}  {'GRACE-final':>12}")
    print("-"*85)
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
    out = RESULTS_DIR / "e5_deq_leakyrelu.json"
    with open(out, 'w') as f:
        json.dump({'config': CFG,
                   'per_seed': {str(k): v for k, v in all_results.items()},
                   'summary':  {str(k): v for k, v in summary.items()}}, f, indent=2)
    print(f"\nSaved: {out}")

    try:
        make_plots(all_results, summary, CFG, RESULTS_DIR)
    except Exception as e:
        print(f"Plot error: {e}")

    return summary


if __name__ == "__main__":
    main()
