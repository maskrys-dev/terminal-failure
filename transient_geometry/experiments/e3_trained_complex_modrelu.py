"""
E3: Trained Complex modReLU Recurrent Classifier — Pre-Collapse Under Regime Shift
====================================================================================
Train a shared-weight complex recurrent classifier end-to-end on MNIST.
Then at test time sweep the recurrent gain lambda far past the training regime.

Architecture:
    z_{t+1} = modReLU(lambda * W z_t + U x + b),   z_0 = 0
    logits   = V_re @ z_re + V_im @ z_im + c     (real projection)

Implementation: complex weights stored as real 2x-width vectors and computed
using real arithmetic (avoids autograd issues with torch.complex on Windows).
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
RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "e3_trained_complex"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).parent.parent.parent / "data"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CFG = dict(
    hidden_dim=128,          # complex units → 2*128 real dims in features
    input_dim=784,
    n_classes=10,
    T_train=10,              # short horizon — prevents learning long-term robustness
    lambda_train=0.85,       # firmly contractive during training
    modrelu_bias=0.0,              # no dead zone at init — states can activate immediately
    T_test=40,               # long horizon at test time
    early_K=8,
    lambda_test_vals=[0.7, 0.85, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0],
    lr=1e-3,
    epochs=30,
    batch_size=256,
    n_train=5000,
    n_test=2000,
    seeds=[0, 1, 2, 3, 4],
    grace_tau=1.2,
    grace_alpha=10.0,
    stop_tau=1.5,
    mag_clip=50.0,
)

# ─────────────────────────────────────────────────────────────────────────────
# Real 2×2-block complex multiplication
# ─────────────────────────────────────────────────────────────────────────────

def cplx_linear(x_re, x_im, W_re, W_im):
    """
    Complex linear: [re, im] = W_re @ x_re - W_im @ x_im,  W_re @ x_im + W_im @ x_re
    All tensors real. Full autograd support.
    """
    out_re = F.linear(x_re, W_re) - F.linear(x_im, W_im)
    out_im = F.linear(x_re, W_im) + F.linear(x_im, W_re)
    return out_re, out_im


def modrelu_real(z_re, z_im, bias, leaky=0.01):
    """
    Leaky modReLU in real [re, im] form.
    scale = max(|z| + bias, 0) / |z|  + leaky * min(|z| + bias, 0) / |z|
    (leaky avoids total gradient death in the dead zone)
    """
    mag   = torch.sqrt(z_re**2 + z_im**2).clamp(min=1e-8)
    act   = mag + bias
    scale = (torch.clamp(act, min=0.0) + leaky * torch.clamp(act, max=0.0)) / mag
    return z_re * scale, z_im * scale


def mag_clip_real(z_re, z_im, threshold=50.0):
    """Magnitude clip in [re, im] form."""
    mag  = torch.sqrt(z_re**2 + z_im**2)
    mask = mag > threshold
    if mask.any():
        scale = torch.where(mask, threshold / mag.clamp(min=1e-8),
                            torch.ones_like(mag))
        z_re = z_re * scale
        z_im = z_im * scale
    return z_re, z_im

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class TrainedComplexRNN(nn.Module):
    """
    Trained complex recurrent classifier using real 2x-block arithmetic.

    State:  (z_re, z_im) ∈ R^d × R^d  (logically z ∈ C^d)

    Recurrence:
        pre_re, pre_im = lam * W_complex @ z + U_complex @ x + b
        z_re, z_im     = modReLU(pre_re, pre_im)

    Logits:
        logits = V_re @ z_re + V_im @ z_im + c
    """

    def __init__(self, input_dim, hidden_dim, n_classes=10,
                 lam_init=0.85, modrelu_bias=-0.5, mag_clip_val=50.0):
        super().__init__()
        d = hidden_dim
        self.hidden_dim   = d
        self.lam          = lam_init
        self.modrelu_bias = modrelu_bias
        self.mag_clip_val = mag_clip_val

        # ── Recurrent weight W ∈ C^{d×d} → stored as W_re, W_im ──────────
        # Initialise with spectral radius ≈ 1 so lam controls the actual sr
        W_c = torch.randn(d, d, dtype=torch.complex64) / np.sqrt(d)
        eigs = torch.linalg.eigvals(W_c)
        sr   = torch.max(torch.abs(eigs)).item()
        W_c  = W_c / max(sr, 1e-8)
        self.W_re = nn.Parameter(W_c.real.clone())
        self.W_im = nn.Parameter(W_c.imag.clone())

        # ── Input projection U ∈ C^{d×input_dim} ─────────────────────────
        self.U_re = nn.Parameter(torch.randn(d, input_dim) / np.sqrt(input_dim) * 0.5)
        self.U_im = nn.Parameter(torch.randn(d, input_dim) / np.sqrt(input_dim) * 0.5)

        # ── Recurrent bias b ∈ C^d ────────────────────────────────────────
        self.b_re = nn.Parameter(torch.zeros(d))
        self.b_im = nn.Parameter(torch.zeros(d))

        # ── Output head: Re[V z_T] + c ────────────────────────────────────
        self.V_re = nn.Parameter(torch.randn(n_classes, d) * 0.1 / np.sqrt(d))
        self.V_im = nn.Parameter(torch.randn(n_classes, d) * 0.1 / np.sqrt(d))
        self.c    = nn.Parameter(torch.zeros(n_classes))

    def step(self, z_re, z_im, Ux_re, Ux_im, lam):
        """Single recurrent step — full autograd via real arithmetic."""
        # lam * W z
        Wz_re, Wz_im = cplx_linear(z_re, z_im, self.W_re, self.W_im)
        # pre = lam * Wz + Ux + b
        pre_re = lam * Wz_re + Ux_re + self.b_re
        pre_im = lam * Wz_im + Ux_im + self.b_im
        # modReLU
        z_re, z_im = modrelu_real(pre_re, pre_im, self.modrelu_bias)
        # Magnitude clip
        z_re, z_im = mag_clip_real(z_re, z_im, self.mag_clip_val)
        return z_re, z_im

    def rollout(self, x, T, lam=None, return_trajectory=False):
        """
        Roll out for T steps.

        Parameters
        ----------
        x : real Tensor [batch, input_dim]
        T : int
        lam : float — gain (defaults to self.lam)
        return_trajectory : if True, return list of (z_re, z_im) pairs

        Returns
        -------
        z_re, z_im : Tensors [batch, d]
        trajectory (optional): list of (z_re, z_im) pairs
        """
        if lam is None:
            lam = self.lam
        batch = x.shape[0]

        # Input projection (constant across timesteps)
        Ux_re, Ux_im = cplx_linear(x, torch.zeros_like(x), self.U_re, self.U_im)

        z_re = torch.zeros(batch, self.hidden_dim, device=x.device)
        z_im = torch.zeros(batch, self.hidden_dim, device=x.device)

        trajectory = [] if return_trajectory else None

        for _ in range(T):
            z_re, z_im = self.step(z_re, z_im, Ux_re, Ux_im, lam)
            if return_trajectory:
                trajectory.append((z_re, z_im))

        if return_trajectory:
            return (z_re, z_im), trajectory
        return z_re, z_im

    def forward(self, x, T=None, lam=None):
        if T is None:
            T = 10
        z_re, z_im = self.rollout(x, T, lam=lam)
        # logits = V_re @ z_re + V_im @ z_im + c
        logits = F.linear(z_re, self.V_re) + F.linear(z_im, self.V_im) + self.c
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# GRACE and adaptive stopping
# ─────────────────────────────────────────────────────────────────────────────

def to_feat(z_re, z_im):
    return torch.cat([z_re, z_im], dim=-1)


def grace_readout(trajectory, tau=1.2, alpha=10.0):
    """GRACE over (z_re, z_im) trajectory."""
    z0_re, z0_im = trajectory[0]
    norm_0 = (z0_re**2 + z0_im**2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)

    weighted_re = torch.zeros_like(z0_re)
    weighted_im = torch.zeros_like(z0_im)
    weight_sum  = torch.zeros(z0_re.shape[0], 1, device=z0_re.device)

    for z_re, z_im in trajectory:
        norm_t = (z_re**2 + z_im**2).mean(dim=-1, keepdim=True).sqrt()
        g_t    = norm_t / norm_0
        w_t    = torch.exp(-alpha * torch.clamp(g_t - tau, min=0.0))
        weighted_re += w_t * z_re
        weighted_im += w_t * z_im
        weight_sum  += w_t

    w = weight_sum.clamp(min=1e-8)
    feat = to_feat(weighted_re / w, weighted_im / w)
    return feat.cpu().numpy()


def adaptive_stop(trajectory, tau_stop=1.5):
    """Adaptive stopping over (z_re, z_im) trajectory."""
    z0_re, z0_im = trajectory[0]
    batch  = z0_re.shape[0]
    device = z0_re.device
    norm_0 = (z0_re**2 + z0_im**2).mean(dim=-1).sqrt().clamp(min=1e-8)

    res_re  = trajectory[-1][0].clone()
    res_im  = trajectory[-1][1].clone()
    stopped = torch.zeros(batch, dtype=torch.bool, device=device)

    for z_re, z_im in trajectory:
        norm_t = (z_re**2 + z_im**2).mean(dim=-1).sqrt()
        growth = norm_t / norm_0
        just_stopped = (~stopped) & (growth > tau_stop)
        if just_stopped.any():
            res_re[just_stopped] = z_re[just_stopped]
            res_im[just_stopped] = z_im[just_stopped]
            stopped |= just_stopped
        if stopped.all():
            break

    return to_feat(res_re, res_im).cpu().numpy()


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
    X_tr = torch.stack([tr[i][0] for i in torch.randperm(len(tr), generator=rng)[:n_train]])
    y_tr = torch.tensor([tr[i][1] for i in range(n_train)])
    X_te = torch.stack([te[i][0] for i in torch.randperm(len(te), generator=rng)[:n_test]])
    y_te = torch.tensor([te[i][1] for i in range(n_test)])
    return X_tr, y_tr, X_te, y_te

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, X_tr, y_tr, cfg, verbose=True):
    model.train()
    opt   = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'])
    X_tr  = X_tr.to(DEVICE)
    y_tr  = y_tr.to(DEVICE)

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
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{cfg['epochs']}  "
                  f"loss={total_loss/len(X_tr):.4f}  acc={n_correct/len(X_tr):.3f}")
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
        head_logits_all = []
        norm_traj_all   = []

        for i in range(0, len(X_te), cfg['batch_size']):
            xb = X_te[i:i + cfg['batch_size']]
            (z_re, z_im), traj = model.rollout(xb, cfg['T_test'],
                                               lam=lam, return_trajectory=True)
            logits = F.linear(z_re, model.V_re) + F.linear(z_im, model.V_im) + model.c
            head_logits_all.append(logits.cpu())
            for t, (zr, zi) in enumerate(traj):
                all_traj[t].append((zr.detach(), zi.detach()))
            norms = [(zr**2 + zi**2).mean().sqrt().item() for zr, zi in traj]
            norm_traj_all.append(norms)

        # Consolidate trajectory
        trajectory = [
            (torch.cat([b[0] for b in all_traj[t]], dim=0),
             torch.cat([b[1] for b in all_traj[t]], dim=0))
            for t in range(cfg['T_test'])
        ]
        mean_norm_traj = np.mean(norm_traj_all, axis=0).tolist()

        # Features
        feat_final = to_feat(*trajectory[-1]).cpu().numpy()
        feat_early = np.stack([to_feat(*trajectory[t]).cpu().numpy()
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

        per_t_acc = [probe(to_feat(*trajectory[t]).cpu().numpy())
                     for t in range(cfg['T_test'])]

        results[lam] = {
            'head': float(acc_head), 'final': float(acc_final),
            'early': float(acc_early), 'grace': float(acc_grace),
            'adaptive': float(acc_adaptive),
            'per_t': per_t_acc, 'norm_traj': mean_norm_traj,
        }
        gain = acc_grace - acc_final
        marker = ' <-- GRACE wins' if gain > 0.005 else ''
        print(f"head={acc_head:.3f}  final={acc_final:.3f}  "
              f"early={acc_early:.3f}  grace={acc_grace:.3f}  "
              f"norm_T={mean_norm_traj[-1]:.1f}{marker}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(all_results, summary, cfg, save_dir):
    import matplotlib.pyplot as plt
    lams    = cfg['lambda_test_vals']
    metrics = ['final', 'early', 'grace', 'adaptive']
    colors  = {'final': '#e74c3c', 'early': '#f39c12',
               'adaptive': '#3498db', 'grace': '#2ecc71'}
    labels  = {'final': f'Final (z_{cfg["T_test"]})',
               'early': f'Early window (t=1..{cfg["early_K"]})',
               'adaptive': 'Adaptive stopping', 'grace': 'GRACE'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Trained Complex modReLU Recurrent System — Pre-Collapse Under Gain Sweep\n"
        f"(trained: T={cfg['T_train']}, lam={cfg['lambda_train']}  |  "
        f"tested: T={cfg['T_test']}, lam up to {max(lams)})",
        fontsize=12, fontweight='bold')

    # Panel 1: accuracy sweep
    ax = axes[0]
    for m in metrics:
        mn = [summary[l][m]['mean'] for l in lams]
        sd = [summary[l][m]['std']  for l in lams]
        ax.plot(lams, mn, 'o-', color=colors[m], label=labels[m], lw=2)
        ax.fill_between(lams, [a-b for a,b in zip(mn,sd)],
                              [a+b for a,b in zip(mn,sd)], alpha=0.15, color=colors[m])
    ax.axvline(1.0, color='gray', ls='--', alpha=0.6, label='rho=1.0')
    ax.axvline(cfg['lambda_train'], color='k', ls=':', alpha=0.5,
               label=f'lam_train={cfg["lambda_train"]}')
    ax.set_xlabel('Test-time gain lambda'); ax.set_ylabel('Accuracy')
    ax.set_title('Readout accuracy vs. lambda'); ax.legend(fontsize=8)
    ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)

    # Panel 2: per-timestep at highest degrade lambda
    ax2 = axes[1]
    baseline = summary[lams[0]]['final']['mean']
    tgt = next((l for l in lams if l>1.0 and summary[l]['final']['mean'] < baseline-0.02), lams[-1])
    for s in cfg['seeds']:
        ax2.plot(range(1, cfg['T_test']+1), all_results[s][tgt]['per_t'],
                 alpha=0.25, color='#95a5a6', lw=1)
    mean_pt = np.mean([[all_results[s][tgt]['per_t'][t]
                        for t in range(cfg['T_test'])] for s in cfg['seeds']], axis=0)
    ax2.plot(range(1, cfg['T_test']+1), mean_pt, 'k-', lw=2.5, label=f'Mean (lam={tgt})')
    for m in ['grace', 'final', 'adaptive']:
        ax2.axhline(summary[tgt][m]['mean'], color=colors[m], ls='--', lw=1.5,
                    label=f"{m}={summary[tgt][m]['mean']:.3f}")
    ax2.set_xlabel('Timestep t'); ax2.set_ylabel('Probe accuracy')
    ax2.set_title(f'Per-timestep accuracy at lam={tgt}')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # Panel 3: GRACE advantage bar chart
    ax3 = axes[2]
    gains = [summary[l]['grace']['mean'] - summary[l]['final']['mean'] for l in lams]
    bars  = ax3.bar(range(len(lams)), gains,
                    color=['#2ecc71' if g > 0 else '#e74c3c' for g in gains])
    ax3.axhline(0, color='k', lw=0.8)
    ax3.set_xticks(range(len(lams)))
    ax3.set_xticklabels([f'{l:.1f}' for l in lams], rotation=45, fontsize=8)
    ax3.set_xlabel('Test-time gain lambda')
    ax3.set_ylabel('GRACE - Final accuracy')
    ax3.set_title('GRACE advantage over final-state readout')
    ax3.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(save_dir / 'e3_trained_complex.png', dpi=150, bbox_inches='tight')
    print(f"Figure saved.")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    all_results = {}
    for seed in CFG['seeds']:
        print(f"\n{'='*65}")
        print(f"Seed {seed} | T_train={CFG['T_train']} lam_train={CFG['lambda_train']}"
              f" | T_test={CFG['T_test']} lam up to {max(CFG['lambda_test_vals'])}")
        print(f"{'='*65}")
        torch.manual_seed(seed);  np.random.seed(seed)
        X_tr, y_tr, X_te, y_te = load_mnist(CFG['n_train'], CFG['n_test'], seed=seed)
        model = TrainedComplexRNN(
            input_dim=CFG['input_dim'], hidden_dim=CFG['hidden_dim'],
            n_classes=CFG['n_classes'], lam_init=CFG['lambda_train'],
            modrelu_bias=CFG['modrelu_bias'], mag_clip_val=CFG['mag_clip'],
        ).to(DEVICE)
        print("Training...")
        train_model(model, X_tr, y_tr, CFG)
        print("\nEvaluating...")
        all_results[seed] = evaluate_regime_sweep(model, X_te, y_te, CFG)

    # Aggregate
    metrics = ['final', 'early', 'grace', 'adaptive']
    summary = {}
    for lam in CFG['lambda_test_vals']:
        summary[lam] = {
            m: {'mean': float(np.mean([all_results[s][lam][m] for s in CFG['seeds']])),
                'std':  float(np.std( [all_results[s][lam][m] for s in CFG['seeds']]))}
            for m in metrics
        }

    print("\n\n" + "="*80)
    print("AGGREGATE  (5 seeds)")
    print(f"{'lam':>6}  {'final':>14}  {'early':>14}  {'grace':>14}  {'adaptive':>14}  {'GRACE-final':>12}")
    print("-"*82)
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
    out = RESULTS_DIR / "e3_trained_complex.json"
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
