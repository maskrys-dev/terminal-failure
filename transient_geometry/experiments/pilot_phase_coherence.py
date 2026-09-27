"""
Pilot: Phase Coherence Stopping
Uses EXACT same system parameters as the main paper (c1_c2_readout.py).
K diverse heads trained on different timesteps + bootstrap.
Phase coherence = mean resultant length of tanh-normalised head outputs.
"""
import os, sys, json
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeClassifierCV
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from transient_geometry.system import ComplexDynamicalSystem

RESULTS_DIR = ROOT / "results" / "pilot_phase_coherence"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR    = ROOT / "data"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Exact params from c1_c2_readout.py ───────────────────────────────────────
CFG = dict(
    hidden_dim     = 128,      # SAME as main paper
    input_dim      = 784,
    n_classes      = 10,
    modrelu_bias   = -0.5,     # SAME as main paper
    noise_std      = 0.0,
    spectral_radius= 1.4,      # well into grace-period regime
    T              = 40,
    train_subset   = 5000,
    test_subset    = 2000,
    seed           = 0,
    # Phase coherence heads
    K_heads        = 10,       # spread evenly over t=2..T//2
    bootstrap_frac = 0.7,
    coherence_thr  = 0.50,
    norm_stop_tau  = 1.5,
)

# ── Data (same as main paper) ─────────────────────────────────────────────────
def get_data():
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tf)
    te = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    rng = np.random.RandomState(CFG['seed'])
    tr_idx = rng.permutation(len(tr))[:CFG['train_subset']]
    te_idx = rng.permutation(len(te))[:CFG['test_subset']]
    X_tr = torch.stack([tr[i][0] for i in tr_idx]).to(DEVICE)
    y_tr = torch.tensor([tr[i][1] for i in tr_idx]).numpy()
    X_te = torch.stack([te[i][0] for i in te_idx]).to(DEVICE)
    y_te = torch.tensor([te[i][1] for i in te_idx]).numpy()
    return X_tr, y_tr, X_te, y_te

# ── Ridge probe helper ────────────────────────────────────────────────────────
def ridge_acc(X_tr, y_tr, X_te, y_te):
    clf = RidgeClassifierCV(alphas=[1e-2, 0.1, 1.0, 10.0, 100.0])
    clf.fit(X_tr, y_tr)
    return clf.score(X_te, y_te), clf

def to_feat(z):
    """complex [batch, d] -> real [batch, 2d]"""
    return torch.cat([z.real, z.imag], dim=-1).cpu().numpy()

# ── Build diverse heads ───────────────────────────────────────────────────────
def build_heads(traj_tr, y_tr):
    """
    K heads, each:
      - trained on a DIFFERENT timestep (spread evenly over t=2..T//2)
      - trained on a DIFFERENT 70% bootstrap subsample
    Stores W_re, W_im: the summed class-weight direction as [hidden] vectors.
    """
    T = traj_tr.shape[0]   # [T, n_train, hidden]
    n = len(y_tr)
    n_boot = int(n * CFG['bootstrap_frac'])
    rng    = np.random.default_rng(CFG['seed'] + 77)
    # Timesteps spread over t=2..T//2 (avoid very first transient)
    t_slots = np.linspace(2, T // 2, CFG['K_heads'], dtype=int)
    heads   = []
    print(f"\nTraining {CFG['K_heads']} diverse heads:")
    for k, t_k in enumerate(t_slots):
        idx  = rng.choice(n, size=n_boot, replace=False)
        feat = to_feat(traj_tr[t_k])[idx]
        y_k  = y_tr[idx]
        acc, clf = ridge_acc(feat, y_k, feat, y_k)   # in-sample just to check
        # Extract complex direction from coef
        coef = clf.coef_   # [n_classes, 2*hidden] or [1, 2*hidden]
        if coef.shape[0] == 1:
            coef = np.tile(coef, (CFG['n_classes'], 1))
        h = CFG['hidden_dim']
        w_re = torch.tensor(coef[:, :h].sum(0), dtype=torch.float32, device=DEVICE)
        w_im = torch.tensor(coef[:, h: ].sum(0), dtype=torch.float32, device=DEVICE)
        # L2 normalise so heads differ in direction not scale
        nrm = (w_re**2 + w_im**2).sum().sqrt().clamp(min=1e-8)
        heads.append({'W_re': w_re/nrm, 'W_im': w_im/nrm, 't': int(t_k)})
        print(f"  Head {k}: t={t_k:2d}, in-sample acc={acc:.3f}")
    return heads

# ── Phase coherence at one timestep ──────────────────────────────────────────
def coherence(z_t, heads):
    """
    z_t: complex [batch, hidden]
    Each head computes a complex dot product -> scalar per sample.
    tanh(|s|) * s/|s| normalises magnitude, preserves phase.
    MRL across heads, averaged over batch.
    """
    phases = []
    for h in heads:
        # complex dot: (w_re - i*w_im)^T (z_re + i*z_im)
        s_re = z_t.real @ h['W_re'] + z_t.imag @ h['W_im']
        s_im = z_t.real @ h['W_im'] - z_t.imag @ h['W_re']
        mag  = (s_re**2 + s_im**2).sqrt().clamp(min=1e-8)
        a_re = torch.tanh(mag) * s_re / mag
        a_im = torch.tanh(mag) * s_im / mag
        phases.append(torch.atan2(a_im, a_re))   # [batch]
    phases = torch.stack(phases, 0)   # [K, batch]
    mrl = torch.abs(torch.exp(1j * phases).mean(0)).mean().item()
    return mrl

# ── Norm-based adaptive stop ──────────────────────────────────────────────────
def norm_stop(traj, tau=1.5):
    batch   = traj[0].shape[0]
    norm_0  = torch.abs(traj[0]).mean(-1).clamp(min=1e-8)
    result  = traj[-1].clone()
    stopped = torch.zeros(batch, dtype=torch.bool, device=DEVICE)
    for z in traj:
        g = torch.abs(z).mean(-1) / norm_0
        new = (~stopped) & (g > tau)
        if new.any():
            result[new] = z[new]
            stopped |= new
        if stopped.all():
            break
    return result

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(CFG['seed']); np.random.seed(CFG['seed'])

    print("Loading data...")
    X_tr, y_tr, X_te, y_te = get_data()

    print(f"Building ComplexDynamicalSystem (rho={CFG['spectral_radius']})...")
    sys_ = ComplexDynamicalSystem(
        input_dim=CFG['input_dim'], hidden_dim=CFG['hidden_dim'],
        spectral_radius=CFG['spectral_radius'],
        noise_std=CFG['noise_std'], modrelu_bias=CFG['modrelu_bias'],
        seed=CFG['seed'],
    ).to(DEVICE)

    with torch.no_grad():
        traj_tr = sys_(X_tr, T=CFG['T'])   # [T, n_train, hidden] complex
        traj_te = sys_(X_te, T=CFG['T'])

    # Check: what does per-timestep accuracy look like with standard ridge?
    print("\nPer-timestep ridge accuracy (sanity check):")
    n = len(y_te); split = n // 2
    best_t, best_acc = 0, 0.0
    pt_acc = []
    for t in range(CFG['T']):
        f_tr = to_feat(traj_tr[t])
        f_te = to_feat(traj_te[t])
        acc, _ = ridge_acc(f_tr, y_tr, f_te[:split], y_te[:split])
        pt_acc.append(acc)
        if acc > best_acc:
            best_acc, best_t = acc, t
    print(f"  Best t={best_t}, acc={best_acc:.3f}  |  "
          f"t=0: {pt_acc[0]:.3f}  t={CFG['T']-1}: {pt_acc[-1]:.3f}")

    # Build heads
    heads = build_heads(traj_tr, y_tr)

    # Phase coherence trajectory
    print("\nComputing C(t):")
    c_traj = [coherence(traj_te[t], heads) for t in range(CFG['T'])]
    print(f"  C(t) range: {min(c_traj):.3f} -> {max(c_traj):.3f}")
    print(f"  Peak at t={np.argmax(c_traj)}, min at t={np.argmin(c_traj)}")

    # Stopping strategies
    f_tr  = to_feat(traj_tr[-1]); f_te = to_feat(traj_te[-1])
    acc_final, _ = ridge_acc(f_tr, y_tr, f_te[:split], y_te[:split])

    z_ns = norm_stop(list(traj_te), tau=CFG['norm_stop_tau'])
    acc_norm, _ = ridge_acc(to_feat(traj_tr[-1]), y_tr,
                            to_feat(z_ns)[:split], y_te[:split])

    # Phase-coherence stop: first t where C drops below previous peak by >15%
    peak_c = max(c_traj)
    t_stop = CFG['T'] - 1
    for t in range(1, CFG['T']):
        if c_traj[t] < peak_c * (1 - 0.15):
            t_stop = max(0, t - 1)
            break
    acc_coh, _ = ridge_acc(to_feat(traj_tr[t_stop]), y_tr,
                           to_feat(traj_te[t_stop])[:split], y_te[:split])

    print(f"\n{'='*55}")
    print(f"PILOT RESULTS  rho={CFG['spectral_radius']}, T={CFG['T']}")
    print(f"{'='*55}")
    print(f"  Final state (t={CFG['T']}):     {acc_final:.3f}")
    print(f"  Norm-based stop:           {acc_norm:.3f}")
    print(f"  Phase-coherence stop(t={t_stop:2d}): {acc_coh:.3f}")
    print(f"  Oracle best (t={best_t:2d}):      {best_acc:.3f}")
    print(f"\n  C(t) peak={peak_c:.3f} at t={np.argmax(c_traj)}")
    print(f"  C(t) at phase stop t={t_stop}: {c_traj[t_stop]:.3f}")
    print(f"  Corr(C(t), accuracy): "
          f"{np.corrcoef(c_traj, pt_acc)[0,1]:.3f}")

    # Save + plot
    out = RESULTS_DIR / "pilot_phase_coherence.json"
    with open(out, 'w') as f:
        json.dump({'pt_acc': pt_acc, 'c_traj': c_traj,
                   'acc_final': acc_final, 'acc_norm': acc_norm,
                   'acc_coh': acc_coh, 'oracle': best_acc,
                   't_stop': t_stop, 'best_t': best_t}, f, indent=2)

    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"Phase Coherence Pilot  rho={CFG['spectral_radius']}, "
                     f"K={CFG['K_heads']} heads", fontsize=12, fontweight='bold')

        ts = range(1, CFG['T'] + 1)
        ax = axes[0]
        ax.plot(ts, pt_acc, 'o-', color='#2c3e50', lw=2, ms=3, label='Accuracy')
        ax2 = ax.twinx()
        ax2.plot(ts, c_traj, 's--', color='#9b59b6', lw=1.5, ms=3, label='C(t)')
        ax2.set_ylabel('Phase coherence C(t)', color='#9b59b6')
        ax2.tick_params(axis='y', labelcolor='#9b59b6')
        for label, val, col in [('Final', acc_final, '#e74c3c'),
                                  ('Norm stop', acc_norm, '#3498db'),
                                  ('Phase stop', acc_coh, '#9b59b6'),
                                  ('Oracle', best_acc, '#2ecc71')]:
            ax.axhline(val, color=col, ls='--', lw=1.5, label=f'{label}={val:.3f}')
        if t_stop < CFG['T'] - 1:
            ax.axvline(t_stop + 1, color='#9b59b6', ls=':', alpha=0.7)
        ax.set_xlabel('Timestep t'); ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy & phase coherence')
        lines1, lbl1 = ax.get_legend_handles_labels()
        lines2, lbl2 = ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2, lbl1+lbl2, fontsize=7, loc='lower left')
        ax.grid(alpha=0.3)

        ax3 = axes[1]
        methods = ['Final', 'Norm\nstop', 'Phase\nstop', 'Oracle']
        accs    = [acc_final, acc_norm, acc_coh, best_acc]
        cols    = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71']
        bars    = ax3.bar(methods, accs, color=cols, width=0.5)
        for bar, a in zip(bars, accs):
            ax3.text(bar.get_x() + bar.get_width()/2, a + 0.005,
                     f'{a:.3f}', ha='center', va='bottom', fontweight='bold')
        ax3.set_ylabel('Accuracy'); ax3.set_title('Strategy comparison')
        ax3.set_ylim(0, min(1.0, max(accs) + 0.08)); ax3.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'pilot_phase_coherence.png', dpi=150, bbox_inches='tight')
        print(f"\nFigure saved.")
        plt.close()
    except Exception as e:
        print(f"Plot error: {e}")

if __name__ == "__main__":
    main()
