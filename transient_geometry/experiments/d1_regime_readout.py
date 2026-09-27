"""
Experiment D1: Regime-aware trajectory readout.

D1-A: Soft norm-growth penalty weighting.
    w̃_t = exp(-α · max(0, g_t - τ))
    w_t = w̃_t / Σ_s w̃_s
    z_D1 = Σ_t w_t z_t

Key question: Can a smooth descriptor-weighted readout beat naive
baselines by using more of the path when safe, while suppressing
destructive late states?

Usage:
    python -m transient_geometry.experiments.d1_regime_readout
"""

import os, sys, json, time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transient_geometry.system import ComplexDynamicalSystem
from transient_geometry.probes import LinearProbe, extract_timestep_features


# =============================================================================
# Configuration
# =============================================================================

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'results', 'd1')

CONFIG = {
    'input_dim': 784,
    'hidden_dim': 128,
    'rollout_steps': 40,
    'modrelu_bias': -0.5,
    'noise_std': 0.0,

    'spectral_radii': [1.0, 1.2, 1.30, 1.35, 1.40, 1.45, 1.50, 1.60],
    'num_seeds': 3,

    'batch_size': 512,
    'train_n': 5000,
    'test_n': 5000,

    'early_K': 10,

    # D1-A hyperparameter grid
    'tau_values': [1.2, 1.5, 2.0],
    'alpha_values': [2, 5, 10],
}


# =============================================================================
# Data
# =============================================================================

def get_data(config):
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    root = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), 'data')
    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    rng = np.random.RandomState(0)
    if config['train_n'] < len(train_ds):
        idx = rng.permutation(len(train_ds))[:config['train_n']]
        train_ds = torch.utils.data.Subset(train_ds, idx)
    if config['test_n'] < len(test_ds):
        idx = rng.permutation(len(test_ds))[:config['test_n']]
        test_ds = torch.utils.data.Subset(test_ds, idx)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=False)
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)
    return train_ld, test_ld


# =============================================================================
# D1 core: collect trajectory + compute weighted readouts
# =============================================================================

@torch.no_grad()
def collect_trajectory_and_readouts(system, loader, T, K, device):
    """
    Single forward pass per batch. Returns:
      - per-timestep complex states (for D1 weighting)
      - per-sample norm trajectories (for computing g_t)
      - standard readout features
      - labels
    """
    all_z_real = []       # list of [T, batch, 2d] arrays
    all_norms = []        # list of [T, batch] arrays
    all_final = []
    all_trans = []
    all_early = []
    all_labels = []

    for xb, yb in loader:
        xb = xb.to(device)
        batch = xb.shape[0]
        d = system.hidden_dim

        x_complex = xb.to(torch.complex64)
        Ux = torch.matmul(x_complex, system.U.T)

        z = torch.zeros(batch, d, dtype=torch.complex64, device=device)
        z_sum = torch.zeros_like(z)
        z_early_sum = torch.zeros_like(z)

        batch_z_real = np.zeros((T, batch, 2 * d), dtype=np.float32)
        batch_norms = np.zeros((T, batch), dtype=np.float32)

        for t in range(T):
            pre = torch.matmul(z, system.W.T) + Ux + system.b
            if system.noise_std > 0:
                noise = torch.complex(
                    torch.randn_like(pre.real) * system.noise_std,
                    torch.randn_like(pre.imag) * system.noise_std)
                pre = pre + noise
            z = system.modrelu(pre)
            mag = torch.abs(z)
            mask = mag > 50.0
            if mask.any():
                z = torch.where(mask, z * (50.0 / mag.clamp(min=1e-8)), z)

            z_sum += z
            if t < K:
                z_early_sum += z

            # Store per-timestep real features and norms
            z_real_t = torch.cat([z.real, z.imag], dim=-1).cpu().numpy()
            batch_z_real[t] = z_real_t
            batch_norms[t] = torch.abs(z).mean(dim=-1).cpu().numpy()  # [batch]

        z_mean = z_sum / T
        z_early = z_early_sum / K

        all_z_real.append(batch_z_real)
        all_norms.append(batch_norms)
        all_final.append(torch.cat([z.real, z.imag], dim=-1).cpu().numpy())
        all_trans.append(torch.cat([z_mean.real, z_mean.imag], dim=-1).cpu().numpy())
        all_early.append(torch.cat([z_early.real, z_early.imag], dim=-1).cpu().numpy())
        all_labels.append(yb.numpy())

    # Concatenate across batches: z_real is [T, N, 2d], norms is [T, N]
    z_real = np.concatenate(all_z_real, axis=1)       # [T, N, 2d]
    norms = np.concatenate(all_norms, axis=1)          # [T, N]

    return {
        'z_real': z_real,
        'norms': norms,
        'final': np.concatenate(all_final),
        'transient_mean': np.concatenate(all_trans),
        'early_window': np.concatenate(all_early),
        'labels': np.concatenate(all_labels),
    }


def compute_d1_features(z_real, norms, tau, alpha):
    """
    D1-A: Soft norm-growth penalty weighting.

    g_t[i] = norms[t, i] / norms[0, i]   (per-sample growth)
    w̃_t[i] = exp(-α · max(0, g_t[i] - τ))
    w_t[i] = w̃_t[i] / Σ_s w̃_s[i]       (normalized per sample)
    z_D1[i] = Σ_t w_t[i] · z_real[t, i]

    Args:
        z_real: [T, N, 2d] — per-timestep real features
        norms: [T, N] — per-sample norms
        tau: threshold
        alpha: penalty strength

    Returns:
        features: [N, 2d]
        weights: [T, N] — normalized weights (for visualization)
    """
    T, N, feat_dim = z_real.shape

    # Per-sample norm growth
    g = norms / np.maximum(norms[0:1, :], 1e-10)  # [T, N]

    # Unnormalized weights
    w_raw = np.exp(-alpha * np.maximum(0.0, g - tau))  # [T, N]

    # Normalize per sample
    w_sum = np.sum(w_raw, axis=0, keepdims=True)  # [1, N]
    w = w_raw / np.maximum(w_sum, 1e-10)           # [T, N]

    # Weighted sum: z_D1[i] = Σ_t w[t,i] * z_real[t,i]
    features = np.einsum('tn,tnf->nf', w, z_real)  # [N, 2d]

    return features, w


# =============================================================================
# Main
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_ld, test_ld = get_data(CONFIG)
    print(f"Train: {len(train_ld.dataset)} | Test: {len(test_ld.dataset)}")

    T = CONFIG['rollout_steps']
    K = CONFIG['early_K']
    srs = CONFIG['spectral_radii']
    seeds = CONFIG['num_seeds']
    total = len(srs) * seeds

    taus = CONFIG['tau_values']
    alphas = CONFIG['alpha_values']

    print(f"\n{'='*70}")
    print(f"D1: {len(srs)} SR × {seeds} seeds = {total} runs")
    print(f"D1-A grid: τ={taus}, α={alphas}")
    print(f"{'='*70}\n")

    results = []
    weight_profiles = {}  # (sr, tau, alpha) -> mean weight profile [T]
    run_idx = 0

    for sr in srs:
        for seed in range(seeds):
            run_idx += 1
            t0 = time.time()
            print(f"[{run_idx}/{total}] ρ={sr:.2f}, seed={seed} ... ", end="", flush=True)

            system = ComplexDynamicalSystem(
                input_dim=CONFIG['input_dim'], hidden_dim=CONFIG['hidden_dim'],
                spectral_radius=sr, noise_std=CONFIG['noise_std'],
                modrelu_bias=CONFIG['modrelu_bias'], seed=seed,
            ).to(device)

            train_data = collect_trajectory_and_readouts(system, train_ld, T, K, device)
            test_data = collect_trajectory_and_readouts(system, test_ld, T, K, device)

            # Baseline probes
            accs = {}
            for name in ['final', 'transient_mean', 'early_window']:
                probe = LinearProbe()
                probe.fit(train_data[name], train_data['labels'])
                accs[name] = probe.score(test_data[name], test_data['labels'])

            # Best-timestep (sample t=0,1,4,9,14,19,24,29,34,39)
            best_acc, best_t = 0, 0
            for t in [0, 1, 4, 9, 14, 19, 24, 29, 34, 39]:
                if t >= T:
                    continue
                probe = LinearProbe()
                probe.fit(train_data['z_real'][t], train_data['labels'])
                a = probe.score(test_data['z_real'][t], test_data['labels'])
                if a > best_acc:
                    best_acc, best_t = a, t
            accs['best_timestep'] = best_acc

            # Adaptive ×1.5 (using batch-averaged norms)
            mean_norms = train_data['norms'].mean(axis=1)  # [T]
            ratio = mean_norms / max(mean_norms[0], 1e-10)
            exceeded = np.where(ratio > 1.5)[0]
            stop_t = max(0, exceeded[0] - 1) if len(exceeded) > 0 else T - 1
            probe = LinearProbe()
            probe.fit(train_data['z_real'][stop_t], train_data['labels'])
            accs['adaptive_1.5'] = probe.score(test_data['z_real'][stop_t], test_data['labels'])

            # D1-A: sweep (τ, α)
            d1_accs = {}
            for tau in taus:
                for alpha in alphas:
                    key = f'd1_t{tau}_a{alpha}'
                    tr_feat, tr_w = compute_d1_features(
                        train_data['z_real'], train_data['norms'], tau, alpha)
                    te_feat, te_w = compute_d1_features(
                        test_data['z_real'], test_data['norms'], tau, alpha)

                    probe = LinearProbe()
                    probe.fit(tr_feat, train_data['labels'])
                    d1_accs[key] = probe.score(te_feat, test_data['labels'])

                    # Store weight profiles (mean across samples)
                    wp_key = (sr, tau, alpha)
                    if wp_key not in weight_profiles:
                        weight_profiles[wp_key] = []
                    weight_profiles[wp_key].append(tr_w.mean(axis=1))  # [T]

            elapsed = time.time() - t0

            # Find best D1 config
            best_d1_key = max(d1_accs, key=d1_accs.get)
            best_d1_acc = d1_accs[best_d1_key]

            result = {
                'spectral_radius': sr, 'seed': seed,
                'acc_final': accs['final'],
                'acc_transient_mean': accs['transient_mean'],
                'acc_early_window': accs['early_window'],
                'acc_best_timestep': accs['best_timestep'],
                'acc_adaptive': accs['adaptive_1.5'],
                'adaptive_stop_t': int(stop_t + 1),
                'd1_accs': {k: float(v) for k, v in d1_accs.items()},
                'best_d1': best_d1_key,
                'best_d1_acc': float(best_d1_acc),
                'elapsed': elapsed,
            }
            results.append(result)

            print(f"F={accs['final']:.4f} E={accs['early_window']:.4f} "
                  f"A={accs['adaptive_1.5']:.4f}@t={stop_t+1} "
                  f"D1={best_d1_acc:.4f}({best_d1_key}) ({elapsed:.0f}s)")

    # Save
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Average weight profiles
    avg_profiles = {}
    for k, v_list in weight_profiles.items():
        avg_profiles[str(k)] = np.mean(v_list, axis=0).tolist()

    with open(os.path.join(RESULTS_DIR, 'weight_profiles.json'), 'w') as f:
        json.dump(avg_profiles, f, indent=2)

    # =========================================================================
    # Plot 1: Accuracy vs SR — all methods
    # =========================================================================

    sns.set_theme(style='whitegrid', font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    sr_arr = np.array(srs)

    # Baselines
    styles = [
        ('acc_final', 'Final-state', '#2196F3', 'o', '--', 1.5),
        ('acc_transient_mean', 'Transient-mean', '#F44336', 's', '--', 1.5),
        ('acc_early_window', 'Early-window (K=10)', '#4CAF50', '^', '-', 2.5),
        ('acc_adaptive', 'Adaptive (×1.5)', '#FF9800', 'D', '-', 2.5),
        ('acc_best_timestep', 'Best-timestep oracle', '#9E9E9E', 'x', ':', 1.5),
    ]

    for key, label, color, marker, ls, lw in styles:
        means = [np.mean([r[key] for r in results if r['spectral_radius'] == sr])
                 for sr in srs]
        ax.plot(sr_arr, means, f'{marker}{ls}', color=color, lw=lw, ms=7, label=label, alpha=0.85)

    # Best D1 per SR (use τ=1.5, α=5 as the anchor)
    anchor_key = 'd1_t1.5_a5'
    d1_means = [np.mean([r['d1_accs'].get(anchor_key, 0) for r in results
                         if r['spectral_radius'] == sr]) for sr in srs]
    ax.plot(sr_arr, d1_means, 'P-', color='#6A1B9A', lw=3, ms=9,
            label='D1-weighted (τ=1.5, α=5)', zorder=10)

    ax.set_xlabel('Spectral Radius ρ(W)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('D1: Regime-Aware Trajectory Readout')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 2: Temporal weight profiles
    # =========================================================================

    ax = axes[1]
    tau, alpha = 1.5, 5
    profile_srs = [1.0, 1.3, 1.5]
    cmap = plt.cm.coolwarm
    colors_prof = [cmap(0.1), cmap(0.5), cmap(0.9)]

    for i, sr in enumerate(profile_srs):
        wp_key = str((sr, tau, alpha))
        if wp_key in avg_profiles:
            profile = np.array(avg_profiles[wp_key])
            ax.plot(range(1, T + 1), profile, '-', lw=2.5, color=colors_prof[i],
                    label=f'ρ={sr:.1f}', alpha=0.9)

    ax.set_xlabel('Timestep t')
    ax.set_ylabel('Mean Weight w_t')
    ax.set_title(f'D1 Weight Profile (τ={tau}, α={alpha})')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'd1_results.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {path}")

    # =========================================================================
    # Summary
    # =========================================================================

    print(f"\n{'='*90}")
    print("D1 SUMMARY (anchor: τ=1.5, α=5)")
    print(f"{'='*90}")
    print(f"\n{'SR':>6} | {'Final':>7} | {'Trans':>7} | {'Early':>7} | "
          f"{'Adap':>7} | {'D1':>7} | {'D1-Final':>9}")
    print("-" * 72)

    for sr in srs:
        runs = [r for r in results if r['spectral_radius'] == sr]
        f = np.mean([r['acc_final'] for r in runs])
        t = np.mean([r['acc_transient_mean'] for r in runs])
        e = np.mean([r['acc_early_window'] for r in runs])
        a = np.mean([r['acc_adaptive'] for r in runs])
        d = np.mean([r['d1_accs'].get(anchor_key, 0) for r in runs])
        print(f"{sr:>6.2f} | {f:>7.4f} | {t:>7.4f} | {e:>7.4f} | "
              f"{a:>7.4f} | {d:>7.4f} | {d-f:>+9.4f}")

    # Best D1 config per SR
    print(f"\n{'='*90}")
    print("BEST D1 CONFIG PER SR")
    print(f"{'='*90}")
    for sr in srs:
        runs = [r for r in results if r['spectral_radius'] == sr]
        # Average D1 accs across seeds for each config
        all_configs = set()
        for r in runs:
            all_configs.update(r['d1_accs'].keys())
        config_means = {}
        for cfg in all_configs:
            config_means[cfg] = np.mean([r['d1_accs'].get(cfg, 0) for r in runs])
        best_cfg = max(config_means, key=config_means.get)
        print(f"  ρ={sr:.2f}: {best_cfg} → {config_means[best_cfg]:.4f}")

    print(f"\nAll outputs saved to: {RESULTS_DIR}")
    print("Done.")


if __name__ == '__main__':
    main()
