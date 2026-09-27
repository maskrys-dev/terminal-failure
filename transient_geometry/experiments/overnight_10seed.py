"""
Overnight 10-seed confirmation run.

Consolidates C1/C2 + D1 into a single run with 10 seeds for proper
statistical significance. This is the publication-quality version.

Usage:
    python -m transient_geometry.experiments.overnight_10seed
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
from transient_geometry.probes import LinearProbe


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'results', 'overnight_10seed')

CONFIG = {
    'input_dim': 784,
    'hidden_dim': 128,
    'rollout_steps': 40,
    'modrelu_bias': -0.5,
    'noise_std': 0.0,

    'spectral_radii': [0.5, 1.0, 1.20, 1.30, 1.35, 1.40, 1.45, 1.50, 1.60],
    'num_seeds': 10,

    'batch_size': 512,
    'train_n': 5000,
    'test_n': 5000,
    'early_K': 10,

    # D1 anchor config
    'd1_tau': 1.5,
    'd1_alpha': 5,
    # D1 best config
    'd1_tau_best': 1.2,
    'd1_alpha_best': 10,
}


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


@torch.no_grad()
def collect_full_trajectory(system, loader, T, K, device):
    """Collect full trajectory for D1 + all baseline readouts."""
    all_z_real = []
    all_norms = []
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

            z_real_t = torch.cat([z.real, z.imag], dim=-1).cpu().numpy()
            batch_z_real[t] = z_real_t
            batch_norms[t] = torch.abs(z).mean(dim=-1).cpu().numpy()

        z_mean = z_sum / T
        z_early = z_early_sum / K

        all_z_real.append(batch_z_real)
        all_norms.append(batch_norms)
        all_final.append(torch.cat([z.real, z.imag], dim=-1).cpu().numpy())
        all_trans.append(torch.cat([z_mean.real, z_mean.imag], dim=-1).cpu().numpy())
        all_early.append(torch.cat([z_early.real, z_early.imag], dim=-1).cpu().numpy())
        all_labels.append(yb.numpy())

    return {
        'z_real': np.concatenate(all_z_real, axis=1),
        'norms': np.concatenate(all_norms, axis=1),
        'final': np.concatenate(all_final),
        'transient_mean': np.concatenate(all_trans),
        'early_window': np.concatenate(all_early),
        'labels': np.concatenate(all_labels),
    }


def compute_d1_features(z_real, norms, tau, alpha):
    T, N, feat_dim = z_real.shape
    g = norms / np.maximum(norms[0:1, :], 1e-10)
    w_raw = np.exp(-alpha * np.maximum(0.0, g - tau))
    w_sum = np.sum(w_raw, axis=0, keepdims=True)
    w = w_raw / np.maximum(w_sum, 1e-10)
    features = np.einsum('tn,tnf->nf', w, z_real)
    return features


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

    print(f"\n{'='*80}")
    print(f"OVERNIGHT 10-SEED: {len(srs)} SR × {seeds} seeds = {total} runs")
    print(f"{'='*80}\n")

    results = []
    t_start = time.time()

    for idx, sr in enumerate(srs):
        for seed in range(seeds):
            run_num = idx * seeds + seed + 1
            t0 = time.time()
            print(f"[{run_num}/{total}] ρ={sr:.2f} s={seed} ... ", end="", flush=True)

            system = ComplexDynamicalSystem(
                input_dim=CONFIG['input_dim'], hidden_dim=CONFIG['hidden_dim'],
                spectral_radius=sr, noise_std=CONFIG['noise_std'],
                modrelu_bias=CONFIG['modrelu_bias'], seed=seed,
            ).to(device)

            train_data = collect_full_trajectory(system, train_ld, T, K, device)
            test_data = collect_full_trajectory(system, test_ld, T, K, device)

            # Baselines
            accs = {}
            for name in ['final', 'transient_mean', 'early_window']:
                probe = LinearProbe()
                probe.fit(train_data[name], train_data['labels'])
                accs[name] = probe.score(test_data[name], test_data['labels'])

            # Best-timestep
            best_acc, best_t = 0, 0
            for t in [0, 1, 4, 9, 14, 19, 29, 39]:
                if t >= T:
                    continue
                probe = LinearProbe()
                probe.fit(train_data['z_real'][t], train_data['labels'])
                a = probe.score(test_data['z_real'][t], test_data['labels'])
                if a > best_acc:
                    best_acc, best_t = a, t
            accs['best_timestep'] = best_acc

            # Adaptive ×1.5
            mean_norms = train_data['norms'].mean(axis=1)
            ratio = mean_norms / max(mean_norms[0], 1e-10)
            exceeded = np.where(ratio > 1.5)[0]
            stop_t = max(0, exceeded[0] - 1) if len(exceeded) > 0 else T - 1
            probe = LinearProbe()
            probe.fit(train_data['z_real'][stop_t], train_data['labels'])
            accs['adaptive_1.5'] = probe.score(test_data['z_real'][stop_t], test_data['labels'])

            # D1 anchor (τ=1.5, α=5)
            tr_d1 = compute_d1_features(train_data['z_real'], train_data['norms'],
                                        CONFIG['d1_tau'], CONFIG['d1_alpha'])
            te_d1 = compute_d1_features(test_data['z_real'], test_data['norms'],
                                        CONFIG['d1_tau'], CONFIG['d1_alpha'])
            probe = LinearProbe()
            probe.fit(tr_d1, train_data['labels'])
            accs['d1_anchor'] = probe.score(te_d1, test_data['labels'])

            # D1 best (τ=1.2, α=10)
            tr_d1b = compute_d1_features(train_data['z_real'], train_data['norms'],
                                         CONFIG['d1_tau_best'], CONFIG['d1_alpha_best'])
            te_d1b = compute_d1_features(test_data['z_real'], test_data['norms'],
                                         CONFIG['d1_tau_best'], CONFIG['d1_alpha_best'])
            probe = LinearProbe()
            probe.fit(tr_d1b, train_data['labels'])
            accs['d1_best'] = probe.score(te_d1b, test_data['labels'])

            elapsed = time.time() - t0
            result = {
                'spectral_radius': sr, 'seed': seed,
                'acc_final': accs['final'],
                'acc_transient_mean': accs['transient_mean'],
                'acc_early_window': accs['early_window'],
                'acc_best_timestep': accs['best_timestep'],
                'acc_adaptive': accs['adaptive_1.5'],
                'adaptive_stop_t': int(stop_t + 1),
                'acc_d1_anchor': accs['d1_anchor'],
                'acc_d1_best': accs['d1_best'],
                'elapsed': elapsed,
            }
            results.append(result)

            print(f"F={accs['final']:.4f} E={accs['early_window']:.4f} "
                  f"A={accs['adaptive_1.5']:.4f} D1={accs['d1_anchor']:.4f} "
                  f"D1b={accs['d1_best']:.4f} ({elapsed:.0f}s)")

    total_time = time.time() - t_start
    print(f"\nTotal runtime: {total_time/60:.1f} min")

    # Save
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # =========================================================================
    # Summary with CI
    # =========================================================================

    print(f"\n{'='*100}")
    print(f"10-SEED SUMMARY (mean ± std)")
    print(f"{'='*100}")
    print(f"\n{'SR':>6} | {'Final':>13} | {'Early':>13} | {'Adaptive':>13} | "
          f"{'D1-anchor':>13} | {'D1-best':>13}")
    print("-" * 90)

    for sr in srs:
        runs = [r for r in results if r['spectral_radius'] == sr]
        for key, label in [
            ('acc_final', 'F'), ('acc_early_window', 'E'),
            ('acc_adaptive', 'A'), ('acc_d1_anchor', 'D1a'), ('acc_d1_best', 'D1b')
        ]:
            vals = [r[key] for r in runs]
        f_m, f_s = np.mean([r['acc_final'] for r in runs]), np.std([r['acc_final'] for r in runs])
        e_m, e_s = np.mean([r['acc_early_window'] for r in runs]), np.std([r['acc_early_window'] for r in runs])
        a_m, a_s = np.mean([r['acc_adaptive'] for r in runs]), np.std([r['acc_adaptive'] for r in runs])
        d_m, d_s = np.mean([r['acc_d1_anchor'] for r in runs]), np.std([r['acc_d1_anchor'] for r in runs])
        b_m, b_s = np.mean([r['acc_d1_best'] for r in runs]), np.std([r['acc_d1_best'] for r in runs])
        print(f"{sr:>6.2f} | {f_m:.4f}±{f_s:.4f} | {e_m:.4f}±{e_s:.4f} | "
              f"{a_m:.4f}±{a_s:.4f} | {d_m:.4f}±{d_s:.4f} | {b_m:.4f}±{b_s:.4f}")

    # =========================================================================
    # Main figure: accuracy with error bands
    # =========================================================================

    sns.set_theme(style='whitegrid', font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    sr_arr = np.array(srs)
    strategies = [
        ('acc_final', 'Final-state', '#2196F3', 'o', '--', 1.5),
        ('acc_transient_mean', 'Transient-mean', '#F44336', 's', '--', 1.5),
        ('acc_early_window', 'Early-window (K=10)', '#4CAF50', '^', '-', 2),
        ('acc_adaptive', 'Adaptive (×1.5)', '#FF9800', 'D', '-', 2),
        ('acc_d1_anchor', 'D1 (τ=1.5, α=5)', '#6A1B9A', 'P', '-', 2.5),
        ('acc_d1_best', 'D1 (τ=1.2, α=10)', '#E91E63', 'H', '-', 2.5),
    ]

    for key, label, color, marker, ls, lw in strategies:
        means = np.array([np.mean([r[key] for r in results if r['spectral_radius'] == sr])
                          for sr in srs])
        stds = np.array([np.std([r[key] for r in results if r['spectral_radius'] == sr])
                         for sr in srs])
        ax.plot(sr_arr, means, f'{marker}{ls}', color=color, lw=lw, ms=7, label=label, alpha=0.9)
        ax.fill_between(sr_arr, means - stds, means + stds, color=color, alpha=0.1)

    ax.set_xlabel('Spectral Radius ρ(W)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('10-Seed Confirmation: All Readout Strategies')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)

    # Panel 2: D1 advantage over early-window with CI
    ax = axes[1]
    for key, label, color in [
        ('acc_d1_anchor', 'D1 (τ=1.5, α=5) − Early', '#6A1B9A'),
        ('acc_d1_best', 'D1 (τ=1.2, α=10) − Early', '#E91E63'),
    ]:
        gap_means = []
        gap_stds = []
        for sr in srs:
            runs = [r for r in results if r['spectral_radius'] == sr]
            gaps = [r[key] - r['acc_early_window'] for r in runs]
            gap_means.append(np.mean(gaps))
            gap_stds.append(np.std(gaps))
        gap_means = np.array(gap_means)
        gap_stds = np.array(gap_stds)
        ax.plot(sr_arr, gap_means, 'o-', color=color, lw=2.5, ms=7, label=label)
        ax.fill_between(sr_arr, gap_means - gap_stds, gap_means + gap_stds,
                        color=color, alpha=0.15)

    ax.axhline(y=0, color='black', lw=0.8)
    ax.set_xlabel('Spectral Radius ρ(W)')
    ax.set_ylabel('D1 − Early Window Accuracy')
    ax.set_title('D1 Advantage with 10-Seed Confidence Bands')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Publication-Quality 10-Seed Results', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'overnight_10seed.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {path}")
    print("Done.")


if __name__ == '__main__':
    main()
