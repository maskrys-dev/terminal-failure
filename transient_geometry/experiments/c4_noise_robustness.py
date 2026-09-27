"""
C4: Light robustness test — noise injection.

Tests whether the pre-collapse regime and GRACE survive mild dynamical noise.
Complex modReLU only, at ρ = 1.4 and 1.5.

Usage:
    python -m transient_geometry.experiments.c4_noise_robustness
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
    os.path.dirname(os.path.abspath(__file__)))), 'results', 'c4_noise')

CONFIG = {
    'input_dim': 784,
    'hidden_dim': 128,
    'rollout_steps': 40,
    'modrelu_bias': -0.5,

    'spectral_radii': [1.0, 1.40, 1.50],
    'noise_levels':   [0.0, 0.01, 0.05, 0.10],
    'num_seeds': 5,

    'batch_size': 512,
    'train_n': 5000,
    'test_n': 5000,
    'early_K': 10,

    'd1_tau': 1.2,
    'd1_alpha': 10,
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
def collect_trajectory(system, loader, T, K, noise_std, device):
    all_z, all_norms = [], []
    all_final, all_early, all_labels = [], [], []

    for xb, yb in loader:
        xb = xb.to(device)
        batch = xb.shape[0]
        d = system.hidden_dim

        x_complex = xb.to(torch.complex64)
        Ux = torch.matmul(x_complex, system.U.T)
        z = torch.zeros(batch, d, dtype=torch.complex64, device=device)
        z_early_sum = torch.zeros_like(z)

        batch_z = np.zeros((T, batch, 2 * d), dtype=np.float32)
        batch_norms = np.zeros((T, batch), dtype=np.float32)

        for t in range(T):
            pre = torch.matmul(z, system.W.T) + Ux + system.b
            # Inject noise
            if noise_std > 0:
                noise = torch.complex(
                    torch.randn_like(pre.real) * noise_std,
                    torch.randn_like(pre.imag) * noise_std)
                pre = pre + noise
            z = system.modrelu(pre)
            mag = torch.abs(z)
            mask = mag > 50.0
            if mask.any():
                z = torch.where(mask, z * (50.0 / mag.clamp(min=1e-8)), z)

            if t < K:
                z_early_sum += z
            batch_z[t] = torch.cat([z.real, z.imag], dim=-1).cpu().numpy()
            batch_norms[t] = torch.abs(z).mean(dim=-1).cpu().numpy()

        z_early = z_early_sum / K
        all_z.append(batch_z)
        all_norms.append(batch_norms)
        all_final.append(torch.cat([z.real, z.imag], dim=-1).cpu().numpy())
        all_early.append(torch.cat([z_early.real, z_early.imag], dim=-1).cpu().numpy())
        all_labels.append(yb.numpy())

    return {
        'z': np.concatenate(all_z, axis=1),
        'norms': np.concatenate(all_norms, axis=1),
        'final': np.concatenate(all_final),
        'early_window': np.concatenate(all_early),
        'labels': np.concatenate(all_labels),
    }


def compute_d1(z, norms, tau, alpha):
    T, N, d = z.shape
    g = norms / np.maximum(norms[0:1, :], 1e-10)
    w_raw = np.exp(-alpha * np.maximum(0.0, g - tau))
    w_sum = np.sum(w_raw, axis=0, keepdims=True)
    w = w_raw / np.maximum(w_sum, 1e-10)
    return np.einsum('tn,tnf->nf', w, z)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_ld, test_ld = get_data(CONFIG)
    T = CONFIG['rollout_steps']
    K = CONFIG['early_K']
    srs = CONFIG['spectral_radii']
    noises = CONFIG['noise_levels']
    seeds = CONFIG['num_seeds']
    total = len(srs) * len(noises) * seeds

    print(f"\n{'='*80}")
    print(f"C4 NOISE ROBUSTNESS: {len(srs)} SR × {len(noises)} noise × {seeds} seeds = {total} runs")
    print(f"{'='*80}\n")

    results = []
    run_idx = 0

    for sr in srs:
        for noise_std in noises:
            for seed in range(seeds):
                run_idx += 1
                t0 = time.time()
                print(f"[{run_idx}/{total}] ρ={sr:.2f} σ={noise_std:.2f} s={seed} ... ",
                      end="", flush=True)

                system = ComplexDynamicalSystem(
                    input_dim=CONFIG['input_dim'], hidden_dim=CONFIG['hidden_dim'],
                    spectral_radius=sr, noise_std=0.0,
                    modrelu_bias=CONFIG['modrelu_bias'], seed=seed,
                ).to(device)

                train_data = collect_trajectory(system, train_ld, T, K, noise_std, device)
                test_data = collect_trajectory(system, test_ld, T, K, noise_std, device)

                # Final
                probe = LinearProbe()
                probe.fit(train_data['final'], train_data['labels'])
                acc_final = probe.score(test_data['final'], test_data['labels'])

                # Early-window
                probe = LinearProbe()
                probe.fit(train_data['early_window'], train_data['labels'])
                acc_early = probe.score(test_data['early_window'], test_data['labels'])

                # Adaptive ×1.5
                mean_norms = train_data['norms'].mean(axis=1)
                ratio = mean_norms / max(mean_norms[0], 1e-10)
                exceeded = np.where(ratio > 1.5)[0]
                stop_t = max(0, exceeded[0] - 1) if len(exceeded) > 0 else T - 1
                probe = LinearProbe()
                probe.fit(train_data['z'][stop_t], train_data['labels'])
                acc_adaptive = probe.score(test_data['z'][stop_t], test_data['labels'])

                # GRACE
                tr_d1 = compute_d1(train_data['z'], train_data['norms'],
                                   CONFIG['d1_tau'], CONFIG['d1_alpha'])
                te_d1 = compute_d1(test_data['z'], test_data['norms'],
                                   CONFIG['d1_tau'], CONFIG['d1_alpha'])
                probe = LinearProbe()
                probe.fit(tr_d1, train_data['labels'])
                acc_grace = probe.score(te_d1, test_data['labels'])

                elapsed = time.time() - t0
                result = {
                    'spectral_radius': sr, 'noise_std': noise_std, 'seed': seed,
                    'acc_final': acc_final, 'acc_early': acc_early,
                    'acc_adaptive': acc_adaptive, 'acc_grace': acc_grace,
                    'adaptive_stop_t': int(stop_t + 1), 'elapsed': elapsed,
                }
                results.append(result)
                print(f"F={acc_final:.4f} E={acc_early:.4f} "
                      f"A={acc_adaptive:.4f} G={acc_grace:.4f} ({elapsed:.0f}s)")

    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*90}")
    print("C4 NOISE ROBUSTNESS SUMMARY")
    print(f"{'='*90}")

    for sr in srs:
        print(f"\n  ρ = {sr:.2f}")
        print(f"  {'σ_noise':>8} | {'Final':>13} | {'Early':>13} | "
              f"{'Adaptive':>13} | {'GRACE':>13} | {'Grace−Early':>12}")
        print(f"  {'-'*80}")
        for noise_std in noises:
            runs = [r for r in results
                    if r['spectral_radius'] == sr and r['noise_std'] == noise_std]
            f_m = np.mean([r['acc_final'] for r in runs])
            f_s = np.std([r['acc_final'] for r in runs])
            e_m = np.mean([r['acc_early'] for r in runs])
            e_s = np.std([r['acc_early'] for r in runs])
            a_m = np.mean([r['acc_adaptive'] for r in runs])
            a_s = np.std([r['acc_adaptive'] for r in runs])
            g_m = np.mean([r['acc_grace'] for r in runs])
            g_s = np.std([r['acc_grace'] for r in runs])
            gap = g_m - e_m
            print(f"  {noise_std:>8.2f} | {f_m:.3f}±{f_s:.3f} | {e_m:.3f}±{e_s:.3f} | "
                  f"{a_m:.3f}±{a_s:.3f} | {g_m:.3f}±{g_s:.3f} | {gap:>+12.4f}")

    # Plot: 1×3 panels, one per SR
    sns.set_theme(style='whitegrid', font_scale=1.1)
    fig, axes = plt.subplots(1, len(srs), figsize=(6 * len(srs), 6))

    strategies = [
        ('acc_final', 'Final-state', '#2196F3', 'o', '--'),
        ('acc_early', 'Early-window', '#4CAF50', '^', '-'),
        ('acc_adaptive', 'Adaptive', '#FF9800', 'D', '-'),
        ('acc_grace', 'GRACE', '#6A1B9A', 'P', '-'),
    ]

    for col, sr in enumerate(srs):
        ax = axes[col] if len(srs) > 1 else axes
        for key, label, color, marker, ls in strategies:
            means = []
            stds = []
            for ns in noises:
                runs = [r for r in results
                        if r['spectral_radius'] == sr and r['noise_std'] == ns]
                means.append(np.mean([r[key] for r in runs]))
                stds.append(np.std([r[key] for r in runs]))
            means = np.array(means)
            stds = np.array(stds)
            ax.errorbar(noises, means, yerr=stds, fmt=f'{marker}{ls}',
                        color=color, lw=2, ms=8, capsize=4, label=label)

        ax.set_xlabel('Noise σ')
        ax.set_ylabel('Test Accuracy')
        ax.set_title(f'ρ = {sr:.2f}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('C4: Noise Robustness', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'c4_noise_robustness.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {path}")
    print("Done.")


if __name__ == '__main__':
    main()
