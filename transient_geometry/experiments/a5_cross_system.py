"""
Experiment A5: Cross-system replication.

Tests whether the pre-collapse grace period generalises across
dynamical system classes:
  1. Complex-valued reservoir (modReLU) — the main system
  2. Real-valued ESN (tanh) — proves it's not a complex-number artifact
  3. Linear dynamical system (no nonlinearity) — spectral instability alone

Usage:
    python -m transient_geometry.experiments.a5_cross_system
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

from transient_geometry.system import ComplexDynamicalSystem, RealESN, LinearDynamicalSystem
from transient_geometry.probes import LinearProbe


# =============================================================================
# Configuration
# =============================================================================

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'results', 'a5_cross_system')

CONFIG = {
    'input_dim': 784,
    'hidden_dim': 128,
    'rollout_steps': 40,
    'noise_std': 0.0,

    'spectral_radii': [0.5, 1.0, 1.20, 1.30, 1.35, 1.40, 1.45, 1.50, 1.60],
    'num_seeds': 3,

    'batch_size': 512,
    'train_n': 5000,
    'test_n': 5000,
    'early_K': 10,

    'systems': {
        'complex_modrelu': {
            'class': 'ComplexDynamicalSystem',
            'kwargs': {'modrelu_bias': -0.5},
            'label': 'Complex (modReLU)',
            'color': '#2196F3',
        },
        'real_esn': {
            'class': 'RealESN',
            'kwargs': {'leak_rate': 1.0},
            'label': 'Real ESN (tanh)',
            'color': '#F44336',
        },
        'linear': {
            'class': 'LinearDynamicalSystem',
            'kwargs': {},
            'label': 'Linear (no nonlin.)',
            'color': '#4CAF50',
        },
    },
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
# Build system
# =============================================================================

SYSTEM_CLASSES = {
    'ComplexDynamicalSystem': ComplexDynamicalSystem,
    'RealESN': RealESN,
    'LinearDynamicalSystem': LinearDynamicalSystem,
}


def build_system(sys_config, input_dim, hidden_dim, sr, seed, device):
    cls = SYSTEM_CLASSES[sys_config['class']]
    kwargs = dict(sys_config['kwargs'])
    system = cls(
        input_dim=input_dim, hidden_dim=hidden_dim,
        spectral_radius=sr, noise_std=0.0, seed=seed,
        **kwargs
    ).to(device)
    return system


# =============================================================================
# Extract readouts
# =============================================================================

def extract_readouts(system, loader, T, K, device):
    all_final, all_trans, all_early, all_labels = [], [], [], []
    norm_sum = np.zeros(T)
    n_batches = 0
    for xb, yb in loader:
        xb = xb.to(device)
        res = system.forward_all_readouts(xb, T, K)
        all_final.append(res['final'])
        all_trans.append(res['transient_mean'])
        all_early.append(res['early_window'])
        norm_sum += res['norms']
        n_batches += 1
        all_labels.append(yb.numpy())
    return {
        'final': np.concatenate(all_final),
        'transient_mean': np.concatenate(all_trans),
        'early_window': np.concatenate(all_early),
        'norms': norm_sum / n_batches,
        'labels': np.concatenate(all_labels),
    }


# =============================================================================
# Main sweep
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
    systems_cfg = CONFIG['systems']

    total = len(systems_cfg) * len(srs) * seeds
    print(f"\n{'='*70}")
    print(f"A5: {len(systems_cfg)} systems × {len(srs)} SR × {seeds} seeds = {total} runs")
    print(f"{'='*70}\n")

    all_results = {sys_name: [] for sys_name in systems_cfg}
    run_idx = 0

    for sys_name, sys_cfg in systems_cfg.items():
        print(f"\n--- {sys_cfg['label']} ---")

        for sr in srs:
            for seed in range(seeds):
                run_idx += 1
                t0 = time.time()
                print(f"[{run_idx}/{total}] {sys_name} ρ={sr:.2f} s={seed} ... ",
                      end="", flush=True)

                system = build_system(sys_cfg, CONFIG['input_dim'],
                                      CONFIG['hidden_dim'], sr, seed, device)

                train_data = extract_readouts(system, train_ld, T, K, device)
                test_data = extract_readouts(system, test_ld, T, K, device)

                accs = {}
                for name in ['final', 'transient_mean', 'early_window']:
                    probe = LinearProbe()
                    probe.fit(train_data[name], train_data['labels'])
                    accs[name] = probe.score(test_data[name], test_data['labels'])

                elapsed = time.time() - t0
                result = {
                    'system': sys_name,
                    'spectral_radius': sr,
                    'seed': seed,
                    'acc_final': accs['final'],
                    'acc_transient_mean': accs['transient_mean'],
                    'acc_early_window': accs['early_window'],
                    'norms': train_data['norms'].tolist(),
                    'elapsed': elapsed,
                }
                all_results[sys_name].append(result)

                print(f"F={accs['final']:.4f} T={accs['transient_mean']:.4f} "
                      f"E={accs['early_window']:.4f} ({elapsed:.0f}s)")

    # Save
    flat_results = []
    for sys_name, rlist in all_results.items():
        flat_results.extend(rlist)
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(flat_results, f, indent=2)

    # =========================================================================
    # Plotting
    # =========================================================================

    sns.set_theme(style='whitegrid', font_scale=1.1)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    for col, (sys_name, sys_cfg) in enumerate(systems_cfg.items()):
        results = all_results[sys_name]
        sr_list = sorted(set(r['spectral_radius'] for r in results))

        # Top row: Accuracy curves
        ax = axes[0, col]
        for key, label, color, ls, lw in [
            ('acc_final', 'Final-state', '#2196F3', '--', 2),
            ('acc_transient_mean', 'Transient-mean', '#F44336', '-', 2),
            ('acc_early_window', 'Early-window', '#4CAF50', '-', 2.5),
        ]:
            means = [np.mean([r[key] for r in results if r['spectral_radius'] == sr])
                     for sr in sr_list]
            stds = [np.std([r[key] for r in results if r['spectral_radius'] == sr])
                    for sr in sr_list]
            ax.plot(sr_list, means, f'o{ls}', color=color, lw=lw, ms=6, label=label)
            ax.fill_between(sr_list, np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds), color=color, alpha=0.1)

        ax.set_xlabel('Spectral Radius ρ(W)')
        ax.set_ylabel('Test Accuracy')
        ax.set_title(f'{sys_cfg["label"]}')
        ax.legend(fontsize=8, loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=1.0, color='gray', ls='--', alpha=0.4)

        # Bottom row: Early-window advantage
        ax = axes[1, col]
        gap = [np.mean([r['acc_early_window'] - r['acc_final']
                        for r in results if r['spectral_radius'] == sr])
               for sr in sr_list]
        ax.plot(sr_list, gap, 'o-', color=sys_cfg['color'], lw=2.5, ms=7)
        ax.fill_between(sr_list, 0, gap, color=sys_cfg['color'], alpha=0.15)
        ax.axhline(y=0, color='black', lw=0.8)
        ax.set_xlabel('Spectral Radius ρ(W)')
        ax.set_ylabel('Early − Final Accuracy')
        ax.set_title(f'{sys_cfg["label"]}: Grace Period Advantage')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Cross-System Replication: The Pre-Collapse Grace Period',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'a5_cross_system.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {path}")

    # Summary table
    print(f"\n{'='*90}")
    print("CROSS-SYSTEM SUMMARY")
    print(f"{'='*90}")
    for sys_name, sys_cfg in systems_cfg.items():
        results = all_results[sys_name]
        sr_list = sorted(set(r['spectral_radius'] for r in results))
        print(f"\n  {sys_cfg['label']}")
        print(f"  {'SR':>6} | {'Final':>7} | {'Trans':>7} | {'Early':>7} | {'Gap':>7}")
        print(f"  {'-'*45}")
        for sr in sr_list:
            runs = [r for r in results if r['spectral_radius'] == sr]
            f = np.mean([r['acc_final'] for r in runs])
            t = np.mean([r['acc_transient_mean'] for r in runs])
            e = np.mean([r['acc_early_window'] for r in runs])
            print(f"  {sr:>6.2f} | {f:>7.4f} | {t:>7.4f} | {e:>7.4f} | {e-f:>+7.4f}")

    print(f"\nAll outputs saved to: {RESULTS_DIR}")
    print("Done.")


if __name__ == '__main__':
    main()
