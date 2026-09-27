"""
Refined sweep: densify the transition region and measure the transient grace period.

Key additions over pilot_sweep:
    1. Dense spectral radius grid between 1.2–1.5
    2. Best-timestep accuracy: max_t probe_accuracy(t)  
    3. Early-window accuracy: mean probe accuracy over t=1..K
    4. Collapse time: first t where probe accuracy < threshold
    5. Short-horizon amplification (first 10 steps only)

Usage:
    python -m transient_geometry.refined_sweep
"""

import os
import sys
import json
import time

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transient_geometry.system import ComplexDynamicalSystem, compute_empirical_regime_descriptors
from transient_geometry.probes import LinearProbe, extract_timestep_features


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # System
    'input_dim': 784,
    'hidden_dim': 128,
    'rollout_steps': 40,
    'modrelu_bias': -0.5,
    'noise_std': 0.0,

    # Sweep — dense in the transition region
    'spectral_radii': [0.5, 0.8, 1.0, 1.2, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.60],
    'num_seeds': 5,

    # Data
    'dataset': 'MNIST',
    'batch_size': 512,
    'train_subset': 10000,

    # Grace period analysis
    'collapse_threshold': 0.70,     # Accuracy below this = "collapsed"
    'early_window': 10,             # First K timesteps for early-window metric
    'probe_timesteps': list(range(0, 40)),  # Probe at all timesteps for key SRs

    # Separability curves for these SRs (full t=1..40)
    'separability_radii': [0.5, 1.0, 1.25, 1.35, 1.45, 1.50],
    'separability_seed': 0,

    # Regime descriptors
    'perturbation_epsilon': 0.01,
    'regime_descriptor_batch': 512,

    # Output  
    'results_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', 'refined'),
}


# =============================================================================
# Data loading
# =============================================================================

def get_data_loaders(config):
    """Load MNIST with flatten transform. Subsamples training set."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])

    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    n_train = config.get('train_subset', len(train_dataset))
    if n_train < len(train_dataset):
        indices = np.random.RandomState(0).permutation(len(train_dataset))[:n_train]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    return train_loader, test_loader


# =============================================================================
# Fast feature extraction (final + transient-mean only)
# =============================================================================

def extract_features_fast(system, loader, T, device):
    """Use forward_features for fast final + transient-mean extraction."""
    all_final, all_transient, all_labels = [], [], []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        final_f, trans_f = system.forward_features(x_batch, T)
        all_final.append(final_f)
        all_transient.append(trans_f)
        all_labels.append(y_batch.numpy())
    return (np.concatenate(all_final), np.concatenate(all_transient),
            np.concatenate(all_labels))


# =============================================================================
# Full trajectory extraction (for separability-over-time)
# =============================================================================

def collect_trajectories(system, loader, T, device):
    """Collect full trajectories on CPU. Returns [T, N, d] complex tensor + labels."""
    trajs, labels = [], []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        traj = system(x_batch, T)  # [T, batch, d]
        trajs.append(traj.cpu())
        labels.append(y_batch.numpy())
    return torch.cat(trajs, dim=1), np.concatenate(labels)


def compute_separability_curve(train_traj, train_labels, test_traj, test_labels, T):
    """Probe accuracy at each timestep. Returns [T] array."""
    accuracies = np.zeros(T)
    for t in range(T):
        train_feats = extract_timestep_features(train_traj, t)
        test_feats = extract_timestep_features(test_traj, t)
        probe = LinearProbe()
        probe.fit(train_feats, train_labels)
        accuracies[t] = probe.score(test_feats, test_labels)
    return accuracies


def extract_grace_period_metrics(accuracies, early_window, collapse_threshold):
    """
    From a separability curve, extract:
        - best_timestep: argmax of accuracy
        - best_accuracy: max accuracy
        - early_window_accuracy: mean accuracy over first K steps  
        - collapse_time: first t where accuracy < threshold (None if never)
        - grace_period: collapse_time - 1 (or T if no collapse)
    """
    best_t = int(np.argmax(accuracies))
    best_acc = float(accuracies[best_t])
    early_acc = float(np.mean(accuracies[:early_window]))

    below = np.where(accuracies < collapse_threshold)[0]
    collapse_t = int(below[0]) if len(below) > 0 else len(accuracies)
    grace_period = collapse_t  # Number of usable timesteps

    return {
        'best_timestep': best_t + 1,  # 1-indexed
        'best_accuracy': best_acc,
        'early_window_accuracy': early_acc,
        'collapse_time': collapse_t + 1 if collapse_t < len(accuracies) else None,
        'grace_period': grace_period,
    }


# =============================================================================
# Short-horizon amplification
# =============================================================================

def compute_short_horizon_amplification(system, x_batch, T_short=10, epsilon=0.01):
    """Amplification measured only over first T_short steps."""
    traj = system(x_batch, T_short)
    perturbation = torch.randn_like(x_batch) * epsilon
    traj_pert = system(x_batch + perturbation, T_short)

    state_diff = traj[-1] - traj_pert[-1]
    state_divergence = torch.abs(state_diff).mean(dim=-1)
    input_pert_mag = torch.abs(perturbation).mean(dim=-1)
    amp = (state_divergence / input_pert_mag.clamp(min=1e-8)).mean().item()
    return amp


# =============================================================================
# Main sweep
# =============================================================================

def run_refined_sweep(config):
    """Run refined sweep with grace period metrics."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_loader, test_loader = get_data_loaders(config)
    print(f"Train: {len(train_loader.dataset)} | Test: {len(test_loader.dataset)}")

    T = config['rollout_steps']
    total = len(config['spectral_radii']) * config['num_seeds']

    print(f"\n{'='*70}")
    print(f"REFINED SWEEP: {len(config['spectral_radii'])} spectral radii × "
          f"{config['num_seeds']} seeds = {total} runs")
    print(f"{'='*70}\n")

    results = []
    run_idx = 0

    for sr in config['spectral_radii']:
        for seed in range(config['num_seeds']):
            run_idx += 1
            t0 = time.time()
            print(f"[{run_idx}/{total}] ρ={sr:.2f}, seed={seed} ... ", end="", flush=True)

            system = ComplexDynamicalSystem(
                input_dim=config['input_dim'],
                hidden_dim=config['hidden_dim'],
                spectral_radius=sr,
                noise_std=config['noise_std'],
                modrelu_bias=config['modrelu_bias'],
                seed=seed,
            ).to(device)

            # --- Fast features for main metrics ---
            train_final, train_trans, train_labels = extract_features_fast(
                system, train_loader, T, device)
            test_final, test_trans, test_labels = extract_features_fast(
                system, test_loader, T, device)

            probe_f = LinearProbe()
            probe_f.fit(train_final, train_labels)
            acc_final = probe_f.score(test_final, test_labels)

            probe_t = LinearProbe()
            probe_t.fit(train_trans, train_labels)
            acc_transient = probe_t.score(test_trans, test_labels)

            # --- Regime descriptors ---
            regime_x = []
            count = 0
            for xb, _ in train_loader:
                regime_x.append(xb)
                count += xb.shape[0]
                if count >= config['regime_descriptor_batch']:
                    break
            regime_x = torch.cat(regime_x)[:config['regime_descriptor_batch']].to(device)

            regime = compute_empirical_regime_descriptors(system, regime_x, T,
                                                          config['perturbation_epsilon'])

            # --- Short-horizon amplification ---
            short_amp = compute_short_horizon_amplification(
                system, regime_x, T_short=10, epsilon=config['perturbation_epsilon'])

            elapsed = time.time() - t0
            gap = acc_transient - acc_final

            result = {
                'spectral_radius': sr,
                'seed': seed,
                'acc_final_state': acc_final,
                'acc_transient_mean': acc_transient,
                'gap': gap,
                'perturbation_amplification': regime['perturbation_amplification'],
                'short_horizon_amplification': short_amp,
                'mean_final_norm': regime['mean_final_norm'],
                'phase_coherence': regime['phase_coherence'],
                'elapsed_seconds': elapsed,
            }
            results.append(result)

            print(f"final={acc_final:.4f} trans={acc_transient:.4f} "
                  f"gap={gap:+.4f} amp_full={regime['perturbation_amplification']:.1f} "
                  f"amp_10={short_amp:.2f} ({elapsed:.1f}s)")

    return results


# =============================================================================
# Separability analysis with grace period
# =============================================================================

def run_separability_analysis(config, device):
    """Full separability-over-time for selected spectral radii."""
    print(f"\n{'='*70}")
    print("SEPARABILITY ANALYSIS: Grace period measurement")
    print(f"{'='*70}\n")

    train_loader, test_loader = get_data_loaders(config)
    T = config['rollout_steps']
    sep_results = {}

    for sr in config['separability_radii']:
        seed = config['separability_seed']
        print(f"  ρ={sr:.2f}, seed={seed} ... ", end="", flush=True)
        t0 = time.time()

        system = ComplexDynamicalSystem(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            spectral_radius=sr,
            noise_std=config['noise_std'],
            modrelu_bias=config['modrelu_bias'],
            seed=seed,
        ).to(device)

        train_traj, train_labels = collect_trajectories(system, train_loader, T, device)
        test_traj, test_labels = collect_trajectories(system, test_loader, T, device)

        accs = compute_separability_curve(train_traj, train_labels, test_traj, test_labels, T)

        metrics = extract_grace_period_metrics(
            accs, config['early_window'], config['collapse_threshold'])

        elapsed = time.time() - t0
        print(f"best={metrics['best_accuracy']:.4f}@t={metrics['best_timestep']} "
              f"early={metrics['early_window_accuracy']:.4f} "
              f"grace={metrics['grace_period']} "
              f"collapse={metrics['collapse_time']} ({elapsed:.1f}s)")

        sep_results[f"{sr:.2f}"] = {
            'spectral_radius': sr,
            'accuracies': accs.tolist(),
            'metrics': metrics,
        }

    return sep_results


# =============================================================================
# Plotting
# =============================================================================

def plot_refined_phase_diagram(results, output_dir):
    """Phase diagram with dense transition region."""
    sns.set_theme(style='whitegrid', font_scale=1.2)

    spectral_radii = sorted(set(r['spectral_radius'] for r in results))
    acc_f = {sr: [] for sr in spectral_radii}
    acc_t = {sr: [] for sr in spectral_radii}
    for r in results:
        acc_f[r['spectral_radius']].append(r['acc_final_state'])
        acc_t[r['spectral_radius']].append(r['acc_transient_mean'])

    srs = np.array(spectral_radii)
    f_mean = np.array([np.mean(acc_f[sr]) for sr in spectral_radii])
    f_std = np.array([np.std(acc_f[sr]) for sr in spectral_radii])
    t_mean = np.array([np.mean(acc_t[sr]) for sr in spectral_radii])
    t_std = np.array([np.std(acc_t[sr]) for sr in spectral_radii])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: Phase diagram ---
    ax = axes[0, 0]
    ax.plot(srs, f_mean, 'o-', color='#2196F3', lw=2, ms=6, label='Final-state', zorder=3)
    ax.fill_between(srs, f_mean - f_std, f_mean + f_std, color='#2196F3', alpha=0.15)
    ax.plot(srs, t_mean, 's-', color='#F44336', lw=2, ms=6, label='Transient-mean', zorder=3)
    ax.fill_between(srs, t_mean - t_std, t_mean + t_std, color='#F44336', alpha=0.15)
    ax.axvline(x=1.0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Spectral Radius ρ(W)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Refined Phase Diagram')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Gap ---
    ax = axes[0, 1]
    gap = t_mean - f_mean
    colors = ['#4CAF50' if g > 0.005 else '#FF9800' if g > 0 else '#F44336' for g in gap]
    ax.bar(srs, gap, width=0.03, color=colors, alpha=0.85, zorder=3)
    ax.axhline(y=0, color='black', lw=0.8)
    ax.set_xlabel('Spectral Radius ρ(W)')
    ax.set_ylabel('Transient − Final Accuracy')
    ax.set_title('Transient Advantage (Grace Period Signal)')
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Amplification comparison ---
    ax = axes[1, 0]
    amp_full = {sr: [] for sr in spectral_radii}
    amp_short = {sr: [] for sr in spectral_radii}
    for r in results:
        amp_full[r['spectral_radius']].append(r['perturbation_amplification'])
        amp_short[r['spectral_radius']].append(r['short_horizon_amplification'])

    af = np.array([np.mean(amp_full[sr]) for sr in spectral_radii])
    a10 = np.array([np.mean(amp_short[sr]) for sr in spectral_radii])
    ax.plot(srs, af, 'D-', color='#9C27B0', lw=2, ms=5, label='Full (T=40)')
    ax.plot(srs, a10, 'o-', color='#00BCD4', lw=2, ms=5, label='Short (T=10)')
    ax.set_xlabel('Spectral Radius ρ(W)')
    ax.set_ylabel('Perturbation Amplification')
    ax.set_title('Full vs Short-horizon Amplification')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Phase coherence ---
    ax = axes[1, 1]
    coh = {sr: [] for sr in spectral_radii}
    for r in results:
        coh[r['spectral_radius']].append(r['phase_coherence'])
    c_mean = np.array([np.mean(coh[sr]) for sr in spectral_radii])
    c_std = np.array([np.std(coh[sr]) for sr in spectral_radii])
    ax.plot(srs, c_mean, '^-', color='#FF5722', lw=2, ms=6)
    ax.fill_between(srs, c_mean - c_std, c_mean + c_std, color='#FF5722', alpha=0.15)
    ax.set_xlabel('Spectral Radius ρ(W)')
    ax.set_ylabel('Phase Coherence')
    ax.set_title('Phase Coherence vs Stability')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'refined_phase_diagram.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPhase diagram saved: {path}")


def plot_grace_period(sep_results, output_dir, collapse_threshold):
    """Plot separability curves showing the grace period."""
    sns.set_theme(style='whitegrid', font_scale=1.2)

    # Color map: blue → red for increasing SR
    srs_sorted = sorted(sep_results.keys(), key=lambda k: float(k))
    n = len(srs_sorted)
    cmap = plt.cm.coolwarm
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Panel 1: Separability curves ---
    ax = axes[0]
    for i, key in enumerate(srs_sorted):
        data = sep_results[key]
        accs = data['accuracies']
        sr = data['spectral_radius']
        ax.plot(range(1, len(accs) + 1), accs, '-', lw=2, color=colors[i],
                label=f'ρ={sr:.2f}')

    ax.axhline(y=collapse_threshold, color='gray', ls=':', lw=1.5,
               label=f'Collapse threshold ({collapse_threshold})')
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('Linear Probe Accuracy')
    ax.set_title('Separability Over Time: The Grace Period')
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Grace period length ---
    ax = axes[1]
    sr_vals = []
    gp_vals = []
    for key in srs_sorted:
        data = sep_results[key]
        sr_vals.append(data['spectral_radius'])
        gp_vals.append(data['metrics']['grace_period'])

    ax.bar(range(len(sr_vals)), gp_vals, color=[colors[i] for i in range(len(sr_vals))],
           alpha=0.85)
    ax.set_xticks(range(len(sr_vals)))
    ax.set_xticklabels([f'{s:.2f}' for s in sr_vals], rotation=45)
    ax.set_xlabel('Spectral Radius ρ(W)')
    ax.set_ylabel('Grace Period (timesteps before collapse)')
    ax.set_title(f'Grace Period Length (threshold={collapse_threshold})')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(output_dir, 'grace_period_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grace period plot saved: {path}")


# =============================================================================
# Summary table
# =============================================================================

def print_summary(results, sep_results):
    """Print the refined summary with grace period metrics."""
    spectral_radii = sorted(set(r['spectral_radius'] for r in results))

    agg = {sr: {'final': [], 'trans': [], 'amp': [], 'amp10': [], 'coh': []}
           for sr in spectral_radii}
    for r in results:
        sr = r['spectral_radius']
        agg[sr]['final'].append(r['acc_final_state'])
        agg[sr]['trans'].append(r['acc_transient_mean'])
        agg[sr]['amp'].append(r['perturbation_amplification'])
        agg[sr]['amp10'].append(r['short_horizon_amplification'])
        agg[sr]['coh'].append(r['phase_coherence'])

    print(f"\n{'='*80}")
    print("REFINED SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'SR':>6} | {'Final':>7} | {'Trans':>7} | {'Gap':>7} | "
          f"{'Amp40':>7} | {'Amp10':>7} | {'Coh':>6} | {'Grace':>6}")
    print("-" * 72)

    for sr in spectral_radii:
        f_m = np.mean(agg[sr]['final'])
        t_m = np.mean(agg[sr]['trans'])
        gap = t_m - f_m
        a40 = np.mean(agg[sr]['amp'])
        a10 = np.mean(agg[sr]['amp10'])
        coh = np.mean(agg[sr]['coh'])

        sr_key = f"{sr:.2f}"
        grace = sep_results.get(sr_key, {}).get('metrics', {}).get('grace_period', '—')

        print(f"{sr:>6.2f} | {f_m:>7.4f} | {t_m:>7.4f} | {gap:>+7.4f} | "
              f"{a40:>7.1f} | {a10:>7.2f} | {coh:>6.3f} | {str(grace):>6}")


# =============================================================================
# Main
# =============================================================================

def main():
    config = CONFIG.copy()
    os.makedirs(config['results_dir'], exist_ok=True)

    # Save config
    with open(os.path.join(config['results_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Main sweep ---
    results = run_refined_sweep(config)

    with open(os.path.join(config['results_dir'], 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    plot_refined_phase_diagram(results, config['results_dir'])

    # --- Separability analysis ---
    sep_results = run_separability_analysis(config, device)

    with open(os.path.join(config['results_dir'], 'separability.json'), 'w') as f:
        json.dump(sep_results, f, indent=2)

    plot_grace_period(sep_results, config['results_dir'], config['collapse_threshold'])

    # --- Summary ---
    print_summary(results, sep_results)

    print(f"\nAll outputs saved to: {config['results_dir']}")
    print("Done.")


if __name__ == '__main__':
    main()
