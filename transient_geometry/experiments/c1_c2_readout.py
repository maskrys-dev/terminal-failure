"""
Experiments C1 + C2: Readout comparison and adaptive stopping.

C1: Compare final-state, transient-mean, early-window, best-timestep oracle
C2: Compare fixed-time vs adaptive stopping via norm-growth rule

Usage:
    python -m transient_geometry.experiments.c1_c2_readout
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


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    'input_dim': 784,
    'hidden_dim': 128,
    'rollout_steps': 40,
    'modrelu_bias': -0.5,
    'noise_std': 0.0,

    # Sweep — anchors + dense transition
    'spectral_radii': [0.5, 1.0, 1.2, 1.30, 1.35, 1.40, 1.45, 1.50, 1.60],
    'num_seeds': 3,

    # Data
    'batch_size': 512,
    'train_subset': 5000,

    # Readout
    'early_K': 10,

    # Adaptive stopping
    'norm_growth_thresholds': [1.5, 2.0, 3.0, 5.0],

    # Output
    'results_dir': os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), 'results', 'c1_c2'),
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
    n_train = config.get('train_subset', len(train_ds))
    if n_train < len(train_ds):
        idx = rng.permutation(len(train_ds))[:n_train]
        train_ds = torch.utils.data.Subset(train_ds, idx)
    # Also subsample test for speed
    n_test = config.get('test_subset', 5000)
    if n_test < len(test_ds):
        idx = rng.permutation(len(test_ds))[:n_test]
        test_ds = torch.utils.data.Subset(test_ds, idx)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=False)
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)
    return train_ld, test_ld


# =============================================================================
# Feature extraction — lean approach
# =============================================================================

def extract_main_readouts(system, loader, T, K, device):
    """Fast: final + transient-mean + early-window via forward_all_readouts."""
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
        # Don't store per_timestep — too much memory
    return {
        'final': np.concatenate(all_final),
        'transient_mean': np.concatenate(all_trans),
        'early_window': np.concatenate(all_early),
        'norms': norm_sum / n_batches,
        'labels': np.concatenate(all_labels),
    }


def extract_at_timesteps(system, loader, T, timesteps, device):
    """Extract features at multiple timesteps in ONE forward pass per batch.
    
    Returns dict: {t: (features, labels)} for each t in timesteps.
    """
    from transient_geometry.probes import extract_timestep_features
    ts_set = set(timesteps)
    accum = {t: ([], []) for t in timesteps}
    
    for xb, yb in loader:
        xb = xb.to(device)
        traj = system(xb, T)  # [T, batch, d]
        lab = yb.numpy()
        for t in timesteps:
            if t < T:
                feats = extract_timestep_features(traj, t)
                accum[t][0].append(feats)
                accum[t][1].append(lab)
    
    return {t: (np.concatenate(accum[t][0]), np.concatenate(accum[t][1]))
            for t in timesteps if accum[t][0]}


# =============================================================================
# Adaptive stopping rule
# =============================================================================

def adaptive_stop_time(norms, threshold):
    """
    Find the first timestep where the norm exceeds threshold × initial norm.
    Returns the last "safe" timestep (0-indexed).
    """
    ratio = norms / max(norms[0], 1e-10)
    exceeded = np.where(ratio > threshold)[0]
    if len(exceeded) == 0:
        return len(norms) - 1
    return max(0, exceeded[0] - 1)


# =============================================================================
# Main
# =============================================================================

def run_c1_c2(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_ld, test_ld = get_data(config)
    print(f"Train: {len(train_ld.dataset)} | Test: {len(test_ld.dataset)}")

    T = config['rollout_steps']
    K = config['early_K']
    total = len(config['spectral_radii']) * config['num_seeds']
    # Timesteps to probe for best-timestep and adaptive
    probe_ts = sorted(set([0, 1, 4, 9, 14, 19, 24, 29, 34, 39]))

    print(f"\n{'='*70}")
    print(f"C1+C2: {len(config['spectral_radii'])} SR × {config['num_seeds']} seeds = {total} runs")
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

            # Step 1: Get main readouts + norms (fast path, no full traj stored)
            train_data = extract_main_readouts(system, train_ld, T, K, device)
            test_data = extract_main_readouts(system, test_ld, T, K, device)

            # C1: Main readout probes
            readout_accs = {}
            for name in ['final', 'transient_mean', 'early_window']:
                probe = LinearProbe()
                probe.fit(train_data[name], train_data['labels'])
                readout_accs[name] = probe.score(test_data[name], test_data['labels'])

            # Step 2: Determine which timesteps we need for best-t + adaptive
            adaptive_stop_ts = set()
            for thresh in config['norm_growth_thresholds']:
                st = adaptive_stop_time(train_data['norms'], thresh)
                adaptive_stop_ts.add(st)
            
            all_needed_ts = sorted(set(probe_ts) | adaptive_stop_ts)

            # Step 3: Probe at needed timesteps (ONE forward pass per dataset)
            all_needed_ts = sorted(t for t in (set(probe_ts) | adaptive_stop_ts) if t < T)
            train_ts = extract_at_timesteps(system, train_ld, T, all_needed_ts, device)
            test_ts = extract_at_timesteps(system, test_ld, T, all_needed_ts, device)
            
            t_accs = {}
            for t in all_needed_ts:
                if t in train_ts and t in test_ts:
                    probe = LinearProbe()
                    probe.fit(train_ts[t][0], train_ts[t][1])
                    t_accs[t] = probe.score(test_ts[t][0], test_ts[t][1])

            # Best-timestep oracle
            best_t_idx = max((t for t in probe_ts if t < T), key=lambda t: t_accs.get(t, 0))
            best_t_acc = t_accs.get(best_t_idx, 0)
            readout_accs['best_timestep'] = best_t_acc

            # C2: Adaptive stopping
            adaptive_accs = {}
            for thresh in config['norm_growth_thresholds']:
                stop_t = adaptive_stop_time(train_data['norms'], thresh)
                acc = t_accs.get(stop_t, readout_accs['final'])
                adaptive_accs[f'adaptive_{thresh}'] = {
                    'accuracy': acc,
                    'stop_time': stop_t + 1,
                }

            elapsed = time.time() - t0

            result = {
                'spectral_radius': sr,
                'seed': seed,
                'acc_final': readout_accs['final'],
                'acc_transient_mean': readout_accs['transient_mean'],
                'acc_early_window': readout_accs['early_window'],
                'acc_best_timestep': readout_accs['best_timestep'],
                'best_t': best_t_idx + 1,
                'adaptive': adaptive_accs,
                'norms': train_data['norms'].tolist(),
                'elapsed': elapsed,
            }
            results.append(result)

            best_adap = max(adaptive_accs.items(), key=lambda x: x[1]['accuracy'])
            print(f"F={readout_accs['final']:.4f} T={readout_accs['transient_mean']:.4f} "
                  f"E={readout_accs['early_window']:.4f} B={best_t_acc:.4f}@t={best_t_idx+1} "
                  f"A={best_adap[1]['accuracy']:.4f}@t={best_adap[1]['stop_time']} "
                  f"({elapsed:.0f}s)")

    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_c1(results, output_dir):
    """C1: Readout comparison across regimes."""
    sns.set_theme(style='whitegrid', font_scale=1.2)

    spectral_radii = sorted(set(r['spectral_radius'] for r in results))

    # Aggregate by SR
    metrics = {sr: {'final': [], 'trans': [], 'early': [], 'best': []}
               for sr in spectral_radii}
    for r in results:
        sr = r['spectral_radius']
        metrics[sr]['final'].append(r['acc_final'])
        metrics[sr]['trans'].append(r['acc_transient_mean'])
        metrics[sr]['early'].append(r['acc_early_window'])
        metrics[sr]['best'].append(r['acc_best_timestep'])

    srs = np.array(spectral_radii)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: All readout strategies
    ax = axes[0]
    styles = [
        ('final', 'Final-state (z_T)', '#2196F3', 'o'),
        ('trans', 'Transient-mean', '#F44336', 's'),
        ('early', 'Early-window (K=10)', '#4CAF50', '^'),
        ('best', 'Best-timestep oracle', '#FF9800', 'D'),
    ]
    for key, label, color, marker in styles:
        means = [np.mean(metrics[sr][key]) for sr in spectral_radii]
        stds = [np.std(metrics[sr][key]) for sr in spectral_radii]
        ax.plot(srs, means, f'{marker}-', color=color, lw=2, ms=7, label=label)
        ax.fill_between(srs, np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds), color=color, alpha=0.1)
    ax.axvline(x=1.0, color='gray', ls='--', alpha=0.4)
    ax.set_xlabel('Spectral Radius ρ(W)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('C1: Readout Strategy Comparison')
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)

    # Panel 2: Advantage over final-state
    ax = axes[1]
    for key, label, color, marker in styles[1:]:
        gaps = [np.mean(metrics[sr][key]) - np.mean(metrics[sr]['final'])
                for sr in spectral_radii]
        ax.plot(srs, gaps, f'{marker}-', color=color, lw=2, ms=7, label=label)
    ax.axhline(y=0, color='black', lw=0.8)
    ax.set_xlabel('Spectral Radius ρ(W)')
    ax.set_ylabel('Accuracy − Final-state Accuracy')
    ax.set_title('Advantage Over Final-State Readout')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'c1_readout_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nC1 plot saved: {path}")


def plot_c2(results, config, output_dir):
    """C2: Adaptive stopping comparison."""
    sns.set_theme(style='whitegrid', font_scale=1.2)

    spectral_radii = sorted(set(r['spectral_radius'] for r in results))
    thresholds = config['norm_growth_thresholds']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Accuracy of adaptive vs baselines
    ax = axes[0]

    # Final-state baseline
    f_means = [np.mean([r['acc_final'] for r in results if r['spectral_radius'] == sr])
               for sr in spectral_radii]
    ax.plot(spectral_radii, f_means, 'o--', color='#2196F3', lw=2, ms=6,
            label='Final-state (t=40)', alpha=0.7)

    # Early-window baseline
    e_means = [np.mean([r['acc_early_window'] for r in results if r['spectral_radius'] == sr])
               for sr in spectral_radii]
    ax.plot(spectral_radii, e_means, '^--', color='#4CAF50', lw=2, ms=6,
            label='Early-window (t=1-10)', alpha=0.7)

    # Adaptive rules
    colors_adap = ['#FF5722', '#E91E63', '#9C27B0', '#673AB7']
    for i, thresh in enumerate(thresholds):
        key = f'adaptive_{thresh}'
        a_means = [np.mean([r['adaptive'][key]['accuracy']
                            for r in results if r['spectral_radius'] == sr])
                   for sr in spectral_radii]
        ax.plot(spectral_radii, a_means, 's-', color=colors_adap[i], lw=2, ms=6,
                label=f'Adaptive (×{thresh})', alpha=0.85)

    ax.axvline(x=1.0, color='gray', ls='--', alpha=0.4)
    ax.set_xlabel('Spectral Radius ρ(W)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('C2: Adaptive Stopping vs Baselines')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)

    # Panel 2: When does adaptive stop?
    ax = axes[1]
    for i, thresh in enumerate(thresholds):
        key = f'adaptive_{thresh}'
        stop_means = [np.mean([r['adaptive'][key]['stop_time']
                               for r in results if r['spectral_radius'] == sr])
                      for sr in spectral_radii]
        ax.plot(spectral_radii, stop_means, 's-', color=colors_adap[i], lw=2, ms=6,
                label=f'Adaptive (×{thresh})')
    ax.axhline(y=40, color='gray', ls=':', alpha=0.5, label='T=40 (full)')
    ax.set_xlabel('Spectral Radius ρ(W)')
    ax.set_ylabel('Stop Time (timestep)')
    ax.set_title('Adaptive Stop Time by Regime')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'c2_adaptive_stopping.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"C2 plot saved: {path}")


def plot_c3_extended_range(results, output_dir):
    """C3: Extended usable operating range."""
    sns.set_theme(style='whitegrid', font_scale=1.2)

    spectral_radii = sorted(set(r['spectral_radius'] for r in results))
    threshold = 0.75

    strategies = {
        'Final-state': lambda r: r['acc_final'],
        'Transient-mean': lambda r: r['acc_transient_mean'],
        'Early-window': lambda r: r['acc_early_window'],
        'Best adaptive': lambda r: max(v['accuracy'] for v in r['adaptive'].values()),
    }
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF5722']

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for (name, fn), color in zip(strategies.items(), colors):
        means = [np.mean([fn(r) for r in results if r['spectral_radius'] == sr])
                 for sr in spectral_radii]
        ax.plot(spectral_radii, means, 'o-', color=color, lw=2.5, ms=7, label=name)

    ax.axhline(y=threshold, color='black', ls=':', lw=1.5, label=f'Usable threshold ({threshold})')
    ax.axvline(x=1.0, color='gray', ls='--', alpha=0.4)
    ax.set_xlabel('Spectral Radius ρ(W)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('C3: Extended Usable Operating Range')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'c3_extended_range.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"C3 plot saved: {path}")


# =============================================================================
# Summary
# =============================================================================

def print_summary(results, config):
    """Print C1+C2 summary table."""
    spectral_radii = sorted(set(r['spectral_radius'] for r in results))

    print(f"\n{'='*90}")
    print("C1: READOUT COMPARISON")
    print(f"{'='*90}")
    print(f"\n{'SR':>6} | {'Final':>7} | {'Trans':>7} | {'Early':>7} | "
          f"{'Best':>7} | {'Best_t':>6} | {'Winner':>12}")
    print("-" * 75)

    for sr in spectral_radii:
        runs = [r for r in results if r['spectral_radius'] == sr]
        f = np.mean([r['acc_final'] for r in runs])
        t = np.mean([r['acc_transient_mean'] for r in runs])
        e = np.mean([r['acc_early_window'] for r in runs])
        b = np.mean([r['acc_best_timestep'] for r in runs])
        bt = int(np.round(np.mean([r['best_t'] for r in runs])))

        vals = {'Final': f, 'Trans': t, 'Early': e, 'Best': b}
        winner = max(vals, key=vals.get)

        print(f"{sr:>6.2f} | {f:>7.4f} | {t:>7.4f} | {e:>7.4f} | "
              f"{b:>7.4f} | {'t='+str(bt):>6} | {winner:>12}")

    print(f"\n{'='*90}")
    print("C2: ADAPTIVE STOPPING")
    print(f"{'='*90}")
    print(f"\n{'SR':>6} | {'Final':>7}", end="")
    for thresh in config['norm_growth_thresholds']:
        print(f" | {'×'+str(thresh)+' acc':>9} {'@t':>4}", end="")
    print()
    print("-" * (16 + 16 * len(config['norm_growth_thresholds'])))

    for sr in spectral_radii:
        runs = [r for r in results if r['spectral_radius'] == sr]
        f = np.mean([r['acc_final'] for r in runs])
        print(f"{sr:>6.2f} | {f:>7.4f}", end="")
        for thresh in config['norm_growth_thresholds']:
            key = f'adaptive_{thresh}'
            acc = np.mean([r['adaptive'][key]['accuracy'] for r in runs])
            st = int(np.round(np.mean([r['adaptive'][key]['stop_time'] for r in runs])))
            print(f" | {acc:>9.4f} {'t='+str(st):>4}", end="")
        print()


# =============================================================================
# Entry point
# =============================================================================

def main():
    config = CONFIG.copy()
    os.makedirs(config['results_dir'], exist_ok=True)

    with open(os.path.join(config['results_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)

    results = run_c1_c2(config)

    with open(os.path.join(config['results_dir'], 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist()
                  if hasattr(x, 'tolist') else x)

    plot_c1(results, config['results_dir'])
    plot_c2(results, config, config['results_dir'])
    plot_c3_extended_range(results, config['results_dir'])
    print_summary(results, config)

    print(f"\nAll outputs saved to: {config['results_dir']}")
    print("Done.")


if __name__ == '__main__':
    main()
