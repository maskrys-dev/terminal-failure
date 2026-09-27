"""
Experiment A3: Dataset robustness — FashionMNIST replication.

Runs the same C1/C2/C3 analysis on FashionMNIST to prove
the grace period regime is not MNIST-specific.

Usage:
    python -m transient_geometry.experiments.a3_fashion
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
    os.path.dirname(os.path.abspath(__file__)))), 'results', 'a3_fashion')

CONFIG = {
    'input_dim': 784,
    'hidden_dim': 128,
    'rollout_steps': 40,
    'modrelu_bias': -0.5,
    'noise_std': 0.0,

    # Same sweep as C1/C2
    'spectral_radii': [0.5, 1.0, 1.20, 1.30, 1.35, 1.40, 1.45, 1.50, 1.60],
    'num_seeds': 3,

    'batch_size': 512,
    'train_n': 5000,
    'test_n': 5000,

    'early_K': 10,
    'norm_thresholds': [1.5, 2.0, 3.0],

    # Grace period
    'grace_srs': [1.0, 1.35, 1.40, 1.45, 1.50],
}


# =============================================================================
# Data
# =============================================================================

def get_fashion_data(config):
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    root = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), 'data')
    train_ds = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    rng = np.random.RandomState(42)
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
# Feature extraction
# =============================================================================

def extract_readouts(system, loader, T, K, device):
    """Main readouts + norms in one pass."""
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


def extract_multi_timestep(system, loader, T, timesteps, device):
    """Extract features at multiple timesteps in one forward pass."""
    accum = {t: ([], []) for t in timesteps}
    for xb, yb in loader:
        xb = xb.to(device)
        traj = system(xb, T)
        lab = yb.numpy()
        for t in timesteps:
            if t < T:
                feats = extract_timestep_features(traj, t)
                accum[t][0].append(feats)
                accum[t][1].append(lab)
    return {t: (np.concatenate(accum[t][0]), np.concatenate(accum[t][1]))
            for t in timesteps if accum[t][0]}


def adaptive_stop_time(norms, threshold):
    ratio = norms / max(norms[0], 1e-10)
    exceeded = np.where(ratio > threshold)[0]
    if len(exceeded) == 0:
        return len(norms) - 1
    return max(0, exceeded[0] - 1)


# =============================================================================
# Main sweep
# =============================================================================

def run_sweep(config, device, train_ld, test_ld):
    T = config['rollout_steps']
    K = config['early_K']
    srs = config['spectral_radii']
    seeds = config['num_seeds']
    total = len(srs) * seeds
    probe_ts = sorted(set([0, 1, 4, 9, 14, 19, 24, 29, 34, 39]))

    results = []
    run_idx = 0

    for sr in srs:
        for seed in range(seeds):
            run_idx += 1
            t0 = time.time()
            print(f"[{run_idx}/{total}] ρ={sr:.2f}, seed={seed} ... ", end="", flush=True)

            system = ComplexDynamicalSystem(
                input_dim=config['input_dim'], hidden_dim=config['hidden_dim'],
                spectral_radius=sr, noise_std=config['noise_std'],
                modrelu_bias=config['modrelu_bias'], seed=seed,
            ).to(device)

            # Main readouts
            train_data = extract_readouts(system, train_ld, T, K, device)
            test_data = extract_readouts(system, test_ld, T, K, device)

            readout_accs = {}
            for name in ['final', 'transient_mean', 'early_window']:
                probe = LinearProbe()
                probe.fit(train_data[name], train_data['labels'])
                readout_accs[name] = probe.score(test_data[name], test_data['labels'])

            # Adaptive stop times
            adap_ts = set()
            for thresh in config['norm_thresholds']:
                adap_ts.add(adaptive_stop_time(train_data['norms'], thresh))
            
            needed_ts = sorted(t for t in (set(probe_ts) | adap_ts) if t < T)
            
            train_ts = extract_multi_timestep(system, train_ld, T, needed_ts, device)
            test_ts = extract_multi_timestep(system, test_ld, T, needed_ts, device)
            
            t_accs = {}
            for t in needed_ts:
                if t in train_ts and t in test_ts:
                    probe = LinearProbe()
                    probe.fit(train_ts[t][0], train_ts[t][1])
                    t_accs[t] = probe.score(test_ts[t][0], test_ts[t][1])

            # Best-timestep
            best_t = max((t for t in probe_ts if t < T), key=lambda t: t_accs.get(t, 0))
            readout_accs['best_timestep'] = t_accs.get(best_t, 0)

            # Adaptive
            adaptive = {}
            for thresh in config['norm_thresholds']:
                st = adaptive_stop_time(train_data['norms'], thresh)
                adaptive[f'adaptive_{thresh}'] = {
                    'accuracy': t_accs.get(st, readout_accs['final']),
                    'stop_time': st + 1,
                }

            elapsed = time.time() - t0
            best_adap = max(adaptive.items(), key=lambda x: x[1]['accuracy'])

            result = {
                'spectral_radius': sr, 'seed': seed,
                'acc_final': readout_accs['final'],
                'acc_transient_mean': readout_accs['transient_mean'],
                'acc_early_window': readout_accs['early_window'],
                'acc_best_timestep': readout_accs['best_timestep'],
                'best_t': best_t + 1,
                'adaptive': adaptive,
                'norms': train_data['norms'].tolist(),
                'elapsed': elapsed,
            }
            results.append(result)

            print(f"F={readout_accs['final']:.4f} T={readout_accs['transient_mean']:.4f} "
                  f"E={readout_accs['early_window']:.4f} B={readout_accs['best_timestep']:.4f}@t={best_t+1} "
                  f"A={best_adap[1]['accuracy']:.4f}@t={best_adap[1]['stop_time']} ({elapsed:.0f}s)")

    return results


# =============================================================================
# Grace period analysis
# =============================================================================

def run_grace_period(config, device, train_ld, test_ld):
    """Separability over time for key SRs."""
    T = config['rollout_steps']
    grace_results = {}
    
    for sr in config['grace_srs']:
        print(f"\n  Grace ρ={sr:.2f} ... ", end="", flush=True)
        t0 = time.time()
        
        system = ComplexDynamicalSystem(
            input_dim=config['input_dim'], hidden_dim=config['hidden_dim'],
            spectral_radius=sr, noise_std=0.0,
            modrelu_bias=config['modrelu_bias'], seed=0,
        ).to(device)
        
        # Full trajectory
        all_ts = list(range(T))
        train_ts = extract_multi_timestep(system, train_ld, T, all_ts, device)
        test_ts = extract_multi_timestep(system, test_ld, T, all_ts, device)
        
        accs = np.zeros(T)
        for t in range(T):
            probe = LinearProbe()
            probe.fit(train_ts[t][0], train_ts[t][1])
            accs[t] = probe.score(test_ts[t][0], test_ts[t][1])
        
        collapse_70 = np.where(accs < 0.60)[0]  # FashionMNIST baseline is lower
        collapse_t = int(collapse_70[0]) + 1 if len(collapse_70) > 0 else None
        
        elapsed = time.time() - t0
        print(f"best={accs.max():.4f}@t={accs.argmax()+1} "
              f"collapse@60%={'t='+str(collapse_t) if collapse_t else 'never'} ({elapsed:.0f}s)")
        
        grace_results[f"{sr:.2f}"] = {
            'sr': sr, 'accs': accs.tolist(),
            'best_t': int(accs.argmax()) + 1,
            'best_acc': float(accs.max()),
            'collapse_t': collapse_t,
        }
    
    return grace_results


# =============================================================================
# Plotting
# =============================================================================

def plot_comparison(mnist_path, fashion_results, grace_results, output_dir):
    """Side-by-side comparison: MNIST vs FashionMNIST."""
    sns.set_theme(style='whitegrid', font_scale=1.1)
    
    # Load MNIST results
    mnist_results = None
    if os.path.exists(mnist_path):
        with open(mnist_path) as f:
            mnist_results = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # --- Panel 1: C3 Extended range (FashionMNIST) ---
    ax = axes[0, 0]
    srs = sorted(set(r['spectral_radius'] for r in fashion_results))
    
    strategies = {
        'Final-state': lambda r: r['acc_final'],
        'Transient-mean': lambda r: r['acc_transient_mean'],
        'Early-window': lambda r: r['acc_early_window'],
        'Best adaptive': lambda r: max(v['accuracy'] for v in r['adaptive'].values()),
    }
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF5722']
    
    for (name, fn), color in zip(strategies.items(), colors):
        means = [np.mean([fn(r) for r in fashion_results if r['spectral_radius'] == sr])
                 for sr in srs]
        ax.plot(srs, means, 'o-', color=color, lw=2.5, ms=7, label=name)
    
    ax.axhline(y=0.65, color='black', ls=':', lw=1.5, label='Usable (65%)')
    ax.set_xlabel('Spectral Radius ρ(W)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('FashionMNIST: Extended Usable Range')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)
    
    # --- Panel 2: MNIST vs FashionMNIST final/early comparison ---
    ax = axes[0, 1]
    
    # FashionMNIST
    f_final = [np.mean([r['acc_final'] for r in fashion_results if r['spectral_radius'] == sr])
               for sr in srs]
    f_early = [np.mean([r['acc_early_window'] for r in fashion_results if r['spectral_radius'] == sr])
               for sr in srs]
    ax.plot(srs, f_final, 'o--', color='#2196F3', lw=2, ms=6, label='Fashion: Final', alpha=0.8)
    ax.plot(srs, f_early, 's-', color='#4CAF50', lw=2.5, ms=7, label='Fashion: Early-window')
    
    # MNIST (if available)
    if mnist_results:
        m_srs = sorted(set(r['spectral_radius'] for r in mnist_results))
        m_final = [np.mean([r['acc_final'] for r in mnist_results if r['spectral_radius'] == sr])
                   for sr in m_srs]
        m_early = [np.mean([r['acc_early_window'] for r in mnist_results if r['spectral_radius'] == sr])
                   for sr in m_srs]
        ax.plot(m_srs, m_final, 'o--', color='#90CAF9', lw=1.5, ms=5, label='MNIST: Final', alpha=0.6)
        ax.plot(m_srs, m_early, 's-', color='#A5D6A7', lw=1.5, ms=5, label='MNIST: Early-window', alpha=0.6)
    
    ax.set_xlabel('Spectral Radius ρ(W)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Cross-Dataset: Final vs Early-Window')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # --- Panel 3: Grace period curves (FashionMNIST) ---
    ax = axes[1, 0]
    if grace_results:
        cmap = plt.cm.coolwarm
        gkeys = sorted(grace_results.keys(), key=float)
        gcolors = [cmap(i / max(len(gkeys) - 1, 1)) for i in range(len(gkeys))]
        for i, key in enumerate(gkeys):
            d = grace_results[key]
            ax.plot(range(1, len(d['accs']) + 1), d['accs'], '-', lw=2.5,
                    color=gcolors[i], label=f"ρ={d['sr']:.2f}", alpha=0.9)
        ax.axhline(y=0.60, color='gray', ls=':', lw=1.5, label='Collapse (60%)')
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('Linear Probe Accuracy')
    ax.set_title('FashionMNIST: The Grace Period')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)
    
    # --- Panel 4: C2 Adaptive stopping (FashionMNIST) ---
    ax = axes[1, 1]
    f_means_final = [np.mean([r['acc_final'] for r in fashion_results if r['spectral_radius'] == sr])
                     for sr in srs]
    ax.plot(srs, f_means_final, 'o--', color='#2196F3', lw=2, ms=6, label='Final-state', alpha=0.7)
    
    adap_colors = ['#FF5722', '#E91E63', '#9C27B0']
    for i, thresh in enumerate([1.5, 2.0, 3.0]):
        key = f'adaptive_{thresh}'
        a_means = [np.mean([r['adaptive'][key]['accuracy']
                            for r in fashion_results if r['spectral_radius'] == sr])
                   for sr in srs]
        ax.plot(srs, a_means, 's-', color=adap_colors[i], lw=2, ms=6,
                label=f'Adaptive (×{thresh})', alpha=0.85)
    
    ax.set_xlabel('Spectral Radius ρ(W)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('FashionMNIST: Adaptive Stopping')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Dataset Robustness: FashionMNIST Replication', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'a3_fashion_results.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {path}")


def print_summary(results, config):
    srs = sorted(set(r['spectral_radius'] for r in results))
    
    print(f"\n{'='*90}")
    print("FashionMNIST: READOUT COMPARISON")
    print(f"{'='*90}")
    print(f"\n{'SR':>6} | {'Final':>7} | {'Trans':>7} | {'Early':>7} | "
          f"{'Best':>7} | {'Best_t':>6}")
    print("-" * 60)
    
    for sr in srs:
        runs = [r for r in results if r['spectral_radius'] == sr]
        f = np.mean([r['acc_final'] for r in runs])
        t = np.mean([r['acc_transient_mean'] for r in runs])
        e = np.mean([r['acc_early_window'] for r in runs])
        b = np.mean([r['acc_best_timestep'] for r in runs])
        bt = int(np.round(np.mean([r['best_t'] for r in runs])))
        print(f"{sr:>6.2f} | {f:>7.4f} | {t:>7.4f} | {e:>7.4f} | {b:>7.4f} | {'t='+str(bt):>6}")
    
    print(f"\n{'='*90}")
    print("FashionMNIST: ADAPTIVE STOPPING")
    print(f"{'='*90}")
    print(f"\n{'SR':>6} | {'Final':>7}", end="")
    for thresh in config['norm_thresholds']:
        print(f" | {'×'+str(thresh)+' acc':>9} {'@t':>4}", end="")
    print()
    print("-" * (16 + 16 * len(config['norm_thresholds'])))
    
    for sr in srs:
        runs = [r for r in results if r['spectral_radius'] == sr]
        f = np.mean([r['acc_final'] for r in runs])
        print(f"{sr:>6.2f} | {f:>7.4f}", end="")
        for thresh in config['norm_thresholds']:
            key = f'adaptive_{thresh}'
            acc = np.mean([r['adaptive'][key]['accuracy'] for r in runs])
            st = int(np.round(np.mean([r['adaptive'][key]['stop_time'] for r in runs])))
            print(f" | {acc:>9.4f} {'t='+str(st):>4}", end="")
        print()


# =============================================================================
# Entry
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Dataset: FashionMNIST\n")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    train_ld, test_ld = get_fashion_data(CONFIG)
    print(f"Train: {len(train_ld.dataset)} | Test: {len(test_ld.dataset)}")
    
    # Phase 1: Main C1/C2 sweep
    print(f"\n{'='*70}")
    print(f"C1+C2 SWEEP: {len(CONFIG['spectral_radii'])} SR × {CONFIG['num_seeds']} seeds")
    print(f"{'='*70}")
    
    results = run_sweep(CONFIG, device, train_ld, test_ld)
    
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print_summary(results, CONFIG)
    
    # Phase 2: Grace period
    print(f"\n{'='*70}")
    print("GRACE PERIOD ANALYSIS")
    print(f"{'='*70}")
    
    grace = run_grace_period(CONFIG, device, train_ld, test_ld)
    
    with open(os.path.join(RESULTS_DIR, 'grace_period.json'), 'w') as f:
        json.dump(grace, f, indent=2)
    
    # Phase 3: Comparison plots
    mnist_path = os.path.join(os.path.dirname(RESULTS_DIR), 'c1_c2', 'results.json')
    plot_comparison(mnist_path, results, grace, RESULTS_DIR)
    
    print(f"\nAll outputs saved to: {RESULTS_DIR}")
    print("Done.")


if __name__ == '__main__':
    main()
