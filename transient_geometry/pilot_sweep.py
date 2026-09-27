"""
Pilot sweep: coarse phase diagram for GO/NO-GO decision.

Sweeps spectral radius across 8 values × 5 seeds.
Compares final-state vs transient-mean readout accuracy on MNIST.
Computes empirical regime descriptors alongside nominal spectral radius.

Expected runtime: ~15-25 minutes on RTX Pro 1000 (8GB).

Usage:
    python -m transient_geometry.pilot_sweep
    
    or from the CVNNs directory:
    python transient_geometry/pilot_sweep.py
"""

import os
import sys
import json
import time
from datetime import datetime

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torchvision import datasets, transforms

# Add parent dir to path so imports work when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transient_geometry.system import ComplexDynamicalSystem, compute_empirical_regime_descriptors
from transient_geometry.probes import (
    LinearProbe,
    extract_final_state_features,
    extract_transient_mean_features,
    extract_timestep_features,
)


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # System
    'input_dim': 784,           # MNIST flattened
    'hidden_dim': 128,          # Complex hidden dimension
    'rollout_steps': 40,        # Number of timesteps T
    'modrelu_bias': -0.5,       # modReLU dead-zone bias
    'noise_std': 0.0,           # No noise for pilot

    # Sweep
    'spectral_radii': [0.2, 0.5, 0.8, 0.95, 1.0, 1.05, 1.2, 1.5],
    'num_seeds': 5,

    # Data
    'dataset': 'MNIST',
    'batch_size': 256,
    'train_subset': 10000,  # Use 10K training samples (full 60K is wasteful for pilot)

    # Regime descriptors
    'perturbation_epsilon': 0.01,
    'regime_descriptor_batch': 512,   # Subsample for speed

    # Separability over time (diagnostic)
    'separability_radii': [0.5, 1.0, 1.5],  # 3 regimes for time-resolved analysis
    'separability_seed': 0,                   # Single seed for diagnostic

    # Output
    'results_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', 'pilot'),
}


# =============================================================================
# Data loading
# =============================================================================

def get_data_loaders(config):
    """Load MNIST with flatten transform. Subsamples training set for speed."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),  # Flatten to 784
    ])

    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    train_dataset = datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform,
    )

    # Subsample training set for pilot speed
    n_train = config.get('train_subset', len(train_dataset))
    if n_train < len(train_dataset):
        indices = np.random.RandomState(0).permutation(len(train_dataset))[:n_train]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0,
    )

    return train_loader, test_loader


# =============================================================================
# Feature extraction (batched over full dataset)
# =============================================================================

def extract_all_features_fast(system, loader, T, device):
    """
    Run all data through the system using the fast path (no full trajectory).
    
    Returns
    -------
    features_final : ndarray [N, 2d]
    features_transient : ndarray [N, 2d]
    labels : ndarray [N]
    """
    all_final = []
    all_transient = []
    all_labels = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        final_feats, transient_feats = system.forward_features(x_batch, T)
        all_final.append(final_feats)
        all_transient.append(transient_feats)
        all_labels.append(y_batch.numpy())

    return (
        np.concatenate(all_final, axis=0),
        np.concatenate(all_transient, axis=0),
        np.concatenate(all_labels, axis=0),
    )


# =============================================================================
# Time-resolved separability (diagnostic)
# =============================================================================

def compute_separability_over_time(system, train_loader, test_loader, T, device):
    """
    Train a linear probe at each timestep and return accuracy curve.
    
    Returns
    -------
    accuracies : ndarray [T]
        Test accuracy at each timestep.
    """
    # Collect all trajectories (expensive in memory but T=40, d=128 is fine)
    train_trajs = []
    train_labels = []
    test_trajs = []
    test_labels = []

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        traj = system(x_batch, T)  # [T, batch, d]
        train_trajs.append(traj.cpu())
        train_labels.append(y_batch.numpy())

    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        traj = system(x_batch, T)
        test_trajs.append(traj.cpu())
        test_labels.append(y_batch.numpy())

    train_traj_full = torch.cat(train_trajs, dim=1)  # [T, N_train, d]
    test_traj_full = torch.cat(test_trajs, dim=1)    # [T, N_test, d]
    train_labels_full = np.concatenate(train_labels)
    test_labels_full = np.concatenate(test_labels)

    accuracies = np.zeros(T)
    for t in range(T):
        train_feats = extract_timestep_features(train_traj_full, t)
        test_feats = extract_timestep_features(test_traj_full, t)

        probe = LinearProbe(max_iter=500, C=0.1)  # Fast probes for diagnostic
        probe.fit(train_feats, train_labels_full)
        accuracies[t] = probe.score(test_feats, test_labels_full)

    return accuracies


# =============================================================================
# Main sweep
# =============================================================================

def run_pilot_sweep(config):
    """Run the full pilot sweep and return results."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Configuration: {json.dumps(config, indent=2, default=str)}")

    # Load data
    print("\nLoading data...")
    train_loader, test_loader = get_data_loaders(config)
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Test:  {len(test_loader.dataset)} samples")

    results = []
    T = config['rollout_steps']
    total_runs = len(config['spectral_radii']) * config['num_seeds']

    print(f"\n{'='*60}")
    print(f"PILOT SWEEP: {len(config['spectral_radii'])} spectral radii × "
          f"{config['num_seeds']} seeds = {total_runs} runs")
    print(f"{'='*60}\n")

    run_idx = 0
    for sr in config['spectral_radii']:
        for seed in range(config['num_seeds']):
            run_idx += 1
            t_start = time.time()

            print(f"[{run_idx}/{total_runs}] λ={sr:.2f}, seed={seed} ... ", end="", flush=True)

            # Create system with this (spectral_radius, seed) pair
            system = ComplexDynamicalSystem(
                input_dim=config['input_dim'],
                hidden_dim=config['hidden_dim'],
                spectral_radius=sr,
                noise_std=config['noise_std'],
                modrelu_bias=config['modrelu_bias'],
                seed=seed,
            ).to(device)

            # Extract features using fast path (no full trajectory stored)
            train_final, train_transient, train_labels = extract_all_features_fast(
                system, train_loader, T, device)
            test_final, test_transient, test_labels = extract_all_features_fast(
                system, test_loader, T, device)

            # Train and evaluate probes
            probe_final = LinearProbe()
            probe_final.fit(train_final, train_labels)
            acc_final = probe_final.score(test_final, test_labels)

            probe_transient = LinearProbe()
            probe_transient.fit(train_transient, train_labels)
            acc_transient = probe_transient.score(test_transient, test_labels)

            # Compute empirical regime descriptors (on a subsample for speed)
            regime_batch_size = min(config['regime_descriptor_batch'],
                                    len(train_loader.dataset))
            regime_x = []
            count = 0
            for x_batch, _ in train_loader:
                regime_x.append(x_batch)
                count += x_batch.shape[0]
                if count >= regime_batch_size:
                    break
            regime_x = torch.cat(regime_x, dim=0)[:regime_batch_size].to(device)

            regime = compute_empirical_regime_descriptors(
                system, regime_x, T, config['perturbation_epsilon'])

            elapsed = time.time() - t_start
            gap = acc_transient - acc_final

            result = {
                'spectral_radius': sr,
                'seed': seed,
                'acc_final_state': acc_final,
                'acc_transient_mean': acc_transient,
                'gap': gap,
                'perturbation_amplification': regime['perturbation_amplification'],
                'mean_final_norm': regime['mean_final_norm'],
                'phase_coherence': regime['phase_coherence'],
                'elapsed_seconds': elapsed,
            }
            results.append(result)

            print(f"final={acc_final:.4f} transient={acc_transient:.4f} "
                  f"gap={gap:+.4f} amp={regime['perturbation_amplification']:.2f} "
                  f"coh={regime['phase_coherence']:.3f} ({elapsed:.1f}s)")

    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_phase_diagram(results, output_dir):
    """
    Plot the killer figure: accuracy vs spectral radius.
    Two lines: final-state vs transient-mean, with CI bands.
    """
    sns.set_theme(style='whitegrid', font_scale=1.2)

    # Aggregate results
    spectral_radii = sorted(set(r['spectral_radius'] for r in results))
    
    acc_final_by_sr = {sr: [] for sr in spectral_radii}
    acc_trans_by_sr = {sr: [] for sr in spectral_radii}
    for r in results:
        acc_final_by_sr[r['spectral_radius']].append(r['acc_final_state'])
        acc_trans_by_sr[r['spectral_radius']].append(r['acc_transient_mean'])

    srs = np.array(spectral_radii)
    final_mean = np.array([np.mean(acc_final_by_sr[sr]) for sr in spectral_radii])
    final_std = np.array([np.std(acc_final_by_sr[sr]) for sr in spectral_radii])
    trans_mean = np.array([np.mean(acc_trans_by_sr[sr]) for sr in spectral_radii])
    trans_std = np.array([np.std(acc_trans_by_sr[sr]) for sr in spectral_radii])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: Main phase diagram ---
    ax = axes[0, 0]
    ax.plot(srs, final_mean, 'o-', color='#2196F3', linewidth=2, markersize=6,
            label='Final-state readout', zorder=3)
    ax.fill_between(srs, final_mean - final_std, final_mean + final_std,
                     color='#2196F3', alpha=0.15)
    ax.plot(srs, trans_mean, 's-', color='#F44336', linewidth=2, markersize=6,
            label='Transient-mean readout', zorder=3)
    ax.fill_between(srs, trans_mean - trans_std, trans_mean + trans_std,
                     color='#F44336', alpha=0.15)
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='ρ(W) = 1')
    ax.set_xlabel('Spectral Radius ρ(W)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Phase Diagram: Readout Accuracy vs Stability')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Gap (transient advantage) ---
    ax = axes[0, 1]
    gap_mean = trans_mean - final_mean
    gap_std = np.sqrt(trans_std**2 + final_std**2) / np.sqrt(2)  # approximate
    ax.bar(srs, gap_mean, width=0.06, color=['#4CAF50' if g > 0 else '#FF9800' for g in gap_mean],
           alpha=0.8, zorder=3)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Spectral Radius ρ(W)')
    ax.set_ylabel('Transient − Final Accuracy')
    ax.set_title('Transient Advantage by Regime')
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Perturbation amplification ---
    ax = axes[1, 0]
    amp_by_sr = {sr: [] for sr in spectral_radii}
    for r in results:
        amp_by_sr[r['spectral_radius']].append(r['perturbation_amplification'])
    amp_mean = np.array([np.mean(amp_by_sr[sr]) for sr in spectral_radii])
    amp_std = np.array([np.std(amp_by_sr[sr]) for sr in spectral_radii])
    ax.plot(srs, amp_mean, 'D-', color='#9C27B0', linewidth=2, markersize=6)
    ax.fill_between(srs, amp_mean - amp_std, amp_mean + amp_std,
                     color='#9C27B0', alpha=0.15)
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Spectral Radius ρ(W)')
    ax.set_ylabel('Perturbation Amplification')
    ax.set_title('Empirical Regime Descriptor: Sensitivity')
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Phase coherence ---
    ax = axes[1, 1]
    coh_by_sr = {sr: [] for sr in spectral_radii}
    for r in results:
        coh_by_sr[r['spectral_radius']].append(r['phase_coherence'])
    coh_mean = np.array([np.mean(coh_by_sr[sr]) for sr in spectral_radii])
    coh_std = np.array([np.std(coh_by_sr[sr]) for sr in spectral_radii])
    ax.plot(srs, coh_mean, '^-', color='#FF5722', linewidth=2, markersize=6)
    ax.fill_between(srs, coh_mean - coh_std, coh_mean + coh_std,
                     color='#FF5722', alpha=0.15)
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Spectral Radius ρ(W)')
    ax.set_ylabel('Phase Coherence')
    ax.set_title('Empirical Regime Descriptor: Phase Coherence')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'pilot_phase_diagram.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPhase diagram saved: {path}")
    return path


def plot_separability_over_time(sep_results, output_dir):
    """
    Plot time-resolved separability for diagnostic regimes.
    """
    sns.set_theme(style='whitegrid', font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'contractive': '#2196F3', 'near_critical': '#F44336', 'unstable': '#FF9800'}
    for label, (sr, accs) in sep_results.items():
        ax.plot(range(1, len(accs) + 1), accs, '-', linewidth=2,
                color=colors.get(label, 'gray'),
                label=f'{label} (ρ={sr:.1f})')

    ax.set_xlabel('Timestep t')
    ax.set_ylabel('Linear Probe Accuracy')
    ax.set_title('Separability Over Time by Regime')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, 'pilot_separability_over_time.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Separability plot saved: {path}")
    return path


# =============================================================================
# GO / NO-GO analysis
# =============================================================================

def analyse_go_nogo(results):
    """
    Evaluate the three GO/NO-GO gates.
    """
    spectral_radii = sorted(set(r['spectral_radius'] for r in results))

    # Aggregate
    acc_final_by_sr = {sr: [] for sr in spectral_radii}
    acc_trans_by_sr = {sr: [] for sr in spectral_radii}
    amp_by_sr = {sr: [] for sr in spectral_radii}
    coh_by_sr = {sr: [] for sr in spectral_radii}

    for r in results:
        acc_final_by_sr[r['spectral_radius']].append(r['acc_final_state'])
        acc_trans_by_sr[r['spectral_radius']].append(r['acc_transient_mean'])
        amp_by_sr[r['spectral_radius']].append(r['perturbation_amplification'])
        coh_by_sr[r['spectral_radius']].append(r['phase_coherence'])

    print("\n" + "=" * 60)
    print("GO / NO-GO ANALYSIS")
    print("=" * 60)

    # --- Summary table ---
    print(f"\n{'SR':>6} | {'Final':>8} | {'Transient':>9} | {'Gap':>7} | {'Amp':>8} | {'Coh':>6}")
    print("-" * 55)
    for sr in spectral_radii:
        f_mean = np.mean(acc_final_by_sr[sr])
        t_mean = np.mean(acc_trans_by_sr[sr])
        gap = t_mean - f_mean
        amp = np.mean(amp_by_sr[sr])
        coh = np.mean(coh_by_sr[sr])
        marker = " ◄" if sr == 1.0 else ""
        print(f"{sr:>6.2f} | {f_mean:>8.4f} | {t_mean:>9.4f} | {gap:>+7.4f} | "
              f"{amp:>8.2f} | {coh:>6.3f}{marker}")

    # --- Gate 1: Meaningful divergence between readout types ---
    print(f"\n{'─'*50}")
    print("GATE 1: Meaningful divergence between readout types?")
    gaps = []
    for sr in spectral_radii:
        f_mean = np.mean(acc_final_by_sr[sr])
        t_mean = np.mean(acc_trans_by_sr[sr])
        gaps.append(t_mean - f_mean)
    max_gap = max(gaps)
    max_gap_sr = spectral_radii[np.argmax(gaps)]
    any_significant = any(abs(g) > 0.005 for g in gaps)

    if any_significant:
        print(f"  ✅ PASS — Max gap = {max_gap:+.4f} at ρ={max_gap_sr:.2f}")
    else:
        print(f"  ❌ FAIL — Max gap = {max_gap:+.4f} (< 0.5% everywhere)")
        print("  → Consider: different nonlinearity, input injection strength, trajectory length")

    # --- Gate 2: Regime descriptors move systematically ---
    print(f"\n{'─'*50}")
    print("GATE 2: Geometric metrics move with divergence?")
    amp_values = [np.mean(amp_by_sr[sr]) for sr in spectral_radii]
    coh_values = [np.mean(coh_by_sr[sr]) for sr in spectral_radii]
    amp_range = max(amp_values) - min(amp_values)
    coh_range = max(coh_values) - min(coh_values)

    amp_moves = amp_range > 0.1
    coh_moves = coh_range > 0.01

    if amp_moves or coh_moves:
        print(f"  ✅ PASS — Amplification range: {amp_range:.2f}, "
              f"Coherence range: {coh_range:.3f}")
    else:
        print(f"  ❌ FAIL — Metrics are flat across regimes")
        print("  → System may not be sensitive enough to spectral radius changes")

    # --- Gate 3: Can we identify a critical region? ---
    print(f"\n{'─'*50}")
    print("GATE 3: Can we identify a near-critical region?")
    # Check if transient readout peaks at an intermediate SR
    trans_means = [np.mean(acc_trans_by_sr[sr]) for sr in spectral_radii]
    peak_idx = np.argmax(trans_means)
    peak_sr = spectral_radii[peak_idx]
    is_intermediate = 0 < peak_idx < len(spectral_radii) - 1

    if is_intermediate:
        print(f"  ✅ PASS — Transient readout peaks at ρ={peak_sr:.2f} "
              f"(intermediate, not endpoint)")
    else:
        edge = "minimum" if peak_idx == 0 else "maximum"
        print(f"  ⚠️  PARTIAL — Peak at ρ={peak_sr:.2f} ({edge} of sweep range)")
        print("  → Consider: extending sweep range, finer resolution near ρ=1")

    # --- Overall verdict ---
    print(f"\n{'='*60}")
    all_pass = any_significant and (amp_moves or coh_moves) and is_intermediate
    if all_pass:
        print("🟢 OVERALL: GO — All gates pass. Proceed to full sweep.")
    elif any_significant:
        print("🟡 OVERALL: CONDITIONAL GO — Signal exists. Refine parameterisation.")
    else:
        print("🔴 OVERALL: NO-GO — Investigate system design before proceeding.")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    config = CONFIG.copy()
    
    # Create output directory
    os.makedirs(config['results_dir'], exist_ok=True)

    # Save config
    config_path = os.path.join(config['results_dir'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Config saved: {config_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ─── Main sweep ───
    results = run_pilot_sweep(config)

    # Save raw results
    results_path = os.path.join(config['results_dir'], 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")

    # ─── Phase diagram ───
    plot_phase_diagram(results, config['results_dir'])

    # ─── Separability over time (diagnostic) ───
    print("\nComputing time-resolved separability (diagnostic)...")
    train_loader, test_loader = get_data_loaders(config)
    sep_labels = ['contractive', 'near_critical', 'unstable']
    sep_results = {}

    for label, sr in zip(sep_labels, config['separability_radii']):
        print(f"  {label} (ρ={sr}) ...", end=" ", flush=True)
        system = ComplexDynamicalSystem(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            spectral_radius=sr,
            noise_std=config['noise_std'],
            modrelu_bias=config['modrelu_bias'],
            seed=config['separability_seed'],
        ).to(device)

        accs = compute_separability_over_time(
            system, train_loader, test_loader, config['rollout_steps'], device)
        sep_results[label] = (sr, accs)
        print(f"peak={accs.max():.4f} at t={accs.argmax()+1}")

    # Save separability results
    sep_path = os.path.join(config['results_dir'], 'separability.json')
    sep_save = {k: (v[0], v[1].tolist()) for k, v in sep_results.items()}
    with open(sep_path, 'w') as f:
        json.dump(sep_save, f, indent=2)

    plot_separability_over_time(sep_results, config['results_dir'])

    # ─── GO / NO-GO analysis ───
    analyse_go_nogo(results)

    print(f"\nAll outputs saved to: {config['results_dir']}")
    print("Done.")


if __name__ == '__main__':
    main()
