"""
D1 Cross-System: Regime-aware trajectory readout on Complex modReLU + Real ESN.

Shows that D1 soft norm-growth weighting works across system classes,
not just the complex reservoir.

Usage:
    python -m transient_geometry.experiments.d1_cross_system
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

from transient_geometry.system import ComplexDynamicalSystem, RealESN
from transient_geometry.probes import LinearProbe


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'results', 'd1_cross_system')

CONFIG = {
    'input_dim': 784,
    'hidden_dim': 128,
    'rollout_steps': 40,
    'noise_std': 0.0,

    'spectral_radii': [0.5, 1.0, 1.20, 1.30, 1.35, 1.40, 1.45, 1.50, 1.60],
    'num_seeds': 5,

    'batch_size': 512,
    'train_n': 5000,
    'test_n': 5000,
    'early_K': 10,

    # D1 configs to test
    'd1_configs': [
        {'tau': 1.2, 'alpha': 5,  'label': 'D1 (τ=1.2, α=5)'},
        {'tau': 1.2, 'alpha': 10, 'label': 'D1 (τ=1.2, α=10)'},
        {'tau': 1.5, 'alpha': 5,  'label': 'D1 (τ=1.5, α=5)'},
        {'tau': 1.5, 'alpha': 10, 'label': 'D1 (τ=1.5, α=10)'},
    ],

    'systems': [
        {
            'name': 'complex_modrelu',
            'label': 'Complex (modReLU)',
            'class': 'complex',
            'kwargs': {'modrelu_bias': -0.5},
            'color': '#2196F3',
        },
        {
            'name': 'real_esn',
            'label': 'Real ESN (tanh)',
            'class': 'real',
            'kwargs': {'leak_rate': 1.0},
            'color': '#F44336',
        },
    ],
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


def build_system(sys_cfg, input_dim, hidden_dim, sr, seed, device):
    if sys_cfg['class'] == 'complex':
        system = ComplexDynamicalSystem(
            input_dim=input_dim, hidden_dim=hidden_dim,
            spectral_radius=sr, noise_std=0.0, seed=seed,
            **sys_cfg['kwargs']
        ).to(device)
    else:
        system = RealESN(
            input_dim=input_dim, hidden_dim=hidden_dim,
            spectral_radius=sr, noise_std=0.0, seed=seed,
            **sys_cfg['kwargs']
        ).to(device)
    return system


@torch.no_grad()
def collect_trajectory_complex(system, loader, T, K, device):
    """Full trajectory collection for complex system."""
    all_z_real, all_norms = [], []
    all_final, all_trans, all_early, all_labels = [], [], [], []

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
            batch_z_real[t] = torch.cat([z.real, z.imag], dim=-1).cpu().numpy()
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


@torch.no_grad()
def collect_trajectory_real(system, loader, T, K, device):
    """Full trajectory collection for real ESN."""
    all_z_real, all_norms = [], []
    all_final, all_trans, all_early, all_labels = [], [], [], []

    for xb, yb in loader:
        xb = xb.to(device)
        batch = xb.shape[0]
        d = system.hidden_dim

        Ux = torch.matmul(xb, system.U.T)
        h = torch.zeros(batch, d, device=device)
        h_sum = torch.zeros_like(h)
        h_early_sum = torch.zeros_like(h)

        batch_z_real = np.zeros((T, batch, d), dtype=np.float32)
        batch_norms = np.zeros((T, batch), dtype=np.float32)

        for t in range(T):
            pre = torch.matmul(h, system.W.T) + Ux + system.b
            if system.noise_std > 0:
                pre = pre + torch.randn_like(pre) * system.noise_std
            h_new = torch.tanh(pre)
            h = system.leak_rate * h_new + (1 - system.leak_rate) * h

            h_sum += h
            if t < K:
                h_early_sum += h
            batch_z_real[t] = h.cpu().numpy()
            batch_norms[t] = h.norm(dim=-1).cpu().numpy()

        h_mean = h_sum / T
        h_early = h_early_sum / K
        all_z_real.append(batch_z_real)
        all_norms.append(batch_norms)
        all_final.append(h.cpu().numpy())
        all_trans.append(h_mean.cpu().numpy())
        all_early.append(h_early.cpu().numpy())
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
    return features, w


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
    d1_cfgs = CONFIG['d1_configs']
    systems = CONFIG['systems']

    total = len(systems) * len(srs) * seeds
    print(f"\n{'='*80}")
    print(f"D1 CROSS-SYSTEM: {len(systems)} systems × {len(srs)} SR × {seeds} seeds = {total} runs")
    print(f"D1 configs: {len(d1_cfgs)}")
    print(f"{'='*80}\n")

    all_results = {s['name']: [] for s in systems}
    weight_profiles = {}  # (sys_name, sr, tau, alpha) -> list of [T] profiles
    run_idx = 0

    for sys_cfg in systems:
        sys_name = sys_cfg['name']
        collect_fn = collect_trajectory_complex if sys_cfg['class'] == 'complex' else collect_trajectory_real
        print(f"\n--- {sys_cfg['label']} ---")

        for sr in srs:
            for seed in range(seeds):
                run_idx += 1
                t0 = time.time()
                print(f"[{run_idx}/{total}] {sys_name} ρ={sr:.2f} s={seed} ... ",
                      end="", flush=True)

                system = build_system(sys_cfg, CONFIG['input_dim'],
                                      CONFIG['hidden_dim'], sr, seed, device)

                train_data = collect_fn(system, train_ld, T, K, device)
                test_data = collect_fn(system, test_ld, T, K, device)

                # Baselines
                accs = {}
                for name in ['final', 'transient_mean', 'early_window']:
                    probe = LinearProbe()
                    probe.fit(train_data[name], train_data['labels'])
                    accs[name] = probe.score(test_data[name], test_data['labels'])

                # Adaptive ×1.5
                mean_norms = train_data['norms'].mean(axis=1)
                ratio = mean_norms / max(mean_norms[0], 1e-10)
                exceeded = np.where(ratio > 1.5)[0]
                stop_t = max(0, exceeded[0] - 1) if len(exceeded) > 0 else T - 1
                probe = LinearProbe()
                probe.fit(train_data['z_real'][stop_t], train_data['labels'])
                accs['adaptive'] = probe.score(test_data['z_real'][stop_t], test_data['labels'])

                # D1 configs
                d1_accs = {}
                for d1c in d1_cfgs:
                    key = f"d1_t{d1c['tau']}_a{d1c['alpha']}"
                    tr_feat, tr_w = compute_d1_features(
                        train_data['z_real'], train_data['norms'],
                        d1c['tau'], d1c['alpha'])
                    te_feat, te_w = compute_d1_features(
                        test_data['z_real'], test_data['norms'],
                        d1c['tau'], d1c['alpha'])

                    probe = LinearProbe()
                    probe.fit(tr_feat, train_data['labels'])
                    d1_accs[key] = probe.score(te_feat, test_data['labels'])

                    wp_key = (sys_name, sr, d1c['tau'], d1c['alpha'])
                    if wp_key not in weight_profiles:
                        weight_profiles[wp_key] = []
                    weight_profiles[wp_key].append(tr_w.mean(axis=1))

                elapsed = time.time() - t0
                best_d1 = max(d1_accs, key=d1_accs.get)
                best_d1_acc = d1_accs[best_d1]

                result = {
                    'system': sys_name,
                    'spectral_radius': sr, 'seed': seed,
                    'acc_final': accs['final'],
                    'acc_transient_mean': accs['transient_mean'],
                    'acc_early_window': accs['early_window'],
                    'acc_adaptive': accs['adaptive'],
                    'adaptive_stop_t': int(stop_t + 1),
                    'd1_accs': {k: float(v) for k, v in d1_accs.items()},
                    'best_d1': best_d1,
                    'best_d1_acc': float(best_d1_acc),
                    'elapsed': elapsed,
                }
                all_results[sys_name].append(result)

                print(f"F={accs['final']:.4f} E={accs['early_window']:.4f} "
                      f"A={accs['adaptive']:.4f} D1={best_d1_acc:.4f}({best_d1}) "
                      f"({elapsed:.0f}s)")

    # Save
    flat = []
    for rlist in all_results.values():
        flat.extend(rlist)
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(flat, f, indent=2)

    avg_profiles = {}
    for k, v_list in weight_profiles.items():
        avg_profiles[str(k)] = np.mean(v_list, axis=0).tolist()
    with open(os.path.join(RESULTS_DIR, 'weight_profiles.json'), 'w') as f:
        json.dump(avg_profiles, f, indent=2)

    # =========================================================================
    # Plots: 2×2 — accuracy curves + weight profiles for each system
    # =========================================================================

    sns.set_theme(style='whitegrid', font_scale=1.1)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    anchor_key = 'd1_t1.5_a5'

    for col, sys_cfg in enumerate(systems):
        sys_name = sys_cfg['name']
        results = all_results[sys_name]
        sr_list = sorted(set(r['spectral_radius'] for r in results))

        # Top row: Accuracy
        ax = axes[0, col]
        styles = [
            ('acc_final', 'Final-state', '#2196F3', 'o', '--', 1.5),
            ('acc_transient_mean', 'Trans-mean', '#9E9E9E', 's', '--', 1.5),
            ('acc_early_window', 'Early-window', '#4CAF50', '^', '-', 2.0),
            ('acc_adaptive', 'Adaptive (×1.5)', '#FF9800', 'D', '-', 2.0),
        ]
        for key, label, color, marker, ls, lw in styles:
            means = [np.mean([r[key] for r in results if r['spectral_radius'] == sr])
                     for sr in sr_list]
            ax.plot(sr_list, means, f'{marker}{ls}', color=color, lw=lw, ms=6,
                    label=label, alpha=0.85)

        # D1 anchor
        d1_means = [np.mean([r['d1_accs'].get(anchor_key, 0)
                             for r in results if r['spectral_radius'] == sr])
                    for sr in sr_list]
        ax.plot(sr_list, d1_means, 'P-', color='#6A1B9A', lw=3, ms=9,
                label='D1 (τ=1.5, α=5)', zorder=10)

        ax.set_xlabel('Spectral Radius ρ(W)')
        ax.set_ylabel('Test Accuracy')
        ax.set_title(f'{sys_cfg["label"]}')
        ax.legend(fontsize=7, loc='lower left')
        ax.grid(True, alpha=0.3)

        # Bottom row: Weight profiles
        ax = axes[1, col]
        tau, alpha = 1.5, 5
        profile_srs = [1.0, 1.3, 1.5]
        cmap = plt.cm.coolwarm
        colors_prof = [cmap(0.1), cmap(0.5), cmap(0.9)]

        for i, psr in enumerate(profile_srs):
            wp_key = str((sys_name, psr, tau, alpha))
            if wp_key in avg_profiles:
                profile = np.array(avg_profiles[wp_key])
                ax.plot(range(1, T + 1), profile, '-', lw=2.5, color=colors_prof[i],
                        label=f'ρ={psr:.1f}', alpha=0.9)

        ax.set_xlabel('Timestep t')
        ax.set_ylabel('Mean Weight w_t')
        ax.set_title(f'{sys_cfg["label"]}: Weight Profile (τ={tau}, α={alpha})')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('D1 Cross-System: complex modReLU vs Real ESN',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'd1_cross_system.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {path}")

    # Summary
    print(f"\n{'='*90}")
    print("D1 CROSS-SYSTEM SUMMARY (anchor: τ=1.5, α=5)")
    print(f"{'='*90}")
    for sys_cfg in systems:
        sys_name = sys_cfg['name']
        results = all_results[sys_name]
        sr_list = sorted(set(r['spectral_radius'] for r in results))
        print(f"\n  {sys_cfg['label']}")
        print(f"  {'SR':>6} | {'Final':>7} | {'Early':>7} | {'Adap':>7} | "
              f"{'D1':>7} | {'D1−Final':>9} | {'D1−Early':>9}")
        print(f"  {'-'*70}")
        for sr in sr_list:
            runs = [r for r in results if r['spectral_radius'] == sr]
            f = np.mean([r['acc_final'] for r in runs])
            e = np.mean([r['acc_early_window'] for r in runs])
            a = np.mean([r['acc_adaptive'] for r in runs])
            d = np.mean([r['d1_accs'].get(anchor_key, 0) for r in runs])
            print(f"  {sr:>6.2f} | {f:>7.4f} | {e:>7.4f} | {a:>7.4f} | "
                  f"{d:>7.4f} | {d-f:>+9.4f} | {d-e:>+9.4f}")

    print(f"\nAll outputs saved to: {RESULTS_DIR}")
    print("Done.")


if __name__ == '__main__':
    main()
