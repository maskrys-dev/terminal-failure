"""
D1 ESN hyperparameter sweep.

The default D1 hyperparameters (τ=1.2-1.5, α=5-10) are tuned for the
complex modReLU system's aggressive norm growth. The real ESN's tanh
bounds norms more gently, so D1 needs higher τ and lower α.

Usage:
    python -m transient_geometry.experiments.d1_esn_tune
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

from transient_geometry.system import RealESN
from transient_geometry.probes import LinearProbe


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'results', 'd1_esn_tune')

CONFIG = {
    'input_dim': 784,
    'hidden_dim': 128,
    'rollout_steps': 40,
    'noise_std': 0.0,

    'spectral_radii': [1.0, 1.30, 1.40, 1.50, 1.60],
    'num_seeds': 5,

    'batch_size': 512,
    'train_n': 5000,
    'test_n': 5000,
    'early_K': 10,

    # Wide D1 grid for ESN tuning
    'd1_configs': [
        # Original complex-tuned
        {'tau': 1.2, 'alpha': 5},
        {'tau': 1.5, 'alpha': 5},
        # Higher τ for bounded dynamics
        {'tau': 2.0, 'alpha': 2},
        {'tau': 2.0, 'alpha': 5},
        {'tau': 2.5, 'alpha': 2},
        {'tau': 2.5, 'alpha': 5},
        {'tau': 3.0, 'alpha': 1},
        {'tau': 3.0, 'alpha': 2},
        {'tau': 5.0, 'alpha': 1},
        {'tau': 5.0, 'alpha': 2},
        # Very gentle
        {'tau': 10.0, 'alpha': 0.5},
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


@torch.no_grad()
def collect_trajectory(system, loader, T, K, device):
    all_z, all_norms = [], []
    all_final, all_trans, all_early, all_labels = [], [], [], []

    for xb, yb in loader:
        xb = xb.to(device)
        batch = xb.shape[0]
        d = system.hidden_dim

        Ux = torch.matmul(xb, system.U.T)
        h = torch.zeros(batch, d, device=device)
        h_sum = torch.zeros_like(h)
        h_early_sum = torch.zeros_like(h)

        batch_z = np.zeros((T, batch, d), dtype=np.float32)
        batch_norms = np.zeros((T, batch), dtype=np.float32)

        for t in range(T):
            pre = torch.matmul(h, system.W.T) + Ux + system.b
            h_new = torch.tanh(pre)
            h = system.leak_rate * h_new + (1 - system.leak_rate) * h

            h_sum += h
            if t < K:
                h_early_sum += h
            batch_z[t] = h.cpu().numpy()
            batch_norms[t] = h.norm(dim=-1).cpu().numpy()

        h_mean = h_sum / T
        h_early = h_early_sum / K
        all_z.append(batch_z)
        all_norms.append(batch_norms)
        all_final.append(h.cpu().numpy())
        all_trans.append(h_mean.cpu().numpy())
        all_early.append(h_early.cpu().numpy())
        all_labels.append(yb.numpy())

    return {
        'z': np.concatenate(all_z, axis=1),
        'norms': np.concatenate(all_norms, axis=1),
        'final': np.concatenate(all_final),
        'transient_mean': np.concatenate(all_trans),
        'early_window': np.concatenate(all_early),
        'labels': np.concatenate(all_labels),
    }


def compute_d1(z, norms, tau, alpha):
    T, N, d = z.shape
    g = norms / np.maximum(norms[0:1, :], 1e-10)
    w_raw = np.exp(-alpha * np.maximum(0.0, g - tau))
    w_sum = np.sum(w_raw, axis=0, keepdims=True)
    w = w_raw / np.maximum(w_sum, 1e-10)
    features = np.einsum('tn,tnf->nf', w, z)
    return features, w


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_ld, test_ld = get_data(CONFIG)
    T = CONFIG['rollout_steps']
    K = CONFIG['early_K']
    srs = CONFIG['spectral_radii']
    seeds = CONFIG['num_seeds']
    d1_cfgs = CONFIG['d1_configs']
    total = len(srs) * seeds

    print(f"\n{'='*80}")
    print(f"D1 ESN TUNE: {len(srs)} SR × {seeds} seeds = {total} runs")
    print(f"D1 configs: {len(d1_cfgs)}")
    print(f"{'='*80}\n")

    results = []
    run_idx = 0

    for sr in srs:
        for seed in range(seeds):
            run_idx += 1
            t0 = time.time()
            print(f"[{run_idx}/{total}] ρ={sr:.2f} s={seed} ... ", end="", flush=True)

            system = RealESN(
                input_dim=CONFIG['input_dim'], hidden_dim=CONFIG['hidden_dim'],
                spectral_radius=sr, noise_std=0.0, leak_rate=1.0, seed=seed,
            ).to(device)

            train_data = collect_trajectory(system, train_ld, T, K, device)
            test_data = collect_trajectory(system, test_ld, T, K, device)

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
            probe.fit(train_data['z'][stop_t], train_data['labels'])
            accs['adaptive'] = probe.score(test_data['z'][stop_t], test_data['labels'])

            # All D1 configs
            d1_accs = {}
            for d1c in d1_cfgs:
                key = f"t{d1c['tau']}_a{d1c['alpha']}"
                tr_feat, _ = compute_d1(train_data['z'], train_data['norms'],
                                        d1c['tau'], d1c['alpha'])
                te_feat, _ = compute_d1(test_data['z'], test_data['norms'],
                                        d1c['tau'], d1c['alpha'])
                probe = LinearProbe()
                probe.fit(tr_feat, train_data['labels'])
                d1_accs[key] = probe.score(te_feat, test_data['labels'])

            elapsed = time.time() - t0
            best_key = max(d1_accs, key=d1_accs.get)
            best_acc = d1_accs[best_key]

            result = {
                'spectral_radius': sr, 'seed': seed,
                'acc_final': accs['final'],
                'acc_trans': accs['transient_mean'],
                'acc_early': accs['early_window'],
                'acc_adaptive': accs['adaptive'],
                'adaptive_stop_t': int(stop_t + 1),
                'd1_accs': {k: float(v) for k, v in d1_accs.items()},
                'best_d1': best_key,
                'best_d1_acc': float(best_acc),
                'elapsed': elapsed,
            }
            results.append(result)

            print(f"F={accs['final']:.4f} E={accs['early_window']:.4f} "
                  f"A={accs['adaptive']:.4f} D1={best_acc:.4f}({best_key}) "
                  f"({elapsed:.0f}s)")

    # Save
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Summary: best D1 per SR
    print(f"\n{'='*90}")
    print("ESN D1 HYPERPARAMETER SUMMARY")
    print(f"{'='*90}")

    # Full grid
    d1_keys = [f"t{c['tau']}_a{c['alpha']}" for c in d1_cfgs]
    print(f"\n{'Config':>15}", end="")
    for sr in srs:
        print(f" | ρ={sr:.2f}", end="")
    print(f" | {'Mean':>7}")
    print("-" * (17 + 10 * len(srs) + 10))

    for key in d1_keys:
        print(f"  {key:>13}", end="")
        vals_per_sr = []
        for sr in srs:
            runs = [r for r in results if r['spectral_radius'] == sr]
            mean_acc = np.mean([r['d1_accs'][key] for r in runs])
            vals_per_sr.append(mean_acc)
            print(f" | {mean_acc:.4f}", end="")
        print(f" | {np.mean(vals_per_sr):.4f}")

    # Baselines
    print(f"\n  {'Final':>13}", end="")
    for sr in srs:
        runs = [r for r in results if r['spectral_radius'] == sr]
        print(f" | {np.mean([r['acc_final'] for r in runs]):.4f}", end="")
    print()
    print(f"  {'Early':>13}", end="")
    for sr in srs:
        runs = [r for r in results if r['spectral_radius'] == sr]
        print(f" | {np.mean([r['acc_early'] for r in runs]):.4f}", end="")
    print()
    print(f"  {'Adaptive':>13}", end="")
    for sr in srs:
        runs = [r for r in results if r['spectral_radius'] == sr]
        print(f" | {np.mean([r['acc_adaptive'] for r in runs]):.4f}", end="")
    print()
    print(f"  {'Trans-mean':>13}", end="")
    for sr in srs:
        runs = [r for r in results if r['spectral_radius'] == sr]
        print(f" | {np.mean([r['acc_trans'] for r in runs]):.4f}", end="")
    print()

    # Best per SR
    print(f"\n  BEST D1 per SR:")
    for sr in srs:
        runs = [r for r in results if r['spectral_radius'] == sr]
        best_overall = {}
        for key in d1_keys:
            best_overall[key] = np.mean([r['d1_accs'][key] for r in runs])
        winner = max(best_overall, key=best_overall.get)
        print(f"    ρ={sr:.2f}: {winner} → {best_overall[winner]:.4f}")

    # Plot
    sns.set_theme(style='whitegrid', font_scale=1.1)
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # Plot baselines
    baseline_styles = [
        ('acc_final', 'Final-state', '#2196F3', 'o', '--', 1.5),
        ('acc_early', 'Early-window', '#4CAF50', '^', '-', 2.0),
        ('acc_adaptive', 'Adaptive (×1.5)', '#FF9800', 'D', '-', 2.0),
        ('acc_trans', 'Trans-mean', '#9E9E9E', 's', '--', 1.5),
    ]
    for key, label, color, marker, ls, lw in baseline_styles:
        means = [np.mean([r[key] for r in results if r['spectral_radius'] == sr])
                 for sr in srs]
        ax.plot(srs, means, f'{marker}{ls}', color=color, lw=lw, ms=7, label=label)

    # Plot D1 configs with color gradient
    cmap = plt.cm.viridis
    for i, key in enumerate(d1_keys):
        color = cmap(i / len(d1_keys))
        means = [np.mean([r['d1_accs'][key] for r in results if r['spectral_radius'] == sr])
                 for sr in srs]
        lw = 2.5 if i >= len(d1_keys) - 4 else 1.5
        alpha = 0.9 if i >= len(d1_keys) - 4 else 0.5
        ax.plot(srs, means, 'x-', color=color, lw=lw, ms=6, alpha=alpha,
                label=f'D1 {key}')

    ax.set_xlabel('Spectral Radius ρ(W)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Real ESN: D1 Hyperparameter Sweep')
    ax.legend(fontsize=7, loc='lower left', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'd1_esn_tune.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {path}")
    print("Done.")


if __name__ == '__main__':
    main()
