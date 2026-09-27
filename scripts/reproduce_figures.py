"""Regenerate and stage preprint figures from saved results, without training."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    for script in ['paper/generate_figures.py', 'paper/generate_arxiv_assets.py',
                   'results/e7_cifar10/plot_e7.py']:
        subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)
    mapping = {
        'figures/fig_cifar10_regime.png': 'fig_cifar10_regime.png',
        'figures/fig_cifar10_regime.pdf': 'fig_cifar10_regime.pdf',
        'results/e12_cifar10_resnet18/e12_cifar10_resnet18.png': 'fig_cifar10_resnet18_regime.png',
        'results/e13_seq_cifar10_lstm/e13_seq_cifar10_lstm.png': 'fig13_seq_cifar10_lstm.png',
        'results/e13_seq_cifar10_lstm/e13_seq_cifar10_lstm_timestep_audit.png': 'fig14_seq_cifar10_lstm_timestep_audit.png',
    }
    for source, name in mapping.items():
        shutil.copy2(ROOT/source, ROOT/'paper/figures'/name)
    print('Preprint figures and modern tables ready in paper/.')
