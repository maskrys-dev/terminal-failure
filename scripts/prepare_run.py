"""Create a fresh SwiGLU or attention run directory without altering evidence."""
from pathlib import Path
import argparse
import shutil

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--family', choices=['swiglu','attention'], required=True)
    args = parser.parse_args()
    target = args.destination.resolve()
    if target.exists():
        parser.error('Destination must not exist; existing files will not be overwritten.')
    if target == ROOT or ROOT in target.parents:
        parser.error('Choose a directory outside this repository.')
    selected = list((ROOT/'transient_geometry').rglob('*.py'))
    selected += list((ROOT/'Rebuttal').glob('*.md'))
    selected += [ROOT/'requirements.txt', ROOT/'LICENSE', ROOT/'NOTICE.md']
    selected += list((ROOT/'results').glob('rebuttal_recursive*/config_frozen.json'))
    if args.family == 'attention':
        selected += [p for p in (ROOT/'results/modern_iterative_panel/protocol').iterdir() if p.is_file()]
        # E17 verifies these exact published E16 artifacts before any execution.
        locked = ROOT/'results/rebuttal_recursive_tinyimagenet_paired_horizon'
        selected += [locked/name for name in ['aggregate_summary.json','RUN_REPORT.md','paired_horizon_result.png']]
    for source in set(selected):
        if not source.is_file():
            parser.error(f'Required input missing: {source.relative_to(ROOT)}')
    target.mkdir(parents=True)
    for source in set(selected):
        out = target/source.relative_to(ROOT)
        out.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,out)
    print(f'Prepared {args.family} workspace: {target}')
    print('Place Tiny ImageNet at data/tiny-imagenet-200/ in that workspace.')

if __name__ == '__main__':
    main()
