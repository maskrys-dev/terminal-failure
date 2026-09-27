"""Verify published file hashes; requires only the Python standard library."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints',action='store_true')
    args=parser.parse_args()
    entries=json.loads((ROOT/'docs/release_manifest.json').read_text(encoding='utf8'))['files']
    if args.checkpoints:
        entries += json.loads((ROOT/'docs/checkpoint_manifest.json').read_text(encoding='utf8'))['files']
    failures=[]
    for item in entries:
        path=ROOT/item['path']
        if not path.is_file() or path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
            failures.append(item['path'])
    if failures:
        raise SystemExit('Missing or changed files:\n'+'\n'.join(failures))
    print(f'PASS: {len(entries)} files match their published hashes.')

if __name__ == '__main__':
    main()
