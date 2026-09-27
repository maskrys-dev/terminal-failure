"""Package only the local LaTeX inputs and graphics used by the preprint."""
from pathlib import Path
import re
import zipfile

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
OUT = ROOT / 'output' / 'arxiv'


def dependencies():
    found = set()
    pending = [PAPER / 'main.tex']
    while pending:
        path = pending.pop().resolve()
        path.relative_to(PAPER)
        if path in found:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        found.add(path)
        if path.suffix != '.tex':
            continue
        source = re.sub(r'(?<!\\)%.*', '', path.read_text(encoding='utf-8'))
        for name in re.findall(r'\\(?:input|include)\{([^}]+)\}', source):
            target = PAPER / name
            pending.append(target if target.suffix else target.with_suffix('.tex'))
        for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', source):
            target = PAPER / 'figures' / name
            # Must match main.tex's declared preference: PNG before PDF.
            candidates = [target] if target.suffix else [target.with_suffix(ext) for ext in ('.png', '.pdf')]
            match = next((p for p in candidates if p.is_file()), None)
            if match is None:
                raise FileNotFoundError(name)
            pending.append(match)
        for group in re.findall(r'\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}', source):
            for name in group.split(','):
                local = PAPER / (name.strip() + '.sty')
                if local.is_file():
                    pending.append(local)
    return sorted(found)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'terminal_failure_arxiv_source.zip'
    files = dependencies()
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(PAPER).as_posix())
    print(f'{target}: {len(files)} files, {target.stat().st_size:,} bytes')
