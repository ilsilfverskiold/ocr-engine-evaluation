"""Read the reconciled published result files."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'results'


def source_image_hash(path):
    """Verify present image bytes; use the pinned corpus identity when absent."""
    path = Path(path)
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        relative = path.resolve().relative_to(ROOT).as_posix()
        for entry in json.loads((ROOT / 'documents/manifest.json').read_text()):
            if entry['file'] == relative:
                return entry['sha256']
    except (ValueError, FileNotFoundError):
        pass
    raise FileNotFoundError(f'Image absent and not pinned in corpus manifest: {path}')

def read_text(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding='utf-8')

def current_timestamp():
    candidates = sorted((RESULTS_DIR / 'runs').glob('*_results.jsonl'))
    if not candidates:
        raise FileNotFoundError(
            'No reconciled results in results/runs/')
    return candidates[-1].name.removesuffix('_results.jsonl')

def location(path):
    return str(Path(path))


def exists(path):
    return Path(path).exists()


def glob_paths(directory, pattern):
    """Return matching immediate children from the reconciled result tree."""
    return sorted(Path(directory).glob(pattern))
