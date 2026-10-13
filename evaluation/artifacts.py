"""Read the published result bundle, with local working files taking precedence."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'results' / 'current.zip'


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

@lru_cache(maxsize=2)
def _archive(path, mtime, size):
    return zipfile.ZipFile(path)

def archive():
    stat = BUNDLE.stat()
    return _archive(str(BUNDLE), stat.st_mtime_ns, stat.st_size)

def read_text(path):
    path = Path(path)
    if path.is_file():
        return path.read_text(encoding='utf-8')
    try:
        key = path.resolve().relative_to(ROOT).as_posix()
        return archive().read(key).decode('utf-8')
    except (FileNotFoundError, KeyError, ValueError):
        raise FileNotFoundError(path) from None

def current_timestamp():
    return json.loads(archive().read('manifest.json'))['run']

def location(path):
    path = Path(path)
    if path.is_file():
        return str(path)
    return f"results/current.zip!/{path.resolve().relative_to(ROOT).as_posix()}"


def exists(path):
    path = Path(path)
    if path.exists():
        return True
    try:
        return path.resolve().relative_to(ROOT).as_posix() in archive().NameToInfo
    except (FileNotFoundError, ValueError):
        return False


def glob_paths(directory, pattern):
    """Local and archived immediate children, deduplicated by logical path."""
    from fnmatch import fnmatch
    directory = Path(directory)
    paths = set(directory.glob(pattern))
    try:
        prefix = directory.resolve().relative_to(ROOT).as_posix() + '/'
        for name in archive().namelist():
            if name.startswith(prefix):
                tail = name[len(prefix):]
                if '/' not in tail and fnmatch(tail, pattern):
                    paths.add(directory / tail)
    except (FileNotFoundError, ValueError):
        pass
    return sorted(paths)
