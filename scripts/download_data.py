"""
Download the benchmark source images from HuggingFace.

Fetches the 93-document corpus from the dataset
https://huggingface.co/datasets/ilsilfverskiold/ocr-benchmark and rebuilds
the image layout the harness expects:

  documents/<tier>/<doc_type>/<stem>.png          the page image
All ground-truth sidecars, accepted references and saved results ship in Git.
Images are needed for new inference and source-page previews in HTML reports.
Downloading images never replaces the repository's reference files.

Usage:
  python scripts/download_data.py
"""

from __future__ import annotations

import os
import sys
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "ilsilfverskiold/ocr-benchmark"
DOCS_DIR = ROOT / "documents"


def write_row(row: dict, manifest: dict, root: Path = ROOT) -> None:
    """Restore exact published image bytes; never re-encode a reviewed source.

    Missing files are resumable. Existing local edits are refused rather than
    overwritten. Gold, history and proposal folders are not download targets.
    """
    stem = row["stem"]
    entry = manifest[stem]
    relative = Path(entry["file"])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "documents":
        raise ValueError(f"Invalid corpus path for {stem}")
    image = row["image"]
    data = image.get("bytes")
    if data is None and image.get("path"):
        data = Path(image["path"]).read_bytes()
    if not isinstance(data, bytes):
        raise ValueError(f"Dataset did not supply original image bytes: {stem}")
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"Published image differs from pinned manifest: {stem}")
    target = root / relative
    if target.exists() and target.read_bytes() != data:
        raise ValueError(f"Local file differs; preserve/review it before downloading: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(data)


def main() -> None:
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        from datasets import Image, load_dataset
    except ImportError:
        print("ERROR: the 'datasets' package is required. Install with:")
        print("  pip install datasets")
        sys.exit(1)

    manifest = {Path(r["file"]).stem: r for r in json.loads((DOCS_DIR / "manifest.json").read_text())}
    token = os.environ.get("HF_TOKEN")
    print(f"Downloading {REPO_ID} ...")
    ds = load_dataset(REPO_ID, split="train", token=token).cast_column("image", Image(decode=False))

    n = 0
    for row in ds:
        write_row(row, manifest)
        n += 1

    print(f"Rebuilt {n} documents under {DOCS_DIR}")
    print("\nYou can now run engines, e.g.:")
    print("  python run_benchmark.py --engines tesseract,docling --tiers easy --skip-judge")


if __name__ == "__main__":
    main()
