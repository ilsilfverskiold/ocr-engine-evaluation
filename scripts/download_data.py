"""
Download the benchmark documents from HuggingFace.

Fetches the 93-document corpus from the dataset
https://huggingface.co/datasets/ilsilfverskiold/ocr-benchmark and rebuilds
the on-disk layout the harness expects:

  documents/<tier>/<doc_type>/<stem>.png          the page image
  documents/<tier>/<doc_type>/<stem>_gt.txt        plain-text GT
  documents/<tier>/<doc_type>/<stem>_gt.json       table/plain-text GT
  documents/<tier>/<doc_type>/<stem>_structured_gt.json
  documents/<tier>/<doc_type>/<stem>_markdown_gt.md
  documents/<tier>/<doc_type>/<stem>_geometry_gt.json

The reference run results ship with the git repo (results/), so this is only
needed when you want to rerun engines against the documents.

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
    sidecars = {
        f"{stem}_gt.txt": row.get("gt_text"),
        f"{stem}_gt.json": row.get("gt_json"),
        f"{stem}_structured_gt.json": row.get("structured_gt"),
        f"{stem}_markdown_gt.md": row.get("markdown_gt"),
        f"{stem}_geometry_gt.json": row.get("geometry_gt"),
    }
    files = {target: data, **{target.parent / name: value.encode("utf-8")
                            for name, value in sidecars.items() if value is not None}}
    for path, content in files.items():
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Local file differs; preserve/review it before downloading: {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    for path, content in files.items():
        if not path.exists():
            path.write_bytes(content)


def main() -> None:
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
