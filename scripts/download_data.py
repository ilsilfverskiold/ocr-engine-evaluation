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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "ilsilfverskiold/ocr-benchmark"
DOCS_DIR = ROOT / "documents"


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: the 'datasets' package is required. Install with:")
        print("  pip install datasets")
        sys.exit(1)

    existing = list(DOCS_DIR.rglob("*.png")) + list(DOCS_DIR.rglob("*.jpg"))
    if existing:
        print(f"documents/ already holds {len(existing)} images — nothing to do.")
        print(f"Delete the tier folders under {DOCS_DIR} and re-run to force a re-download.")
        return

    token = os.environ.get("HF_TOKEN")
    print(f"Downloading {REPO_ID} ...")
    ds = load_dataset(REPO_ID, split="train", token=token)

    n = 0
    for row in ds:
        out_dir = DOCS_DIR / row["tier"] / row["doc_type"]
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = row["stem"]

        img = row["image"]
        fmt = (img.format or "PNG").lower()
        ext = ".jpg" if fmt in ("jpeg", "jpg") else ".png"
        img.save(out_dir / f"{stem}{ext}")

        sidecars = {
            f"{stem}_gt.txt": row["gt_text"],
            f"{stem}_gt.json": row["gt_json"],
            f"{stem}_structured_gt.json": row["structured_gt"],
            f"{stem}_markdown_gt.md": row["markdown_gt"],
            f"{stem}_geometry_gt.json": row["geometry_gt"],
        }
        for name, content in sidecars.items():
            if content is not None:
                (out_dir / name).write_text(content, encoding="utf-8")

        n += 1

    print(f"Rebuilt {n} documents under {DOCS_DIR}")
    print("\nYou can now run engines, e.g.:")
    print("  python run_benchmark.py --engines tesseract,docling --tiers easy --skip-judge")


if __name__ == "__main__":
    main()
