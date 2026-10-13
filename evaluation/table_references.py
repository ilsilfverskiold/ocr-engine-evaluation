"""Human-selected table references. Proposed patches are never an automatic fallback."""
from __future__ import annotations

import json
from pathlib import Path
from evaluation.artifacts import source_image_hash


def load_gold_table(document_path: Path) -> dict | None:
    return _load_table_version(document_path, "gold")


def load_original_table(document_path: Path) -> dict | None:
    """Frozen pre-review reference for the shipping alternate-reference policy."""
    return _load_table_version(document_path, "old")


def _load_table_version(document_path: Path, version: str) -> dict | None:
    path = document_path.parents[2] / "table_references" / version / f"{document_path.stem}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("page") != document_path.stem or data.get("version") != 1:
        raise ValueError(f"Invalid gold table identity/version: {path}")
    if data.get("image_sha256") != source_image_hash(document_path):
        raise ValueError(f"Gold table source image changed: {path}")
    markup = data.get("table_html")
    if not isinstance(markup, str):
        raise ValueError(f"Gold table HTML must be a string: {path}")
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(markup, "html.parser")
    if markup.strip() and not soup.find("table"):
        raise ValueError(f"Gold reference contains no table: {path}")
    data["path"] = str(path)
    return data
