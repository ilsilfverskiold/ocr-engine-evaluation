"""Build and sample the deterministic GT field checklist.

The legacy module name is retained for the field scorers.
Per-engine verdicts live in kv_schema_code.py. The old LLM verdict client,
substring verifier and verdict cache code were removed; this module makes
no model calls. Existing deterministic checklist caches remain readable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.structured_metrics import is_blank, walk_fields

CACHE_DIR = Path(__file__).parent.parent / "results" / "schema_judge_cache"
MAX_CHECKLIST_ITEMS = 80


@dataclass
class SchemaChecklistItem:
    path: str
    value: str
    is_blank: bool = False


def build_schema_checklist(
    true_json: Any,
    json_schema: dict | None,
    document_stem: str,
    use_cache: bool = True,
    scalar_only: bool = False,
    excluded_paths: set[str] | None = None,
) -> list[SchemaChecklistItem]:
    # Exclusions describe source-verified unscorable fields, never missing
    # engine answers. Filter before sampling and preserve the complete cache.
    excluded = excluded_paths or set()
    suffix = "_kv" if scalar_only else ""
    cache_path = CACHE_DIR / f"{document_stem}{suffix}.json"
    if use_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        items = [SchemaChecklistItem(**i) for i in cached["items"]]
        return _sample_checklist([i for i in items if i.path not in excluded])

    fields = walk_fields(true_json, json_schema)
    fields = [f for f in fields if f.path]
    if scalar_only:
        fields = [f for f in fields if not f.in_array]

    items: list[SchemaChecklistItem] = []
    for f in fields:
        blank = is_blank(f.value)
        value_str = "" if blank else _display_value(f.value)
        items.append(SchemaChecklistItem(
            path=f.path,
            value=value_str,
            is_blank=blank,
        ))

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({
                "document": document_stem,
                "items": [
                    {"path": i.path, "value": i.value, "is_blank": i.is_blank}
                    for i in items
                ],
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return _sample_checklist([i for i in items if i.path not in excluded])


def _display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).strip()


def _sample_checklist(
    items: list[SchemaChecklistItem],
) -> list[SchemaChecklistItem]:
    if len(items) <= MAX_CHECKLIST_ITEMS:
        return items
    step = len(items) / MAX_CHECKLIST_ITEMS
    return [items[int(i * step)] for i in range(MAX_CHECKLIST_ITEMS)]
