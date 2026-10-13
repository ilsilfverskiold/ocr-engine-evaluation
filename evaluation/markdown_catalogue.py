"""Load and validate the benchmark's accepted Markdown reference catalogues.

Catalogues are supplied in results/current.zip. Scoring never generates them.
"""
from __future__ import annotations
import json
from pathlib import Path
from evaluation.artifacts import read_text as artifact_text, exists as artifact_exists
from evaluation.artifacts import source_image_hash as source_hash

ROOT = Path(__file__).resolve().parents[1]

CACHE = ROOT / 'results/markdown_judge_cache'
MARKDOWN_MAX_ITEMS = 40

VERSION = 'markdown-image-catalogue-v1'

TYPES = ['heading', 'paragraph', 'list_item', 'table_header', 'table_row', 'key_value']

SCHEMA = {'type': 'object', 'required': ['items'], 'properties': {'items': {
    # Enforce the count locally: maxItems=40 exceeds Gemini's grammar-state
    # limit for this nested object array, before any generation is attempted.
    'type': 'array', 'items': {'type': 'object',
    'required': ['type', 'content', 'confident', 'notes'], 'properties': {
        'type': {'type': 'string', 'enum': TYPES}, 'content': {'type': 'string'},
        'confident': {'type': 'boolean'}, 'notes': {'type': 'string'}}}}}}

def validate(data):
    import jsonschema
    jsonschema.validate(data, SCHEMA)
    if not 1 <= len(data['items']) <= 40 or any(not i['content'].strip() for i in data['items']):
        raise ValueError('Empty catalogue/content needs manual review')
    return data['items']

def reviewed_items(document_path):
    """Current-image, explicitly reviewed catalogues can establish eligibility."""
    path = CACHE / (Path(document_path).stem + '.json')
    if not artifact_exists(path):
        return None
    data = json.loads(artifact_text(path))
    if data.get('version') != VERSION or data.get('review_status') != 'accepted':
        return None
    if data.get('image_sha256') != source_hash(Path(document_path)):
        raise ValueError(f'Markdown catalogue image mismatch: {document_path}')
    validate({'items': data['items']})
    return data['items']


def _sample(items: list, cap: int) -> list:
    if len(items) <= cap:
        return items
    step = len(items) / cap
    return [items[int(i * step)] for i in range(cap)]


def load_markdown_checklist(document_stem: str) -> list[dict] | None:
    path = CACHE / f"{document_stem}.json"
    if not artifact_exists(path):
        return None
    try:
        record = json.loads(artifact_text(path))
        if record.get('version') == 'markdown-image-catalogue-v1' and record.get('review_status') != 'accepted':
            return None
        items = record.get("items") or []
    except Exception:
        return None
    items = [{**i, "content_format": "plain" if record.get("version") == "markdown-image-catalogue-v1" else "markdown"}
             for i in items if isinstance(i, dict) and i.get("confident", True)]
    if document_stem == "invoice_004":
        for item in items:
            if "Picturebook ☐☐☐☐☐☐☐☐" in item.get("content", ""):
                item["content"] = item["content"].replace("Picturebook ☐☐☐☐☐☐☐☐", "Picturebook")
                item["reference_review"] = "invoice-reference-v1: unreadable replacement-glyph span excluded"
    items = _sample(items, MARKDOWN_MAX_ITEMS)          # sample first, as the judge did
    return [i for i in items if i.get("content") and i.get("type")] or None
