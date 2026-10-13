"""Load and validate the benchmark's accepted Markdown reference catalogues.

Catalogues are supplied in results/current.zip. Scoring never generates them.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from evaluation.artifacts import read_text as artifact_text, exists as artifact_exists

ROOT = Path(__file__).resolve().parents[1]

CACHE = ROOT / 'results/markdown_judge_cache'

VERSION = 'markdown-image-catalogue-v1'

TYPES = ['heading', 'paragraph', 'list_item', 'table_header', 'table_row', 'key_value']

SCHEMA = {'type': 'object', 'required': ['items'], 'properties': {'items': {
    # Enforce the count locally: maxItems=40 exceeds Gemini's grammar-state
    # limit for this nested object array, before any generation is attempted.
    'type': 'array', 'items': {'type': 'object',
    'required': ['type', 'content', 'confident', 'notes'], 'properties': {
        'type': {'type': 'string', 'enum': TYPES}, 'content': {'type': 'string'},
        'confident': {'type': 'boolean'}, 'notes': {'type': 'string'}}}}}}

def source_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

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
