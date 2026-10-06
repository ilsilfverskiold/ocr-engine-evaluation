"""Human-reviewed schema/KV references, independent of table references."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def field_patterns(record: dict) -> set[str]:
    """Version target fields without confusing new array rows with new keys."""
    from evaluation.structured_metrics import walk_fields
    patterns = {re.sub(r'\[\d+\]', '[*]', f.path)
                for f in walk_fields(record['true_json'], record['json_schema']) if f.path}

    def visit(schema, path=''):
        if not isinstance(schema, dict):
            return
        if schema.get('properties'):
            for key, child in schema['properties'].items():
                visit(child, f'{path}.{key}' if path else key)
        elif schema.get('items'):
            visit(schema['items'], path + '[*]')
        elif path:
            patterns.add(path)

    visit(record['json_schema'])
    return patterns


def validate_structured_record(record: dict) -> None:
    from jsonschema.validators import validator_for
    from evaluation.structured_metrics import walk_fields

    if record.get('reference_kind') != 'structured':
        raise ValueError('Expected a structured reference, not table gold')
    if not isinstance(record.get('true_json'), (dict, list)):
        raise ValueError('Expected reference values as a JSON object or array')
    schema = record.get('json_schema')
    if not isinstance(schema, dict):
        raise ValueError('json_schema must be an object')
    try:
        validator_for(schema).check_schema(schema)
    except Exception as exc:
        raise ValueError(f'Invalid schema: {exc}') from exc
    # Refuse non-JSON values, including NaN, even when a permissive decoder accepts them.
    json.dumps(record, allow_nan=False)
    fields = walk_fields(record['true_json'], schema)
    paths = {f.path for f in fields}
    aliases = record.get('kv_aliases')
    if aliases is not None:
        if not isinstance(aliases, dict):
            raise ValueError('KV labels must be an object or null when unreviewed')
        for path, alias in aliases.items():
            if path not in paths:
                raise ValueError(f'KV label points to an unknown field: {path}')
            if not isinstance(alias, dict) or not isinstance(alias.get('is_printed_pair'), bool):
                raise ValueError(f'KV printed-pair flag must be true or false: {path}')
            labels = alias.get('printed_labels')
            if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
                raise ValueError(f'KV printed labels must be a list of strings: {path}')
    excluded = record.get('excluded_fields', [])
    if not isinstance(excluded, list) or not all(isinstance(x, str) and x in paths for x in excluded):
        raise ValueError('Excluded fields must be a list of existing field paths')
    if record.get('shared_field_selection') is True:
        for path in paths:
            if aliases is None or path not in aliases:
                raise ValueError(f'Shared reference needs a mapping for every field: {path}')
            if aliases[path]['is_printed_pair'] != (path not in excluded):
                raise ValueError(f'Schema and KV inclusion must agree: {path}')


def load_gold_structured(document_path: Path) -> dict | None:
    path = document_path.parents[2] / 'structured_references' / 'gold' / f'{document_path.stem}.json'
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('version') != 1 or record.get('page') != document_path.stem:
        raise ValueError(f'Invalid structured gold identity/version: {path}')
    if record.get('image_sha256') != hashlib.sha256(document_path.read_bytes()).hexdigest():
        raise ValueError(f'Structured gold source image changed: {path}')
    validate_structured_record(record)
    record['path'] = str(path)
    return record
