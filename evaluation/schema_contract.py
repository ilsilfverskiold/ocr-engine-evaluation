"""Prepare the schema actually handed to engines, without supplying answers.

The benchmark asks for null when a field is not on the page. Its historical
schemas often forbade null, and eight contained open dictionaries which a
vendor converted to empty closed objects. Explicit target field names are
versioned separately from GT values. The scoring reference is not expanded
with extra blank fields merely because an inference schema is made explicit.
"""
from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path

CONTRACT_VERSION = "schema-contract-v1 (2026-10-02)"
FIELD_FILE = Path(__file__).parent.parent / "documents" / "schema_fields.json"


def contract_id(schema: dict | None) -> str | None:
    if schema is None:
        return None
    return hashlib.sha256(json.dumps(schema, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def stored_contract_status(gt, supplied_schema: dict | None) -> dict:
    """Never judge an earlier extraction against a contract it did not receive."""
    expected = gt.inference_schema
    same = supplied_schema is not None and supplied_schema == expected
    return {"schema_contract_matches": same,
            "schema_expected_contract_id": contract_id(expected),
            "schema_supplied_contract_id": contract_id(supplied_schema),
            "schema_rerun_required": not same}


@lru_cache(maxsize=1)
def _field_definitions() -> dict:
    return json.loads(FIELD_FILE.read_text())["documents"]


def extraction_schema(schema: dict, document_stem: str = "") -> dict:
    """Explicit target fields, with nullable values; never reads true_json."""
    result = copy.deepcopy(schema)

    def has_open_object(node):
        if isinstance(node, list):
            return any(has_open_object(child) for child in node)
        if not isinstance(node, dict):
            return False
        return isinstance(node.get('additionalProperties'), dict) or any(
            has_open_object(child) for child in node.values())

    # These overrides expand historical open dictionaries. A reviewed,
    # explicit replacement schema already defines its targets and must not
    # be forced back into the old document's object/array structure.
    overrides = _field_definitions().get(document_stem, []) if has_open_object(result) else []
    for override in overrides:
        node = result
        for key in override["schema_path"]:
            node = node[key]
        template = node.get("additionalProperties")
        if not isinstance(template, dict):
            raise ValueError(f"{document_stem}: schema override no longer matches its open object")
        node["properties"] = {name: copy.deepcopy(template) for name in override["field_names"]}
        node["additionalProperties"] = False

    def visit(node: dict, root: bool = False) -> None:
        if not isinstance(node, dict):
            return
        kind = node.get("type")
        kinds = kind if isinstance(kind, list) else [kind] if kind else []
        if "object" in kinds:
            if not node.get("properties") and isinstance(node.get("additionalProperties"), dict):
                raise ValueError("Extraction schema has an open object without explicit target fields")
            for child in (node.get("properties") or {}).values():
                visit(child)
        if "array" in kinds:
            visit(node.get("items") or {})
        if not root and kinds and "null" not in kinds:
            node["type"] = kinds + ["null"]
        if not root and "enum" in node and None not in node["enum"]:
            node["enum"] = [*node["enum"], None]

    visit(result, root=True)
    return result


def repair_spanned_financial_rows(reference: dict, table_html: str) -> tuple[dict, bool]:
    """Map numeric cells to their annotated spanning header, preserving values.

    The old corpus builder indexed physical cells against unexpanded headers.
    A colspan=2 year therefore shifted values into the next year or col_3.
    Empty physical padding is not a separate semantic value column.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(table_html or "", "lxml")
    table = soup.find("table")
    if table is None:
        return reference, False
    rows = table.find_all("tr")
    if not rows:
        return reference, False
    header = rows[0].find_all(["th", "td"], recursive=False)
    if not any(int(c.get("colspan", 1)) > 1 for c in header):
        return reference, False
    names = [c.get_text(" ", strip=True) for c in header]
    if names != reference.get("columns"):
        raise ValueError("Financial reference header differs from its annotated table")
    expanded = [name for name, c in zip(names, header) for _ in range(int(c.get("colspan", 1)))]
    corrected = []
    for row in rows[1:]:
        cells = []
        for cell in row.find_all(["th", "td"], recursive=False):
            cells += [cell.get_text(" ", strip=True)] + [""] * (int(cell.get("colspan", 1)) - 1)
        if len(cells) != len(expanded):
            raise ValueError("Financial reference row width differs from expanded headers")
        values = {}
        for name, value in zip(expanded[1:], cells[1:]):
            if value:
                if name in values:
                    raise ValueError("Multiple nonempty cells under one financial header need review")
                values[name] = value
        corrected.append({"label": cells[0], "values": values})
    old_rows = reference.get("rows", [])
    if len(old_rows) != len(corrected) or any(
        old["label"] != new["label"] or sorted(old["values"].values()) != sorted(new["values"].values())
        for old, new in zip(old_rows, corrected)
    ):
        raise ValueError("Financial repair would alter reference text or amounts, rather than their column keys")
    if corrected == old_rows:
        return reference, False
    result = copy.deepcopy(reference)
    result["rows"] = corrected
    # This derived, malformed physical-column CSV is neither schema input nor
    # an extraction target. Keeping it would contradict the repaired rows.
    result.pop("csv", None)
    return result, True


SAMPLE_FILE = FIELD_FILE.with_name("schema_sample.json")


def sample_reference_id(gt) -> str:
    """Invalidate a frozen sample when keys, values or human exclusions change."""
    gold = gt.metadata.get("structured_gold_reference") or {}
    return contract_id({"schema": gt.json_schema, "values": gt.true_json,
                        "excluded": sorted(gold.get("excluded_fields", []))})


@lru_cache(maxsize=1)
def sample_manifest() -> dict:
    return json.loads(SAMPLE_FILE.read_text()) if SAMPLE_FILE.exists() else {}


def sample_entry(gt) -> dict | None:
    if gt.metadata.get("schema_sample_applied"):
        return None
    entry = sample_manifest().get("documents", {}).get(Path(gt.document_path).stem)
    if entry and entry["reference_id"] != sample_reference_id(gt):
        raise ValueError(f"{Path(gt.document_path).stem}: schema sample is stale; rebuild it before inference/scoring")
    return entry


def restrict_schema(schema: dict, keys: list[str]) -> dict:
    """Preserve literal flat field names/types; all engines receive this subset."""
    result = copy.deepcopy(schema)
    properties = result.get("properties", {})
    if len(keys) != len(set(keys)) or any(key not in properties for key in keys):
        raise ValueError("Schema sample has duplicate or unknown keys")
    result["properties"] = {key: properties[key] for key in keys}
    result["required"] = [key for key in schema.get("required", []) if key in keys]
    return result


def sampled_request_schema(gt) -> dict:
    schema = extraction_schema(gt.json_schema, Path(gt.document_path).stem)
    entry = sample_entry(gt)
    return restrict_schema(schema, entry["keys"]) if entry else schema


def schema_scoring_reference(gt):
    """A schema-only view. Never mutate the full gold or KV reference."""
    from dataclasses import replace
    entry = sample_entry(gt)
    if entry is None:
        return gt
    schema = sampled_request_schema(gt)
    metadata = dict(gt.metadata)
    metadata["schema_sample_applied"] = True
    metadata["schema_sample"] = {"version": sample_manifest()["version"],
        "reference_id": entry["reference_id"], "selected": len(entry["keys"]),
        "available": entry["available"], "keys": entry["keys"]}
    return replace(gt, json_schema=schema,
        true_json={key: gt.true_json[key] for key in entry["keys"]}, metadata=metadata)
