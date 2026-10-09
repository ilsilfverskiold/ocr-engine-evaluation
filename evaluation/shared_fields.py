"""One reviewed reference set for KV/schema; cached KV decisions precede value scoring.

KV may consume versioned judge-approved synonyms. Scoring itself is offline.
Schema uses supplied key identities and penalizes unexpected output fields.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import re
from evaluation.artifacts import read_text as artifact_text, exists as artifact_exists, glob_paths
from pathlib import Path

from evaluation.kv_schema_code import FieldVerdict, is_blank_value, value_equal
from evaluation.structured_metrics import normalize_label, walk_fields

VERSION = "shared-fields-v1"
KV_VERSION = "shared-fields-v8-identifiers-native-checkboxes"
CACHE_ROOT = Path(__file__).resolve().parents[1] / "results" / "kv_synonyms"


def reference_id(gt) -> str:
    gold = gt.metadata.get("structured_gold_reference") or {}
    payload = {"reference": gt.true_json, "schema": gt.json_schema,
               "excluded_fields": sorted(gold.get("excluded_fields", [])),
               "aliases": gold.get("kv_aliases", {}), "image_sha256": gold.get("image_sha256")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def reference_fields(gt, leg=None):
    excluded = set((gt.metadata.get("structured_gold_reference") or {}).get("excluded_fields", []))
    if leg == 'key_value':
        excluded.update(path for path, item in gt.metadata.get('kv_label_review', {}).get('fields', {}).items()
                        if item['decision'] != 'keep')
    return [f for f in walk_fields(gt.true_json, None) if f.path and f.path not in excluded]


def load_synonym_decisions(gt) -> dict[str, str | None]:
    path = CACHE_ROOT / (Path(gt.document_path).stem + ".json")
    if not artifact_exists(path):
        return {}
    data = json.loads(artifact_text(path))
    if data.get("reference_id") != reference_id(gt):
        return {}  # A changed reference invalidates previous key decisions.
    allowed = {f.path for f in reference_fields(gt)}
    return {entry["output_key"]: entry["reference_key"] for entry in data.get("matches", [])
            if (entry.get("reference_key") in allowed or entry.get("reference_key") is None)
            and isinstance(entry.get("output_key"), str)}


def load_synonyms(gt) -> dict[str, str]:
    return {key: value for key, value in load_synonym_decisions(gt).items() if value is not None}


def _numbered_line_key(key: str) -> tuple[str, str] | None:
    """Equivalent printed line identifiers, e.g. 'b Taxable amount 4b'.

    Requires the same complete label and explicit number. Never uses values or
    strips a numeric difference, owner or subtotal qualification.
    """
    start = re.fullmatch(r"(\d{1,2}[a-z]?)\s*[—–-]\s*(.+)", key.strip(), re.I)
    if start:
        return start[1].casefold(), normalize_label(start[2])
    end = re.fullmatch(r"(?:(?P<letter>[a-z])\s+)?(?P<label>.+?)\s+(?P<line>\d{1,2}[a-z]?)", key.strip(), re.I)
    if end and (not end['letter'] or end['line'].casefold().endswith(end['letter'].casefold())):
        return end['line'].casefold(), normalize_label(end['label'])
    return None


def match_keys(gt, pairs, leg: str, synonyms: dict[str, str] | None = None, native_matches: dict | None = None):
    fields = reference_fields(gt, leg)
    candidates = defaultdict(set)
    gold = gt.metadata.get("structured_gold_reference") or {}
    for f in fields:
        candidates[normalize_label(f.path)].add(f.path)
        if leg == "key_value":
            for alias in (gold.get("kv_aliases") or {}).get(f.path, {}).get("printed_labels", []):
                candidates[normalize_label(alias)].add(f.path)
    by_path = {f.path: f for f in fields}
    numbered = defaultdict(set)
    for f in fields:
        if re.match(r"^\d{1,2}[a-z]?\s*[—–-]", f.path, re.I):
            numbered[_numbered_line_key(f.path)].add(f.path)
    proposals = [(-1, hit['index'], path, hit['how']) for path, hit in (native_matches or {}).items()
                 if path in by_path]
    unresolved = []
    for i, (key, value) in enumerate(pairs):
        key = str(key)
        if key in gold.get("excluded_fields", []):
            # Human-excluded fields do not need a synonym decision in either leg.
            continue
        if key in by_path:
            proposals.append((0, i, key, "exact key"))
        elif leg == "schema":
            # A supplied schema defines literal JSON property names.
            unresolved.append({"index": i, "key": key})
        else:
            known = candidates.get(normalize_label(key), set())
            if len(known) == 1:
                proposals.append((1, i, next(iter(known)), "unambiguous normalized key/printed alias"))
            elif len(numbered.get(_numbered_line_key(key), set())) == 1:
                proposals.append((1, i, next(iter(numbered[_numbered_line_key(key)])),
                                  "same explicit line number and complete normalized label"))
            else:
                from evaluation.kv_value_judge import pair_identity
                target = (synonyms or {}).get(pair_identity(key, value), (synonyms or {}).get(key))
                if target in by_path:
                    proposals.append((2, i, target, "cached judge-confirmed KV mapping"))
                else:
                    unresolved.append({"index": i, "key": key})
    assigned, used = {}, set()
    for _, i, key, how in sorted(proposals):
        if key not in assigned and i not in used:
            assigned[key] = (i, how)
            used.add(i)
    unresolved = [item for item in unresolved if item['index'] not in used]
    return assigned, unresolved, used


@dataclass
class Result:
    accuracy: float
    unpenalized_accuracy: float
    total: int
    correct: int
    missing: int
    wrong: int
    invented: int
    blank_total: int
    blank_correct: int
    extras: list[dict] = field(default_factory=list)
    unresolved: list[dict] = field(default_factory=list)
    verdicts: list[FieldVerdict] = field(default_factory=list)


def score_shared(gt, pairs: list, leg: str, confirmed_synonyms: dict | None = None, confirmed_groups: list | None = None, native_matches: dict | None = None, pair_context=None) -> Result:
    fields = reference_fields(gt, leg)
    synonyms = load_synonyms(gt) if leg == "key_value" else {}
    if leg == "key_value":
        synonyms.update(confirmed_synonyms or {})
    assigned, unresolved, used = match_keys(gt, pairs, leg, synonyms, native_matches)
    protected = {v['index'] for v in (native_matches or {}).values()}
    groups = {g['reference_key']: g for g in (confirmed_groups or [])
              if g['reference_key'] not in (native_matches or {}) and not protected.intersection(g['indices'])} if leg == "key_value" else {}
    group_indices = {i for g in groups.values() for i in g['indices']}
    unresolved = [item for item in unresolved if item['index'] not in group_indices]
    verdicts = []
    for f in fields:
        blank = is_blank_value(f.value)
        hit = assigned.get(f.path)
        key, value, how = "", "", ("no confirmed key match after value-assisted resolution; see unmatched-pair decisions"
                                    if leg == "key_value" else "no matching supplied schema key")
        if f.path in groups:
            group = groups[f.path]
            key = " + ".join(str(pairs[i][0]) for i in group['indices'])
            value = " | ".join(str(v) for v in group.get('evaluated_values', [pairs[i][1] for i in group['indices']]))
            how = "judge-confirmed value group; request " + group['request_id']
            verdict = "blank_ok" if blank else "correct"
        elif hit is None:
            verdict = "missing"
        else:
            i, how = hit
            key, value = pairs[i]
            if leg == 'key_value' and pair_context:
                from evaluation.kv_schema_code import contextual_value
                interpreted = contextual_value(f.value, value, pair_context[i])
                if interpreted != value:
                    how += '; explicit native checkbox ' + str(interpreted)
                    value = interpreted
            if blank:
                verdict = "blank_ok" if is_blank_value(value) else "invented"
            else:
                verdict = "correct" if value_equal(f.value, value, [f.path]) else "wrong"
        verdicts.append(FieldVerdict(f.path, f.value, blank, verdict, str(key), str(value), how,
                                     printed_pair=True if leg == "key_value" else None))
    extras = []
    if leg == "schema":
        # Explicitly excluded reference fields are still legitimate requested
        # keys. Their presence is neither credit nor an unexpected-key penalty.
        all_paths = {f.path for f in walk_fields(gt.true_json, gt.json_schema) if f.path}
        counts = Counter()
        for i, (key, value) in enumerate(pairs):
            counts[key] += 1
            allowed = key in all_paths or _contract_allows_key(str(key), gt.inference_schema or {})
            if not allowed or counts[key] > 1:
                extras.append({"key": key, "value": value,
                               "reason": "duplicate key" if counts[key] > 1 else "unexpected key"})
    counts = Counter(v.verdict for v in verdicts)
    correct = counts["correct"] + counts["blank_ok"]
    total = len(fields)
    return Result(correct / (total + len(extras)) if total + len(extras) else 0,
                  correct / total if total else 0, total, counts["correct"], counts["missing"],
                  counts["wrong"], counts["invented"], sum(v.is_blank for v in verdicts),
                  counts["blank_ok"], extras, unresolved, verdicts)


def _contract_allows_key(path: str, schema: dict) -> bool:
    if path in schema.get("properties", {}):
        return True
    # Flattening uses dots and array indices, while a flat reviewed key may
    # itself contain dots. Literal properties therefore take precedence.
    parts = [p for p in re.split(r"\.|\[(\d+)\]", path) if p]
    node = schema
    for part in parts:
        if not isinstance(node, dict):
            return bool(node)
        kind = node.get("type")
        if kind == "array" or isinstance(kind, list) and "array" in kind:
            if not part.isdigit():
                return False
            node = node.get("items", {})
            continue
        props = node.get("properties", {})
        if part in props:
            node = props[part]
            continue
        patterns = [v for pattern, v in node.get("patternProperties", {}).items() if re.search(pattern, part)]
        if patterns:
            node = patterns[0]
            continue
        allowed = node.get("additionalProperties", True)
        if allowed is False:
            return False
        node = allowed
    return True


def schema_compliance(text: str, schema: dict | None) -> dict:
    """Separate usability diagnostic; recovering values does not fix the JSON."""
    import jsonschema
    duplicates = []
    def obj(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                duplicates.append(key)
            result[key] = value
        return result
    try:
        data = json.loads(text, object_pairs_hook=obj,
                          parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {x}")))
    except (ValueError, TypeError) as exc:
        return {"valid_json": False, "schema_valid": False, "errors": [str(exc)]}
    errors = [f"duplicate key: {key}" for key in duplicates]
    if schema:
        validator = jsonschema.validators.validator_for(schema)(schema)
        errors += [f"{e.json_path}: {e.message}" for e in validator.iter_errors(data)]
    return {"valid_json": not duplicates, "schema_valid": not errors if schema else None, "errors": errors}
