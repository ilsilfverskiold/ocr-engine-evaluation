"""KV judge request journaling, spending limits and the per-document rulings file.

One file per document, keyed by its reference fields. Under each field:
`expected` (the reference value), `synonyms` (label means this field on this
document, any engine), `pairs` (label only with this exact value) and
`output_rulings` (accepted/rejected decisions about one engine output, with the
pairs looked at; reused only for that identical output). Full requests and
replies stay in the local, untracked calls/ journal.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from evaluation.artifacts import read_text as artifact_text, exists as artifact_exists
from pathlib import Path
from threading import RLock

from evaluation.shared_fields import reference_fields, reference_id
from evaluation.structured_metrics import normalize_label

MODEL = "gemini-2.5-flash"
PRICES = {"input_per_million": .30, "output_per_million": 2.50,
          "source": "https://ai.google.dev/gemini-api/docs/pricing", "checked": "2026-10-06"}
CONTROL_CONFIG = {"temperature": 0, "max_output_tokens": 4096, "thinking_budget": 1024}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def decode_reply(text: str) -> dict:
    """JSON's usual last-key-wins behavior must not conceal duplicate decisions."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON property: {key}")
            result[key] = value
        return result
    reply = json.loads(text, object_pairs_hook=unique)
    if not isinstance(reply, dict):
        raise ValueError("Judge response must be an object")
    return reply


VERSION = "kv-field-rulings-v1"
CACHE_ROOT = Path(__file__).resolve().parents[1] / "results/kv_synonyms/value_assisted"
_locks = defaultdict(RLock)
_budget_lock = RLock()
_reserved_usd = 0.0


def pair_identity(key, value) -> tuple[str, str]:
    return str(key), digest(value)


def _path(gt):
    return CACHE_ROOT / (Path(gt.document_path).stem + ".json")


def _read(gt):
    path = _path(gt)
    if artifact_exists(path):
        record = json.loads(artifact_text(path))
        if record.get("version") == VERSION and record.get("reference_id") == reference_id(gt):
            return record
    return {"version": VERSION, "reference_id": reference_id(gt), "fields": {}}


def repeated_label(path, gt) -> bool:
    """A label printed for several reference fields (item vs summary "Net worth")
    is never a value-independent synonym; it stays an exact key+value pair."""
    leaf = lambda key: normalize_label(key.split(' — ')[-1])
    return sum(leaf(f.path) == leaf(path) for f in reference_fields(gt)) > 1


def _entries(record, gt, path, kind):
    field = record["fields"].setdefault(path, {})
    if "expected" not in field:
        field["expected"] = next(f.value for f in reference_fields(gt) if f.path == path)
    return field.setdefault(kind, [])


def _identities(record):
    return {pair_identity(e["key"], e["value"]): path
            for path, kinds in record["fields"].items()
            for kind in ("synonyms", "pairs") for e in kinds.get(kind, [])}


def add_synonym(record, gt, path, key, value, request_id):
    """A judge-approved label for a field. Repeated labels are never synonyms;
    their approval stays with the output ruling it came from."""
    key = str(key)
    if any(e["key"] == key for e in record["fields"].get(path, {}).get("synonyms", [])):
        return
    if repeated_label(path, gt) or pair_identity(key, value) in _identities(record):
        return
    _entries(record, gt, path, "synonyms").append({"key": key, "value": value, "request_id": request_id})


def looked_at(pairs, output_ids):
    """Engine pairs by 1-based output ID (the numbering the judge saw)."""
    return [{"output_id": i, "key": str(pairs[i - 1][0]), "value": pairs[i - 1][1]} for i in output_ids]


def add_output_ruling(record, gt, path, ruling):
    if ruling not in record["fields"].get(path, {}).get("output_rulings", []):
        _entries(record, gt, path, "output_rulings").append(ruling)


def output_rulings(record, output):
    """This output's reviews, as the matcher uses them (reference field + indices)."""
    return [{"reference_key": path,
             "indices": [p["output_id"] - 1 for p in r["looked_at"]] if r["verdict"] == "accepted" else [],
             **{k: v for k, v in r.items() if k not in ("output", "engine", "verdict", "looked_at")}}
            for path, kinds in record["fields"].items()
            for r in kinds.get("output_rulings", []) if r["output"] == output]


def _cache_maps(record):
    """Label -> field (unambiguous synonyms) plus exact (label, value) -> field."""
    aliases = defaultdict(set)
    for path, kinds in record["fields"].items():
        for e in kinds.get("synonyms", []):
            aliases[e["key"]].add(path)
    # Conflicting synonyms never resolve by last-write-wins.
    result = {key: next(iter(paths)) for key, paths in aliases.items() if len(paths) == 1}
    result.update(_identities(record))
    return result


def validate(request, reply):
    from evaluation.kv_value_groups import validate
    return validate(request, reply)


def _ask(request):
    """One journaled SDK attempt. Interrupted/invalid calls are never auto-retried."""
    global _reserved_usd
    path = CACHE_ROOT / "calls" / (request["id"] + ".json")
    if artifact_exists(path):
        saved = json.loads(artifact_text(path))
        if "raw_text" not in saved:
            raise RuntimeError(f"Unfinished paid request needs review: {path}")
        return validate(request, decode_reply(saved["raw_text"])), 0.0
    reserve = request["planning_reserve_usd"]
    budget = float(os.environ.get("OCR_KV_JUDGE_BUDGET_USD", "2"))
    with _budget_lock:
        if not 0 < budget < float("inf") or _reserved_usd + reserve > budget:
            raise RuntimeError(f"KV judge planning reserve would exceed ${budget:.2f}; remaining pairs stay pending")
        _reserved_usd += reserve
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"request": request, "reserved_usd": reserve}
    with path.open("x") as stream:
        json.dump(record, stream, ensure_ascii=False)
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"],
                         http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)))
    try:
        response = client.models.generate_content(model=MODEL, contents=request["prompt"],
            config=types.GenerateContentConfig(temperature=0, max_output_tokens=CONTROL_CONFIG["max_output_tokens"],
                thinking_config=types.ThinkingConfig(thinking_budget=CONTROL_CONFIG["thinking_budget"]),
                response_mime_type="application/json", response_schema=request["response_schema"]))
        usage = response.usage_metadata
        inputs = getattr(usage, "prompt_token_count", 0) or 0
        outputs = (getattr(usage, "candidates_token_count", 0) or 0) + (getattr(usage, "thoughts_token_count", 0) or 0)
        cost = (inputs * PRICES["input_per_million"] + outputs * PRICES["output_per_million"]) / 1_000_000
        record.update(raw_text=response.text, raw_response=response.model_dump(mode="json"),
                      input_tokens=inputs, output_tokens=outputs, cost_usd=cost)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        with _budget_lock:
            _reserved_usd += cost - reserve
        entries = validate(request, decode_reply(response.text))
        return entries, cost
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        raise
    finally:
        client.close()


def _save(gt, record):
    path = _path(gt)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    temp.replace(path)


def estimate_saved(sources):
    """Estimate the current contextual evaluator, not the legacy pair-only plan."""
    from evaluation.kv_value_groups import plan_saved
    plan = plan_saved(sources)
    return {key: value for key, value in plan.items() if key != 'requests'}
