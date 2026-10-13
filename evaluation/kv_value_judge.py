"""KV judge request journaling, spending limits and saved-decision compatibility."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from functools import lru_cache
from evaluation.artifacts import read_text as artifact_text, exists as artifact_exists, glob_paths
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


VERSION = "kv-value-assisted-v1"
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
            _limit_repeated_aliases(record['decisions'], [f.path for f in reference_fields(gt)])
            return record
    return {"version": VERSION, "reference_id": reference_id(gt), "decisions": []}


def _limit_repeated_aliases(entries, keys):
    # Repeated contextual labels remain exact-pair confirmations even when the
    # model overstates their suitability as unconditional aliases.
    leaf = lambda key: normalize_label(key.split(' — ')[-1])
    counts = Counter(leaf(key) for key in keys)
    for entry in entries:
        target = entry.get('reference_key')
        if target and entry.get('synonym') and counts[leaf(target)] > 1:
            entry['synonym'] = False
            entry['alias_scope_reason'] = 'repeated reference label; exact key/value confirmation only'


@lru_cache(maxsize=4)
def _saved_calls(root):
    """Index earlier paid replies once on startup, independent of batch grouping."""
    by_reference = defaultdict(list)
    for path in sorted(glob_paths(root / 'calls', '*.json')):
        saved = json.loads(artifact_text(path))
        request = saved.get('request', {})
        if request.get('version') == VERSION and saved.get('raw_text'):
            try:
                entries = validate(request, decode_reply(saved['raw_text']))
            except (ValueError, TypeError):
                continue  # Preserve malformed paid replies for manual review.
            by_reference[request['reference_id']].extend(entries)
    return by_reference


def _recover_saved(gt):
    record = _read(gt)
    known = {pair_identity(d['key'], d['value']) for d in record['decisions']}
    added = False
    for entry in _saved_calls(CACHE_ROOT).get(record['reference_id'], []):
        identity = pair_identity(entry['key'], entry['value'])
        if identity not in known:
            record['decisions'].append(entry)
            known.add(identity)
            added = True
    if added:
        _save(gt, record)


def _cache_maps(record):
    decisions, aliases = {}, defaultdict(set)
    for entry in record["decisions"]:
        decisions[pair_identity(entry["key"], entry["value"])] = entry["reference_key"]
        if entry.get("synonym") and entry["reference_key"] is not None:
            aliases[entry["key"]].add(entry["reference_key"])
    # Conflicting aliases never resolve by last-write-wins.
    result = {key: next(iter(refs)) for key, refs in aliases.items() if len(refs) == 1}
    result.update({key: value for key, value in decisions.items() if value is not None})
    return decisions, result


def validate(request, reply):
    if request.get('version') == 'kv-context-groups-v2':
        from evaluation.kv_value_groups import validate
        return validate(request, reply)
    pairs = request["output_pairs"]
    if set(reply) != {str(p["id"]) for p in pairs}:
        raise ValueError("Missing or invented pair IDs")
    refs = {r["id"]: r["key"] for r in request["reference_fields"]}
    entries = []
    for pair in pairs:
        row = reply[str(pair["id"])]
        if not isinstance(row, dict) or set(row) != {"reference_id", "synonym"}:
            raise ValueError("Invalid pair verdict")
        rid = row["reference_id"]
        if type(rid) is not int or rid not in [0, *refs]:
            raise ValueError("Invented reference field ID")
        if type(row["synonym"]) is not bool or (not rid and row["synonym"]):
            raise ValueError("Invalid synonym flag")
        # A real field with a wrong value is an incorrect pair, not a broken
        # JSON batch. Keep the proposal for audit, reject credit, and preserve
        # valid decisions for the other pairs without paying for a retry.
        eligible = rid in pair['candidate_ids']
        entries.append({"key": pair["key"], "value": pair["value"],
                        "reference_key": refs.get(rid) if eligible else None,
                        "synonym": row["synonym"] if eligible else False,
                        "rejected_value_proposal": refs.get(rid) if rid and not eligible else None,
                        "request_id": request["id"]})
    _limit_repeated_aliases(entries, list(refs.values()))
    return entries


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
