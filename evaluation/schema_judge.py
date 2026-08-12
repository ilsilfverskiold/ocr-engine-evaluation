"""Schema leg — LLM judge.

The deterministic field-accuracy scorer required engine output to arrive under
the exact schema key it was asked for.  That penalised engines whose JSON
shape differed (different nesting, different key casing) and made native-KV
engines like Textract unscorable without a hand-written label mapping.

This judge replaces it.  The checklist is built from the ground-truth
``true_json`` — every leaf field becomes one checkable item — and an LLM
decides whether each value appears correctly in the engine's output, whatever
format that output is in (JSON, FORMS pairs, raw text).

Stage 1  Build the checklist deterministically from ``true_json`` + the
         document's ``json_schema``.  No LLM call; cached once per document.
Stage 2  For each (document, engine), show the judge the checklist and the
         engine's output.  Exact substring matches are settled locally;
         the LLM handles the rest.  Verdicts are cached.
Stage 3  ``field_accuracy = correct / total``.

Cost: one Gemini 2.5 Pro call per (document, engine) pair at stage 2,
minus any items settled by exact match.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.structured_metrics import (
    Field,
    FieldVerdict,
    is_blank,
    normalize_string,
    walk_fields,
)

MODEL = "gemini-2.5-pro"   # override with --judge-model (env OCR_JUDGE_MODEL)

PRICING = {
    "gemini-2.5-flash":      {"input": 0.30 / 1_000_000, "output": 2.50 / 1_000_000},
    "gemini-2.5-pro":        {"input": 1.25 / 1_000_000, "output": 10.00 / 1_000_000},
    "gemini-3.5-flash":      {"input": 1.50 / 1_000_000, "output": 9.00 / 1_000_000},
    "gemini-3.5-flash-lite": {"input": 0.30 / 1_000_000, "output": 2.50 / 1_000_000},
    "gemini-3.6-flash":      {"input": 1.50 / 1_000_000, "output": 7.50 / 1_000_000},
}


def _model() -> str:
    return os.environ.get("OCR_JUDGE_MODEL") or MODEL


def _thinking_config(model: str):
    from google.genai import types
    if model == "gemini-2.5-pro":
        return None          # dynamic thinking — the pre-override behaviour
    if model in ("gemini-2.5-flash", "gemini-3.5-flash"):
        return types.ThinkingConfig(thinking_budget=0)
    if model in ("gemini-3.5-flash-lite", "gemini-3.6-flash"):
        return types.ThinkingConfig(thinking_level="MINIMAL")
    return None

CACHE_DIR = Path(__file__).parent.parent / "results" / "schema_judge_cache"

MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 30

MAX_CHECKLIST_ITEMS = 80


# ---------------------------------------------------------------------------
# Stage 1 — build the checklist from GT, deterministically
# ---------------------------------------------------------------------------

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
) -> list[SchemaChecklistItem]:
    suffix = "_kv" if scalar_only else ""
    cache_path = CACHE_DIR / f"{document_stem}{suffix}.json"
    if use_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        items = [SchemaChecklistItem(**i) for i in cached["items"]]
        return _sample_checklist(items)

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
    return _sample_checklist(items)


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


# ---------------------------------------------------------------------------
# Stage 2 — judge each field against the engine output
# ---------------------------------------------------------------------------

_VERDICT_PROMPT = """\
You are checking whether an OCR engine correctly extracted specific field \
values from a document.

Below is a numbered list of fields with their expected values, taken from \
the document's ground truth. The engine's output follows — it may be JSON, \
key-value pairs, or plain text.

For each numbered field, decide:

- "correct"   — the value appears in the output and matches (same meaning, \
same digits; minor formatting differences like whitespace, casing, currency \
symbols, or date format are OK)
- "wrong"     — a value for this field is present but incorrect (wrong \
digits, wrong name, wrong amount)
- "missing"   — no value for this field appears in the output at all
- "blank_ok"  — the expected value is blank/null AND the engine also left \
it blank or omitted it (both agree the field is empty)
- "invented"  — the expected value is blank/null BUT the engine filled in \
a value (hallucinated content for an empty field)

Important: match by VALUE, not by field name. The engine may use different \
field names, different nesting, or no field names at all. What matters is \
whether the correct value is present and attributable to the right concept.

FIELDS TO CHECK:
{items}

ENGINE OUTPUT:
```
{engine_text}
```

Output JSON only, one entry per numbered field, same order:
{{"verdicts": [{{"n": 1, "verdict": "correct"}}]}}"""


_VERDICT_SCHEMA = {
    "type": "OBJECT",
    "required": ["verdicts"],
    "properties": {
        "verdicts": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["n", "verdict"],
                "properties": {
                    "n": {"type": "INTEGER"},
                    "verdict": {
                        "type": "STRING",
                        "enum": [
                            "correct", "wrong", "missing",
                            "blank_ok", "invented",
                        ],
                    },
                },
            },
        }
    },
}


@dataclass
class SchemaFieldVerdict:
    path: str
    expected: str
    verdict: str   # correct | wrong | missing | blank_ok | invented
    is_blank: bool = False


@dataclass
class SchemaJudgeResult:
    field_accuracy: float | None
    items: list[SchemaFieldVerdict] = field(default_factory=list)
    checklist_size: int = 0
    cost_usd: float = 0.0
    error: str | None = None

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.verdict] = counts.get(item.verdict, 0) + 1
        return counts


def _make_client():
    from google import genai
    return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def _call(client, contents, schema) -> tuple[dict, object]:
    from google.genai import types

    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            model = _model()
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0,
                    thinking_config=_thinking_config(model),
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            return json.loads(response.text), getattr(response, "usage_metadata", None)
        except Exception as exc:
            last = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP_SECONDS)
    raise RuntimeError(f"schema judge failed after {MAX_RETRIES} attempts: {last}")


def _cost(usage) -> float:
    if usage is None:
        return 0.0
    prices = PRICING.get(_model(), PRICING[MODEL])
    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
    return prompt_tokens * prices["input"] + output_tokens * prices["output"]


def _verify_locally(item: SchemaChecklistItem, engine_text: str) -> str | None:
    if item.is_blank:
        return None
    if not item.value:
        return None

    normed = re.sub(r"[^a-z0-9]", "", item.value.lower())
    text_normed = re.sub(r"[^a-z0-9]", "", engine_text.lower())
    if not normed:
        return None
    if normed in text_normed:
        return "correct"
    return None


def _verdict_cache_path(document_stem: str, engine: str) -> Path:
    return CACHE_DIR / "verdicts" / f"{document_stem}__{engine}.json"


def judge_schema(
    true_json: Any,
    json_schema: dict | None,
    engine_text: str,
    engine: str,
    document_stem: str,
    structured_output: dict | None = None,
    client=None,
    use_cache: bool = True,
    scalar_only: bool = False,
) -> SchemaJudgeResult:
    try:
        cache_path = _verdict_cache_path(document_stem, engine)
        if use_cache and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            items = [SchemaFieldVerdict(**i) for i in cached["items"]]
            correct = sum(
                1 for i in items if i.verdict in ("correct", "blank_ok")
            )
            return SchemaJudgeResult(
                field_accuracy=correct / len(items) if items else None,
                items=items,
                checklist_size=len(items),
                cost_usd=0.0,
            )

        checklist = build_schema_checklist(
            true_json, json_schema, document_stem, use_cache,
            scalar_only=scalar_only,
        )
        if not checklist:
            return SchemaJudgeResult(
                field_accuracy=None, error="empty schema checklist"
            )

        output_for_judge = _prepare_engine_output(
            engine_text, structured_output
        )

        verdicts: dict[int, str] = {}
        undecided: list[tuple[int, SchemaChecklistItem]] = []
        for idx, item in enumerate(checklist):
            local = _verify_locally(item, output_for_judge)
            if local:
                verdicts[idx] = local
            else:
                undecided.append((idx, item))

        verdict_cost = 0.0
        if undecided:
            if client is None:
                client = _make_client()

            listing = "\n".join(
                f"{n}. {_format_item(item)}"
                for n, (_, item) in enumerate(undecided, 1)
            )
            data, usage = _call(
                client,
                [_VERDICT_PROMPT.format(
                    items=listing,
                    engine_text=output_for_judge[:60_000],
                )],
                _VERDICT_SCHEMA,
            )
            verdict_cost = _cost(usage)
            returned = {
                v.get("n"): v.get("verdict") for v in data.get("verdicts", [])
            }
            for n, (idx, item) in enumerate(undecided, 1):
                v = returned.get(n, "missing")
                if v not in ("correct", "wrong", "missing", "blank_ok", "invented"):
                    v = "missing"
                verdicts[idx] = v

        results = [
            SchemaFieldVerdict(
                path=item.path,
                expected=item.value,
                verdict=verdicts.get(idx, "missing"),
                is_blank=item.is_blank,
            )
            for idx, item in enumerate(checklist)
        ]

        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({
                    "document": document_stem,
                    "engine": engine,
                    "items": [
                        {
                            "path": i.path,
                            "expected": i.expected,
                            "verdict": i.verdict,
                            "is_blank": i.is_blank,
                        }
                        for i in results
                    ],
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        correct = sum(
            1 for r in results if r.verdict in ("correct", "blank_ok")
        )
        return SchemaJudgeResult(
            field_accuracy=correct / len(results) if results else None,
            items=results,
            checklist_size=len(results),
            cost_usd=verdict_cost,
        )
    except Exception as exc:
        return SchemaJudgeResult(
            field_accuracy=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def _format_item(item: SchemaChecklistItem) -> str:
    if item.is_blank:
        return f"{item.path}: (should be blank/null/empty)"
    return f"{item.path}: {item.value}"


def _prepare_engine_output(
    engine_text: str,
    structured_output: dict | None,
) -> str:
    parts: list[str] = []

    forms = (structured_output or {}).get("forms")
    if forms:
        parts.append("=== FORMS KEY-VALUE PAIRS ===")
        for pair in forms:
            key = pair.get("key", "")
            value = pair.get("value", "")
            parts.append(f"{key}: {value}")

    tables = (structured_output or {}).get("tables")
    if tables:
        parts.append("\n=== TABLES ===")
        for i, table in enumerate(tables):
            html = table.get("html", "") if isinstance(table, dict) else ""
            if html:
                parts.append(f"Table {i}: {html}")

    if parts:
        parts.append(f"\n=== RAW TEXT ===\n{engine_text}")
        return "\n".join(parts)

    return engine_text or ""
