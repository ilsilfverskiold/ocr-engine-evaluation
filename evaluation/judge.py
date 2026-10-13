"""Judge text against cached image checklists where reliable text GT is unavailable."""

from __future__ import annotations

from evaluation.artifacts import read_text as artifact_text, exists as artifact_exists

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "gemini-3.6-flash"
VERDICT_VERSION = "silver-document-v2-no-substring-2026-10-06"

PRICING = {
    "gemini-2.5-flash":      {"input": 0.30 / 1_000_000, "output": 2.50 / 1_000_000},
    "gemini-2.5-pro":        {"input": 1.25 / 1_000_000, "output": 10.00 / 1_000_000},
    "gemini-3.5-flash":      {"input": 1.50 / 1_000_000, "output": 9.00 / 1_000_000},
    "gemini-3.5-flash-lite": {"input": 0.30 / 1_000_000, "output": 2.50 / 1_000_000},
    # gemini-3.6-flash list price (ai.google.dev/gemini-api/docs/pricing, read
    # 2026-10-01): $0.75 / $3.75 per M tokens through 2026-12-31, $1.50 / $7.50
    # from 2027-01-01. The reference run (2026-08) was costed at the latter,
    # i.e. 2x the price in force; recorded judge costs are upper bounds.
    "gemini-3.6-flash":      {"input": 0.75 / 1_000_000, "output": 3.75 / 1_000_000},
}

THINKING_BUDGET_MODELS = {"gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.5-flash"}
THINKING_LEVEL_MODELS = {"gemini-3.5-flash-lite", "gemini-3.6-flash"}

# Where stage-1 checklists are cached. They are reused across every engine, so
# a full run builds each document's checklist once rather than fourteen times.
CACHE_DIR = Path(__file__).parent.parent / "results" / "judge_checklists"

MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 30

# Sample a fixed, evenly spaced checklist so dense pages do not dominate cost or coverage.
MAX_CHECKLIST_ITEMS = 40


# Stage 1 — build the checklist from the image, engine-blind

_CHECKLIST_PROMPT = """\
You are cataloguing the verifiable content of a document image so that OCR \
engines can later be tested against it.

List the specific, checkable facts printed on this page. A fact is something \
with an exact written form that a reader could confirm by looking: a field and \
its value, a heading, a table cell, a total, an identifier, a date, a name, an \
amount.

Rules:
- Transcribe each value EXACTLY as printed, including punctuation, casing and \
formatting. Do not normalise, correct or tidy anything.
- Give each item a short label describing what it is.
- Cover the whole page: header, body, tables, footer, stamps, marginalia.
- Include values that are hard to read, and mark those with "confident": false.
- Do NOT include anything you cannot actually see on the page.
- Do NOT include prose sentences from body paragraphs. This catalogues discrete \
values, not narrative text.
- Aim for the values that would distinguish a careful reader from a careless \
one: reference codes, digits, names, amounts.

Output JSON only:
{"items": [{"label": "...", "value": "...", "confident": true}]}"""


_VERDICT_PROMPT = """\
You are checking whether an OCR engine's output captured specific values from a \
document.

Below is a numbered list of values that are printed on the page, and the text \
an OCR engine produced from that page. For each numbered value, decide:

- "present_correct" — the value appears in the output and matches
- "present_wrong"   — the value appears but is wrong (a wrong digit, a \
misspelling, a mangled code)
- "absent"          — the value does not appear in the output at all

Check the complete value for the labeled item, not a substring inside another \
value. Ignore harmless whitespace, casing and layout differences, but preserve \
digits, decimal separators, signs, percentages and identifier punctuation when \
they change meaning. 100 does not match 1000; 12.34 does not match 1234; -100 \
does not match 100. A correct subtotal does not establish a correct total. \
A value split across lines may match only if the complete value and its \
association with the labeled item are preserved.

VALUES TO CHECK:
{items}

ENGINE OUTPUT:
```
{engine_text}
```

Output JSON only, one entry per numbered value, same order:
{{"verdicts": [{{"n": 1, "verdict": "present_correct"}}]}}"""


_CHECKLIST_SCHEMA = {
    "type": "OBJECT",
    "required": ["items"],
    "properties": {
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["label", "value", "confident"],
                "properties": {
                    "label": {"type": "STRING"},
                    "value": {"type": "STRING"},
                    "confident": {"type": "BOOLEAN"},
                },
            },
        }
    },
}

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
                        "enum": ["present_correct", "present_wrong", "absent"],
                    },
                },
            },
        }
    },
}


@dataclass
class ChecklistItem:
    label: str
    value: str
    confident: bool = True


@dataclass
class ItemVerdict:
    label: str
    value: str
    verdict: str          # present_correct | present_wrong | absent
    confident: bool = True


@dataclass
class JudgeResult:
    """Per-item verdicts are kept, not just the total, so any number can be
    traced back to the items behind it."""
    item_accuracy: float | None
    items: list[ItemVerdict] = field(default_factory=list)
    checklist_size: int = 0
    cost_usd: float = 0.0
    error: str | None = None

    def summary(self) -> dict:
        counts = {"present_correct": 0, "present_wrong": 0, "absent": 0}
        for item in self.items:
            counts[item.verdict] = counts.get(item.verdict, 0) + 1
        return counts


def make_client():
    from google import genai
    return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def _thinking_config(model: str):
    from google.genai import types
    if model in THINKING_BUDGET_MODELS:
        return types.ThinkingConfig(thinking_budget=0)
    if model in THINKING_LEVEL_MODELS:
        return types.ThinkingConfig(thinking_level="MINIMAL")
    return None


def _call(client, contents, schema, model: str = DEFAULT_MODEL) -> tuple[dict, object]:
    """One structured call, retried on transient failure."""
    from google.genai import types

    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
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
        except Exception as exc:                     # noqa: BLE001
            last = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP_SECONDS)
    raise RuntimeError(f"judge call failed after {MAX_RETRIES} attempts: {last}")


def _cost(usage, model: str = DEFAULT_MODEL) -> float:
    if usage is None:
        return 0.0
    prices = PRICING.get(model, PRICING[DEFAULT_MODEL])
    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
    thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0
    return (prompt_tokens * prices["input"]
            + (output_tokens + thinking_tokens) * prices["output"])


def _sample_checklist(items: list[ChecklistItem]) -> list[ChecklistItem]:
    """At most MAX_CHECKLIST_ITEMS, evenly spaced and order-preserving."""
    if len(items) <= MAX_CHECKLIST_ITEMS:
        return items
    step = len(items) / MAX_CHECKLIST_ITEMS
    return [items[int(i * step)] for i in range(MAX_CHECKLIST_ITEMS)]


def build_checklist(
    document_path: Path, client=None, use_cache: bool = True,
    model: str | None = None,
) -> tuple[list[ChecklistItem], float]:
    """Stage 1. Engine-blind, cached — every engine is checked against the same list."""
    model = model or os.environ.get("OCR_JUDGE_MODEL") or DEFAULT_MODEL
    cache_path = CACHE_DIR / f"{Path(document_path).stem}.json"
    if use_cache and artifact_exists(cache_path):
        cached = json.loads(artifact_text(cache_path))
        # Capped on the way out, not on the way in: a checklist cached before
        # the cap still yields the same sampled list every engine sees.
        return _sample_checklist([ChecklistItem(**i) for i in cached["items"]]), 0.0

    if client is None:
        client = make_client()

    from google.genai import types

    mime = "image/jpeg" if Path(document_path).suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    contents = [
        types.Part.from_bytes(data=Path(document_path).read_bytes(), mime_type=mime),
        _CHECKLIST_PROMPT,
    ]
    data, usage = _call(client, contents, _CHECKLIST_SCHEMA, model=model)
    items = [
        ChecklistItem(
            label=str(i.get("label", "")).strip(),
            value=str(i.get("value", "")).strip(),
            confident=bool(i.get("confident", True)),
        )
        for i in data.get("items", [])
        if str(i.get("value", "")).strip()
    ]

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"document": str(document_path), "items": [vars(i) for i in items]},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    # The full list is cached; the sampled one is what gets judged.
    return _sample_checklist(items), _cost(usage, model=model)


def judge_document(
    document_path: Path,
    engine_text: str,
    engine_name: str = "",
    client=None,
    use_cache: bool = True,
    model: str | None = None,
) -> JudgeResult:
    """Stages 1-3 for one (document, engine) pair.

    The model never returns a score. It returns observations; the arithmetic
    below is ours.
    """
    model = model or os.environ.get("OCR_JUDGE_MODEL") or DEFAULT_MODEL
    try:
        if client is None:
            client = make_client()

        items, checklist_cost = build_checklist(document_path, client, use_cache, model=model)
        if not items:
            return JudgeResult(item_accuracy=None, error="empty checklist")

        # Every sampled item needs a contextual verdict. The old punctuation-
        # stripped substring shortcut could accept 100 in 1000 or the wrong field.
        listing = "\n".join(
            f"{n}. {item.label}: {item.value}" for n, item in enumerate(items, 1)
        )
        data, usage = _call(
            client,
            [_VERDICT_PROMPT.format(items=listing, engine_text=engine_text[:60_000])],
            _VERDICT_SCHEMA,
            model=model,
        )
        verdict_cost = _cost(usage, model=model)
        returned = {v.get("n"): v.get("verdict") for v in data.get("verdicts", [])}
        verdicts = {idx: returned.get(idx + 1) or "absent" for idx in range(len(items))}

        results = [
            ItemVerdict(label=i.label, value=i.value,
                        verdict=verdicts.get(n, "absent"), confident=i.confident)
            for n, i in enumerate(items)
        ]
        correct = sum(1 for r in results if r.verdict == "present_correct")

        return JudgeResult(
            item_accuracy=correct / len(results),
            items=results,
            checklist_size=len(results),
            cost_usd=checklist_cost + verdict_cost,
        )
    except Exception as exc:                          # noqa: BLE001
        return JudgeResult(item_accuracy=None, error=f"{type(exc).__name__}: {exc}")
