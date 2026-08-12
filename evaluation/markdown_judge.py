"""Markdown leg — LLM judge.

The deterministic scorer compares engine output against hand-written markdown
GT.  Those references are over-segmented: lease_000 has 52 paragraph blocks
where a human sees 12 logical elements.  Engines that correctly merge text
into flowing prose cannot match the fragmented blocks and score low.

This judge replaces the deterministic scorer as headline.  Like the table
judge it builds its own reference from the page image — a checklist of
structural elements (headings, paragraphs, list items, table rows) that a
reader would recognise — and then checks each engine's markdown against it.

Stage 1  Show a capable model the page image.  It catalogues every
         structural element: type, content, position.  Cached once per
         document; every engine is measured against the identical checklist.
Stage 2  For each (document, engine), show the checklist and the engine's
         markdown output.  The LLM judges each item.  Verdicts are cached.
Stage 3  ``element_accuracy = (correct + 0.5*partial) / total``.

Cost: one Stage 1 call per document (shared, cached), one Stage 2 call per
(document, engine) pair (cached).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CHECKLIST_MODEL = "gemini-2.5-pro"
DEFAULT_VERDICT_MODEL = "gemini-3.6-flash"

PRICING = {
    "gemini-2.5-flash":      {"input": 0.30 / 1_000_000, "output": 2.50 / 1_000_000},
    "gemini-2.5-pro":        {"input": 1.25 / 1_000_000, "output": 10.00 / 1_000_000},
    "gemini-3.1-flash-lite": {"input": 0.25 / 1_000_000, "output": 1.50 / 1_000_000},
    "gemini-3.5-flash":      {"input": 1.50 / 1_000_000, "output": 9.00 / 1_000_000},
    "gemini-3.5-flash-lite": {"input": 0.30 / 1_000_000, "output": 2.50 / 1_000_000},
    "gemini-3.6-flash":      {"input": 1.50 / 1_000_000, "output": 7.50 / 1_000_000},
}

THINKING_BUDGET_MODELS = {"gemini-2.5-flash", "gemini-3.5-flash"}
THINKING_MUST_THINK_MODELS = {"gemini-2.5-pro"}
THINKING_LEVEL_MODELS = {"gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.6-flash"}

CACHE_DIR = Path(__file__).parent.parent / "results" / "markdown_judge_cache"

MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 30

MAX_CHECKLIST_ITEMS = 40


# ---------------------------------------------------------------------------
# Stage 1 — build the structural element checklist from the image
# ---------------------------------------------------------------------------

_CHECKLIST_PROMPT = """\
You are cataloguing every structural element on this document page so that \
OCR engines can later be tested on markdown conversion.

List every element a reader would recognise as a distinct structural unit. \
An element is one logical block of content — what would become one markdown \
block in an ideal conversion.

Element types:
- "heading": section titles, document titles, any text that functions as a \
heading (bold standalone lines count).
- "paragraph": body text blocks. For paragraphs longer than 50 words, \
include only the first ~50 words as a fingerprint — enough to identify it.
- "list_item": individual items in a bulleted or numbered list.
- "table_row": one row of a table, with cell values separated by " | ". \
For charts and graphs with labeled data points, represent the data as \
table rows (one row per data series or category, with columns for each \
axis value) — this is what a correct markdown conversion would produce.
- "key_value": a printed label and its filled-in value on the page, like \
"Date: 2025-12-31" or "Name: John Smith". Common in forms.

For each element record:
- "type": one of the types above.
- "content": the text content. Transcribe exactly as printed — do not \
normalise, correct, or reformat. For paragraphs longer than 50 words, \
include only the first ~50 words.
- "notes": very brief context, e.g. "section 3 heading" or "second row \
of the totals table".

List elements in reading order (top to bottom, left to right).

Do NOT list:
- Page numbers, headers/footers that are not part of the document content.
- Empty or blank elements.
- Decorative lines or separators.

Output JSON only:
{"items": [{"type": "heading", "content": "...", "notes": "..."}]}"""


_VERDICT_PROMPT = """\
You are checking whether an OCR engine correctly converted a document page \
to markdown.

Below is a numbered list of structural elements that appear on the page, \
with their expected type and content. The engine's markdown output follows.

For each numbered element, decide:

- "correct"    — the content appears in the output and is rendered as the \
right structural type (heading as heading, list item as list item, table \
row as table row, paragraph as paragraph text). Minor differences in \
wording, whitespace, punctuation, or casing are OK. Key-value items are \
correct if the label and value both appear, in any format.
- "wrong_type" — the content is present but rendered as the wrong \
structural type (e.g. a heading rendered as a plain paragraph, a \
list item rendered as a heading, or a table rendered as plain text \
without markdown table syntax). If the text content is there but \
has no markdown formatting at all, that is wrong_type, not absent.
- "partial"    — the element is present but the content is noticeably \
incomplete (significant portions missing) or has meaningful errors \
(wrong numbers, wrong names). Minor OCR errors are NOT partial — they \
are still correct.
- "absent"     — the content does not appear in the output at all, or \
only a trivial fragment appears.

ELEMENTS TO CHECK:
{items}

ENGINE MARKDOWN OUTPUT:
```
{engine_text}
```

Output JSON only, one entry per numbered element, same order:
{{"verdicts": [{{"n": 1, "verdict": "correct"}}]}}"""


_CHECKLIST_SCHEMA = {
    "type": "OBJECT",
    "required": ["items"],
    "properties": {
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["type", "content"],
                "properties": {
                    "type": {
                        "type": "STRING",
                        "enum": ["heading", "paragraph", "list_item",
                                 "table_row", "key_value"],
                    },
                    "content": {"type": "STRING"},
                    "notes": {"type": "STRING"},
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
                        "enum": ["correct", "wrong_type", "partial", "absent"],
                    },
                },
            },
        }
    },
}


@dataclass
class MarkdownChecklistItem:
    type: str
    content: str
    notes: str = ""


@dataclass
class MarkdownElementVerdict:
    type: str
    content: str
    notes: str
    verdict: str   # correct | wrong_type | partial | absent


@dataclass
class MarkdownJudgeResult:
    element_accuracy: float | None
    items: list[MarkdownElementVerdict] = field(default_factory=list)
    checklist_size: int = 0
    full_checklist_size: int = 0
    sampled: bool = False
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    checklist_model: str | None = None
    verdict_model: str | None = None
    error: str | None = None

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.verdict] = counts.get(item.verdict, 0) + 1
        return counts


def _make_client():
    from google import genai
    return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def _thinking_config(model: str):
    from google.genai import types
    if model in THINKING_MUST_THINK_MODELS:
        return types.ThinkingConfig(thinking_budget=1024)
    if model in THINKING_BUDGET_MODELS:
        return types.ThinkingConfig(thinking_budget=0)
    if model in THINKING_LEVEL_MODELS:
        return types.ThinkingConfig(thinking_level="MINIMAL")
    return None


def _call(client, contents, schema, model: str) -> tuple[dict, object]:
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
        except Exception as exc:
            last = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP_SECONDS)
    raise RuntimeError(f"markdown judge failed after {MAX_RETRIES} attempts: {last}")


def _cost(usage, model: str) -> float:
    if usage is None:
        return 0.0
    prices = PRICING.get(model, PRICING[DEFAULT_VERDICT_MODEL])
    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
    thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0
    return (prompt_tokens * prices["input"]
            + (output_tokens + thinking_tokens) * prices["output"])


def _sample_checklist(
    items: list[MarkdownChecklistItem],
) -> list[MarkdownChecklistItem]:
    if len(items) <= MAX_CHECKLIST_ITEMS:
        return items
    step = len(items) / MAX_CHECKLIST_ITEMS
    return [items[int(i * step)] for i in range(MAX_CHECKLIST_ITEMS)]


# ---------------------------------------------------------------------------
# Stage 1 — build checklist
# ---------------------------------------------------------------------------

def build_markdown_checklist(
    document_path: Path, client=None, use_cache: bool = True,
    model: str = DEFAULT_CHECKLIST_MODEL,
) -> tuple[list[MarkdownChecklistItem], float, int, bool, object]:
    """Stage 1.  Engine-blind, cached per document.

    Returns (sampled_items, cost, full_count, was_sampled, usage_metadata).
    """
    cache_path = CACHE_DIR / f"{Path(document_path).stem}.json"
    if use_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        items = [MarkdownChecklistItem(**i) for i in cached["items"]
                 if str(i.get("content", "")).strip()]
        full_count = len(items)
        sampled = full_count > MAX_CHECKLIST_ITEMS
        return _sample_checklist(items), 0.0, full_count, sampled, None

    if client is None:
        client = _make_client()

    from google.genai import types

    suffix = Path(document_path).suffix.lower()
    if suffix == ".pdf":
        mime = "application/pdf"
    elif suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    else:
        mime = "image/png"

    contents = [
        types.Part.from_bytes(
            data=Path(document_path).read_bytes(), mime_type=mime
        ),
        _CHECKLIST_PROMPT,
    ]
    data, usage = _call(client, contents, _CHECKLIST_SCHEMA, model=model)
    items = [
        MarkdownChecklistItem(
            type=str(i.get("type", "paragraph")).strip(),
            content=str(i.get("content", "")).strip(),
            notes=str(i.get("notes", "")).strip(),
        )
        for i in data.get("items", [])
        if str(i.get("content", "")).strip()
    ]

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({
                "document": str(document_path),
                "model": model,
                "items": [
                    {"type": i.type, "content": i.content, "notes": i.notes}
                    for i in items
                ],
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    full_count = len(items)
    sampled = full_count > MAX_CHECKLIST_ITEMS
    return _sample_checklist(items), _cost(usage, model), full_count, sampled, usage


# ---------------------------------------------------------------------------
# Stage 2 — judge each element against the engine output
# ---------------------------------------------------------------------------

def _verdict_cache_path(document_stem: str, engine: str) -> Path:
    return CACHE_DIR / "verdicts" / f"{document_stem}__{engine}.json"


def _format_element(n: int, item: MarkdownChecklistItem) -> str:
    return f"{n}. [{item.type}] {item.content}"


def judge_markdown(
    document_path: Path,
    engine_text: str,
    engine_name: str,
    client=None,
    use_cache: bool = True,
    checklist_model: str = DEFAULT_CHECKLIST_MODEL,
    verdict_model: str | None = None,
) -> MarkdownJudgeResult:
    """Stages 1-3 for one (document, engine) pair.

    ``verdict_model`` honours the --judge-model override (OCR_JUDGE_MODEL);
    the checklist model stays fixed because checklists are cached and shared.
    """
    verdict_model = (verdict_model or os.environ.get("OCR_JUDGE_MODEL")
                     or DEFAULT_VERDICT_MODEL)
    try:
        cache_path = _verdict_cache_path(
            Path(document_path).stem, engine_name
        )
        if use_cache and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            items = [MarkdownElementVerdict(**i) for i in cached["items"]
                     if str(i.get("content", "")).strip()]
            correct = sum(1 for i in items if i.verdict == "correct")
            partial = sum(1 for i in items
                          if i.verdict in ("wrong_type", "partial"))
            return MarkdownJudgeResult(
                element_accuracy=(correct + 0.5 * partial) / len(items) if items else None,
                items=items,
                checklist_size=len(items),
                cost_usd=0.0,
                checklist_model=cached.get("checklist_model"),
                verdict_model=cached.get("verdict_model"),
            )

        if client is None:
            client = _make_client()

        items, checklist_cost, full_count, was_sampled, cl_usage = build_markdown_checklist(
            document_path, client, use_cache, model=checklist_model,
        )
        if not items:
            return MarkdownJudgeResult(
                element_accuracy=None, error="no structural elements found on page"
            )

        listing = "\n".join(
            _format_element(n, item)
            for n, item in enumerate(items, 1)
        )
        data, usage = _call(
            client,
            [_VERDICT_PROMPT.format(
                items=listing,
                engine_text=(engine_text or "")[:60_000],
            )],
            _VERDICT_SCHEMA,
            model=verdict_model,
        )
        verdict_cost = _cost(usage, verdict_model)
        returned = {
            v.get("n"): v.get("verdict")
            for v in data.get("verdicts", [])
        }
        results = []
        for n, item in enumerate(items, 1):
            v = returned.get(n, "absent")
            if v not in ("correct", "wrong_type", "partial", "absent"):
                v = "absent"
            results.append(MarkdownElementVerdict(
                type=item.type,
                content=item.content,
                notes=item.notes,
                verdict=v,
            ))

        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({
                    "document": str(document_path),
                    "engine": engine_name,
                    "checklist_model": checklist_model,
                    "verdict_model": verdict_model,
                    "items": [
                        {
                            "type": i.type,
                            "content": i.content,
                            "notes": i.notes,
                            "verdict": i.verdict,
                        }
                        for i in results
                    ],
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        correct = sum(1 for r in results if r.verdict == "correct")
        partial_count = sum(1 for r in results
                            if r.verdict in ("wrong_type", "partial"))
        total = len(results)
        accuracy = (correct + 0.5 * partial_count) / total if total else None

        def _tokens(u, attr):
            return getattr(u, attr, 0) or 0

        return MarkdownJudgeResult(
            element_accuracy=accuracy,
            items=results,
            checklist_size=total,
            full_checklist_size=full_count,
            sampled=was_sampled,
            cost_usd=checklist_cost + verdict_cost,
            prompt_tokens=(
                _tokens(cl_usage, "prompt_token_count")
                + _tokens(usage, "prompt_token_count")
            ),
            output_tokens=(
                _tokens(cl_usage, "candidates_token_count")
                + _tokens(usage, "candidates_token_count")
            ),
            thinking_tokens=(
                _tokens(cl_usage, "thoughts_token_count")
                + _tokens(usage, "thoughts_token_count")
            ),
            checklist_model=checklist_model,
            verdict_model=verdict_model,
        )
    except Exception as exc:
        return MarkdownJudgeResult(
            element_accuracy=None,
            error=f"{type(exc).__name__}: {exc}",
        )
