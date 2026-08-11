"""Table leg — LLM judge.

TEDS compares engine output against hand-built ``table_html`` references.
Those references are inconsistent: five are derived from schema field names
rather than printed page columns, markdown-converted references lose
colspan/rowspan, and some document types have no table reference at all.

This judge replaces TEDS as the headline scorer.  Like the silver-GT judge
it builds its own reference from the page image — Gemini identifies every
table cell on the page — and then checks each engine's output against that
reference.  The result is independent of whatever ``table_html`` the corpus
ships.

Stage 1  Show Gemini the page image.  It catalogues every table cell:
         row index, column index, exact value.  Cached once per document;
         every engine is measured against the identical checklist.
Stage 2  For each (document, engine), show the checklist and the engine's
         table output.  Exact substring matches are settled locally; the
         LLM handles the rest.  Verdicts are cached.
Stage 3  ``cell_accuracy = correct / total``.

Cost: one Gemini 2.5 Pro image call per document at stage 1 (shared), one
text call per (document, engine) at stage 2.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "gemini-3.6-flash"

PRICING = {
    "gemini-2.5-flash":      {"input": 0.30 / 1_000_000, "output": 2.50 / 1_000_000},
    "gemini-2.5-pro":        {"input": 1.25 / 1_000_000, "output": 10.00 / 1_000_000},
    "gemini-3.5-flash":      {"input": 1.50 / 1_000_000, "output": 9.00 / 1_000_000},
    "gemini-3.5-flash-lite": {"input": 0.30 / 1_000_000, "output": 2.50 / 1_000_000},
    "gemini-3.6-flash":      {"input": 1.50 / 1_000_000, "output": 7.50 / 1_000_000},
}

THINKING_BUDGET_MODELS = {"gemini-2.5-flash", "gemini-3.5-flash"}
THINKING_MUST_THINK_MODELS = {"gemini-2.5-pro"}
THINKING_LEVEL_MODELS = {"gemini-3.5-flash-lite", "gemini-3.6-flash"}

CACHE_DIR = Path(__file__).parent.parent / "results" / "table_judge_cache"

MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 30

MAX_CHECKLIST_ITEMS = 60


# ---------------------------------------------------------------------------
# Stage 1 — build the cell checklist from the image, engine-blind
# ---------------------------------------------------------------------------

_CHECKLIST_PROMPT = """\
You are cataloguing every table cell on this document page so that OCR \
engines can later be tested on table extraction.

For each table on the page, list every cell. A cell is one value occupying \
one position in the table grid.

Include:
- Column header cells and row label cells.
- Data cells: numbers, text, dates, codes, amounts.
Skip empty or blank cells — only catalogue cells that contain visible text.

For each cell record:
- "label": a short description, e.g. "Table 1, Row 2, Col 3" or \
"Header: Amount".
- "value": the EXACT text printed in the cell. Do not normalise, correct \
or reformat anything. Transcribe numbers, punctuation, casing and \
whitespace exactly as they appear.
- "row": 1-based row index from the top of the table (header row is 1).
- "col": 1-based column index from the left.

If a cell spans multiple columns or rows, list it once at its top-left \
position.

If the page has multiple tables, include all cells from each table \
and prefix labels with the table number.

If the page has NO tables at all, return {"items": []}.

Output JSON only:
{"items": [{"label": "...", "value": "...", "row": 1, "col": 1}]}"""


_VERDICT_PROMPT = """\
You are checking whether an OCR engine correctly extracted table cells from \
a document.

Below is a numbered list of cells that appear in the page's tables, with \
their expected values and positions (row, column). The engine's output \
follows — it may be a Markdown pipe table, HTML table, or plain text.

For each numbered cell, decide:

- "correct"       — the value appears in the output in approximately the \
right table position (same row/column or clearly the same cell, even if \
rows/columns shifted by one). Minor formatting differences (whitespace, \
casing, currency symbols, date formats) are OK.
- "wrong"         — a value is present in approximately the right position \
but it is incorrect (wrong digits, wrong text).
- "misplaced"     — the value appears somewhere in the output but in the \
wrong position or outside any table structure.
- "absent"        — the value does not appear in the output at all.

CELLS TO CHECK:
{items}

ENGINE OUTPUT:
```
{engine_text}
```

Output JSON only, one entry per numbered cell, same order:
{{"verdicts": [{{"n": 1, "verdict": "correct"}}]}}"""


_CHECKLIST_SCHEMA = {
    "type": "OBJECT",
    "required": ["items"],
    "properties": {
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["label", "value", "row", "col"],
                "properties": {
                    "label": {"type": "STRING"},
                    "value": {"type": "STRING"},
                    "row": {"type": "INTEGER"},
                    "col": {"type": "INTEGER"},
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
                        "enum": [
                            "correct", "wrong", "misplaced",
                            "absent",
                        ],
                    },
                },
            },
        }
    },
}


@dataclass
class TableChecklistItem:
    label: str
    value: str
    row: int
    col: int
    empty: bool = False


@dataclass
class TableCellVerdict:
    label: str
    value: str
    row: int
    col: int
    verdict: str   # correct | wrong | misplaced | absent
    empty: bool = False


@dataclass
class TableJudgeResult:
    cell_accuracy: float | None
    items: list[TableCellVerdict] = field(default_factory=list)
    checklist_size: int = 0
    full_checklist_size: int = 0
    sampled: bool = False
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
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


def _call(client, contents, schema, model: str = DEFAULT_MODEL) -> tuple[dict, object]:
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
    raise RuntimeError(f"table judge failed after {MAX_RETRIES} attempts: {last}")


def _cost(usage, model: str = DEFAULT_MODEL) -> float:
    if usage is None:
        return 0.0
    prices = PRICING.get(model, PRICING[DEFAULT_MODEL])
    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
    thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0
    return (prompt_tokens * prices["input"]
            + (output_tokens + thinking_tokens) * prices["output"])


def _sample_checklist(
    items: list[TableChecklistItem],
) -> list[TableChecklistItem]:
    if len(items) <= MAX_CHECKLIST_ITEMS:
        return items
    step = len(items) / MAX_CHECKLIST_ITEMS
    return [items[int(i * step)] for i in range(MAX_CHECKLIST_ITEMS)]


def build_table_checklist(
    document_path: Path, client=None, use_cache: bool = True,
    model: str = DEFAULT_MODEL,
) -> tuple[list[TableChecklistItem], float, int, bool]:
    """Stage 1.  Engine-blind, cached — every engine is checked against the same list.

    Returns (sampled_items, cost, full_count, was_sampled).
    """
    cache_path = CACHE_DIR / f"{Path(document_path).stem}.json"
    if use_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        items = [TableChecklistItem(**i) for i in cached["items"]
                 if str(i.get("value", "")).strip()]
        full_count = len(items)
        sampled = full_count > MAX_CHECKLIST_ITEMS
        return _sample_checklist(items), 0.0, full_count, sampled

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
        TableChecklistItem(
            label=str(i.get("label", "")).strip(),
            value=str(i.get("value", "")).strip(),
            row=int(i.get("row", 0)),
            col=int(i.get("col", 0)),
            empty=bool(i.get("empty", False)),
        )
        for i in data.get("items", [])
        if str(i.get("value", "")).strip()
    ]

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({
                "document": str(document_path),
                "items": [
                    {
                        "label": i.label, "value": i.value,
                        "row": i.row, "col": i.col, "empty": i.empty,
                    }
                    for i in items
                ],
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    full_count = len(items)
    sampled = full_count > MAX_CHECKLIST_ITEMS
    return _sample_checklist(items), _cost(usage, model=model), full_count, sampled


# ---------------------------------------------------------------------------
# Stage 2 — judge each cell against the engine output
# ---------------------------------------------------------------------------


def _verdict_cache_path(document_stem: str, engine: str) -> Path:
    return CACHE_DIR / "verdicts" / f"{document_stem}__{engine}.json"


def _format_cell(n: int, item: TableChecklistItem) -> str:
    pos = f"(row {item.row}, col {item.col})"
    if item.empty:
        return f"{n}. {item.label} {pos}: (empty cell)"
    return f"{n}. {item.label} {pos}: {item.value}"


def _prepare_engine_output(
    engine_text: str,
    structured_output: dict | None,
) -> str:
    parts: list[str] = []
    tables = (structured_output or {}).get("tables") or []
    for i, table in enumerate(tables):
        html = (table or {}).get("html")
        if html:
            parts.append(f"=== TABLE {i} (HTML) ===\n{html}")
    if parts:
        parts.append(f"\n=== RAW TEXT ===\n{engine_text}")
        return "\n".join(parts)
    return engine_text or ""


def judge_table(
    document_path: Path,
    engine_text: str,
    engine_name: str,
    structured_output: dict | None = None,
    client=None,
    use_cache: bool = True,
    model: str = DEFAULT_MODEL,
) -> TableJudgeResult:
    """Stages 1-3 for one (document, engine) pair."""
    try:
        cache_path = _verdict_cache_path(
            Path(document_path).stem, engine_name
        )
        if use_cache and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            items = [TableCellVerdict(**i) for i in cached["items"]
                     if str(i.get("value", "")).strip()]
            correct = sum(
                1 for i in items
                if i.verdict == "correct"
            )
            return TableJudgeResult(
                cell_accuracy=correct / len(items) if items else None,
                items=items,
                checklist_size=len(items),
                cost_usd=0.0,
            )

        if client is None:
            client = _make_client()

        items, checklist_cost, full_count, was_sampled = build_table_checklist(
            document_path, client, use_cache, model=model,
        )
        if not items:
            return TableJudgeResult(
                cell_accuracy=None, error="no table cells found on page"
            )

        output_for_judge = _prepare_engine_output(
            engine_text, structured_output
        )

        listing = "\n".join(
            _format_cell(n, item)
            for n, item in enumerate(items, 1)
        )
        data, usage = _call(
            client,
            [_VERDICT_PROMPT.format(
                items=listing,
                engine_text=output_for_judge[:60_000],
            )],
            _VERDICT_SCHEMA,
            model=model,
        )
        verdict_cost = _cost(usage, model=model)
        returned = {
            v.get("n"): v.get("verdict")
            for v in data.get("verdicts", [])
        }
        verdicts: dict[int, str] = {}
        for n, (idx, item) in enumerate(enumerate(items), 1):
            v = returned.get(n, "absent")
            if v not in (
                "correct", "wrong", "misplaced",
                "absent",
            ):
                v = "absent"
            verdicts[idx] = v

        results = [
            TableCellVerdict(
                label=item.label,
                value=item.value,
                row=item.row,
                col=item.col,
                verdict=verdicts.get(idx, "absent"),
                empty=item.empty,
            )
            for idx, item in enumerate(items)
        ]

        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({
                    "document": str(document_path),
                    "engine": engine_name,
                    "items": [
                        {
                            "label": i.label,
                            "value": i.value,
                            "row": i.row,
                            "col": i.col,
                            "verdict": i.verdict,
                            "empty": i.empty,
                        }
                        for i in results
                    ],
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        correct = sum(
            1 for r in results
            if r.verdict == "correct"
        )
        return TableJudgeResult(
            cell_accuracy=correct / len(results) if results else None,
            items=results,
            checklist_size=len(results),
            full_checklist_size=full_count,
            sampled=was_sampled,
            cost_usd=checklist_cost + verdict_cost,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            thinking_tokens=getattr(usage, "thoughts_token_count", 0) or 0,
        )
    except Exception as exc:
        return TableJudgeResult(
            cell_accuracy=None,
            error=f"{type(exc).__name__}: {exc}",
        )
