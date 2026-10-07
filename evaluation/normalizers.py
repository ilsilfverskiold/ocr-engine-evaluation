"""
Per-engine output normalizers — the "post-processing layer" that real document
pipelines have to build and maintain themselves.

Every engine hands back something different: Textract returns a block graph
(rendered here as key-value pairs + cell grids), the layout parsers return
markdown with pipe tables, the VLMs return whatever the prompt asked for, and
the text-first engines return a wall of text. Downstream systems want one of
four shapes:

    TEXT      plain text                      (every engine)
    MARKDOWN  markdown in reading order       (layout parsers, VLMs)
    TABLES    list of row-dicts / CSV         (structure-producing engines)
    FIELDS    {label: value} key-value pairs  (Textract FORMS, schema-JSON VLMs)

This module converts each engine's native output into those shapes, so the
benchmark can compare engines on what a consumer actually gets rather than on
raw strings. Where an engine cannot produce a shape, the converter returns
None — that absence is a real, reportable result, not an error.

Table parsing (markdown/TSV → HTML) already exists in table_extractor.py and is
reused here rather than reimplemented.
"""

from __future__ import annotations

import csv
import io
import json
import re
from typing import Any

from evaluation.table_extractor import extract_table_html

# ---------------------------------------------------------------------------
# Capability map — derived from the registry, never restated
# ---------------------------------------------------------------------------
#
# These used to be hand-maintained name sets. A set that disagrees with
# engines.py silently changes what an engine is scored on — an engine the
# registry enters in the markdown leg but that is missing from the set here
# returns None and looks like it produced nothing. Deriving them means
# `engines.py` stays the one place an engine is declared.

from engines import ENGINES as _ENGINES

# Engines that emit markdown from their default conversion.
MARKDOWN_ENGINES = frozenset(
    name for name, spec in _ENGINES.items() if "markdown" in spec.native_legs
)

# Engines that return schema-conforming JSON when prompted with a schema.
SCHEMA_CAPABLE_ENGINES = frozenset(
    name for name, spec in _ENGINES.items() if "schema" in spec.prompted_legs
)

# Engines whose runner returns a structured block graph rather than text: their
# tables and fields are read from structured_output, not parsed out of a string.
NATIVE_STRUCTURE_ENGINES = frozenset(
    name for name, spec in _ENGINES.items()
    if "tables" in spec.native_legs and spec.table_format is None
)


# ---------------------------------------------------------------------------
# Textract block graph → tables / fields
# ---------------------------------------------------------------------------

def textract_tables(structured_output: dict) -> list[list[dict[str, str]]] | None:
    """Textract TABLES (rendered as HTML by the runner) → list of row-dicts per table.

    This is the traversal every Textract user writes: the API gives cells with
    row/column indices, not records. Real scans force two fixes that a naive
    walk gets wrong:
      * duplicate/blank column headers (financial statements repeat "£'000"),
      * the first column being a row label rather than a value.
    """
    tables = (structured_output or {}).get("tables") or []
    if not tables:
        return None
    out = []
    for table in tables:
        html = table.get("html") if isinstance(table, dict) else None
        if html:
            rows = html_table_to_rows(html)
            if rows:
                out.append(rows)
    return out or None


def html_table_to_rows(html: str) -> list[dict[str, str]]:
    """Parse an HTML table into row-dicts keyed by (disambiguated) header names."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    grid = [
        [cell.get_text(strip=True) for cell in tr.find_all(["td", "th"])]
        for tr in soup.find_all("tr")
    ]
    grid = [r for r in grid if any(c for c in r)]
    if len(grid) < 2:
        return []

    header = grid[0]
    columns = _disambiguate_columns(header)
    return [
        {col: val for col, val in zip(columns, row) if val}
        for row in grid[1:]
    ]


def _disambiguate_columns(header: list[str]) -> list[str]:
    """Blank headers become col_N; repeated headers get _1, _2 suffixes."""
    counts: dict[str, int] = {}
    columns = []
    for i, raw in enumerate(header):
        name = raw.strip() or f"col_{i}"
        if raw.strip() and header.count(raw) > 1:
            counts[name] = counts.get(name, 0) + 1
            name = f"{name}_{counts[name]}"
        columns.append(name)
    return columns


def textract_fields(structured_output: dict) -> dict[str, str] | None:
    """Textract FORMS key-value pairs → a {label: value} dict.

    Normalizes what the API leaves raw: trailing colons on labels, whitespace,
    and duplicate labels (suffixed rather than silently overwritten).
    """
    pairs = (structured_output or {}).get("forms") or []
    if not pairs:
        return None
    fields: dict[str, str] = {}
    for pair in pairs:
        key = _clean_label(pair.get("key", ""))
        if not key:
            continue
        value = (pair.get("value") or "").strip()
        if key in fields and fields[key] != value:
            n = 2
            while f"{key}_{n}" in fields:
                n += 1
            key = f"{key}_{n}"
        fields[key] = value
    return fields or None


def _clean_label(label: str) -> str:
    return re.sub(r"\s+", " ", (label or "").strip()).rstrip(":").strip()


# ---------------------------------------------------------------------------
# Markdown-emitting engines → tables
# ---------------------------------------------------------------------------

def markdown_tables(raw_text: str, engine: str) -> list[list[dict[str, str]]] | None:
    """Markdown pipe tables / TSV → row-dicts, via the existing table extractor."""
    html = extract_table_html(raw_text or "", engine)
    if not html:
        return None
    rows = html_table_to_rows(html)
    return [rows] if rows else None


def markdown_text(raw_text: str, engine: str) -> str | None:
    """The engine's markdown, or None if it only emits flat text."""
    if engine not in MARKDOWN_ENGINES:
        return None
    return raw_text


def markdown_for_scoring(raw_text: str, engine: str, structured_output=None) -> str | None:
    """Select the same emitted Markdown for scoring and evidence display."""
    rendered = (structured_output or {}).get("markdown")
    if rendered:
        return rendered
    if _ENGINES[engine].answers("markdown") == "prompted":
        return raw_text
    return markdown_text(raw_text, engine)


# ---------------------------------------------------------------------------
# Schema-JSON engines → fields
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def parse_schema_json(raw_text: str) -> dict | None:
    """Parse a VLM's JSON reply, tolerating code fences and surrounding prose."""
    if not raw_text:
        return None
    candidate = _FENCE_RE.sub("", raw_text).strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} span.
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    span = candidate[start:end + 1]
    try:
        parsed = json.loads(span)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    for repair in (_escape_inner_quotes, _drop_valueless_entries,
                   lambda s: _drop_valueless_entries(_escape_inner_quotes(s))):
        repaired = repair(span)
        if repaired == span:
            continue
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def _drop_valueless_entries(span: str) -> str:
    """Remove object entries that are a bare string with no value.

    GLM-OCR sometimes emits a footnote from the page as an entry with no
    value — `"ENDING VALUE": "$706,538.11", "Net Change in Portfolio is the
    difference between..."` — which is invalid JSON and threw away the whole
    reply, including the dozens of fields it had extracted correctly. Only
    strings sitting immediately before a closing brace are removed, so
    elements of a string array are untouched.
    """
    return re.sub(r',\s*"(?:[^"\\]|\\.)*"\s*(?=\})', "", span)


def _escape_inner_quotes(span: str) -> str:
    """Escape stray double quotes inside JSON string values.

    A page that prints inch marks defeats every model that does not escape
    them: GLM-OCR returned `"item_desc": "12" Marble ... 15" Wooden Stand"`,
    which is invalid JSON, so a reply with every field read correctly parsed
    to nothing and scored zero. A quote is treated as content when the text
    after it does not continue the JSON grammar (`,` `}` `]` or `:` next).
    The reply is still recorded as invalid JSON — this only stops one
    unescaped character from erasing the whole extraction.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    for i, char in enumerate(span):
        if escaped:
            out.append(char)
            escaped = False
            continue
        if char == "\\":
            out.append(char)
            escaped = True
            continue
        if char == '"':
            if not in_string:
                in_string = True
            else:
                following = span[i + 1:i + 12].lstrip()
                if following[:1] in {",", "}", "]", ":", ""}:
                    in_string = False
                else:
                    out.append("\\")          # content, not a delimiter
        out.append(char)
    return "".join(out)


def flatten_fields(obj: Any, prefix: str = "") -> dict[str, str]:
    """Flatten nested JSON to dotted paths, so schema JSON is comparable to
    flat key-value output: {"header": {"invoice_no": "X"}} → {"header.invoice_no": "X"}.
    Arrays are indexed: items[0].item_desc.
    """
    flat: dict[str, str] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            flat.update(flatten_fields(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            flat.update(flatten_fields(value, f"{prefix}[{i}]"))
    else:
        flat[prefix] = "" if obj is None else str(obj)
    return flat


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def rows_to_csv(rows: list[dict[str, str]]) -> str:
    """Row-dicts → CSV text (the shape analysts actually load)."""
    if not rows:
        return ""
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def normalize(engine: str, raw_text: str, structured_output: dict | None = None) -> dict:
    """Convert one engine's output into the four consumer shapes.

    Returns {"text", "markdown", "tables", "fields"} where unavailable shapes
    are None. `tables` is a list of tables, each a list of row-dicts.
    """
    structured_output = structured_output or {}

    if engine in NATIVE_STRUCTURE_ENGINES:
        tables = textract_tables(structured_output)
        fields = textract_fields(structured_output)
    else:
        tables = markdown_tables(raw_text, engine)
        parsed = parse_schema_json(raw_text) if engine in SCHEMA_CAPABLE_ENGINES else None
        fields = flatten_fields(parsed) if parsed else None

    return {
        "text": raw_text,
        "markdown": markdown_text(raw_text, engine),
        "tables": tables,
        "fields": fields,
    }
