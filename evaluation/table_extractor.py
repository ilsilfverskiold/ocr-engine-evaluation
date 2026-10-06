"""
Table extractor — converts each engine's raw text output into an HTML table
string suitable for TEDS scoring. This is the deterministic conversion code the
benchmark publishes so that fixed pipelines can compete in the table leg.

Which strategy an engine needs is declared once, in `EngineSpec.table_format`:

  - "markdown"  pipe tables → parsed via the `markdown` library (tables
                extension) + BeautifulSoup
  - "html"      the model already emits <table> markup; strip presentation
                attributes and use it directly
  - "tsv"       split on tabs, build <table><tr><td> rows
  - None        flat text, no recoverable structure → return None, which the
                table leg records as a real result rather than an error

That declaration used to live here as three name sets, which meant the registry
could say an engine competes in the table leg while this module silently
returned None for it and scored it 0.000. An engine unknown to the registry is
sniffed rather than assumed flat, so the converter is usable outside the
benchmark's own engine list.

Note: glm_ocr and paddleocr_vl were once (incorrectly) treated as flat-text
engines and excluded from table scoring. Inspection of their saved outputs
shows glm_ocr emits markdown pipe tables and paddleocr_vl emits HTML directly.
"""

from __future__ import annotations

import re
from html import escape

MARKDOWN, HTML, TSV = "markdown", "html", "tsv"


def _declared_format(engine: str) -> str | None:
    from engines import ENGINES

    spec = ENGINES.get(engine)
    return spec.table_format if spec else "auto"


def _sniff_format(text: str) -> str | None:
    """Best guess for an engine the registry does not know about."""
    if re.search(r"<table[\s>]", text or "", re.IGNORECASE):
        return HTML
    if re.search(r"^\s*\|.*\|\s*$", text or "", re.MULTILINE):
        return MARKDOWN
    if any(line.count("\t") >= 1 for line in (text or "").splitlines()):
        return TSV
    return None


def extract_table_html(text: str, engine: str) -> str | None:
    """
    Return an HTML <table>...</table> string extracted from `text`,
    or None if the engine produces flat text with no recoverable structure.
    """
    table_format = _declared_format(engine)
    if table_format == "auto":
        table_format = _sniff_format(text)
    if table_format == MARKDOWN:
        return _markdown_to_html(_clean_markdown(text, engine))
    if table_format == HTML:
        return _html_to_html(text)
    if table_format == TSV:
        return _tsv_to_html(text)
    return None  # flat-text engines


# ---------------------------------------------------------------------------
# Markdown path
# ---------------------------------------------------------------------------

def _clean_markdown(text: str, engine: str) -> str:
    """Pre-process markdown before parsing, handling engine-specific noise."""
    # An answer wrapped entirely in ```markdown fences parses as a code block,
    # so its tables disappear. See markdown_leg.strip_wrapping_fence.
    from evaluation.markdown_leg import strip_wrapping_fence
    text = strip_wrapping_fence(text)
    # A pipe table glued to the line above it ("**Dependents:**\n| a | b |")
    # is one paragraph to the markdown parser and its rows vanish; Gemini
    # writes every 1040 table that way. Open a block before the table.
    lines = text.split("\n")
    out: list[str] = []
    for i, line in enumerate(lines):
        is_row = line.lstrip().startswith("|")
        prev = out[-1] if out else ""
        if is_row and prev.strip() and not prev.lstrip().startswith("|"):
            out.append("")
        out.append(line)
    text = "\n".join(out)
    if engine == "marker":
        # Strip inline HTML tags (<b>, <i>, etc.)
        text = re.sub(r"<[^>]+>", "", text)
        # Replace LaTeX like $\overline{c}$ with just the inner letter/number
        text = re.sub(r"\$\\overline\{([^}]+)\}\$", r"\1", text)
        # Fix spaces inserted inside parenthesised negatives: (4, 878) → (4,878)
        text = re.sub(r"\((\d[\d,]*),\s+(\d[\d,]*)\)", r"(\1,\2)", text)
    return text


def _markdown_to_html(text: str) -> str | None:
    """Parse markdown pipe tables → first <table> HTML string, or None."""
    try:
        import markdown
        from bs4 import BeautifulSoup

        html = markdown.markdown(text, extensions=["tables", "fenced_code"])
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table")
        if table is None:
            return None
        # Normalise: strip alignment classes/styles added by the markdown lib
        for tag in table.find_all(True):
            tag.attrs = {k: v for k, v in tag.attrs.items() if k in ("colspan", "rowspan")}
        return str(table)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Native HTML path
# ---------------------------------------------------------------------------

def _html_to_html(text: str) -> str | None:
    """Take the model's own <table> markup, stripped of presentation attributes.

    PaddleOCR-VL emits styled HTML (border=1, inline text-align styles); the
    presentation attributes carry no structural meaning. Keep rowspan/colspan:
    removing them shifts numeric cells relative to their headers.
    """
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(text, "lxml")
        table = soup.find("table")
        if table is None:
            return None
        for tag in table.find_all(True):
            tag.attrs = {k: v for k, v in tag.attrs.items() if k in ("colspan", "rowspan")}
        return str(table)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# TSV path
# ---------------------------------------------------------------------------

def _tsv_to_html(text: str) -> str | None:
    """
    Convert tab-separated content to an HTML table.
    Lines with fewer than 2 tab-delimited tokens are treated as
    section headers / blank separators and wrapped in a single full-width cell.
    """
    all_lines = [l for l in text.splitlines() if l.strip()]
    if not all_lines:
        return None

    # Isolate the table before measuring it. Taking the mode over the whole
    # page makes the table invisible whenever prose outnumbers it: mineru
    # returns a full invoice as text with one tab-separated table inside, the
    # mode came out at one column, and the extractor reported "not a table" —
    # so a correctly parsed table scored zero. Use the longest run of
    # consecutive tab-bearing lines instead.
    runs: list[list[str]] = []
    current: list[str] = []
    for line in all_lines:
        if "\t" in line:
            current.append(line)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    # The first table on the page, matching the markdown and HTML paths
    # (both take soup.find("table")) and the references, which transcribe the
    # document's first table. Taking the longest run instead would score an
    # invoice's SUMMARY block against its ITEMS reference.
    lines = next((run for run in runs if len(run) >= 2), [])
    if len(lines) < 2:
        return None

    # Determine the most common column count (mode) to use as table width
    from collections import Counter
    col_counts = Counter(len(l.split("\t")) for l in lines)
    expected_cols = col_counts.most_common(1)[0][0]
    if expected_cols < 2:
        return None  # not a table

    rows_html = []
    for line in lines:
        cells = line.split("\t")
        if len(cells) < 2:
            # Section header — span full width
            rows_html.append(
                f'<tr><td colspan="{expected_cols}">{cells[0].strip()}</td></tr>'
            )
        else:
            # Pad or trim to expected width
            while len(cells) < expected_cols:
                cells.append("")
            cells = cells[:expected_cols]
            tds = "".join(f"<td>{c.strip()}</td>" for c in cells)
            rows_html.append(f"<tr>{tds}</tr>")

    return f"<table>{''.join(rows_html)}</table>"


def extract_all_tables_html(text: str, engine: str) -> list[str]:
    """Every table in the engine's text, not just the first.

    A financial statement comes back from mistral as one pipe table per
    section and from textract as several table objects; scoring only the first
    charged those engines for segmenting the page. The tables leg treats each
    of these — and their concatenation — as candidates.
    """
    table_format = _declared_format(engine)
    if table_format == "auto":
        table_format = _sniff_format(text)
    elif table_format is not None and _sniff_format(text) == TSV:
        # A declared Markdown engine may return native TSV on some pages.
        table_format = TSV
    if table_format == TSV:
        tables, run = [], []
        def flush():
            if len(run) >= 2:
                tables.append("<table>" + "".join(
                    "<tr>" + "".join(f"<td>{escape(cell.strip())}</td>" for cell in line.split("\t"))
                    + "</tr>" for line in run) + "</table>")
            run.clear()
        for line in (text or "").splitlines():
            if "\t" in line:
                run.append(line)
            else:
                flush()
        flush()
        return tables
    if table_format == HTML:
        source = text or ""
    elif table_format == MARKDOWN:
        return _markdown_tables_preserving_cells(text or "")
    else:
        return []
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(source, "lxml")
        out = []
        for table in soup.find_all("table"):
            for tag in table.find_all(True):
                tag.attrs = {k: v for k, v in tag.attrs.items() if k in ("colspan", "rowspan")}
            out.append(str(table))
        return out
    except Exception:
        return []


def _pipe_cells(line: str) -> list[str]:
    """Split unescaped pipes, preserving explicit empty cells."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    return [cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", line)]


def _markdown_tables_preserving_cells(text: str) -> list[str]:
    """Keep missing cells missing; Markdown renderers otherwise pad/drop them.

    Embedded HTML tables remain valid tables. Actual duplicate tables are
    retained; only callers selecting native HTML versus its export deduplicate.
    """
    import markdown
    from evaluation.markdown_leg import strip_wrapping_fence
    text = strip_wrapping_fence(text)
    found = []
    for match in re.finditer(r"<table\b.*?</table>", text, re.I | re.S):
        found.append((match.start(), match.group()))
    masked = re.sub(r"<table\b.*?</table>", lambda m: " " * len(m.group()), text, flags=re.I | re.S)
    lines = masked.splitlines(keepends=True)
    offsets, offset = [], 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)
    i = 0
    in_code = False
    while i + 1 < len(lines):
        if re.match(r"\s*(```|~~~)", lines[i]):
            in_code = not in_code
            i += 1
            continue
        separators = _pipe_cells(lines[i + 1])
        if in_code or "|" not in lines[i] or not all(re.fullmatch(r":?-+:?", c) for c in separators):
            i += 1
            continue
        header = _pipe_cells(lines[i])
        if len(header) != len(separators):
            i += 1
            continue
        start, rows = i, [header]
        i += 2
        while i < len(lines) and "|" in lines[i] and lines[i].strip():
            rows.append(_pipe_cells(lines[i]))
            i += 1
        rendered = []
        for ri, row in enumerate(rows):
            tag = "th" if ri == 0 else "td"
            rendered.append("<tr>" + "".join(f"<{tag}>{markdown.markdown(c)}</{tag}>" for c in row) + "</tr>")
        found.append((offsets[start], "<table>" + "".join(rendered) + "</table>"))
    return [markup for _, markup in sorted(found)]
