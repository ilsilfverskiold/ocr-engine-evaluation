"""Extract HTML tables and cell grids from native, Markdown or TSV outputs."""

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


# Markdown path


# Native HTML path


# TSV path


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


def html_grid(h: str) -> list[list[str]]:
    from bs4 import BeautifulSoup
    # Spans locate cells; discarding them shifts amounts into another year.
    # Repeat the merged cell across its covered positions for anchor lookup.
    occupied = {}
    trs = BeautifulSoup(h, "lxml").find_all("tr")
    for i, tr in enumerate(trs):
        j = 0
        for c in tr.find_all(["td", "th"], recursive=False):
            while (i, j) in occupied:
                j += 1
            try:
                height, width = int(c.get("rowspan", 1)), int(c.get("colspan", 1))
            except (ValueError, TypeError):
                height = width = 1
            if not (1 <= height <= 1000 and 1 <= width <= 1000):
                raise ValueError("Invalid table span")
            value = c.get_text(" ", strip=True)
            for r in range(i, min(i + height, len(trs))):
                for col in range(j, j + width):
                    occupied[r, col] = value
            j += width
    return [[occupied.get((i, j), "") for j in range(
        1 + max((col for row, col in occupied if row == i), default=-1))]
        for i in range(len(trs))]


def engine_grids(text: str, engine: str, structured_output: dict | None) -> list[list[list[str]]]:
    out = []
    for t in (structured_output or {}).get("tables") or []:
        if isinstance(t, dict):
            for html in dict.fromkeys(t.get(key) for key in ("html", "cell_html")):
                if html:
                    g = html_grid(html)
                    if g:
                        out.append(g)
    try:
        for h in extract_all_tables_html(text or "", engine):
            g = html_grid(h)
            if g:
                out.append(g)
    except Exception:
        pass
    return out
