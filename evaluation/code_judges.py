"""Deterministic verdicts against cached reference checklists.

The active Markdown scorer delegates to markdown_structure: locate full or
partial content spans, validate their Markdown/HTML type, and allocate output
content once. Header semantics and ordered table cells are retained. The
headline is (correct + 0.5 * partial) / reference items. Missing catalogues do
not trigger model calls. Legacy table helpers below are not the gold-table
scorer; current tables use table_cells through table_leg.
"""

from __future__ import annotations

from evaluation.artifacts import read_text as artifact_text, exists as artifact_exists, glob_paths

import html as _html
import json
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

SCORER_VERSION = "code-v5 (2026-10-02)"
# Tables: which checklist tables are scored on a human-eligible page.
TABLE_POLICY = {
    "human_marked_tables_only": True,       # only checklist tables that correspond to the human table reference
}
MARKDOWN_CACHE_DIR = Path(__file__).parent.parent / "results" / "markdown_judge_cache"
TABLE_CACHE_DIR = Path(__file__).parent.parent / "results" / "table_judge_cache"
# The judges of the reference run scored a capped, evenly spaced sample of
# each checklist (40 elements for markdown, 60 cells for tables). The code
# scores the same sample so the rescored numbers are on the same denominators.
MARKDOWN_MAX_ITEMS = 40
TABLE_MAX_ITEMS = 60

# Source-image audit 2026-10-02: the SROIE receipt has an unlabeled first
# column before RM/Code and before Amount/Tax. The cached model checklist
# dropped that blank header slot. Preserve the cache; correct only coordinates
# at read time, identically for every engine, without changing any cell value.
_TABLE_HEADER_CORRECTIONS = {
    "sroie_002": {
        "Table 1, Header: RM": ("RM", 2),
        "Table 1, Header: Code": ("Code", 3),
        "Table 2, Header: Amount": ("Amount", 2),
        "Table 2, Header: Tax": ("Tax", 3),
    },
}


def _sample(items: list, cap: int) -> list:
    if len(items) <= cap:
        return items
    step = len(items) / cap
    return [items[int(i * step)] for i in range(cap)]


# ---------------------------------------------------------------- checklists
def load_markdown_checklist(document_stem: str) -> list[dict] | None:
    path = MARKDOWN_CACHE_DIR / f"{document_stem}.json"
    if not artifact_exists(path):
        return None
    try:
        record = json.loads(artifact_text(path))
        if record.get('version') == 'markdown-image-catalogue-v1' and record.get('review_status') != 'accepted':
            return None
        items = record.get("items") or []
    except Exception:
        return None
    items = [{**i, "content_format": "plain" if record.get("version") == "markdown-image-catalogue-v1" else "markdown"}
             for i in items if isinstance(i, dict) and i.get("confident", True)]
    if document_stem == "invoice_004":
        for item in items:
            if "Picturebook ☐☐☐☐☐☐☐☐" in item.get("content", ""):
                item["content"] = item["content"].replace("Picturebook ☐☐☐☐☐☐☐☐", "Picturebook")
                item["reference_review"] = "invoice-reference-v1: unreadable replacement-glyph span excluded"
    items = _sample(items, MARKDOWN_MAX_ITEMS)          # sample first, as the judge did
    return [i for i in items if i.get("content") and i.get("type")] or None


def checklist_is_table(items: list[dict] | None) -> bool:
    """Does the image checklist describe a real grid? Pages with a human table
    reference are eligible regardless; for checklist-only eligibility the
    catalogue must hold at least 3 rows x 3 columns and 9 non-empty cells.
    This keeps receipts' item tables and report tables in, and keeps out the
    2-column label/value lists Gemini also catalogues as "tables" (form
    fields, e-mail headers, checkbox lists) — rendering those as lines is not
    a table failure.
    """
    ne = [c for c in (items or []) if not c.get("empty") and tnorm(c.get("value"))]
    if len(ne) < 9:
        return False
    rows = {c.get("row") for c in ne}
    cols = {c.get("col") for c in ne}
    return len(rows) >= 3 and len(cols) >= 3


def human_marked_tables(items: list[dict], reference_html: str, threshold: float = 0.5) -> list[dict]:
    """The checklist tables that correspond to a table the human marked: at
    least `threshold` of a table's non-empty cells appear in the reference
    HTML's text. On tax_001 the dependents grid scores 0.95 and the income
    lines 0.08; an invoice's items table 1.0 and its summary block 0.42. If
    no table passes, every table is kept (the page is eligible, so something
    on it is the table)."""
    import collections
    ref = tnorm(re.sub(r"<[^>]+>", " ", reference_html or ""))
    by: dict[int, list[dict]] = collections.defaultdict(list)
    for it in items:
        m = re.search(r"Table (\d+)", str(it.get("label") or ""))
        by[int(m.group(1)) if m else 1].append(it)
    keep: list[dict] = []
    for t, its in sorted(by.items()):
        ne = [tnorm(i.get("value")) for i in its if not i.get("empty") and tnorm(i.get("value"))]
        if ne and sum(1 for c in ne if c in ref) / len(ne) >= threshold:
            keep += its
    return keep or items


def load_table_checklist(document_stem: str, reference_html: str | None = None,
                         *, sampled: bool = True) -> list[dict] | None:
    """The page's cell checklist, sampled to TABLE_MAX_ITEMS as the judge did.
    With `reference_html` (the human table reference) and the policy on, only
    the checklist tables that correspond to it are kept before sampling."""
    path = TABLE_CACHE_DIR / f"{document_stem}.json"
    if not artifact_exists(path):
        return None
    try:
        items = json.loads(artifact_text(path)).get("items") or []
    except Exception:
        return None
    items = [i for i in items if isinstance(i, dict)]
    for item in items:
        if document_stem == "invoice_004" and "Picturebook 🡪🡪🡪🡪🡪🡪🡪🡪" in item.get("value", ""):
            item["value"] = item["value"].replace("Picturebook 🡪🡪🡪🡪🡪🡪🡪🡪", "Picturebook")
            item["reference_review"] = "invoice-reference-v1: unreadable replacement-glyph span excluded"
        correction = _TABLE_HEADER_CORRECTIONS.get(document_stem, {}).get(item.get("label"))
        if correction:
            value, col = correction
            if item.get("row") != 1 or item.get("value") != value:
                raise ValueError("Reviewed table header no longer matches its cached source")
            item["col"] = col
    if reference_html and TABLE_POLICY["human_marked_tables_only"]:
        items = human_marked_tables(items, reference_html)
    return (_sample(items, TABLE_MAX_ITEMS) if sampled else items) or None


# ---------------------------------------------------------------- markdown
def unfence(text: str) -> str:
    """An answer wrapped whole in ```markdown ... ``` is markdown, not code;
    so is one that opens with an explicit ```markdown fence and carries a
    note after the closing fence, or never closes it (markdown_leg.
    strip_wrapping_fence has the same rule)."""
    s = (text or "").strip()
    m = re.match(r"^```[a-zA-Z]*\s*\n(.*)\n```\s*$", s, re.S)
    if m:
        return m.group(1)
    from evaluation.markdown_leg import strip_wrapping_fence
    return strip_wrapping_fence(text or "")


def markdown_blocks(text: str) -> list[tuple[str, str, int]]:
    """Actual rendered structure, with block positions rather than guessed lines."""
    from evaluation.markdown_structure import blocks
    return blocks(text)


@dataclass
class MarkdownVerdict:
    index: int
    type: str
    content: str
    verdict: str          # correct | wrong_type | partial | absent
    how: str
    evidence: list[dict] = field(default_factory=list)


@dataclass
class MarkdownCodeResult:
    element_accuracy: float
    checklist_size: int
    correct: int
    partial: int
    wrong_type: int
    absent: int
    verdicts: list[MarkdownVerdict] = field(default_factory=list)
    blocks: int = 0


def score_markdown_code(text: str, items: list[dict]) -> MarkdownCodeResult:
    from evaluation.markdown_structure import match_items
    matches, blocks = match_items(text, items)
    verdicts = [MarkdownVerdict(i + 1, str(item.get("type", "")), str(item.get("content", "")),
                                match.verdict, match.how, match.evidence)
                for i, (item, match) in enumerate(zip(items, matches))]
    counts = {v: sum(m.verdict == v for m in matches) for v in ("correct", "partial", "wrong_type", "absent")}
    n = len(items)
    accuracy = (counts["correct"] + .5 * counts["partial"]) / n if n else 0.0
    return MarkdownCodeResult(round(accuracy, 4), n, counts["correct"], counts["partial"],
                              counts["wrong_type"], counts["absent"], verdicts, len(blocks))


# ---------------------------------------------------------------- tables
def tnorm(s: Any) -> str:
    s = unicodedata.normalize("NFKC", _html.unescape(str(s or ""))).lower()
    return re.sub(r"[^a-z0-9]", "", s)


def _num(s: Any):
    from evaluation.structured_metrics import as_number
    t = unicodedata.normalize("NFKC", str(s or "")).strip()
    # Markdown currency escapes and a math-wrapped minus are presentation.
    t = re.sub(r"\\+(?=[$€£])", "", t)
    t = re.sub(r"^\$\s*-\s*\$\s*", "-", t)
    t = re.sub(r"^RM\s*(?=[-+]?\d)", "", t, flags=re.I)
    # A date such as 31.7.17 is not a thousands-separated number.
    if t.count(".") > 1 and not re.fullmatch(r"[-+]?\d{1,3}(?:\.\d{3})+(?:,\d+)?", t):
        return None
    return as_number(t)


def cell_eq(value: Any, cell: Any) -> bool:
    a, b = _num(value), _num(cell)
    nv, nc = tnorm(value), tnorm(cell)
    if a is not None or b is not None:
        # Never erase an OCR letter in an amount, then fall through to the
        # string-containment rule: 17.5C is not the numeric value 17.50.
        return a is not None and b is not None and abs(a - b) < 1e-9
    if not nv:
        return False
    if nv == nc:
        return True
    if len(nv) >= 3 and nv in nc and len(nc) <= 3 * len(nv):
        return True
    wv = sorted(re.findall(r"[a-z0-9]+", _html.unescape(str(value)).lower()))
    wc = sorted(re.findall(r"[a-z0-9]+", _html.unescape(str(cell)).lower()))
    if len(wv) >= 2 and wv == wc:
        return True
    # "Minor OCR errors are NOT partial — they are still correct": one wrong
    # character in a name or address is the same cell. Never for numbers —
    # one wrong digit is a different value.
    return (len(nv) >= 6 and not nv.isdigit() and not nc.isdigit()
            and SequenceMatcher(None, nv, nc).ratio() >= 0.9)


def _tokens(s: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", _html.unescape(str(s or ""))).lower())


def _inside_cell(value: Any, cell: Any) -> bool:
    """The value's tokens appear contiguously among the cell's tokens
    ("(263)" in "(784) (263)"); never a bare substring ("0.00" is not in
    "CS0008225B")."""
    tv, tc = _tokens(value), _tokens(cell)
    if not tv or len(tc) <= len(tv):
        return False
    return any(tc[i:i + len(tv)] == tv for i in range(len(tc) - len(tv) + 1))


def _html_grid(h: str) -> list[list[str]]:
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
    from evaluation.table_extractor import extract_all_tables_html
    out = []
    for t in (structured_output or {}).get("tables") or []:
        if isinstance(t, dict):
            for html in dict.fromkeys(t.get(key) for key in ("html", "cell_html")):
                if html:
                    g = _html_grid(html)
                    if g:
                        out.append(g)
    try:
        for h in extract_all_tables_html(text or "", engine):
            g = _html_grid(h)
            if g:
                out.append(g)
    except Exception:
        pass
    return out


def _cell(g, i, j):
    return g[i][j] if 0 <= i < len(g) and 0 <= j < len(g[i]) else None


@dataclass
class TableVerdict:
    label: str
    value: str
    row: int
    col: int
    verdict: str        # correct | misplaced | wrong | absent
    how: str


@dataclass
class TableCodeResult:
    cell_accuracy: float
    cells_total: int        # non-empty checklist cells
    cells_empty: int        # excluded
    correct: int
    misplaced: int
    wrong: int
    absent: int
    grids: int
    verdicts: list[TableVerdict] = field(default_factory=list)


def score_tables_code(text: str, engine: str, structured_output: dict | None,
                      items: list[dict], context_items: list[dict] | None = None) -> TableCodeResult:
    import collections
    grids = engine_grids(text, engine, structured_output)
    ntext = tnorm(text)
    by_table: dict[int, list[dict]] = collections.defaultdict(list)
    for it in items:
        m = re.search(r"Table (\d+)", str(it.get("label") or ""))
        by_table[int(m.group(1)) if m else 1].append(it)
    contexts = collections.defaultdict(list)
    for it in context_items if context_items is not None else items:
        m = re.search(r"Table (\d+)", str(it.get("label") or ""))
        contexts[int(m.group(1)) if m else 1].append(it)
    verdicts: list[TableVerdict] = []
    n_empty = sum(1 for it in items if it.get("empty") or not tnorm(it.get("value")))
    for t, its in by_table.items():
        nonempty = [i for i in its if not i.get("empty") and tnorm(i.get("value"))]
        best, best_n = None, 0
        for g in grids:
            n = sum(1 for i in nonempty if any(cell_eq(i["value"], c) for row in g for c in row))
            if n > best_n:
                best, best_n = g, n
        pos = {(i["row"], i["col"]): i["value"] for i in nonempty}
        context = {(i["row"], i["col"]): i["value"] for i in contexts[t]
                   if not i.get("empty") and tnorm(i.get("value"))}
        anchor_reasons = {}

        def anchor_ok(g, i, j, r, c):
            # Named rows and headers take precedence over value neighbours.
            # Neighbouring numbers alone can endorse a wholly swapped column.
            row_label = context.get((r, 1)) if r != 1 and c != 1 else None
            header = context.get((1, c)) if r != 1 else None
            if row_label and sum(cell_eq(row_label, v) for (rr, cc), v in context.items() if cc == 1) == 1:
                locations = [k for k, row in enumerate(g) if any(cell_eq(row_label, v) for v in row)]
                if locations and i not in locations:
                    # A split sub-row may continue a label only across blank
                    # leading cells, never across another named row.
                    continuation_labels = [v for (rr, cc), v in context.items()
                                           if cc == 1 and sum(pr == rr for pr, pc in context) == 1]
                    if not any(k < i and all(not tnorm(_cell(g, n, 0)) or
                               any(cell_eq(label, _cell(g, n, 0)) for label in continuation_labels)
                               for n in range(k + 1, i + 1))
                               for k in locations):
                        anchor_reasons[id(g), i, j, r, c] = (
                            f"named row {row_label!r} is at engine row(s) {[k + 1 for k in locations]}; "
                            f"the value is on row {i + 1}")
                        return False
            if header and sum(cell_eq(header, v) for (rr, cc), v in context.items() if rr == 1) == 1:
                locations = [(k, col) for k, row in enumerate(g) for col, v in enumerate(row) if cell_eq(header, v)]
                if locations:
                    preceding = [k for k, col in locations if k <= i]
                    header_row = max(preceding) if preceding else min(k for k, col in locations)
                    if j not in {col for k, col in locations if k == header_row}:
                        anchor_reasons[id(g), i, j, r, c] = (
                            f"header {header!r} is at engine column(s) "
                            f"{sorted({col + 1 for k, col in locations if k == header_row})}; "
                            f"the value is in column {j + 1}")
                        return False
            return True
        if best:
            width = max(len(r) for r in best)
            tr = [[_cell(best, i, j) or "" for i in range(len(best))] for j in range(width)]

            def in_place(g):
                ok = 0
                for it in nonempty:
                    r, c = it["row"], it["col"]
                    rm = [pv for (pr, pc), pv in pos.items() if pr == r and pc != c]
                    cm = [pv for (pr, pc), pv in pos.items() if pc == c and pr != r]
                    hit = False
                    for i, row in enumerate(g):
                        for j, cv in enumerate(row):
                            if cell_eq(it["value"], cv) and anchor_ok(g, i, j, r, c) and (not rm or any(cell_eq(m, x) for m in rm for x in row)) and (not cm or any(cell_eq(m, _cell(g, k, j) or "") for m in cm for k in range(len(g)))):
                                hit = True
                                break
                        if hit:
                            break
                    ok += hit
                return ok
            if in_place(tr) > in_place(best):
                grids = [tr if g is best else g for g in grids]
                best = tr
        for it in nonempty:
            r, c, v = it["row"], it["col"], it["value"]
            row_mates = [pv for (pr, pc), pv in pos.items() if pr == r and pc != c]
            col_mates = [pv for (pr, pc), pv in pos.items() if pc == c and pr != r]
            found = [(g, i, j) for g in grids for i, row in enumerate(g) for j, cv in enumerate(row) if cell_eq(v, cv)]
            if found:
                ok, how = False, ""
                for g, i, j in found:
                    if not anchor_ok(g, i, j, r, c):
                        continue
                    col_ok = not col_mates or any(cell_eq(m, _cell(g, k, j) or "") for m in col_mates for k in range(len(g)))
                    # row-mates may sit on the row itself or the row above/below (split sub-rows)
                    row_ok = not row_mates or any(cell_eq(m, cv) for m in row_mates for k in (i - 1, i, i + 1) if 0 <= k < len(g) for cv in g[k])
                    if col_ok and row_ok:
                        ok, how = True, f"found at ({i+1},{j+1}) of grid {grids.index(g)+1}; row-mates {'ok' if row_mates else 'none'}, column-mates {'ok' if col_mates else 'none'}"
                        break
                    if g is best and i == r - 1 and j == c - 1:
                        ok, how = True, f"found at ({i+1},{j+1}), expected ({r},{c})"
                        break
                if not ok:
                    how = "found at " + ", ".join(f"({i+1},{j+1})" for _, i, j in found[:4]) + f", expected ({r},{c}); row/column-mates not beside it"
                    conflicts = [anchor_reasons[id(g), i, j, r, c] for g, i, j in found
                                 if (id(g), i, j, r, c) in anchor_reasons]
                    if conflicts:
                        how += "; " + "; ".join(dict.fromkeys(conflicts))
                verdicts.append(TableVerdict(it.get("label", ""), str(v), r, c, "correct" if ok else "misplaced", how))
                continue
            merged = [(g, i, j) for g in grids for i, row in enumerate(g) for j, cv in enumerate(row) if _inside_cell(v, cv)]
            if merged:
                g, i, j = merged[0]
                verdicts.append(TableVerdict(it.get("label", ""), str(v), r, c, "misplaced", f"inside a larger cell at ({i+1},{j+1}) of grid {grids.index(g)+1}: {g[i][j][:60]!r}"))
                continue
            at = None
            if best:
                ri = next((i for i, row in enumerate(best) if row_mates and any(cell_eq(m, cv) for m in row_mates for cv in row)), None)
                cj = next((j for j in range(max(len(x) for x in best)) if col_mates and any(cell_eq(m, _cell(best, k, j) or "") for m in col_mates for k in range(len(best)))), None)
                if ri is not None and cj is not None:
                    at = _cell(best, ri, cj)
                if at is None:
                    at = _cell(best, r - 1, c - 1)
            if at and tnorm(at):
                verdicts.append(TableVerdict(it.get("label", ""), str(v), r, c, "wrong", f"cell at the anchored/expected position holds {at!r}"))
            elif tnorm(v) in ntext:
                verdicts.append(TableVerdict(it.get("label", ""), str(v), r, c, "misplaced", "not in any parsed table; present in the raw text"))
            else:
                verdicts.append(TableVerdict(it.get("label", ""), str(v), r, c, "absent", "not in the tables, not in the text"))
    n = len(verdicts)
    cnt = {k: sum(1 for v in verdicts if v.verdict == k) for k in ("correct", "misplaced", "wrong", "absent")}
    return TableCodeResult(round(cnt["correct"] / n, 4) if n else 0.0, n, n_empty, cnt["correct"], cnt["misplaced"], cnt["wrong"], cnt["absent"], len(grids), verdicts)
