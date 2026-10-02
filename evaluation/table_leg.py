"""Leg 3 — table extraction, scored with TEDS.

**Why this was rewritten.** The previous implementation compared HTML dialects.
A byte-perfect table scored 0.846 because the engine's markdown renderer wrapped
the first row in `<thead><th>` where the reference had plain `<tr><td>`. It also
flattened every cell to an opaque tree label, so a one-character OCR slip in a
long cell cost the same as losing the cell entirely, and cell text containing
`{` or `}` corrupted the bracket-encoded tree it handed to APTED.

What replaced it:

* **Both sides are normalised to a presentation-free grid first** — `thead`,
  `tbody` and `tfoot` unwrapped, `th` rewritten to `td`, all attributes
  dropped, entities unescaped, whitespace collapsed. This is mandatory, not an
  option: if a comparison can be fooled by formatting, it is measuring
  formatting. The regression test asserts that a table differing only in
  `thead`/`th` scores exactly 1.000.
* **Cell content is compared by normalised edit distance**, so a near-correct
  cell costs a fraction of a node rather than a whole one — the standard TEDS
  definition, and the reason TEDS is used instead of exact grid equality.
* **The tree is built as objects**, not parsed back out of a bracket string, so
  cell text can contain any character.

For the supporting TEDS diagnostic only, attributes including spans are
dropped on both sides. The current code-verdict headline preserves spans and
anchors values to named rows/headers. The TEDS simplification is a
deliberate limit of this corpus rather than a claim that spans do not matter:
40 of the 45 references were converted from markdown, which cannot express a
span at all, so scoring spans would measure the reference's dialect and not the
engine. `documents/gt_limitations.json` records the five references that are
weaker still, and the scorer surfaces each document's `table_html_source` so a
number can always be traced to the provenance of what it was compared against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
from typing import Any

from evaluation.ground_truth import GroundTruth
from evaluation.table_extractor import extract_all_tables_html, extract_table_html

# Table references derived from schema field names rather than printed page
# columns. Scored, but the report flags them — see gt_limitations.json,
# id "weak-table-references".
WEAK_TABLE_SOURCES = frozenset({"schema GT record arrays (converted)"})


@dataclass
class TableNode:
    label: str                       # "table" | "tr" | "td"
    text: str = ""                   # cell content, normalised; "" for structure
    children: list["TableNode"] = field(default_factory=list)

    def size(self) -> int:
        return 1 + sum(child.size() for child in self.children)


@dataclass
class TableScore:
    applicable: bool
    reason: str | None = None
    teds: float | None = None
    cell_accuracy: float | None = None       # judge headline when available
    reference_rows: int = 0
    reference_cols: int = 0
    output_rows: int = 0
    output_cols: int = 0
    reference_cells: int = 0
    output_cells: int = 0
    cell_match_rate: float | None = None
    reference_source: str | None = None
    reference_is_weak: bool = False
    judge_cost_usd: float = 0.0
    judge_checklist_size: int = 0
    judge_full_checklist_size: int = 0
    judge_sampled: bool = False
    judge_prompt_tokens: int = 0
    judge_output_tokens: int = 0
    judge_thinking_tokens: int = 0
    judge_model: str | None = None
    judge_error: str | None = None
    # Code verdicts against the image cell checklist (the headline since 2026-10-01)
    checklist_cells: int = 0          # non-empty cells scored
    checklist_cells_empty: int = 0    # empty cells, excluded
    code_counts: dict[str, int] = field(default_factory=dict)
    code_verdicts: list[dict] = field(default_factory=list)
    scorer: str | None = None

    @property
    def score(self) -> float | None:
        if self.cell_accuracy is not None:
            return self.cell_accuracy
        return self.teds

    def as_metrics(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.cell_accuracy is not None:
            d["cell_accuracy"] = self.cell_accuracy
            if self.scorer:
                d["checklist_cells"] = self.checklist_cells
                d["checklist_cells_empty"] = self.checklist_cells_empty
                d["scorer"] = self.scorer
                d.update({f"verdict_{k}": v for k, v in self.code_counts.items()})
            d["judge_checklist_size"] = self.judge_checklist_size
            d["judge_full_checklist_size"] = self.judge_full_checklist_size
            d["judge_sampled"] = self.judge_sampled
        d["teds"] = self.teds
        d["reference_rows"] = self.reference_rows
        d["reference_cols"] = self.reference_cols
        d["output_rows"] = self.output_rows
        d["output_cols"] = self.output_cols
        d["cell_match_rate"] = self.cell_match_rate
        if self.judge_cost_usd:
            d["judge_cost_usd"] = self.judge_cost_usd
            d["judge_prompt_tokens"] = self.judge_prompt_tokens
            d["judge_output_tokens"] = self.judge_output_tokens
            d["judge_thinking_tokens"] = self.judge_thinking_tokens
            d["judge_model"] = self.judge_model
        return d

    def as_detail(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "reference_source": self.reference_source,
            "reference_is_weak": self.reference_is_weak,
            "reference_cells": self.reference_cells,
            "output_cells": self.output_cells,
        }
        if self.code_verdicts:
            d["verdicts"] = self.code_verdicts
        return d


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

def normalize_table_html(html: str) -> str:
    """Reduce a table to the presentation-free grid both sides are compared in.

    Unwrap thead/tbody/tfoot, th -> td, drop every attribute. Applied to the
    reference and the engine's output alike; anything that survives is
    structure, not dialect.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "lxml")
    table = soup.find("table")
    if table is None:
        return ""
    for wrapper in table.find_all(["thead", "tbody", "tfoot"]):
        wrapper.unwrap()
    for header_cell in table.find_all("th"):
        header_cell.name = "td"
    # find_all(True) does not include the table element itself, so its own
    # border/class attributes would survive the strip.
    table.attrs = {}
    for tag in table.find_all(True):
        tag.attrs = {}
    return str(table)


def _cell_text(node) -> str:
    return re.sub(r"\s+", " ", unescape(node.get_text(" ", strip=True))).strip().casefold()


def _table_to_tree(table) -> TableNode | None:
    """One parsed <table> element -> a table/tr/td tree."""
    root = TableNode("table")
    for row in table.find_all("tr"):
        row_node = TableNode("tr")
        for cell in row.find_all("td", recursive=False) or row.find_all("td"):
            row_node.children.append(TableNode("td", _cell_text(cell)))
        root.children.append(row_node)
    return root if root.children else None


def html_to_trees(html: str) -> list[TableNode]:
    """EVERY table in `html` -> a tree each, in document order.

    A page's reference is not always one table: a bank statement ships an
    account summary, a holdings table and a daily-balance grid as three
    sibling <table> elements. Reading only the first — which
    `soup.find("table")` does, and which normalize_table_html did too — meant
    8 of the 40 table documents were scored against a fraction of their own
    reference, and an engine that recovered the other tables perfectly earned
    nothing for them. See score_tables() for how the pieces are combined.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "lxml")
    trees: list[TableNode] = []
    for table in soup.find_all("table"):
        tree = _table_to_tree(BeautifulSoup(
            normalize_table_html(str(table)), "lxml").find("table"))
        if tree is not None:
            trees.append(tree)
    return trees


def html_to_tree(html: str) -> TableNode | None:
    """Normalised HTML -> a table/tr/td tree ready for the edit distance.

    The FIRST table only — kept for callers that compare one table to one
    table. Anything scoring a whole reference wants html_to_trees().
    """
    trees = html_to_trees(html)
    return trees[0] if trees else None


# --------------------------------------------------------------------------
# Tree edit distance
# --------------------------------------------------------------------------

def _cell_cost(a: str, b: str) -> float:
    """Normalised edit distance between two cell strings, in [0, 1]."""
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    import editdistance

    return min(1.0, editdistance.eval(a, b) / max(len(a), len(b)))


def _rename_cost(a: TableNode, b: TableNode) -> float:
    if a.label != b.label:
        return 1.0
    if a.label != "td":
        return 0.0
    return _cell_cost(a.text, b.text)


def _tree_edit_distance(a: TableNode | None, b: TableNode | None) -> float:
    """Zhang-Shasha style distance over these three-level trees.

    The tree is table -> tr -> td, so the general algorithm collapses to a
    sequence alignment of rows, each of which is a sequence alignment of cells.
    Writing it directly keeps cell substitution costs exact and removes the
    bracket-string round trip that used to corrupt cells containing braces.
    """
    if a is None and b is None:
        return 0.0
    if a is None:
        return float(b.size())
    if b is None:
        return float(a.size())

    rows_a, rows_b = a.children, b.children
    dp = [[0.0] * (len(rows_b) + 1) for _ in range(len(rows_a) + 1)]
    for i in range(1, len(rows_a) + 1):
        dp[i][0] = dp[i - 1][0] + rows_a[i - 1].size()
    for j in range(1, len(rows_b) + 1):
        dp[0][j] = dp[0][j - 1] + rows_b[j - 1].size()
    for i in range(1, len(rows_a) + 1):
        for j in range(1, len(rows_b) + 1):
            dp[i][j] = min(
                dp[i - 1][j] + rows_a[i - 1].size(),
                dp[i][j - 1] + rows_b[j - 1].size(),
                dp[i - 1][j - 1] + _row_distance(rows_a[i - 1], rows_b[j - 1]),
            )
    return dp[len(rows_a)][len(rows_b)]


def _row_distance(a: TableNode, b: TableNode) -> float:
    cells_a, cells_b = a.children, b.children
    dp = [[0.0] * (len(cells_b) + 1) for _ in range(len(cells_a) + 1)]
    for i in range(1, len(cells_a) + 1):
        dp[i][0] = i
    for j in range(1, len(cells_b) + 1):
        dp[0][j] = j
    for i in range(1, len(cells_a) + 1):
        for j in range(1, len(cells_b) + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + _rename_cost(cells_a[i - 1], cells_b[j - 1]),
            )
    return dp[len(cells_a)][len(cells_b)]


def _concatenate_tables(htmls: list[str]) -> str:
    """All rows from several tables as one table, for engines that split a
    single reference table into per-section tables."""
    rows: list[str] = []
    for html in htmls:
        tree = html_to_tree(html)
        if not tree:
            continue
        for row in tree.children:
            cells = "".join(f"<td>{c.text}</td>" for c in row.children)
            rows.append(f"<tr>{cells}</tr>")
    return f"<table>{''.join(rows)}</table>"


def teds_trees(hyp_tree: TableNode | None, ref_tree: TableNode | None) -> float:
    """TEDS between two already-parsed tables."""
    if ref_tree is None or hyp_tree is None:
        return 0.0
    distance = _tree_edit_distance(hyp_tree, ref_tree)
    denominator = max(hyp_tree.size(), ref_tree.size(), 1)
    return round(max(0.0, 1.0 - distance / denominator), 4)


def compute_teds(hypothesis_html: str, reference_html: str) -> float:
    """TEDS in [0, 1]; 1.0 is a structurally and textually identical table."""
    return teds_trees(html_to_tree(hypothesis_html), html_to_tree(reference_html))


# --------------------------------------------------------------------------
# Leg entry point
# --------------------------------------------------------------------------

def _grid_shape(tree: TableNode | None) -> tuple[int, int, int]:
    if tree is None:
        return 0, 0, 0
    rows = len(tree.children)
    cols = max((len(r.children) for r in tree.children), default=0)
    cells = sum(len(r.children) for r in tree.children)
    return rows, cols, cells


def _cell_match_rate(hyp: TableNode | None, ref: TableNode | None) -> float | None:
    """Fraction of reference cell strings present anywhere in the output.

    Structure-blind on purpose: it separates "got the numbers, lost the grid"
    from "lost the numbers", which a single TEDS figure cannot.
    """
    if hyp is None or ref is None:
        return None
    ref_cells = [c.text for r in ref.children for c in r.children if c.text]
    if not ref_cells:
        return None
    hyp_cells = {c.text for r in hyp.children for c in r.children if c.text}
    return round(sum(1 for c in ref_cells if c in hyp_cells) / len(ref_cells), 4)


def score_tables(
    output_text: str,
    gt: GroundTruth,
    engine: str,
    structured_output: dict | None = None,
    use_judge: bool | None = None,
    judge_model: str | None = None,
) -> TableScore:
    """Score cached nonempty table cells by code; TEDS supports/falls back.

    Legacy judge arguments are accepted for caller compatibility. This scorer
    makes no model call and never creates a missing image checklist.
    """
    # Human table HTML establishes eligibility. The shared image checklist
    # supplies cells for code verdicts; it cannot add extra eligible pages.
    from evaluation.code_judges import (
        SCORER_VERSION, checklist_is_table, load_table_checklist, score_tables_code, tnorm,
    )
    if not gt.has_gt_for("tables"):
        return TableScore(applicable=False, reason=gt.missing_gt_reason("tables"))
    checklist = load_table_checklist(Path(gt.document_path).stem, gt.table_html) or []
    # The checklist is the headline reference when it describes a grid (or
    # when a human table reference says the page is a table page); otherwise
    # the human table_html alone gives the TEDS diagnostic.
    checklist_nonempty = ([c for c in checklist if not c.get("empty") and tnorm(c.get("value"))]
                          if (checklist_is_table(checklist) or gt.table_html) else [])

    # --- TEDS (supporting metric, computed when table_html exists) ----------
    candidates: list[str] = []
    tables = (structured_output or {}).get("tables") or []
    for table in tables:
        html = (table or {}).get("html")
        if html:
            candidates.append(html)
    if not candidates:
        candidates = extract_all_tables_html(output_text or "", engine)
    if len(candidates) > 1:
        candidates.append(_concatenate_tables(candidates))
    hypothesis_html = candidates[0] if candidates else None

    source = gt.table_html_source
    is_weak = source in WEAK_TABLE_SOURCES
    ref_trees = html_to_trees(gt.table_html or "")
    ref_tree = ref_trees[0] if ref_trees else None
    ref_rows = sum(len(t.children) for t in ref_trees)
    ref_cols = max((max((len(r.children) for r in t.children), default=0)
                    for t in ref_trees), default=0)
    ref_cells = sum(len(r.children) for t in ref_trees for r in t.children)

    if len(candidates) > 1 and gt.table_html:
        hypothesis_html = max(candidates,
                              key=lambda h: compute_teds(h, gt.table_html))

    teds_score: float | None = None
    hyp_rows = hyp_cols = hyp_cells = 0
    cmr: float | None = None

    if hypothesis_html:
        hyp_tree = html_to_tree(hypothesis_html)
        hyp_rows, hyp_cols, hyp_cells = _grid_shape(hyp_tree)
        cmr = _cell_match_rate(hyp_tree, ref_tree)
        hyp_trees = [t for t in (html_to_tree(h) for h in candidates) if t]
        if len(ref_trees) > 1 and hyp_trees:
            total_weight = sum(t.size() for t in ref_trees) or 1
            teds_score = round(sum(
                max(teds_trees(h, ref) for h in hyp_trees) * ref.size()
                for ref in ref_trees
            ) / total_weight, 4)
        else:
            teds_score = compute_teds(hypothesis_html, gt.table_html or "")
    elif not checklist_nonempty:
        return TableScore(
            applicable=True,
            teds=0.0,
            reference_rows=ref_rows,
            reference_cols=ref_cols,
            reference_cells=ref_cells,
            reference_source=source,
            reference_is_weak=is_weak,
            reason="engine produced no recoverable table structure",
        )

    # --- headline: code verdicts against the image cell checklist ---------
    if checklist_nonempty:
        context = load_table_checklist(Path(gt.document_path).stem, gt.table_html, sampled=False)
        res = score_tables_code(output_text or "", engine, structured_output, checklist, context)
        return TableScore(
            applicable=True,
            cell_accuracy=res.cell_accuracy,
            teds=teds_score,
            reference_rows=ref_rows,
            reference_cols=ref_cols,
            reference_cells=ref_cells,
            output_rows=hyp_rows,
            output_cols=hyp_cols,
            output_cells=hyp_cells,
            cell_match_rate=cmr,
            reference_source=source,
            reference_is_weak=is_weak,
            checklist_cells=res.cells_total,
            checklist_cells_empty=res.cells_empty,
            code_counts={"correct": res.correct, "misplaced": res.misplaced,
                         "wrong": res.wrong, "absent": res.absent, "grids": res.grids},
            code_verdicts=[{"label": v.label, "value": v.value, "row": v.row, "col": v.col,
                            "verdict": v.verdict, "how": v.how} for v in res.verdicts],
            scorer=SCORER_VERSION,
            judge_checklist_size=res.cells_total,
            judge_full_checklist_size=len(checklist),
        )

    # --- TEDS only (human table_html, no image checklist) -----------------
    return TableScore(
        applicable=True,
        teds=teds_score,
        reference_rows=ref_rows,
        reference_cols=ref_cols,
        reference_cells=ref_cells,
        output_rows=hyp_rows,
        output_cols=hyp_cols,
        output_cells=hyp_cells,
        cell_match_rate=cmr,
        reference_source=source,
        reference_is_weak=is_weak,
    )
