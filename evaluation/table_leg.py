"""Gold-reference table value credit, with precision/recall/F1 and structure diagnostics.

Logical cells, including blanks, count once; extras reduce diagnostic precision only.
Native tables take precedence over another rendering of the same answer.
Every table/row/cell matches at most once. Shipping alone accepts the better
whole-page comparison against the original or human-gold reference.
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
    rowspan: int = 1
    colspan: int = 1

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
    cell_f1: float | None = None
    value_accuracy: float | None = None
    cell_precision: float | None = None
    cell_recall: float | None = None
    structure_similarity: float | None = None
    cell_counts: dict[str, int] = field(default_factory=dict)
    reference_comparisons: dict[str, Any] = field(default_factory=dict)
    selected_reference: str | None = None
    output_tables: list[str] = field(default_factory=list)

    @property
    def score(self) -> float | None:
        if self.value_accuracy is not None:
            return self.value_accuracy
        if self.cell_f1 is not None:
            return self.cell_f1
        if self.cell_accuracy is not None:
            return self.cell_accuracy
        return self.teds

    def as_metrics(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.cell_f1 is not None:
            d.update(table_value_accuracy=self.value_accuracy, cell_f1=self.cell_f1, cell_precision=self.cell_precision,
                     cell_recall=self.cell_recall, structure_similarity=self.structure_similarity,
                     scorer=self.scorer, selected_reference=self.selected_reference,
                     **self.cell_counts)
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
        # Selected verdicts already live at detail.verdicts; avoid serializing
        # the same large evidence array twice. Keep alternative verdicts.
        d["reference_comparisons"] = {
            name: ({**{k: v for k, v in result.items() if k != "verdicts"},
                    "verdicts_location": "detail.verdicts"}
                   if name == self.selected_reference else result)
            for name, result in self.reference_comparisons.items()
        }
        d["selected_reference"] = self.selected_reference
        d["output_tables"] = self.output_tables
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
        tag.attrs = {k: v for k, v in tag.attrs.items() if k in ("rowspan", "colspan", "data-padding")}
    return str(table)


def _cell_text(node) -> str:
    return re.sub(r"\s+", " ", unescape(node.get_text(" ", strip=True))).strip().casefold()


def _table_to_tree(table) -> TableNode | None:
    """One parsed <table> element -> a table/tr/td tree."""
    root = TableNode("table")
    for row in table.find_all("tr"):
        row_node = TableNode("tr")
        for cell in row.find_all("td", recursive=False) or row.find_all("td"):
            if cell.get("data-padding") == "true":
                continue
            def span(name):
                try:
                    return max(1, min(1000, int(cell.get(name, 1))))
                except (TypeError, ValueError):
                    return 1
            row_node.children.append(TableNode("td", _cell_text(cell),
                                               rowspan=span("rowspan"), colspan=span("colspan")))
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
    if (a.rowspan, a.colspan) != (b.rowspan, b.colspan):
        return 1.0
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


def _structure_score(reference, predicted, mapping: dict[int, int], *, content: bool) -> float:
    """One-to-one table forest similarity; unmatched tables pay full size."""
    from copy import deepcopy
    def tree(table):
        result = deepcopy(html_to_tree(table.html))
        if result is not None and not content:
            for row in result.children:
                for cell in row.children:
                    cell.text = ""
        return result
    refs, preds = [tree(t) for t in reference], [tree(t) for t in predicted]
    denominator = distance = 0.0
    used = set()
    for ri, ref in enumerate(refs):
        pi = mapping.get(ri)
        pred = preds[pi] if pi is not None else None
        if pi is not None:
            used.add(pi)
        distance += _tree_edit_distance(ref, pred)
        denominator += max(ref.size() if ref else 0, pred.size() if pred else 0)
    for pi, pred in enumerate(preds):
        if pi not in used:
            size = pred.size() if pred else 0
            distance += size
            denominator += size
    return max(0.0, 1 - distance / denominator) if denominator else 1.0


def score_tables(
    output_text: str,
    gt: GroundTruth,
    engine: str,
    structured_output: dict | None = None,
    use_judge: bool | None = None,
    judge_model: str | None = None,
) -> TableScore:
    """Score every gold cell. No image checklist or model call is used."""
    import hashlib
    from evaluation.table_cells import VERSION, compare_cells, parse_tables
    if not gt.has_gt_for("tables"):
        return TableScore(applicable=False, reason=gt.missing_gt_reason("tables"))
    native = [t["html"] for t in (structured_output or {}).get("tables", [])
              if isinstance(t, dict) and isinstance(t.get("html"), str) and t["html"].strip()]
    # Never combine native tables with their Markdown export. Duplicates within
    # the chosen representation are real predictions and must remain counted.
    htmls = native if native else extract_all_tables_html(output_text or "", engine)
    predicted = [table for html in htmls for table in parse_tables(html)]
    references = {"gold" if gt.metadata.get("table_gold_reference") else "original": gt.table_html}
    if gt.doc_type == "shipping" and gt.metadata.get("table_original_reference"):
        references["original"] = gt.metadata["table_original_reference"]["table_html"]
    comparisons = {}
    parsed_refs = {}
    for name, markup in references.items():
        ref = parse_tables(markup or "")
        if not ref:
            continue
        parsed_refs[name] = ref
        result = compare_cells(ref, predicted)
        result["structure_similarity"] = _structure_score(ref, predicted, result["table_matches"], content=False)
        result["teds"] = _structure_score(ref, predicted, result["table_matches"], content=True)
        result["reference_sha256"] = hashlib.sha256((markup or "").encode()).hexdigest()
        result["reference_html"] = markup
        comparisons[name] = result
    if not comparisons:
        return TableScore(applicable=False, reason="table reference parsed to no logical cells")
    # In a tie, prefer gold. Dict insertion order above makes this deterministic.
    selected = max(comparisons, key=lambda name: comparisons[name]["score"])
    winner = comparisons[selected]
    ref = parsed_refs[selected]
    counts = {key: winner[key] for key in ("correct", "reference_cells", "output_cells",
              "reference_blank_cells", "output_blank_cells", "correct_blank_cells",
              "partial", "value_credit", "extra_cells", "extra_cell_rate", "extra_tables", "reference_physical_cells")}
    return TableScore(
        applicable=True, value_accuracy=winner["score"], cell_f1=winner["f1"], cell_precision=winner["precision"],
        cell_recall=winner["recall"], structure_similarity=winner["structure_similarity"],
        teds=winner["teds"], reference_cells=winner["reference_cells"],
        output_cells=winner["output_cells"], reference_rows=sum(t.rows for t in ref),
        reference_cols=max(t.cols for t in ref), output_rows=sum(t.rows for t in predicted),
        output_cols=max((t.cols for t in predicted), default=0),
        reference_source=gt.table_html_source, reference_is_weak=gt.table_html_source in WEAK_TABLE_SOURCES,
        cell_counts=counts, code_verdicts=winner["verdicts"], scorer=VERSION,
        reference_comparisons=comparisons, selected_reference=selected,
        output_tables=[t.html for t in predicted],
        reason=None if predicted else "engine produced no recoverable table structure",
    )
