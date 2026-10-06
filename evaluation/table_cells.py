"""Gold-derived table cells, with explicit identity and one-to-one assignment.

HTML spans cover positions but do not create cells. Column remapping uses
unique complete header labels, never the values being graded. Named rows are
aligned by their label; other rows retain their relative order. Content is
NFKC/case/whitespace normalized; digits and punctuation remain significant.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import re
import unicodedata

from bs4 import BeautifulSoup
from rapidfuzz.fuzz import ratio
from scipy.optimize import linear_sum_assignment

VERSION = "gold-table-cells-v1"


def normalize_cell(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


@dataclass(frozen=True)
class Cell:
    row: int
    col: int
    text: str
    rowspan: int = 1
    colspan: int = 1

    @property
    def normalized(self) -> str:
        return normalize_cell(self.text)


@dataclass
class Table:
    cells: list[Cell]
    rows: int
    cols: int
    html: str


def parse_tables(markup: str) -> list[Table]:
    """Parse physical cells; never manufacture padding or duplicate span cells."""
    tables = []
    for node in BeautifulSoup(markup or "", "html.parser").find_all("table"):
        if node.find_parent("table"):
            continue
        cells, occupied = [], set()
        rows = [r for r in node.find_all("tr") if r.find_parent("table") is node]
        for ri, row in enumerate(rows):
            ci = 0
            for cell in row.find_all(["td", "th"], recursive=False):
                # Our Markdown adapter marks absent trailing cells explicitly.
                if cell.get("data-padding") == "true":
                    ci += 1
                    continue
                while (ri, ci) in occupied:
                    ci += 1
                spans = []
                for name in ("rowspan", "colspan"):
                    try:
                        span = int(cell.get(name, 1))
                    except (ValueError, TypeError):
                        span = 1
                    spans.append(max(1, min(span, 1000)))
                rs, cs = spans
                cells.append(Cell(ri, ci, cell.get_text(" ", strip=True), rs, cs))
                occupied.update((r, c) for r in range(ri, ri + rs)
                                for c in range(ci, ci + cs))
                ci += cs
        if cells:
            tables.append(Table(cells, max(len(rows), max(c.row + c.rowspan for c in cells)),
                                max(c.col + c.colspan for c in cells), str(node)))
    return tables


def _assignment(weights: list[list[float]], threshold: float = 0.0) -> dict[int, int]:
    if not weights or not weights[0]:
        return {}
    r, p = linear_sum_assignment(weights, maximize=True)
    return {int(i): int(j) for i, j in zip(r, p) if weights[i][j] > threshold}


def _similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return ratio(a, b) / 100


def _column_map(ref: Table, pred: Table) -> dict[int, int]:
    """Reorder only complete, unique first-row headings on both sides."""
    r = {c.col: c.normalized for c in ref.cells if c.row == 0 and c.colspan == 1}
    p = {c.col: c.normalized for c in pred.cells if c.row == 0 and c.colspan == 1}
    if (len(r) == ref.cols and len(p) == pred.cols and all(r.values())
            and all(p.values()) and len(set(r.values())) == len(r)
            and len(set(p.values())) == len(p)):
        inverse = {value: col for col, value in p.items()}
        shared = {col: inverse[value] for col, value in r.items() if value in inverse}
        # A complete header permutation establishes column identity. Partial
        # overlapping values do not justify moving other columns into place.
        if len(shared) == len(r) == len(p):
            return shared
    return {i: i for i in range(min(ref.cols, pred.cols))}


def _row_map(ref: Table, pred: Table, columns: dict[int, int]) -> dict[int, int]:
    rrows = sorted({c.row for c in ref.cells})
    prows = sorted({c.row for c in pred.cells})
    # Choose a descriptive column in the reference and use its mapped output
    # column. Do not move numeric rows to whichever expected amount matches.
    label_col = max(range(ref.cols), key=lambda col: sum(
        bool(re.search(r"[^\W\d_]", c.normalized))
        for c in ref.cells if c.col == col and c.row > 0), default=0)
    pred_col = columns.get(label_col, label_col)
    rlabels = {c.row: c.normalized for c in ref.cells if c.col == label_col}
    plabels = {c.row: c.normalized for c in pred.cells if c.col == pred_col}
    rcounts, pcounts = Counter(rlabels.values()), Counter(plabels.values())
    weights = []
    for ri in rrows:
        row = []
        for pi in prows:
            a, b = rlabels.get(ri, ""), plabels.get(pi, "")
            named = bool(re.search(r"[^\W\d_]", a))
            sim = _similarity(a, b) if named else 0.0
            # Unique labels anchor whole rows, including when rows reorder.
            anchor = sim if sim >= .75 and rcounts[a] == 1 and pcounts[b] == 1 else 0
            positional = 1 / (1 + abs(ri - pi))
            row.append(10 * anchor + .01 * positional)
        weights.append(row)
    return {rrows[i]: prows[j] for i, j in _assignment(weights).items()}


def _table_similarity(ref: Table, pred: Table) -> float:
    a = Counter(c.normalized for c in ref.cells if c.normalized)
    b = Counter(c.normalized for c in pred.cells if c.normalized)
    shared = sum((a & b).values())
    similarity = 2 * shared / max(1, sum(a.values()) + sum(b.values()))
    # Size tie-break keeps entirely wrong or blank tables inspectably aligned;
    # it cannot give their values credit.
    return similarity + .001 * min(len(ref.cells), len(pred.cells)) / max(len(ref.cells), len(pred.cells))


def compare_cells(reference: list[Table], predicted: list[Table]) -> dict:
    table_map = _assignment([[_table_similarity(r, p) for p in predicted] for r in reference])
    verdicts, used = [], set()
    correct = blanks_correct = 0
    for rt, ref in enumerate(reference):
        pt = table_map.get(rt)
        pred = predicted[pt] if pt is not None else None
        columns = _column_map(ref, pred) if pred else {}
        rows = _row_map(ref, pred, columns) if pred else {}
        positions = {(c.row, c.col): (i, c) for i, c in enumerate(pred.cells)} if pred else {}
        for cell in ref.cells:
            target = positions.get((rows.get(cell.row), columns.get(cell.col)))
            if target is not None and (pt, target[0]) in used:
                target = None
            output = target[1] if target else None
            if target:
                used.add((pt, target[0]))
            equal = output is not None and cell.normalized == output.normalized
            correct += equal
            blanks_correct += equal and not cell.normalized
            verdicts.append({"reference_table": rt, "output_table": pt,
                             "reference": asdict(cell),
                             "output": asdict(output) if output else None,
                             "verdict": "correct" if equal else "wrong" if output else "missing"})
    for pt, pred in enumerate(predicted):
        for ci, cell in enumerate(pred.cells):
            if (pt, ci) not in used:
                verdicts.append({"reference_table": None, "output_table": pt,
                                 "reference": None, "output": asdict(cell), "verdict": "extra"})
    rc, pc = sum(len(t.cells) for t in reference), sum(len(t.cells) for t in predicted)
    return {"correct": correct, "reference_cells": rc, "output_cells": pc,
            "precision": correct / pc if pc else None,
            "recall": correct / rc if rc else None,
            "f1": 2 * correct / (rc + pc) if rc + pc else 1.0,
            "reference_blank_cells": sum(not c.normalized for t in reference for c in t.cells),
            "output_blank_cells": sum(not c.normalized for t in predicted for c in t.cells),
            "correct_blank_cells": blanks_correct, "verdicts": verdicts,
            "table_matches": table_map}
