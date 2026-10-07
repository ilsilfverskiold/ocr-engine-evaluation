"""Gold-derived table cells, with explicit identity and one-to-one assignment.

HTML spans cover positions but do not create cells. Column remapping uses
unique complete header labels, never the values being graded. Named rows are
aligned by their label; other rows retain their relative order. Content is
NFKC/case/whitespace normalized; digits and punctuation remain significant.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import re
import unicodedata

from bs4 import BeautifulSoup
from rapidfuzz.fuzz import ratio
from scipy.optimize import linear_sum_assignment

VERSION = "gold-table-values-v2"


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
    origins: dict = field(default_factory=dict)


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


def _compare_strict(reference: list[Table], predicted: list[Table]) -> dict:
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


def _value_credit(a: str, b: str) -> float:
    """Partial text fidelity; complete numeric values, never digit substrings."""
    from evaluation.code_judges import _num
    a, b = normalize_cell(a), normalize_cell(b)
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    na, nb = _num(a), _num(b)
    if na is not None or nb is not None:
        # A leading-zero identifier is not an amount. Percent is a unit.
        if any(re.fullmatch(r"0\d+", v) for v in (a, b)) or ('%' in a) != ('%' in b):
            return 0.0
        return float(na is not None and nb is not None and abs(na - nb) < 1e-9)
    # Identifiers and dates cannot gain credit by losing punctuation/digits.
    if not re.search(r"[^\W\d_]", a) and not re.search(r"[^\W\d_]", b):
        return 0.0
    sim = ratio(a, b) / 100
    return sim if sim >= .5 else 0.0


def _header(table):
    return [c.normalized for c in sorted(table.cells, key=lambda c: c.col) if c.row == 0]


def _view(table, index):
    """Retain original cell identities while normalizing repeated columns."""
    origins = {(c.row, c.col): [(index, c)] for c in table.cells}
    head = _header(table)
    if any(c.rowspan != 1 or c.colspan != 1 for c in table.cells):
        return Table(table.cells, table.rows, table.cols, table.html, origins)
    width = next((w for w in range(1, len(head)//2 + 1)
                  if len(head) == table.cols and len(head) % w == 0
                  and head == head[:w] * (len(head)//w)
                  and all(re.search(r"[^\W\d_]", x) for x in head[:w])), None)
    if not width:
        return Table(table.cells, table.rows, table.cols, table.html, origins)
    cells, mapped = [], {}
    for c in table.cells:
        group, col = divmod(c.col, width)
        row = 0 if c.row == 0 else group * (table.rows - 1) + c.row
        if c.row == 0 and group:
            mapped[(0, col)].append((index, c))
            continue
        cells.append(Cell(row, col, c.text))
        mapped[(row, col)] = [(index, c)]
    return Table(cells, (table.rows - 1)*(table.cols//width)+1, width, table.html, mapped)


def _transpose(table):
    if any(c.rowspan != 1 or c.colspan != 1 for c in table.cells):
        return table
    return Table([Cell(c.col, c.row, c.text) for c in table.cells], table.cols,
                 table.rows, table.html, {(c.col, c.row): table.origins[(c.row, c.col)] for c in table.cells})


def _header_overlap(a, b):
    a, b = set(filter(None, _header(a))), set(filter(None, _header(b)))
    return len(a & b) / max(1, len(a))


def _views(reference, predicted):
    refs = [_view(t, i) for i, t in enumerate(reference)]
    preds = [_view(t, i) for i, t in enumerate(predicted)]
    for i, table in enumerate(preds):
        transposed = _transpose(table)
        straight = max((_header_overlap(r, table) for r in refs), default=0)
        flipped = max((_header_overlap(r, transposed) for r in refs), default=0)
        # Orientation is established by labels, never by finding graded values.
        if flipped >= .75 and flipped > straight and len(_header(transposed)) >= 2:
            preds[i] = transposed
    # A single reference table may be returned in consecutive chunks with the
    # same headers. Duplicate data rows remain separate output cells.
    grouped = {}
    for table in preds:
        key = tuple(_header(table))
        can_join = (len(key) >= 2 and len(set(key)) == len(key)
                    and sum(tuple(_header(r)) == key for r in refs) == 1)
        if can_join and key in grouped:
            dest = grouped[key]
            offset = dest.rows - 1
            for c in table.cells:
                if c.row == 0:
                    dest.origins[(0, c.col)].extend(table.origins[(0, c.col)])
                else:
                    dest.cells.append(Cell(c.row + offset, c.col, c.text, c.rowspan, c.colspan))
                    dest.origins[(c.row + offset, c.col)] = table.origins[(c.row, c.col)]
            dest.rows += table.rows - 1
        else:
            grouped[key if can_join else ('unique', len(grouped))] = Table(
                list(table.cells), table.rows, table.cols, table.html, dict(table.origins))
    return refs, list(grouped.values())


def _flex_columns(ref, pred):
    rh, ph = _header(ref), _header(pred)
    if rh == ph or len(rh) != ref.cols or len(ph) != pred.cols:
        return {i: i for i in range(min(ref.cols, pred.cols))}
    # Added/missing columns do not shift the other uniquely labelled fields.
    weights = [[_value_credit(a, b) if a and b and rh.count(a) == ph.count(b) == 1 else 0
                for b in ph] for a in rh]
    matched = _assignment(weights, .74)
    if len(matched) >= 2 or len(matched) == len(rh) == len(ph) == 1:
        # An unlabeled column is still a real column. Preserve its position
        # when both headers are blank and labeled mappings do not occupy it.
        for i, label in enumerate(rh):
            if not label and i < len(ph) and not ph[i] and i not in matched.values():
                matched[i] = i
        return matched
    return {i: i for i in range(min(ref.cols, pred.cols))}


def compare_cells(reference: list[Table], predicted: list[Table]) -> dict:
    """Reference-value recall is the headline; extras/strict F1 are diagnostics."""
    strict = _compare_strict(reference, predicted)
    refs, preds = _views(reference, predicted)
    table_map = _assignment([[_table_similarity(r, p) + _header_overlap(r, p) for p in preds] for r in refs])
    used, verdicts = set(), []
    credit = 0.0
    correct = partial = blanks_correct = 0
    for rt, ref in enumerate(refs):
        pt = table_map.get(rt)
        pred = preds[pt] if pt is not None else None
        columns = _flex_columns(ref, pred) if pred else {}
        rows = _row_map(ref, pred, columns) if pred else {}
        # Unique dates also anchor rows; arbitrary repeated numeric amounts do not.
        if pred:
            rdates = {c.text: c.row for c in ref.cells if c.col == 0 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", c.text)}
            pdates = [(c.text, c.row) for c in pred.cells if c.col == columns.get(0)]
            for date, row in rdates.items():
                hits = [pr for text, pr in pdates if text == date]
                if len(hits) == 1:
                    rows[row] = hits[0]
        positions = {(c.row, c.col): c for c in pred.cells} if pred else {}
        for cell in ref.cells:
            output = positions.get((rows.get(cell.row), columns.get(cell.col)))
            sources = pred.origins.get((output.row, output.col), []) if output else []
            ids = {(ti, c.row, c.col) for ti, c in sources}
            if ids & used:
                output, sources, ids = None, [], set()
            if output is not None:
                used.update(ids)
            value = _value_credit(cell.text, output.text) if output is not None else 0.0
            credit += value
            correct += value == 1
            partial += 0 < value < 1
            blanks_correct += value == 1 and not cell.normalized
            rs = ref.origins[(cell.row, cell.col)]
            verdicts.append({"reference_table": rs[0][0], "output_table": sources[0][0] if sources else None,
                "reference": asdict(rs[0][1]), "output": asdict(sources[0][1]) if sources else None,
                "reference_components": [{"table": ti, **asdict(c)} for ti, c in rs],
                "output_components": [{"table": ti, **asdict(c)} for ti, c in sources],
                "credit": value, "verdict": "correct" if value == 1 else "partial" if value else "wrong" if output else "missing",
                "how": "matched table, row label/order and column label/order; normalized complete value or partial text"})
    extra = []
    for ti, table in enumerate(predicted):
        for cell in table.cells:
            if (ti, cell.row, cell.col) not in used:
                extra.append({"reference_table": None, "output_table": ti, "reference": None,
                              "output": asdict(cell), "credit": 0, "verdict": "extra"})
    rc = sum(len(t.cells) for t in refs)
    pc = sum(len(t.cells) for t in predicted)
    return {**strict, "score": credit / rc if rc else 1.0, "value_credit": credit,
            "correct": correct, "partial": partial, "reference_cells": rc, "output_cells": pc,
            "reference_physical_cells": sum(len(t.cells) for t in reference),
            "reference_blank_cells": sum(not c.normalized for t in refs for c in t.cells),
            "correct_blank_cells": blanks_correct,
            "extra_cells": len(extra), "extra_cell_rate": len(extra)/pc if pc else 0.0,
            "extra_tables": max(0, len(predicted) - len(reference)),
            "unmatched_tables": sum(not any(i == ti for i, _, _ in used) for ti in range(len(predicted))),
            "verdicts": verdicts + extra, "alignment_table_matches": table_map,
            "normalization": "repeated column groups, label-confirmed transpose, same-header table chunks; original cell identities retained"}
