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

VERSION = "gold-table-values-v4-sections"


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


def _unit_rows(table):
    """Repeated unit headers, not numeric data, establish ordered value columns."""
    rows = {}
    for row in range(table.rows):
        cells = sorted((c for c in table.cells if c.row == row and c.normalized), key=lambda c: c.col)
        cells = [c for c in cells if c.col > 0]
        if len(cells) >= 2 and all(re.fullmatch(
                r"(?:[$£€¥]|usd|gbp|eur|jpy)(?:\s*['’]?\s*(?:0{3}|m|bn|million|thousand))?", c.normalized)
                for c in cells):
            rows[row] = cells
    return rows


def _sections(table):
    """Split at repeated header bands; retain each source cell exactly once.

    A section title immediately preceding a repeated units/header row travels
    with that section. Never split through a rowspan. This applies equally to
    references and predictions, irrespective of their physical table count.
    """
    units = _unit_rows(table)
    signatures = {}
    for row in range(table.rows):
        cells = sorted((c for c in table.cells if c.row == row), key=lambda c: c.col)
        signature = tuple(c.normalized for c in cells)
        if row in units or (signature == tuple(_header(table))
                            and len([x for x in signature if x]) >= 2
                            and all(not x or re.search(r"[^\W\d_]", x) for x in signature)):
            signatures[row] = signature
    counts = Counter(signatures.values())
    repeated = [row for row, sig in signatures.items() if counts[sig] > 1]
    # Currency rows may have different leading blank spans, but identical units.
    if len(units) > 1:
        repeated = sorted(set(repeated) | set(units))
    boundaries = [0]
    for row in repeated[1:]:
        start = row
        previous = [c for c in table.cells if c.row == row - 1 and c.normalized]
        if len(previous) == 1 and previous[0].col == 0 and re.search(r"[^\W\d_]", previous[0].normalized):
            start -= 1
        if start > boundaries[-1] and not any(c.row < start < c.row + c.rowspan for c in table.cells):
            boundaries.append(start)
    sections = []
    for start, end in zip(boundaries, boundaries[1:] + [table.rows]):
        cells = [Cell(c.row-start, c.col, c.text, c.rowspan, c.colspan)
                 for c in table.cells if start <= c.row < end]
        origins = {(c.row-start, c.col): table.origins[(c.row, c.col)]
                   for c in table.cells if start <= c.row < end}
        sections.append(Table(cells, end-start, table.cols, table.html, origins))
    return sections


def _split_by_labels(table, references):
    """Find section transitions from unique row labels when headers are omitted.

    Only label ownership determines boundaries; amounts never select a section.
    Ambiguous labels and crossing rowspans leave the table unsplit.
    """
    labels = [(i, c.normalized) for i, ref in enumerate(references) for c in ref.cells
              if c.col == 0 and re.search(r"[^\W\d_]", c.normalized)]
    owners = []
    for row in range(table.rows):
        label = next((c.normalized for c in table.cells if c.row == row and c.col == 0), '')
        if not label:
            continue
        best = {}
        for owner, expected in labels:
            strength = _anchor_strength(expected, label)
            if len(expected) >= 4 and label.startswith(expected + ' '):
                strength = max(strength, .9 + .05 * len(expected) / len(label))
            best[owner] = max(best.get(owner, 0), strength)
        ranked = sorted(((score, owner) for owner, score in best.items()), reverse=True)
        if ranked and ranked[0][0] >= .75 and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= .15):
            owners.append((row, ranked[0][1]))
    if not owners:
        return [table]
    boundaries = [0]
    previous_row, previous_owner = owners[0]
    for row, owner in owners[1:]:
        if owner != previous_owner:
            # Include intervening units/title rows with the following section.
            start = previous_row + 1
            if not any(c.row < start < c.row + c.rowspan for c in table.cells):
                boundaries.append(start)
        previous_row, previous_owner = row, owner
    sections = []
    for start, end in zip(boundaries, boundaries[1:] + [table.rows]):
        cells = [Cell(c.row-start, c.col, c.text, c.rowspan, c.colspan)
                 for c in table.cells if start <= c.row < end]
        origins = {(c.row-start, c.col): table.origins[(c.row, c.col)]
                   for c in table.cells if start <= c.row < end}
        sections.append(Table(cells, end-start, table.cols, table.html, origins))
    return sections


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
    refs = [section for table in refs for section in _sections(table)]
    preds = [section for table in preds for section in _sections(table)]
    preds = [section for table in preds for section in _split_by_labels(table, refs)]
    # A single reference table may be returned in consecutive chunks with the
    # same headers. Duplicate data rows remain separate output cells.
    grouped = {}
    for table in preds:
        key = tuple(_header(table))
        # Identical year headers do not make different named sections the same
        # table. Keep chunks separate when their row labels belong elsewhere.
        owner_scores = [sum(max((_anchor_strength(rc.normalized, c.normalized)
                                for rc in ref.cells if rc.col == 0), default=0)
                            for c in table.cells if c.col == 0 and c.normalized)
                        for ref in refs]
        ranked = sorted(enumerate(owner_scores), key=lambda item: item[1], reverse=True)
        owner = ranked[0][0] if ranked and ranked[0][1] > 0 and (len(ranked) == 1 or ranked[0][1] > ranked[1][1] + .5) else None
        can_join = (len(key) >= 2 and len(set(key)) == len(key)
                    and sum(tuple(_header(r)) == key for r in refs) == 1)
        can_join = can_join and (len(refs) == 1 or owner is not None)
        group_key = (owner, key)
        if can_join and group_key in grouped:
            dest = grouped[group_key]
            offset = dest.rows - 1
            for c in table.cells:
                if c.row == 0:
                    dest.origins[(0, c.col)].extend(table.origins[(0, c.col)])
                else:
                    dest.cells.append(Cell(c.row + offset, c.col, c.text, c.rowspan, c.colspan))
                    dest.origins[(c.row + offset, c.col)] = table.origins[(c.row, c.col)]
            dest.rows += table.rows - 1
        else:
            grouped[group_key if can_join else ('unique', len(grouped))] = Table(
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


def _anchor_strength(expected, returned):
    """Identify a record by its label, never by the amount being graded."""
    if expected == returned and expected:
        return 1.0
    if not expected or not returned or not re.search(r"[^\W\d_]", returned):
        return 0.0
    # A description can be split into a name and a quantity/continuation line.
    if len(returned) >= 4 and re.search(r"(?<!\w)" + re.escape(returned) + r"(?!\w)", expected):
        return len(returned) / len(expected)
    sim = _similarity(expected, returned)
    return sim if sim >= .75 else 0.0


def _item_rows(ref, pred, columns):
    """Disjoint adjacent row groups, bounded by item labels and new labels.

    Reference labels establish identity. Empty label cells and fragments of the
    same label may continue an item; another label/total terminates it. Repeated
    or ambiguous labels never justify guessing which item owns an amount.
    """
    fallback = _row_map(ref, pred, columns)
    # Align units with units, never with years or an absent section title.
    ru, pu = _unit_rows(ref), _unit_rows(pred)
    forced = {}
    for rr, cells in ru.items():
        signature = [c.normalized for c in cells]
        hits = [pr for pr, other in pu.items() if [c.normalized for c in other] == signature]
        if len(hits) == 1 and sum([c.normalized for c in other] == signature for other in ru.values()) == 1:
            forced[rr] = hits[0]
    fallback = {rr: pr for rr, pr in fallback.items()
                if rr not in forced and pr not in forced.values()}
    fallback.update(forced)
    groups = {r: [p] for r, p in fallback.items()}
    def identifier(c):
        return bool(re.search(r"[^\W\d_]", c.normalized) or
                    (c.col == 0 and re.fullmatch(r"\d{4,}", c.normalized)))
    # Prefer a stable left-hand record key over long, repeated fee descriptions.
    label_col = max(range(ref.cols), key=lambda col: (
        len({c.normalized for c in ref.cells if c.row > 0 and c.col == col and identifier(c)}),
        -col), default=0)
    if label_col not in columns:
        return groups
    labels = {c.row: c.normalized for c in ref.cells if c.col == label_col and identifier(c)}
    counts = Counter(labels.values())
    plabels = {c.row: c.normalized for c in pred.cells if c.col == columns[label_col]}
    candidates = {}
    for pr, label in plabels.items():
        ranked = sorted(((max(_anchor_strength(text, label),
                         .9 if ru and len(text) >= 4 and label.startswith(text + ' ') else 0), rr)
                         for rr, text in labels.items() if counts[text] == 1), reverse=True)
        if ranked and ranked[0][0] >= .35 and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= .15):
            candidates[pr] = ranked[0]
    starts = {}
    for rr in labels:
        hits = [(strength, pr) for pr, (strength, row) in candidates.items() if row == rr]
        if not hits:
            continue
        best = max(s for s, _ in hits)
        best_rows = [pr for s, pr in hits if s == best]
        if len(best_rows) == 1:
            starts[rr] = best_rows[0]
    header_row = fallback.get(0)
    if header_row in starts.values() and starts.get(0) != header_row:
        fallback.pop(0, None)
        header_row = None
    occupied = set(forced.values()) | ({header_row} if header_row is not None else set())
    anchored = {}
    for rr, start in sorted(starts.items(), key=lambda x: x[1]):
        if rr == 0 and header_row == start:
            anchored[rr] = [start]
            occupied.add(start)
            continue
        rows = [start]
        # A printed item code may precede its description on the row above.
        previous = plabels.get(start - 1, '')
        if (start - 1 not in occupied and start - 1 not in starts.values()
                and re.fullmatch(r"\d{3,}", previous)
                and labels[rr].startswith(previous + ' ')):
            rows.insert(0, start - 1)
        for pr in range(start + 1, pred.rows):
            if pr == header_row or not any(c.normalized for c in pred.cells if c.row == pr):
                break
            if not plabels.get(pr, '') and not any(
                    c.row == rr and c.col != label_col and c.normalized for c in ref.cells):
                break  # a section label does not absorb a following units row
            # A complete record must not absorb a subsequent unlabeled total
            # or numeric-only record. Use label completion and field lengths,
            # not a search for whichever later amount matches the reference.
            label_text = ' '.join(plabels.get(r, '') for r in rows).strip()
            required = [c for c in ref.cells if c.row == rr and c.normalized]
            complete = _similarity(labels[rr], label_text) >= .95 and all(
                sum(len(c.normalized.split()) for c in pred.cells
                    if c.row in rows and c.col == columns.get(rc.col)) >= len(rc.normalized.split())
                for rc in required)
            if complete:
                break
            if pr in starts.values():
                break
            # Another reference item can begin in a secondary label column.
            # Shared codes such as SR do not establish a new item boundary.
            boundary = False
            for pc in (c for c in pred.cells if c.row == pr and re.search(r"[^\W\d_]", c.normalized)):
                own = max((_anchor_strength(c.normalized, pc.normalized) for c in ref.cells if c.row == rr), default=0)
                other = max((_anchor_strength(c.normalized, pc.normalized) for c in ref.cells if c.row > 0 and c.row != rr), default=0)
                if other >= .75 and other > own + .15:
                    boundary = True
                    break
            if boundary:
                break
            label = plabels.get(pr, '')
            if label and not re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", labels[rr]):
                break
            rows.append(pr)
        anchored[rr] = rows
        occupied.update(rows)
    groups = {r: [p] for r, p in forced.items()}
    if 0 in fallback:
        groups[0] = [header_row]
    groups.update(anchored)
    for rr, pr in fallback.items():
        if rr == 0 or rr in anchored or rr in forced:
            continue
        if pr not in occupied:
            groups[rr] = [pr]
            occupied.add(pr)
    return groups


def _column_groups(ref, pred):
    """Match complete header phrases, allowing adjacent headers to combine."""
    columns = _flex_columns(ref, pred)
    # Units retain their left-to-right meaning when an intervening Notes-like
    # column is folded into the label. Do not infer a permutation from amounts.
    ru, pu = _unit_rows(ref), _unit_rows(pred)
    if ref.cols != pred.cols and len(ru) == len(pu) == 1:
        a, b = next(iter(ru.values())), next(iter(pu.values()))
        if [c.normalized for c in a] == [c.normalized for c in b] and a[0].col > 0 and b[0].col > 0:
            columns = {0: 0, **{x.col: y.col for x, y in zip(a, b)}}
            if a[0].col == b[0].col:
                columns.update({i: i for i in range(1, a[0].col)})
            elif b[0].col == 1:
                columns.update({i: 0 for i in range(1, a[0].col)})
    groups = {r: [p] for r, p in columns.items()}
    rh, ph = _header(ref), _header(pred)
    if len(rh) != ref.cols or len(ph) != pred.cols:
        return groups
    for pi, label in enumerate(ph):
        if not label:
            continue
        matches = [(a, b) for a in range(len(rh)) for b in range(a + 2, len(rh) + 1)
                   if all(rh[a:b]) and ' '.join(rh[a:b]) == label]
        if len(matches) == 1 and pred.cols < ref.cols:
            a, b = matches[0]
            for ri in range(a, b):
                groups[ri] = [pi]
    for ri, label in enumerate(rh):
        if not label:
            continue
        matches = [(a, b) for a in range(len(ph)) for b in range(a + 2, len(ph) + 1)
                   if all(ph[a:b]) and ' '.join(ph[a:b]) == label]
        if len(matches) == 1 and pred.cols > ref.cols:
            a, b = matches[0]
            groups[ri] = list(range(a, b))
    return groups


def _partition_values(expected, text):
    """Ordered, non-overlapping word spans within a header-confirmed cell.

    Partition the whole returned value; do not reorder values or search other
    items. Bound the DP for pathological merged paragraphs.
    """
    from functools import lru_cache
    words = list(re.finditer(r"\S+", text))
    if len(words) > 128:
        return []

    @lru_cache(None)
    def solve(i, start):
        if i == len(expected):
            return (0.0, ()) if start == len(words) else (-1.0, ())
        best = (-1.0, ())
        for end in range(start, len(words) + 1):
            a = words[start].start() if start < len(words) else len(text)
            b = words[end - 1].end() if end > start else a
            tail, spans = solve(i + 1, end)
            value = (_value_credit(expected[i], text[a:b]) if a < b else 0.0) + tail
            if tail >= 0 and value > best[0]:
                best = value, ((a, b),) + spans
        return best
    return list(solve(0, 0)[1])


def compare_cells(reference: list[Table], predicted: list[Table]) -> dict:
    """Reference-value recall is the headline; extras/strict F1 are diagnostics."""
    strict = _compare_strict(reference, predicted)
    refs, preds = _views(reference, predicted)
    table_map = _assignment([[_table_similarity(r, p) + _header_overlap(r, p) for p in preds] for r in refs])
    used, verdicts = {}, []
    credit = 0.0
    correct = partial = blanks_correct = 0
    for rt, ref in enumerate(refs):
        pt = table_map.get(rt)
        pred = preds[pt] if pt is not None else None
        columns = _column_groups(ref, pred) if pred else {}
        rows = _item_rows(ref, pred, {k: v[0] for k, v in columns.items()}) if pred else {}
        if pred:
            rdates = {c.text: c.row for c in ref.cells if c.col == 0 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", c.text)}
            pdates = [(c.text, c.row) for c in pred.cells if c.col in columns.get(0, [])]
            for date, row in rdates.items():
                hits = [pr for text, pr in pdates if text == date]
                if len(hits) == 1:
                    rows[row] = hits
        # Build candidate content from the item's rows and identified columns.
        # A shared physical cell is partitioned in reference-column order.
        bundles = {}
        unit_rows = _unit_rows(ref)
        reserved_units = {(pr, col) for rr, cs in unit_rows.items() for pr in rows.get(rr, [])
                          for c in cs for col in columns.get(c.col, [])}
        for cell in ref.cells:
            def label_span(c):
                return c.colspan > 1 and any(
                    rc.row == cell.row and c.col in columns.get(rc.col, [])
                    and _anchor_strength(rc.normalized, c.normalized) >= .35
                    for rc in ref.cells)
            matches = [c for c in pred.cells if c.row in rows.get(cell.row, [])
                       and (cell.row in unit_rows or (c.row, c.col) not in reserved_units)
                       and any(c.col == col or (c.col < col < c.col + c.colspan
                               and (not label_span(c) or not cell.normalized))
                               for col in columns.get(cell.col, []))] if pred else []
            matches.sort(key=lambda c: (c.row, c.col))
            signature = (cell.row, tuple((c.row, c.col) for c in matches))
            bundles.setdefault(signature, ([], matches))[0].append(cell)
        candidates = {}
        for (_, signature), (cells, matches) in bundles.items():
            if not signature:
                continue
            cells.sort(key=lambda c: c.col)
            pieces, offsets, cursor = [], [], 0
            for output in matches:
                value = output.normalized
                if not value:
                    continue
                if pieces:
                    cursor += 1
                pieces.append(value)
                offsets.append((cursor, cursor + len(value), output))
                cursor += len(value)
            joined = ' '.join(pieces)
            spans = _partition_values([c.text for c in cells], joined) if len(cells) > 1 else [(0, len(joined))]
            if not spans:
                spans = [(0, len(joined))]  # unresolved shared cell cannot be reused
            for cell, (start, end) in zip(cells, spans):
                sources = []
                for left, right, output in offsets:
                    lo, hi = max(start, left), min(end, right)
                    if lo >= hi:
                        continue
                    for ti, original in pred.origins[(output.row, output.col)]:
                        sources.append({"table": ti, **asdict(original), "start": lo-left,
                                        "end": hi-left, "matched_text": output.normalized[lo-left:hi-left]})
                if start == end and len(cells) > 1 and len(matches) == 1 and not cell.normalized:
                    # A header/span-confirmed combined field has an empty
                    # component. This is structural evidence, not free padding.
                    sources = [{"table": ti, **asdict(original), "start": start,
                                "end": end, "matched_text": "", "implicit_blank": True}
                               for output in matches for ti, original in pred.origins[(output.row, output.col)]]
                if not joined and len(cells) == 1:
                    sources = [{"table": ti, **asdict(original), "start": 0, "end": 0,
                                "matched_text": ""} for output in matches
                               for ti, original in pred.origins[(output.row, output.col)]]
                candidates[(cell.row, cell.col)] = (joined[start:end], sources)
        for cell in ref.cells:
            text, sources = candidates.get((cell.row, cell.col), ('', []))
            collision = any(any((a < src['end'] and src['start'] < b) or
                                (a == b == src['start'] == src['end'] == 0)
                                for a, b in used.get((src['table'], src['row'], src['col']), []))
                            for src in sources if not src.get("implicit_blank"))
            if collision:
                text, sources = '', []
            for src in sources:
                if src.get("implicit_blank"):
                    used.setdefault((src['table'], src['row'], src['col']), [])
                    continue
                used.setdefault((src['table'], src['row'], src['col']), []).append((src['start'], src['end']))
            value = _value_credit(cell.text, text) if sources else 0.0
            credit += value
            correct += value == 1
            partial += 0 < value < 1
            blanks_correct += value == 1 and not cell.normalized
            rs = ref.origins[(cell.row, cell.col)]
            output = {k: sources[0][k] for k in ('row', 'col', 'text', 'rowspan', 'colspan')} if sources else None
            verdicts.append({"reference_table": rs[0][0], "output_table": sources[0]['table'] if sources else None,
                "reference": asdict(rs[0][1]), "output": output, "matched_text": text,
                "reference_components": [{"table": ti, **asdict(c)} for ti, c in rs],
                "output_components": sources, "output_rows": rows.get(cell.row, []),
                "credit": value, "verdict": "correct" if value == 1 else "partial" if value else "wrong" if sources else "missing",
                "how": "label-associated item rows and columns; ordered, non-overlapping content spans"})
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
            "extra_tables": sum(not any(i == ti for i, _, _ in used) for ti in range(len(predicted))),
            "physical_table_count_difference": len(predicted) - len(reference),
            "unmatched_tables": sum(not any(i == ti for i, _, _ in used) for ti in range(len(predicted))),
            "verdicts": verdicts + extra, "alignment_table_matches": table_map,
            "normalization": "label-associated adjacent item rows; ordered cell spans; repeated columns, transpose and header-defined sections/chunks; source positions retained"}
