"""Rendered Markdown structure, with disjoint content/format matches.

Bold is emphasis, not a heading. Raw tabular-looking text is not a table.
HTML headings/lists/tables are valid embedded structure. Heading depth and
bullet style are ignored; block type and content are checked separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata

from bs4 import BeautifulSoup, NavigableString
import markdown
from rapidfuzz.fuzz import ratio, partial_ratio_alignment

VERSION = "markdown-structure-v3-spans-reflow"


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split()).rstrip(" .:;,-")


def blocks(text: str, *, field_tables: list | None = None, tables: list | None = None) -> list[tuple[str, str, int]]:
    from evaluation.markdown_leg import strip_wrapping_fence
    html = markdown.markdown(strip_wrapping_fence(text or ""), extensions=["tables", "fenced_code", "sane_lists"])
    soup = BeautifulSoup(html, "html.parser")
    result = []

    def add(kind, content):
        if content.strip():
            result.append((kind, content.strip(), len(result) + 1))

    def visit(node):
        if isinstance(node, NavigableString):
            if node.strip():
                add("paragraph", str(node))
            return
        name = node.name or ""
        if re.fullmatch(r"h[1-6]", name):
            add("heading", node.get_text(" ", strip=True))
        elif name == "pre":
            add("code", node.get_text(" ", strip=True))
        elif name == "table":
            table_rows = []
            for row in node.find_all("tr"):
                if row.find_parent("table") is node:
                    cells = row.find_all(["td", "th"], recursive=False)
                    # A row-header <th> among data cells is not a column header.
                    header = bool(row.find_parent("thead")) or (bool(cells) and all(c.name == "th" for c in cells))
                    before = len(result)
                    add("table_header" if header else "table_row", " | ".join(
                        c.get_text(" ", strip=True) for c in cells))
                    if len(result) > before:
                        table_rows.append((before, header, cells))
            if tables is not None and table_rows and all(
                str(c.get(a, "1")) == "1" for row in table_rows for c in row[2] for a in ("rowspan", "colspan")):
                tables.append([row[0] for row in table_rows])
            # A single-record table explicitly associates column labels with
            # values directly below. Multi-row/merged grids need more context;
            # do not invent a field association across them.
            if field_tables is not None and len(table_rows) == 2:
                first, second = table_rows
                if (first[1] and not second[1] and len(first[2]) == len(second[2])
                    and all(str(c.get(a, "1")) == "1" for row in table_rows for c in row[2] for a in ("rowspan", "colspan"))):
                    field_tables.append((first[0], second[0]))
        elif name == "li":
            # Nested list entries remain their own entries, not duplicated in
            # the parent item's text.
            direct = [str(child) for child in node.children if getattr(child, "name", None) not in ("ul", "ol")]
            content = BeautifulSoup("".join(direct), "html.parser").get_text(" ", strip=True)
            add("list_item", re.sub(r"^\[[ xX]\]\s*", "", content))
            for child in node.find_all(["ul", "ol"], recursive=False):
                visit(child)
        elif name == "p":
            add("paragraph", node.get_text(" ", strip=True))
        else:
            for child in node.children:
                visit(child)

    visit(soup)
    return result


def _reference_text(content: str, content_format: str = "plain") -> str:
    # New catalogues contain literal visible text. Only explicitly tagged
    # legacy references may interpret emphasis as Markdown.
    if content_format == "markdown":
        content = BeautifulSoup(markdown.markdown("REFERENCE_INLINE " + content), "html.parser").get_text(" ", strip=True).removeprefix("REFERENCE_INLINE ")
    return normalized(content)


# Keep identifiers, decimals and punctuation. Token boundaries prevent a
# reference 100 matching the interior of 1000. Whitespace around punctuation
# is typography, not a different field.
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(normalized(text))


def _similarity(expected: list[str], output: list[str]) -> float:
    if not expected or not output:
        return float(expected == output)
    return ratio(" ".join(expected), " ".join(output)) / 100


def _numbers(text: list[str]) -> list[str]:
    # Linear token scan: keep signs, decimal punctuation and identifiers.
    # Avoid nested numeric regexes: long identifiers can cause backtracking.
    result = []
    i = 0
    while i < len(text):
        if not any(c.isdigit() for c in text[i]):
            i += 1
            continue
        before = i-1
        if before >= 0 and text[before] in ("$", "€", "£", "¥", "₹"):
            before -= 1
        prefix = text[before] if before >= 0 and text[before] in ("-", "+", "−", ".", ",", "(") else ""
        number = ("-" if prefix == "−" else prefix) + text[i]
        i += 1
        while i+1 < len(text) and text[i] in (".", ",", ":", "/", "-") and any(c.isdigit() for c in text[i+1]):
            number += text[i] + text[i+1]
            i += 2
        if i < len(text) and text[i] == ")":
            number += ")"
        result.append(number)
    return result



@dataclass
class Match:
    verdict: str
    how: str
    output_indices: list[int]
    similarity: float
    evidence: list[dict] = field(default_factory=list)


@dataclass
class _Candidate:
    kind: str
    words: list[str]
    locations: list[tuple[int, int]]  # output block, token offset
    indices: tuple[int, ...]
    cells: list[str] | None = None


def _windows(expected: list[str], words: list[str]):
    """All exact occurrences, plus a boundary-aligned fuzzy local candidate.

    Compare the FULL expected content against each proposed span. A short
    common substring never gets the partial-ratio score as its final score.
    """
    if not words or not expected:
        return
    n = len(expected)
    exact = [(i, i+n) for i in range(len(words)-n+1) if words[i:i+n] == expected]
    if exact:
        yield from exact
        return
    seen = {(0, len(words))}
    text = " ".join(words)
    alignment = partial_ratio_alignment(" ".join(expected), text, score_cutoff=40)
    if alignment:
        starts, ends, pos = [], [], 0
        for word in words:
            starts.append(pos)
            ends.append(pos + len(word))
            pos += len(word) + 1
        start = next((i for i, end in enumerate(ends) if end > alignment.dest_start), 0)
        end = next((i for i, begin in enumerate(starts) if begin >= alignment.dest_end), len(words))
        # Snap to whole tokens; consider neighboring punctuation at boundaries.
        for a in range(max(0, start-1), min(len(words), start+2)):
            for b in range(max(a+1, end-1), min(len(words), end+1)+1):
                seen.add((a, b))
    yield from sorted(seen)


def _cells(text: str) -> list[str]:
    # Checklist delimiters, not a general Markdown table parser. The output
    # cells here already came from the rendered HTML table.
    return [s.strip() for s in re.split(r"(?<!\\)\|", text)]


def _row_similarity(reference: list[str], actual: list[str]) -> tuple[float, bool]:
    # Empty padding cells are presentation in this sampled Markdown catalogue.
    # The separate gold-table leg evaluates actual blank cells and grid shape.
    reference = [c for c in reference if tokens(c)]
    actual = [c for c in actual if tokens(c)]
    scores = [_similarity(tokens(a), tokens(b)) for a, b in zip(reference, actual)]
    score = sum(scores) / max(len(reference), len(actual), 1)
    complete = (len(reference) == len(actual) and all(v >= .85 for v in scores)
                and all(_numbers(tokens(a)) == _numbers(tokens(b)) for a, b in zip(reference, actual)))
    return score, complete


def _compatible(expected: str, actual: str) -> bool:
    return (expected == actual
            or (expected == "table_row" and actual == "table_header")  # legacy generic rows
            or (expected == "key_value" and actual in ("paragraph", "heading", "list_item")))


def _reflow_proposals(items, parsed, tables):
    """Repeated column groups may reflow into one narrow table.

    Requires an explicit repeating header in the reference and matching actual
    table headers. Each group's first cell anchors a unique output row, so
    amounts cannot be borrowed from a different date/item or a different table.
    This is not arbitrary joining of table-row fragments.
    """
    active = None
    for i, item in enumerate(items):
        kind = item.get("type")
        if kind not in ("table_header", "table_row"):
            active = None
            continue
        text = _reference_text(item.get("content", ""), item.get("content_format", "plain"))
        cells = _cells(text)
        pattern = None
        for width in range(1, len(cells)//2+1):
            if (len(cells) % width == 0 and cells == cells[:width] * (len(cells)//width)
                and all(any(c.isalpha() for c in cell) for cell in cells[:width])):
                pattern = cells[:width]
                break
        if pattern:
            active = pattern
        if not active or len(cells) % len(active):
            continue
        for table in tables:
            header = table[0]
            if parsed[header][0] != "table_header":
                continue
            actual_header = [c for c in _cells(parsed[header][1]) if tokens(c)]
            if [tokens(c) for c in active] != [tokens(c) for c in actual_header]:
                continue
            chosen, scores, complete = [], [], True
            if pattern:
                chosen, scores = [header], [1.0]
                total = 1
            else:
                groups = [cells[j:j+len(active)] for j in range(0, len(cells), len(active))]
                total = len(groups)
                for group in groups:
                    anchored = [j for j in table[1:] if len(_cells(parsed[j][1])) == len(active)
                                and tokens(_cells(parsed[j][1])[0]) == tokens(group[0])]
                    if len(anchored) != 1 or anchored[0] in chosen:
                        scores.append(0.0)
                        complete = False
                        continue
                    j = anchored[0]
                    similarity, intact = _row_similarity(group, _cells(parsed[j][1]))
                    chosen.append(j)
                    scores.append(similarity)
                    complete &= intact
            sim = sum(scores)/max(1,total)
            if not chosen or sim < .5:
                continue
            locations = tuple((j,k) for j in chosen for k in range(len(tokens(parsed[j][1]))))
            verdict = "correct" if complete and sim >= .85 else "partial"
            yield (2 if sim >= .85 else 1, sim, verdict == "correct", len(tokens(text)),
                   -len(locations), -i, locations, verdict, "table_header" if pattern else "table_row",
                   sim, f"reflowed repeated-column groups; {len(chosen)}/{total} groups located under matching headers")


def match_items(text: str, items: list[dict]) -> tuple[list[Match], list[tuple[str, str, int]]]:
    field_tables, tables = [], []
    parsed = blocks(text, field_tables=field_tables, tables=tables)
    candidates = []
    for i, (kind, content, _) in enumerate(parsed):
        words = tokens(content)
        candidates.append(_Candidate(kind, words, [(i, j) for j in range(len(words))], (i,),
                                     _cells(content) if kind in ("table_row", "table_header") else None))
    # Permit paragraph splitting, but never join arbitrary table rows/columns
    # or cross a structural boundary. Token ownership also applies to runs.
    for i, (kind, _, _) in enumerate(parsed):
        if kind != "paragraph":
            continue
        words, locations = list(candidates[i].words), list(candidates[i].locations)
        for j in range(i+1, min(len(parsed), i+8)):
            if parsed[j][0] != "paragraph":
                break
            words += candidates[j].words
            locations += candidates[j].locations
            candidates.append(_Candidate(kind, list(words), list(locations), tuple(range(i,j+1))))
    for header, data in field_tables:
        def ranges(candidate):
            start = 0
            for k in range(len(candidate.words)+1):
                if k == len(candidate.words) or candidate.words[k] == "|":
                    yield start, k
                    start = k+1
        head, body = candidates[header], candidates[data]
        for (a,b), (c,d) in zip(ranges(head), ranges(body)):
            if a == b or c == d:
                continue
            candidates.append(_Candidate("key_value", head.words[a:b]+body.words[c:d],
                head.locations[a:b]+body.locations[c:d], (header,data)))
    proposals = []
    for item_index, item in enumerate(items):
        expected_kind = item.get("type")
        reference = _reference_text(str(item.get("content", "")), item.get("content_format", "plain"))
        expected = tokens(reference)
        for candidate in candidates:
            if candidate.kind == "key_value" and expected_kind != "key_value":
                continue
            same = _compatible(expected_kind, candidate.kind)
            if expected_kind == "key_value" and candidate.cells is not None:
                # A field can occupy a cell or adjacent cells within a row.
                # Remove only cell separators, retaining token provenance and
                # all intervening content. Never search other rows for a value.
                keep = [j for j, word in enumerate(candidate.words) if word != "|"]
                candidate = _Candidate(candidate.kind, [candidate.words[j] for j in keep],
                    [candidate.locations[j] for j in keep], candidate.indices, candidate.cells)
                same = True
            row = expected_kind in ("table_row", "table_header") and candidate.cells is not None
            cell_similarity = None
            if row:
                sim, complete = _row_similarity(_cells(reference), candidate.cells)
                # A changed column layout is partial structure, not absent
                # text. Keep cell similarity separate from content evidence.
                cell_similarity = sim
                spans = [(a, b, max(sim, _similarity(expected, candidate.words[a:b])), complete)
                         for a, b in _windows(expected, candidate.words)]
                if complete and sim >= .85:
                    # Full ordered-cell credit consumes and displays the full
                    # row, including content around empty padding cells.
                    spans = [(0, len(candidate.words), sim, True)]
            else:
                spans = [(a, b, _similarity(expected, candidate.words[a:b]),
                          _numbers(expected) == _numbers(candidate.words[a:b]))
                         for a, b in _windows(expected, candidate.words)]
            for a, b, sim, complete in spans:
                if sim < .5:
                    continue
                compatible = same
                # Headings/lists/code must match a complete structural block,
                # not just contain the expected words in a larger heading.
                if compatible and expected_kind in ("heading", "list_item", "code"):
                    whole_sim = _similarity(expected, candidate.words)
                    complete = (complete and whole_sim >= .85
                                and _numbers(expected) == _numbers(candidate.words))
                verdict = ("correct" if sim >= .85 and complete else "partial") if compatible else "wrong_type"
                if sim < .5:
                    continue
                # Character overlap alone ("IRA distributions" versus
                # "separate instructions") is not evidence of present text.
                token_similarity = ratio(expected, candidate.words[a:b]) / 100
                if token_similarity < .5 and sim < .85:
                    continue
                locations = tuple(candidate.locations[a:b])
                if not locations:
                    continue
                # Wrong-type evidence never consumes creditable content.
                # Find the best content FIRST, then assess its type. A weak
                # paragraph resemblance must not hide exact text in a table.
                priority = 2 if sim >= .85 else 1
                proposals.append((priority, sim, verdict == "correct", len(expected), -len(locations), -item_index,
                                  locations, verdict, candidate.kind, cell_similarity, ""))
    proposals.extend(_reflow_proposals(items, parsed, tables))
    used = set()
    conflicts = set()
    matches = [Match("absent", "no matching content span", [], 0.0) for _ in items]
    for priority, sim, _, _, _, negative_index, locations, verdict, kind, cell_similarity, note in sorted(proposals, key=lambda p: p[:6], reverse=True):
        i = -negative_index
        if matches[i].verdict != "absent":
            continue
        if used.intersection(locations):
            conflicts.add(i)
            continue
        if verdict != "wrong_type":
            used.update(locations)
        indices = sorted({block+1 for block, _ in locations})
        evidence = []
        for index in indices:
            offsets = [offset for block, offset in locations if block+1 == index]
            evidence.append({"block": index, "kind": parsed[index-1][0],
                             "token_start": min(offsets), "token_end": max(offsets)+1,
                             "text": " ".join(tokens(parsed[index-1][1])[min(offsets):max(offsets)+1])})
        how = (f"{kind} block(s) {', '.join(map(str, indices))}; "
               f"content similarity {sim:.3f}")
        if cell_similarity is not None:
            how += f"; ordered-cell similarity {cell_similarity:.3f}"
        if note:
            how += "; " + note
        if verdict == "partial":
            how += "; incomplete/different content or cell structure"
        matches[i] = Match(verdict, how, indices, sim, evidence)
    for i in conflicts:
        if matches[i].verdict == "absent":
            matches[i].how = "matching content found but already allocated; no separate occurrence"
    return matches, parsed
