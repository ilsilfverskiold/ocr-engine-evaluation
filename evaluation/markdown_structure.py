"""Rendered Markdown structure, with disjoint content/format matches.

Bold is emphasis, not a heading. Raw tabular-looking text is not a table.
HTML headings/lists/tables are valid embedded structure. Heading depth and
bullet style are ignored; block type and content are checked separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from bs4 import BeautifulSoup, NavigableString
import markdown
from rapidfuzz.fuzz import ratio

VERSION = "markdown-structure-v1"


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split()).rstrip(" .:;,-")


def blocks(text: str) -> list[tuple[str, str, int]]:
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
            for row in node.find_all("tr"):
                if row.find_parent("table") is node:
                    add("table_row", " | ".join(c.get_text(" ", strip=True)
                        for c in row.find_all(["td", "th"], recursive=False)))
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


def _reference_text(content: str) -> str:
    # Checklist text may contain emphasis, but it does not define structure:
    # the checklist's explicit type does. Preserve table-cell separators.
    return normalized(BeautifulSoup(markdown.markdown(content), "html.parser").get_text(" ", strip=True))


@dataclass
class Match:
    verdict: str
    how: str
    output_indices: list[int]
    similarity: float


def match_items(text: str, items: list[dict]) -> tuple[list[Match], list[tuple[str, str, int]]]:
    parsed = blocks(text)
    candidates = [(kind, normalized(content), (i,)) for i, (kind, content, _) in enumerate(parsed)]
    # Adjacent paragraph fragments can form one logical paragraph. Consuming
    # the run consumes every source block, so none can satisfy another item.
    for i, (kind, content, _) in enumerate(parsed):
        if kind != "paragraph":
            continue
        texts = [content]
        for j in range(i + 1, min(len(parsed), i + 8)):
            if parsed[j][0] != "paragraph":
                break
            texts.append(parsed[j][1])
            candidates.append(("paragraph", normalized(" ".join(texts)), tuple(range(i, j + 1))))
    proposals = []
    expected = [_reference_text(str(item.get("content", ""))) for item in items]
    for i, item in enumerate(items):
        kind = item.get("type")
        for ckind, content, indices in candidates:
            if kind == "table_row" and ckind == "table_row":
                content = content.strip(" |")
                expected[i] = expected[i].strip(" |")
            sim = ratio(expected[i], content) / 100 if expected[i] and content else 0
            same = kind == ckind or (kind == "key_value" and ckind == "paragraph")
            if sim >= .5:
                # Real same-type complete matches precede partial matches;
                # wrong-type candidates can only explain a loss, never earn it.
                priority = (3 if sim >= .85 else 2) if same else 1
                proposals.append((priority, sim, -len(indices), i, indices, ckind, same))
    used_items, used_blocks = set(), set()
    matches = [Match("absent", "no matching structural block", [], 0.0) for _ in items]
    for _, sim, _, i, indices, kind, same in sorted(proposals, reverse=True):
        if i in used_items or used_blocks.intersection(indices):
            continue
        verdict = ("correct" if sim >= .85 else "partial") if same else "wrong_type"
        used_items.add(i)
        used_blocks.update(indices)
        matches[i] = Match(verdict, f"{kind} block(s) {', '.join(str(n+1) for n in indices)}; full-content similarity {sim:.3f}",
                           [n + 1 for n in indices], sim)
    return matches, parsed
