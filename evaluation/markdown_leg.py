"""Leg 2 — markdown / reading order.

Ask: *"Convert this document to Markdown. Preserve reading order, headings,
lists and tables. Output only the Markdown."*

The reference encodes **one** valid rendering of the page, so comparing against
it literally would measure our formatting choices. Instead the comparison is
structural: both sides are reduced to a sequence of typed blocks — heading,
list item, table row, paragraph — each carrying its normalised content, and we
ask which blocks are present and whether they arrive in the same order.

Normalised away, on both sides: heading depth, emphasis markers, list bullet
style, table alignment rows, and trailing punctuation. Only actual heading syntax creates a heading. Bold emphasis stays a paragraph;
embedded HTML headings are accepted. Plain TSV is not Markdown table syntax.

Supporting score: `0.8 x element F1 + 0.2 x order`, where element F1 matches blocks by
type and near-identical content (fuzzy, so an OCR typo inside a paragraph does
not delete the whole block), and order is a Kendall-tau over the positions of
matched blocks. The headline is code-scored element accuracy against the
cached image checklist; see code_judges.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from evaluation.ground_truth import GroundTruth

# Two blocks of the same type are the same block above this content similarity.
# Below it they are different content, not a noisy match.
BLOCK_MATCH_THRESHOLD = 0.90

ELEMENT_WEIGHT = 0.8
ORDER_WEIGHT = 0.2

STRUCTURAL_TYPES = ("heading", "list_item", "table_header", "table_row")

_FENCE_RE = re.compile(r"^\s*```")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")
_SETEXT_RE = re.compile(r"^\s{0,3}(=+|-{2,})\s*$")
_BOLD_ONLY_RE = re.compile(r"^\s*(?:\*\*|__)(.+?)(?:\*\*|__)\s*:?\s*$")
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+(.*)$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_ALIGNMENT_ROW_RE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{1,3}|`+|~~)")
_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")


@dataclass(frozen=True)
class Block:
    kind: str      # heading | list_item | table_row | paragraph
    text: str      # normalised content
    index: int     # position in the source document


@dataclass
class MarkdownScore:
    applicable: bool
    reason: str | None = None
    markdown_score: float | None = None      # headline
    element_f1: float | None = None
    element_precision: float | None = None
    element_recall: float | None = None
    order_score: float | None = None
    per_type_recall: dict[str, float] = field(default_factory=dict)
    reference_blocks: int = 0
    output_blocks: int = 0
    matched_blocks: int = 0
    reference_has_structure: bool = True
    missed: list[tuple[str, str]] = field(default_factory=list)
    # Judge fields (when LLM judge is used, these are populated)
    judge_element_accuracy: float | None = None
    judge_cost_usd: float = 0.0
    judge_checklist_size: int = 0
    judge_full_checklist_size: int = 0
    judge_sampled: bool = False
    judge_prompt_tokens: int = 0
    judge_output_tokens: int = 0
    judge_thinking_tokens: int = 0
    judge_checklist_model: str | None = None
    judge_verdict_model: str | None = None
    judge_error: str | None = None
    # Code verdicts against the image checklist (the headline since 2026-10-01)
    code_element_accuracy: float | None = None
    checklist_size: int = 0
    code_counts: dict[str, int] = field(default_factory=dict)
    code_verdicts: list[dict] = field(default_factory=list)
    scorer: str | None = None
    cer: float | None = None
    wer: float | None = None
    text_reference_source: str | None = None

    @property
    def score(self) -> float | None:
        if self.code_element_accuracy is not None:
            return self.code_element_accuracy
        if self.judge_element_accuracy is not None:
            return self.judge_element_accuracy
        return self.markdown_score

    def as_metrics(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.code_element_accuracy is not None:
            d["element_accuracy"] = self.code_element_accuracy
            d["checklist_size"] = self.checklist_size
            d["scorer"] = self.scorer
            d.update({f"verdict_{k}": v for k, v in self.code_counts.items()})
        if self.judge_element_accuracy is not None:
            d["judge_element_accuracy"] = self.judge_element_accuracy
            d["judge_checklist_size"] = self.judge_checklist_size
            d["judge_full_checklist_size"] = self.judge_full_checklist_size
            d["judge_sampled"] = self.judge_sampled
            d["judge_checklist_model"] = self.judge_checklist_model
            d["judge_verdict_model"] = self.judge_verdict_model
        d["markdown_score"] = self.markdown_score
        d["element_f1"] = self.element_f1
        d["element_precision"] = self.element_precision
        d["element_recall"] = self.element_recall
        d["order_score"] = self.order_score
        d["reference_blocks"] = self.reference_blocks
        d["output_blocks"] = self.output_blocks
        d["matched_blocks"] = self.matched_blocks
        d["markdown_cer"] = self.cer
        d["markdown_wer"] = self.wer
        d["markdown_text_reference_source"] = self.text_reference_source
        d.update({f"recall_{k}": v for k, v in self.per_type_recall.items()})
        if self.judge_cost_usd:
            d["judge_cost_usd"] = self.judge_cost_usd
            d["judge_prompt_tokens"] = self.judge_prompt_tokens
            d["judge_output_tokens"] = self.judge_output_tokens
            d["judge_thinking_tokens"] = self.judge_thinking_tokens
        return d

    def as_detail(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "missed_blocks": [
                {"kind": kind, "text": text[:120]} for kind, text in self.missed[:40]
            ],
            "reference_has_structure": self.reference_has_structure,
        }
        if self.code_verdicts:
            # every checklist element with its verdict and the reason, so any
            # number can be traced to the elements behind it
            d["verdicts"] = self.code_verdicts
        return d


def normalize_block_text(text: str) -> str:
    """Strip everything that is a rendering choice rather than content."""
    text = _LINK_RE.sub(r"\1", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _EMPHASIS_RE.sub("", text)
    text = text.replace("\\", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(" .:;,-").strip().casefold()


_HTML_TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
_HTML_ROW_RE = re.compile(r"<tr\b.*?</tr>", re.IGNORECASE | re.DOTALL)
_HTML_CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _html_tables_to_pipe(markdown: str) -> str:
    """Rewrite embedded HTML tables as pipe rows before block parsing.

    Markdown legitimately embeds HTML tables — PaddleOCR-VL's pipeline emits
    its tables that way — and scoring the dialect instead of the structure is
    the same mistake that once cost a byte-perfect table 0.154 in the TEDS
    leg. Same content, same blocks, whichever syntax the engine chose.
    """
    def one(match: re.Match) -> str:
        rows = []
        for row in _HTML_ROW_RE.findall(match.group(0)):
            cells = [_HTML_TAG_RE.sub("", c).strip() for c in _HTML_CELL_RE.findall(row)]
            if any(cells):
                rows.append("| " + " | ".join(cells) + " |")
        return "\n" + "\n".join(rows) + "\n"

    return _HTML_TABLE_RE.sub(one, markdown or "")


def _tsv_tables_to_pipe(markdown: str) -> str:
    """Rewrite tab-separated table blocks as pipe rows.

    The same dialect tolerance HTML tables already get: MinerU renders its
    tables tab-separated, so a correctly recovered table contributed no
    table_row blocks at all and the engine scored 0.200 on a page it had
    parsed. Two or more consecutive lines carrying tabs are a table block.
    """
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if len(run) >= 2:
            for line in run:
                cells = [c.strip() for c in line.split("\t")]
                out.append("| " + " | ".join(cells) + " |")
        else:
            out.extend(run)
        run.clear()

    for line in (markdown or "").splitlines():
        if "\t" in line and line.strip():
            run.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


def strip_wrapping_fence(text: str) -> str:
    """Unwrap an answer that is entirely one code fence.

    Asked for markdown, several models return the whole document inside
    ```markdown ... ``` — qwen3_vl does it consistently. Treated as a code
    block, every heading and table inside becomes invisible and the engine
    scores as though it returned nothing. A fence that opens on the first
    non-empty line and closes on the last is packaging, not content.
    """
    lines = (text or "").strip().splitlines()
    if len(lines) < 2 or not lines[0].lstrip().startswith("```"):
        return text
    if lines[0].strip().lower() in ("```markdown", "```md"):
        # An explicit markdown fence on the first line is packaging even
        # when a note follows the closing fence (qwen3_vl, medical_001:
        # "```markdown … ```\n\nNote: …") or the fence never closes (a
        # truncated answer): drop the opening line and the first closing
        # fence after it, keep everything else.
        body = lines[1:]
        for k, l in enumerate(body):
            if l.strip() == "```":
                del body[k]
                break
        return "\n".join(body)
    if lines[0].strip() != "```":
        # A Python/JSON/etc. example is document content, not Markdown packaging.
        return text
    if not lines[-1].strip().startswith("```"):
        return text
    inner = lines[1:-1]
    # Only unwrap when nothing else in the body opens a fence of its own,
    # so a genuine document that merely starts with a code sample is untouched.
    if any(line.lstrip().startswith("```") for line in inner):
        return text
    return "\n".join(inner)


def parse_blocks(markdown: str) -> list[Block]:
    """Parse semantic Markdown/HTML; emphasis is not promoted to headings."""
    from evaluation.markdown_structure import blocks, normalized
    return [Block(kind, normalized(text), i)
            for i, (kind, text, _) in enumerate(blocks(markdown or ""))]


def _similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    try:
        from rapidfuzz import fuzz
        return fuzz.ratio(a, b) / 100.0
    except ImportError:  # pragma: no cover - rapidfuzz is a hard requirement
        return 1.0 if a == b else 0.0


def _match_blocks(
    reference: list[Block], hypothesis: list[Block]
) -> tuple[list[tuple[Block, Block]], int]:
    """Use the same content/structure contract as the catalogue headline.

    Separate reference paragraphs may occupy disjoint spans of one output
    paragraph. Precision counts an output block only when at least 85% of its
    tokens belong to correct reference matches, so a tiny matching phrase in
    unrelated added prose does not make the whole block correct.
    """
    from evaluation.markdown_structure import match_items, tokens
    # Reconstruct typed HTML rather than reparsing text as Markdown (literal
    # asterisks and list numbers in rendered content must stay literal).
    import html
    tag = {"heading": "h2", "paragraph": "p", "list_item": "li", "code": "pre"}
    def markup(block):
        if block.kind in ("table_row", "table_header"):
            cell = "th" if block.kind == "table_header" else "td"
            return "<table><tr>" + "".join(f"<{cell}>{html.escape(v)}</{cell}>" for v in block.text.split("|")) + "</tr></table>"
        name = tag.get(block.kind, "p")
        return f"<{name}>{html.escape(block.text)}</{name}>"
    text = "\n".join(markup(b) for b in hypothesis)
    matches, _ = match_items(text, [{"type": b.kind, "content": b.text} for b in reference])
    covered: dict[int, set[int]] = {}
    pairs = []
    for ref, match in zip(reference, matches):
        if match.verdict != "correct":
            continue
        first = match.evidence[0]
        # Include within-block position for order when paragraphs are merged.
        index = first["block"] - 1
        base = sum(len(tokens(b.text)) + 1 for b in hypothesis[:index])
        pairs.append((ref, Block(hypothesis[index].kind, hypothesis[index].text,
                                 base + first["token_start"])))
        for span in match.evidence:
            covered.setdefault(span["block"]-1, set()).update(range(span["token_start"], span["token_end"]))
    consumed = 0
    for i, block in enumerate(hypothesis):
        content = {j for j, word in enumerate(tokens(block.text))
                   if word != "|" or block.kind not in ("table_row", "table_header")}
        consumed += (len(covered.get(i, set()) & content) / len(content) >= .85
                     if content else bool(covered.get(i)))
    return pairs, consumed


def _kendall_tau_order(positions: list[int]) -> float:
    if len(positions) <= 1:
        return 1.0
    concordant = discordant = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if positions[i] < positions[j]:
                concordant += 1
            elif positions[i] > positions[j]:
                discordant += 1
    total = len(positions) * (len(positions) - 1) // 2
    return ((concordant - discordant) / total + 1) / 2


def score_markdown(
    output_markdown: str | None,
    gt: GroundTruth,
    engine: str = "",
    use_judge: bool | None = None,
    checklist_model: str | None = None,
    verdict_model: str | None = None,
) -> MarkdownScore:
    """Score cached image-checklist elements by code; block F1/order support it.

    Legacy judge arguments are accepted for caller compatibility. This scorer
    makes no model call and never creates a missing image checklist.
    """
    if not gt.has_gt_for("markdown"):
        return MarkdownScore(applicable=False, reason=gt.missing_gt_reason("markdown"))
    if output_markdown is None:
        return MarkdownScore(
            applicable=False, reason="engine produced no markdown output"
        )

    reference = parse_blocks(gt.markdown or "")
    hypothesis = parse_blocks(output_markdown)

    if not reference:
        if gt.metadata.get('reviewed_markdown_catalogue'):
            # A sampled catalogue establishes structure, not a full transcript.
            # Do not invent a Markdown sidecar or report full-page block recall.
            from evaluation.metrics import compute_cer, compute_wer
            trusted_text = gt.plain_text if gt.has_gt_for('text') else None
            return _score_catalogue(output_markdown, gt, dict(
                output_blocks=len(hypothesis),
                cer=round(compute_cer(output_markdown, trusted_text), 4) if trusted_text else None,
                wer=round(compute_wer(output_markdown, trusted_text), 4) if trusted_text else None,
                text_reference_source='plain_text' if trusted_text else None))
        return MarkdownScore(
            applicable=False, reason="markdown reference parsed to no blocks"
        )

    pairs, consumed = _match_blocks(reference, hypothesis)
    matched = len(pairs)
    # Precision counts every output block that fed a matched reference — an
    # engine that split one annotated block into five is not four blocks wrong.
    precision = consumed / len(hypothesis) if hypothesis else 0.0
    recall = matched / len(reference)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    order = _kendall_tau_order([hyp.index for _, hyp in pairs])

    per_type: dict[str, float] = {}
    for kind in ("heading", "list_item", "table_header", "table_row", "paragraph"):
        ref_of_kind = [b for b in reference if b.kind == kind]
        if not ref_of_kind:
            continue
        hit = sum(1 for ref, _ in pairs if ref.kind == kind)
        per_type[kind] = round(hit / len(ref_of_kind), 4)

    matched_refs = {id(ref) for ref, _ in pairs}
    missed = [(b.kind, b.text) for b in reference if id(b) not in matched_refs]

    # Reading order is only a fact about blocks that were actually matched.
    # With fewer than two of them there is no order to be right about, and
    # Kendall tau returns 1.0 for an empty sequence — so an engine that
    # recovered nothing was collecting the full order weight and landing on a
    # 0.2 floor. Ten engines shared that exact minimum before this was fixed.
    headline = (ELEMENT_WEIGHT * f1 + ORDER_WEIGHT * order) if matched >= 2 else f1

    det_score = round(headline, 4)
    from evaluation.metrics import compute_cer, compute_wer
    text_reference = gt.plain_text if gt.has_gt_for("text") else gt.markdown
    det_kwargs = dict(
        markdown_score=det_score,
        element_f1=round(f1, 4),
        element_precision=round(precision, 4),
        element_recall=round(recall, 4),
        order_score=round(order, 4),
        per_type_recall=per_type,
        reference_blocks=len(reference),
        output_blocks=len(hypothesis),
        matched_blocks=matched,
        reference_has_structure=any(b.kind in STRUCTURAL_TYPES for b in reference),
        missed=missed,
        cer=round(compute_cer(output_markdown, text_reference), 4) if text_reference else None,
        wer=round(compute_wer(output_markdown, text_reference), 4) if text_reference else None,
        text_reference_source="plain_text" if gt.has_gt_for("text") else "markdown" if gt.markdown else None,
    )

    return _score_catalogue(output_markdown, gt, det_kwargs)


def _score_catalogue(output_markdown, gt, det_kwargs):

    # --- headline: code verdicts against the image checklist ---------------
    # The checklist (stage 1) is the cached catalogue Gemini 2.5 Pro built
    # from the page image; stage 2 is evaluation/code_judges.py, not a model.
    # A page without a cached checklist is not scored on this leg and is never
    # sent to a model from here.
    from pathlib import Path
    from evaluation.code_judges import (
        load_markdown_checklist, score_markdown_code,
    )
    from evaluation.markdown_structure import VERSION

    items = gt.metadata.get('reviewed_markdown_catalogue') or load_markdown_checklist(Path(gt.document_path).stem)
    if items is None:
        return MarkdownScore(
            applicable=False,
            reason="no image checklist for markdown (stage 1 never built for this page)",
            **det_kwargs,
        )
    res = score_markdown_code(output_markdown or "", items)
    return MarkdownScore(
        applicable=True,
        code_element_accuracy=res.element_accuracy,
        checklist_size=res.checklist_size,
        code_counts={"correct": res.correct, "partial": res.partial,
                     "wrong_type": res.wrong_type, "absent": res.absent,
                     "engine_blocks": res.blocks},
        code_verdicts=[{"n": v.index, "type": v.type, "content": v.content,
                        "verdict": v.verdict, "how": v.how, "evidence": v.evidence} for v in res.verdicts],
        scorer=VERSION,
        **det_kwargs,
    )
