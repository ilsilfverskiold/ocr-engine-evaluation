"""Leg 6 — geometry.

Ask: *"For every word, return its text and bounding box as
`[{"text": ..., "bbox": [x0, y0, x1, y1]}]`, normalised to 0-1. Output only
JSON."*

Score: match predicted words to reference words by text; a matched word is
**located** when the two boxes contain each other's centers. The headline is
located words / reference words, with unmatched reference words as misses.

Located-rate, not raw IoU, because IoU punishes box-drawing convention:
tesseract boxes the full text line where SynFinTabs boxes the glyphs tightly,
so a word placed *exactly right* scored ~0.45 for being ~2x taller. Mutual
center containment scores position and is indifferent to how generously either
side drew the box. Raw IoU is kept in the metrics as a diagnostic.

Reference boxes exist for 17 of the 93 documents, built from the source
datasets' own annotations (`<stem>_geometry_gt.json`): SROIE 6, FUNSD 6,
SynFinTabs 5. The other 76 documents return "not applicable". SROIE's
annotations are line-level (each sidecar records its granularity), so there
predicted words are merged into lines by vertical overlap before matching and
lines pair by fuzzy text similarity — without that, no word-level engine could
ever match its multi-word lines and everyone was capped near 50%. This is the
leg that turns Textract's per-word-geometry advantage from an assertion into a
measurement.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from evaluation.ground_truth import GroundTruth

BBox = tuple[float, float, float, float]
SCORER_VERSION = "geometry-v4 (2026-10-02; explicit frames and recoverable box syntax)"


@dataclass
class GeometryScore:
    applicable: bool
    reason: str | None = None
    geometry_score: float | None = None      # headline: located words / reference words
    mean_iou_matched: float | None = None    # diagnostic: raw IoU over matched pairs
    match_rate: float | None = None
    reference_words: int = 0
    predicted_words: int = 0
    matched_words: int = 0
    located_words: int = 0
    parse_ok: bool = True
    rescaled_from_thousand: bool = False
    axes_swapped: bool = False
    line_score: float | None = None          # second number: line-level rule
    reference_lines: int = 0
    block_score: float | None = None         # block engines only (diagnostic)
    words_per_box: int | None = None
    worst_matches: list[dict[str, Any]] = field(default_factory=list)
    matches: list[dict[str, Any]] = field(default_factory=list)
    parsed_boxes: list[dict[str, Any]] = field(default_factory=list)
    scored_boxes: list[dict[str, Any]] = field(default_factory=list)
    reference_repairs: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float | None:
        return self.geometry_score

    def as_metrics(self) -> dict[str, Any]:
        return {
            "geometry_scorer": SCORER_VERSION,
            "geometry_score": self.geometry_score,
            "geometry_line_score": self.line_score,
            "geometry_reference_lines": self.reference_lines,
            "geometry_block_score": self.block_score,
            "geometry_words_per_box": self.words_per_box,
            "mean_iou_matched": self.mean_iou_matched,
            "geometry_match_rate": self.match_rate,
            "geometry_reference_words": self.reference_words,
            "geometry_predicted_words": self.predicted_words,
            "geometry_matched_words": self.matched_words,
            "geometry_located_words": self.located_words,
            "geometry_parse_ok": self.parse_ok,
            "geometry_rescaled_from_thousand": self.rescaled_from_thousand,
            "geometry_axes_swapped": self.axes_swapped,
            "geometry_reference_repairs": self.reference_repairs,
        }

    def as_detail(self) -> dict[str, Any]:
        return {"worst_matches": self.worst_matches[:25], "matches": self.matches,
                "parsed_boxes": self.parsed_boxes, "scored_boxes": self.scored_boxes,
                "parsed_box_coordinates": "xyxy after box-key decoding, before scale/axis correction; original keys remain in source call",
                "scored_box_coordinates": "normalized xyxy after recorded scale/axis correction, before line merging; includes unmatched predictions",
                "match_box_coordinates": "normalized xyxy after recorded scale/axis conversion"}


def normalize_word(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    return re.sub(r"[^\w]", "", text).casefold()


def iou(a: BBox, b: BBox) -> float:
    """Intersection over union of two [x0, y0, x1, y1] boxes."""
    ax0, ay0, ax1, ay1 = min(a[0], a[2]), min(a[1], a[3]), max(a[0], a[2]), max(a[1], a[3])
    bx0, by0, bx1, by1 = min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])

    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def center_hit(a: BBox, b: BBox) -> bool:
    """True when each box contains the other's center — position agreement
    regardless of how generously either box was drawn."""
    def _norm(box: BBox) -> BBox:
        return (min(box[0], box[2]), min(box[1], box[3]),
                max(box[0], box[2]), max(box[1], box[3]))

    def _inside(point: tuple[float, float], box: BBox) -> bool:
        return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]

    a, b = _norm(a), _norm(b)
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    return _inside(cb, a) and _inside(ca, b)


def merge_words_into_lines(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group word boxes into text lines by vertical overlap, left to right.

    Used when the reference is line-level (SROIE): a word-level engine's output
    must be lifted to the reference's granularity before matching, or its
    multi-word lines can never match anything.
    """
    lines: list[dict[str, Any]] = []
    for word in sorted(words, key=lambda w: (w["bbox"][1] + w["bbox"][3]) / 2):
        y0, y1 = min(word["bbox"][1], word["bbox"][3]), max(word["bbox"][1], word["bbox"][3])
        placed = None
        for line in lines:
            overlap = min(line["y1"], y1) - max(line["y0"], y0)
            min_h = min(line["y1"] - line["y0"], y1 - y0)
            if min_h > 0 and overlap / min_h >= 0.5:
                placed = line
                break
        if placed is None:
            lines.append({"members": [word], "y0": y0, "y1": y1})
        else:
            placed["members"].append(word)
            placed["y0"], placed["y1"] = min(placed["y0"], y0), max(placed["y1"], y1)

    merged = []
    for line in lines:
        members = sorted(line["members"], key=lambda w: min(w["bbox"][0], w["bbox"][2]))
        merged.append({
            "text": " ".join(str(w["text"]) for w in members),
            "bbox": _union_box([w["bbox"] for w in members]),
            "words": members,
        })
    return merged


def _union_box(boxes: list) -> BBox:
    return (min(min(b[0], b[2]) for b in boxes),
            min(min(b[1], b[3]) for b in boxes),
            max(max(b[0], b[2]) for b in boxes),
            max(max(b[1], b[3]) for b in boxes))


def parse_word_boxes(value: Any) -> list[dict[str, Any]] | None:
    """Accept the leg's JSON shape from a string, a list, or a wrapper object."""
    data = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        text = re.sub(r"^\s*```(?:json)?|```\s*$", "", text, flags=re.MULTILINE).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("["), text.rfind("]")
            data = None
            if start != -1 and end > start:
                try:
                    data = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    data = None
            if data is None:
                # One mangled bracket must not zero the whole page: salvage
                # every individually well-formed {...} item (gemini emitted
                # `"bbox": {...]` mid-array and lost all 90 words to it).
                salvaged = []
                for match in re.finditer(r"\{[^{}]*\}", text):
                    source_item = match.group(0)
                    try:
                        item = json.loads(source_item)
                    except json.JSONDecodeError:
                        # Some replies omit only the opening '[' while
                        # retaining four explicit numbers and the closing ']'.
                        # Recover syntax, never infer a missing coordinate.
                        number = r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?"
                        member = (r'((?<!\\)"(?:bbox|box|bounding_box|bbox_2d|box_2d)"\s*:\s*)'
                                  r'(' + number + r'(?:\s*,\s*' + number + r'){3}\s*\])')
                        repaired, count = re.subn(member, r"\1[\2", source_item)
                        if count != 1:
                            continue
                        try:
                            item = json.loads(repaired)
                        except json.JSONDecodeError:
                            continue
                        item["_geometry_syntax_repair"] = "inserted missing opening box-array bracket; all four coordinates retained"
                    if isinstance(item, dict):
                        salvaged.append(item)
                if not salvaged:
                    return None
                data = salvaged
    if isinstance(data, dict):
        for key in ("words", "items", "boxes", "results"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        return None

    words = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("word") or item.get("text_content")
        if text is None:
            continue
        coords = _extract_box(item)
        if coords is None:
            continue
        word = {"text": str(text), "bbox": coords}
        repair = item.get("_geometry_syntax_repair") or item.get("syntax_repair")
        if repair:
            word["syntax_repair"] = repair
        words.append(word)
    return words or None


def _extract_box(item: dict) -> BBox | None:
    """[x0,y0,x1,y1] from any box shape engines actually emit: our asked
    `bbox`/`box` list, gemini's `box_2d` (documented [ymin,xmin,ymax,xmax]),
    or a named-corner dict."""
    # Qwen's bbox_2d is x/y order, unlike Gemini's box_2d below.
    # Read it directly when the capture adapter could not parse the entire
    # answer: valid objects in a truncated response are still usable boxes.
    box = (item.get("bbox") or item.get("box") or item.get("bounding_box")
           or item.get("bbox_2d"))
    if isinstance(box, dict) and all(k in box for k in ("xmin", "ymin", "xmax", "ymax")):
        box = [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        b2d = item.get("box_2d")
        if isinstance(b2d, (list, tuple)) and len(b2d) == 4:
            box = [b2d[1], b2d[0], b2d[3], b2d[2]]
        else:
            return None
    try:
        return tuple(float(c) for c in box)
    except (TypeError, ValueError):
        return None


def score_geometry(
    predicted: Any, gt: GroundTruth | None = None, reference: Any = None,
    granularity: str | None = None, *, infer_axis_order: bool = False,
    infer_scale: bool = True,
) -> GeometryScore:
    """(engine output, ground truth) -> GeometryScore.

    `reference` may be passed directly so the scorer is testable without a
    corpus document; otherwise it comes from `gt.word_boxes`.
    `granularity` is the reference's unit as the
    sidecar records it — "line" lifts predicted words to lines before matching.
    Known native xyxy outputs are never axis-swapped by their shape. The
    legacy Gemini generic-box dialect may explicitly opt into the recorded
    axis heuristic; key-specific box_2d decoding happens during parsing.
    Set infer_scale=False after an explicitly declared coordinate transform;
    out-of-frame predictions must not trigger a second guessed rescaling.
    """
    if reference is None:
        if gt is None or not gt.has_gt_for("geometry"):
            return GeometryScore(
                applicable=False,
                reason=(gt.missing_gt_reason("geometry") if gt
                        else "no word-box reference supplied"),
            )
        reference = gt.word_boxes
    if granularity is None and gt is not None:
        granularity = gt.word_boxes_granularity

    ref_words = parse_word_boxes(reference)
    if ref_words:
        # A punctuation-only reference (SROIE has a stray ':.') normalises to
        # an empty match key: text-first matching cannot locate it for ANY
        # engine, so it must not sit in anyone's denominator.
        ref_words = [w for w in ref_words if normalize_word(w["text"])]
    if not ref_words:
        return GeometryScore(applicable=False, reason="word-box reference is empty")

    pred_words = parse_word_boxes(predicted)
    if pred_words is None:
        return GeometryScore(
            applicable=True,
            reason="engine output was not parseable word boxes",
            geometry_score=0.0,
            mean_iou_matched=None,
            match_rate=0.0,
            reference_words=len(ref_words),
            predicted_words=0,
            matched_words=0,
            parse_ok=False,
            reference_repairs=gt.metadata.get("geometry_reference_repairs", {}) if gt else {},
        )
    parsed_boxes = pred_words

    # Legacy prompt outputs can ignore the requested 0-1 convention. Retain
    # their recorded scale heuristic by default. This is not unambiguous:
    # malformed normalized boxes can also exceed one. Declared-coordinate
    # callers disable this heuristic and retain out-of-frame predictions.
    rescaled = False
    max_coord = max((abs(c) for w in pred_words for c in w["bbox"]), default=0.0)
    if infer_scale and 1.5 < max_coord <= 1000.0:
        pred_words = [{**w, "bbox": tuple(c / 1000.0 for c in w["bbox"])}
                      for w in pred_words]
        rescaled = True

    # Axis-order guard: gemini emits [ymin, xmin, ymax, xmax] under several key
    # names. Words in this corpus are overwhelmingly wider than tall, so
    # majority-portrait predictions against a majority-landscape reference mean
    # swapped axes — undo it and flag it, never score it as misplacement.
    axes_swapped = False
    def _portrait_share(words: list[dict[str, Any]]) -> float:
        if not words:
            return 0.0
        tall = sum(1 for w in words
                   if abs(w["bbox"][3] - w["bbox"][1]) > abs(w["bbox"][2] - w["bbox"][0]))
        return tall / len(words)
    if infer_axis_order and _portrait_share(pred_words) > 0.7 and _portrait_share(ref_words) < 0.3:
        pred_words = [{**w, "bbox": (w["bbox"][1], w["bbox"][0],
                                     w["bbox"][3], w["bbox"][2])}
                      for w in pred_words]
        axes_swapped = True

    if granularity == "line":
        pred_units = merge_words_into_lines(pred_words)
        pairs = _match_lines(ref_words, pred_units)
    else:
        pred_units = pred_words
        pairs = _match_words(ref_words, pred_units)

    total_iou = 0.0
    matched = 0
    located = 0
    matches: list[dict[str, Any]] = []
    for ref, best in pairs:
        if best is None:
            matches.append({"text": ref["text"], "iou": 0.0, "hit": False,
                            "matched": False, "reference_bbox": ref["bbox"],
                            "predicted_bbox": None, "predicted_text": None})
            continue
        overlap = iou(ref["bbox"], best["bbox"])
        hit = center_hit(ref["bbox"], best["bbox"])
        total_iou += overlap
        matched += 1
        located += hit
        matches.append({"text": ref["text"], "iou": round(overlap, 4),
                        "hit": hit, "matched": True,
                        "reference_bbox": ref["bbox"], "predicted_bbox": best["bbox"],
                        "predicted_text": best["text"]})

    line_score, line_n = line_level_score(ref_words, pred_words)
    return GeometryScore(
        applicable=True,
        geometry_score=round(located / len(ref_words), 4),
        mean_iou_matched=round(total_iou / matched, 4) if matched else None,
        match_rate=round(matched / len(ref_words), 4),
        reference_words=len(ref_words),
        predicted_words=len(pred_units),
        matched_words=matched,
        located_words=located,
        matches=matches,
        parsed_boxes=parsed_boxes,
        scored_boxes=pred_words,
        parse_ok=True,
        rescaled_from_thousand=rescaled,
        axes_swapped=axes_swapped,
        line_score=line_score,
        reference_lines=line_n,
        worst_matches=sorted(matches, key=lambda m: (m["hit"], m["iou"]))[:25],
        reference_repairs=gt.metadata.get("geometry_reference_repairs", {}) if gt else {},
    )


def line_level_score(ref_words: list[dict], pred_words: list[dict]) -> tuple[float, int]:
    """Second number, reported beside the word-level headline: the reference
    merged into visual lines (the same merge applied to predictions), each
    reference line paired with the engine's best line by fuzzy text (>= 0.7,
    each engine line used once), mutual centre containment. This is the only
    fair comparison for engines that return line boxes (marker, MinerU's
    pipeline backend); for word engines it tracks the headline closely.
    """
    from difflib import SequenceMatcher

    ref_lines = merge_words_into_lines(ref_words)
    pred_lines = merge_words_into_lines(pred_words)
    if not ref_lines:
        return 0.0, 0
    used: set[int] = set()
    located = 0
    for rl in ref_lines:
        key = normalize_word(rl["text"])
        best, best_ratio = None, 0.0
        for i, pl in enumerate(pred_lines):
            if i in used:
                continue
            ratio = SequenceMatcher(None, key, normalize_word(pl["text"])).ratio()
            if ratio > best_ratio:
                best, best_ratio = i, ratio
        if best is None or best_ratio < 0.7:
            continue
        used.add(best)
        located += center_hit(rl["bbox"], pred_lines[best]["bbox"])
    return round(located / len(ref_lines), 4), len(ref_lines)


def score_blocks(blocks: list[dict], reference: Any, granularity: str | None) -> GeometryScore:
    """Layout parsers (mineru, paddleocr_vl, deepseek_ocr) return one box per
    paragraph/title/table, never per word. They are NOT APPLICABLE on the
    word leg; this diagnostic says whether their regions sit on the text: a
    reference word/line is located when some block's text contains it and
    the block's box contains its centre. Reported with words-per-box so a
    paragraph-level 0.95 is never read as word geometry.
    """
    from difflib import SequenceMatcher

    ref_words = [w for w in (parse_word_boxes(reference) or []) if normalize_word(w["text"])]
    if not ref_words:
        return GeometryScore(applicable=False, reason="word-box reference is empty")
    blocks = [b for b in (blocks or []) if b.get("bbox") and str(b.get("text") or "").strip()]
    if not blocks:
        return GeometryScore(applicable=False, reason="block engine: no blocks returned",
                             reference_words=len(ref_words), parse_ok=False)

    def text_in(ref_text: str, block_text: str) -> bool:
        r, b = normalize_word(ref_text), normalize_word(block_text)
        if not r or not b:
            return False
        if r in b:
            return True
        if granularity == "line":
            return any(SequenceMatcher(None, r, normalize_word(l)).ratio() >= 0.8
                       for l in str(block_text).split("\n") if normalize_word(l))
        return False

    def inside(box, pt) -> bool:
        x0, y0, x1, y1 = min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])
        return x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1

    located = 0
    for ref in ref_words:
        cx, cy = (ref["bbox"][0] + ref["bbox"][2]) / 2, (ref["bbox"][1] + ref["bbox"][3]) / 2
        if any(text_in(ref["text"], b["text"]) and inside(b["bbox"], (cx, cy)) for b in blocks):
            located += 1
    wpb = sorted(len(str(b["text"]).split()) for b in blocks)[len(blocks) // 2]
    return GeometryScore(
        applicable=False,
        reason="block-level boxes (one per paragraph/title/table): not a word-box engine; see geometry_block_score",
        reference_words=len(ref_words),
        predicted_words=len(blocks),
        block_score=round(located / len(ref_words), 4),
        words_per_box=wpb,
        parse_ok=True,
    )


def _match_words(refs, preds) -> list[tuple[dict, dict | None]]:
    """One-to-one assignment within exact normalized-text buckets.

    Maximize located occurrences first, then overlap. A missing early 'Fax'
    must not consume a later correctly located 'FAX' and cause a cascade of
    misses. Assignment is independent of the reference's reading order and
    never lets one predicted box satisfy two reference occurrences.
    """
    from scipy.optimize import linear_sum_assignment

    buckets: dict[str, list[dict[str, Any]]] = {}
    for word in preds:
        buckets.setdefault(normalize_word(word["text"]), []).append(word)
    reference_buckets = {}
    for index, ref in enumerate(refs):
        reference_buckets.setdefault(normalize_word(ref["text"]), []).append(index)
    matches = [None] * len(refs)
    for text, indices in reference_buckets.items():
        candidates = buckets.get(text) or []
        if not candidates:
            continue
        # One extra hit outweighs the maximum total IoU tie-breaker.
        weight = min(len(indices), len(candidates)) + 1
        rewards = [[weight * center_hit(refs[index]["bbox"], candidate["bbox"])
                    + iou(refs[index]["bbox"], candidate["bbox"])
                    for candidate in candidates] for index in indices]
        row_indices, col_indices = linear_sum_assignment(rewards, maximize=True)
        for row, col in zip(row_indices, col_indices):
            matches[indices[row]] = candidates[col]
    return list(zip(refs, matches))


def _match_lines(refs, lines) -> list[tuple[dict, dict | None]]:
    """Fuzzy segment pairing for line-level references.

    A "line" annotation is often a fragment of the visual line — SROIE cuts
    receipts into column segments ("QTY", "PRICE", "AMOUNT" each their own
    entry). So each reference matches the best contiguous *window of words*
    inside a merged line, scored by fuzzy text similarity (one misread
    character must not erase a segment's location), and every word feeds at
    most one reference.
    """
    from difflib import SequenceMatcher

    # Candidate selection: among windows whose text is close enough (>= 0.7),
    # prefer one that OVERLAPS the reference box, and only then the best text
    # ratio. Ranking by text first let an exact "193.00" anywhere on a receipt
    # beat the slightly misread "1193.00" sitting in the right place; with six
    # identical amounts on the page and every word usable once, each later
    # reference then took the next instance down and the whole column scored
    # as misplaced (PP-OCRv5 on sroie_000: 36 located instead of 44).
    used: set[int] = set()
    out = []
    for ref in refs:
        ref_key = normalize_word(ref["text"])
        best_overlap, best_any = None, None       # (rank, window, box)
        for line in lines:
            words = line.get("words") or [line]
            for start in range(len(words)):
                if id(words[start]) in used:
                    continue
                for end in range(start + 1, len(words) + 1):
                    if id(words[end - 1]) in used:
                        break
                    window = words[start:end]
                    win_key = normalize_word("".join(str(w["text"]) for w in window))
                    if not win_key:
                        continue
                    if len(win_key) > 2 * len(ref_key) + 4:
                        break
                    ratio = SequenceMatcher(None, ref_key, win_key).ratio()
                    if ratio < 0.7:
                        continue
                    box = _union_box([w["bbox"] for w in window])
                    overlap = iou(ref["bbox"], box)
                    rank = (ratio, overlap)
                    if best_any is None or rank > best_any[0]:
                        best_any = (rank, window, box)
                    if overlap > 0 and (best_overlap is None or rank > best_overlap[0]):
                        best_overlap = (rank, window, box)
        chosen = best_overlap or best_any
        if chosen is None:
            out.append((ref, None))
            continue
        _, best_words, best_box = chosen
        used.update(id(w) for w in best_words)
        # Preserve the returned words: a fuzzy line match must not replace
        # the engine's spelling with the reference in the evidence record.
        out.append((ref, {"text": " ".join(str(w["text"]) for w in best_words),
                          "bbox": best_box}))
    return out
