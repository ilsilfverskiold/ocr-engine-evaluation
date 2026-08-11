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
    worst_matches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def score(self) -> float | None:
        return self.geometry_score

    def as_metrics(self) -> dict[str, Any]:
        return {
            "geometry_score": self.geometry_score,
            "mean_iou_matched": self.mean_iou_matched,
            "geometry_match_rate": self.match_rate,
            "geometry_reference_words": self.reference_words,
            "geometry_predicted_words": self.predicted_words,
            "geometry_matched_words": self.matched_words,
            "geometry_located_words": self.located_words,
            "geometry_parse_ok": self.parse_ok,
            "geometry_rescaled_from_thousand": self.rescaled_from_thousand,
            "geometry_axes_swapped": self.axes_swapped,
        }

    def as_detail(self) -> dict[str, Any]:
        return {"worst_matches": self.worst_matches[:25]}


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
                    try:
                        item = json.loads(match.group(0))
                    except json.JSONDecodeError:
                        continue
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
        text = item.get("text") or item.get("word")
        if text is None:
            continue
        coords = _extract_box(item)
        if coords is None:
            continue
        words.append({"text": str(text), "bbox": coords})
    return words or None


def _extract_box(item: dict) -> BBox | None:
    """[x0,y0,x1,y1] from any box shape engines actually emit: our asked
    `bbox`/`box` list, gemini's `box_2d` (documented [ymin,xmin,ymax,xmax]),
    or a named-corner dict."""
    box = item.get("bbox") or item.get("box") or item.get("bounding_box")
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
    granularity: str | None = None,
) -> GeometryScore:
    """(engine output, ground truth) -> GeometryScore.

    `reference` may be passed directly so the scorer is testable without a
    corpus document; otherwise it comes from `gt.word_boxes`, which is None for
    every document in this corpus. `granularity` is the reference's unit as the
    sidecar records it — "line" lifts predicted words to lines before matching.
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
        )

    # The ask says 0-1, but grounding-trained VLMs (qwen, gemini) often emit
    # their native 0-1000 convention regardless. A valid 0-1 box cannot exceed
    # 1, so the conventions are unambiguous — rescale rather than score every
    # box as misplaced, and flag it so the report can say so.
    rescaled = False
    max_coord = max((abs(c) for w in pred_words for c in w["bbox"]), default=0.0)
    if 1.5 < max_coord <= 1000.0:
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
    if _portrait_share(pred_words) > 0.7 and _portrait_share(ref_words) < 0.3:
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
                            "matched": False})
            continue
        overlap = iou(ref["bbox"], best["bbox"])
        hit = center_hit(ref["bbox"], best["bbox"])
        total_iou += overlap
        matched += 1
        located += hit
        matches.append({"text": ref["text"], "iou": round(overlap, 4),
                        "hit": hit, "matched": True})

    return GeometryScore(
        applicable=True,
        geometry_score=round(located / len(ref_words), 4),
        mean_iou_matched=round(total_iou / matched, 4) if matched else None,
        match_rate=round(matched / len(ref_words), 4),
        reference_words=len(ref_words),
        predicted_words=len(pred_units),
        matched_words=matched,
        located_words=located,
        parse_ok=True,
        rescaled_from_thousand=rescaled,
        axes_swapped=axes_swapped,
        worst_matches=sorted(matches, key=lambda m: (m["hit"], m["iou"]))[:25],
    )


def _match_words(refs, preds) -> list[tuple[dict, dict | None]]:
    """Exact normalised-text buckets, best remaining box per reference word.
    Matching on text first is what makes the box comparison meaningful: a high
    overlap with the wrong word is not a hit."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for word in preds:
        buckets.setdefault(normalize_word(word["text"]), []).append(word)
    out = []
    for ref in refs:
        candidates = buckets.get(normalize_word(ref["text"])) or []
        if not candidates:
            out.append((ref, None))
            continue
        best = max(candidates, key=lambda c: iou(ref["bbox"], c["bbox"]))
        candidates.remove(best)
        out.append((ref, best))
    return out


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

    used: set[int] = set()
    out = []
    for ref in refs:
        ref_key = normalize_word(ref["text"])
        best_words, best_box, best_rank = None, None, (0.0, 0.0)
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
                    box = _union_box([w["bbox"] for w in window])
                    rank = (ratio, iou(ref["bbox"], box))
                    if rank > best_rank:
                        best_words, best_box, best_rank = window, box, rank
        if best_words is None or best_rank[0] < 0.7:
            out.append((ref, None))
            continue
        used.update(id(w) for w in best_words)
        out.append((ref, {"text": ref["text"], "bbox": best_box}))
    return out
