from __future__ import annotations

import re
import unicodedata
from collections import Counter

import editdistance


def _markdown_to_plain(text: str) -> str:
    """Render markdown to plain text via the markdown library + HTML stripping."""
    try:
        import markdown
        from bs4 import BeautifulSoup
        html = markdown.markdown(text, extensions=["tables", "fenced_code"])
        return BeautifulSoup(html, "html.parser").get_text(separator=" ")
    except Exception:
        return text


def normalize_text(text: str) -> str:
    # Remove outer Markdown fences before rendering so packaging does not change CER/WER.
    from evaluation.markdown_leg import strip_wrapping_fence

    text = strip_wrapping_fence(text or "")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\(cid:\d+\)", "", text)
    text = _markdown_to_plain(text)
    text = re.sub(r"\\([^\s])", r"\1", text)   # backslash escapes the renderer passes through e.g. \$
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def compute_cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate = edit_distance / len(reference). Lower is better."""
    h = normalize_text(hypothesis)
    r = normalize_text(reference)
    if not r:
        return 0.0 if not h else 1.0
    return editdistance.eval(h, r) / len(r)


def compute_wer(hypothesis: str, reference: str) -> float:
    """Word Error Rate = word-level edit_distance / len(ref_words). Lower is better."""
    h_words = normalize_text(hypothesis).split()
    r_words = normalize_text(reference).split()
    if not r_words:
        return 0.0 if not h_words else 1.0
    return editdistance.eval(h_words, r_words) / len(r_words)


def compute_word_f1(hypothesis: str, reference: str) -> tuple[float, float, float]:
    """
    Token-level precision, recall, F1 (multiset intersection — duplicates count).

    precision: fraction of OCR words that appear in GT  → low = hallucination
    recall:    fraction of GT words captured by OCR     → low = omission
    f1:        harmonic mean

    All three in [0, 1], higher is better.
    """
    hyp = Counter(normalize_text(hypothesis).split())
    ref = Counter(normalize_text(reference).split())
    if not ref and not hyp:
        return 1.0, 1.0, 1.0
    if not hyp:
        return 0.0, 0.0, 0.0
    if not ref:
        return 0.0, 0.0, 0.0
    common = sum((hyp & ref).values())
    precision = common / sum(hyp.values())
    recall = common / sum(ref.values())
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def compute_gt_coverage(hypothesis: str, reference: str) -> float:
    """
    Fraction of unique GT words found anywhere in the OCR output (set-based).

    Unlike recall in compute_word_f1 (multiset), this ignores word frequency —
    a GT word counts as covered if it appears at least once in the output.
    Useful for checking content presence independent of formatting repetition.
    """
    hyp_words = set(normalize_text(hypothesis).split())
    ref_words = set(normalize_text(reference).split())
    if not ref_words:
        return 1.0
    if not hyp_words:
        return 0.0
    return round(sum(1 for w in ref_words if w in hyp_words) / len(ref_words), 4)


def compute_ned(hypothesis: str, reference: str) -> float:
    """Normalized edit distance: 1 - (edit_dist / max_len). Range [0,1], 1=perfect.
    Kept for backwards compatibility; prefer CER + word_f1 for new analysis."""
    h = normalize_text(hypothesis)
    r = normalize_text(reference)
    if not r and not h:
        return 1.0
    if not r or not h:
        return 0.0
    dist = editdistance.eval(h, r)
    return 1.0 - dist / max(len(h), len(r))
