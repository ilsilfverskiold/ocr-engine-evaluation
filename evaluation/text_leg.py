"""Score reference-token coverage, CER/WER and word precision/recall."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evaluation.ground_truth import GroundTruth
from evaluation.metrics import (
    compute_cer,
    compute_gt_coverage,
    compute_ned,
    compute_wer,
    compute_word_f1,
    normalize_text,
)

# An engine emitting this much more than the reference is looping or padding.
# Not a score penalty — coverage cannot see it, so the report shows it instead
# of quietly averaging it away.
REPETITION_RATIO_THRESHOLD = 3.0


@dataclass
class TextScore:
    applicable: bool
    reason: str | None = None
    coverage: float | None = None          # headline
    cer: float | None = None
    wer: float | None = None
    ned: float | None = None
    word_precision: float | None = None
    word_recall: float | None = None
    word_f1: float | None = None
    length_ratio: float | None = None
    repetition_suspected: bool = False
    gt_words: int = 0
    output_words: int = 0
    missed_words: list[str] = field(default_factory=list)
    reference_repairs: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float | None:
        return self.coverage

    def as_metrics(self) -> dict[str, Any]:
        return {
            "gt_coverage": self.coverage,
            "cer": self.cer,
            "wer": self.wer,
            "ned": self.ned,
            "word_precision": self.word_precision,
            "word_recall": self.word_recall,
            "word_f1": self.word_f1,
            "length_ratio": self.length_ratio,
            "repetition_suspected": self.repetition_suspected,
            "gt_words": self.gt_words,
            "output_words": self.output_words,
            "text_reference_repairs": self.reference_repairs,
        }

    def as_detail(self) -> dict[str, Any]:
        """Enough to audit the headline: which reference words were not found."""
        return {"missed_words": self.missed_words[:50]}


def score_text(output_text: str, gt: GroundTruth) -> TextScore:
    """(engine output, ground truth) -> TextScore."""
    if not gt.has_gt_for("text"):
        return TextScore(applicable=False, reason=gt.missing_gt_reason("text"))

    reference = gt.plain_text
    hypothesis = output_text or ""

    coverage = compute_gt_coverage(hypothesis, reference)
    precision, recall, f1 = compute_word_f1(hypothesis, reference)

    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    ratio = round(len(hyp_words) / len(ref_words), 3) if ref_words else None

    hyp_set = set(hyp_words)
    missed = [w for w in dict.fromkeys(ref_words) if w not in hyp_set]

    return TextScore(
        applicable=True,
        coverage=coverage,
        cer=round(compute_cer(hypothesis, reference), 4),
        wer=round(compute_wer(hypothesis, reference), 4),
        ned=round(compute_ned(hypothesis, reference), 4),
        word_precision=precision,
        word_recall=recall,
        word_f1=f1,
        length_ratio=ratio,
        repetition_suspected=bool(ratio and ratio > REPETITION_RATIO_THRESHOLD),
        gt_words=len(ref_words),
        output_words=len(hyp_words),
        missed_words=missed,
        reference_repairs=gt.metadata.get("text_reference_repairs", {}),
    )
