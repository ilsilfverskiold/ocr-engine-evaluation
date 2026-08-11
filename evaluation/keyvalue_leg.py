"""Leg 4 — key-value pairs (LLM-judged).

Ask: *"Extract every printed label and its value from this document as a flat
JSON object of label -> value. Use the label exactly as printed. Use null where
a printed field is blank. Output only JSON."*

Every engine is evaluated the same way: an LLM judge checks whether each
ground-truth field value appears correctly in the engine's output, regardless
of the output format (JSON, FORMS key-value pairs, raw text) or field naming.

This replaced the deterministic label-matching scorer that penalised engines
whose label naming didn't match the GT's schema identifiers.

The checklist is the scalar (non-array) fields from ``true_json`` — arrays
are excluded because they belong to the schema leg. The judge (Gemini 2.5 Pro)
checks each field value against the engine output. Exact substring matches
are settled locally to save cost.

Score: field accuracy = (correct + blank_ok) / total fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.ground_truth import GroundTruth
from evaluation.schema_judge import (
    SchemaChecklistItem,
    SchemaFieldVerdict,
    SchemaJudgeResult,
    _prepare_engine_output,
    _verify_locally,
    judge_schema,
)
from evaluation.structured_metrics import (
    is_blank,
    walk_fields,
)


@dataclass
class KeyValueScore:
    applicable: bool
    reason: str | None = None
    field_accuracy: float | None = None      # headline
    pairs_total: int = 0
    pairs_correct: int = 0
    pairs_missing: int = 0
    pairs_wrong_value: int = 0
    pairs_invented_value: int = 0
    blank_total: int = 0
    blank_correct: int = 0
    judge_cost_usd: float = 0.0
    verdicts: list[SchemaFieldVerdict] = field(default_factory=list)

    @property
    def score(self) -> float | None:
        return self.field_accuracy

    def as_metrics(self) -> dict[str, Any]:
        return {
            "kv_field_accuracy": self.field_accuracy,
            "kv_pairs_total": self.pairs_total,
            "kv_pairs_correct": self.pairs_correct,
            "kv_pairs_missing": self.pairs_missing,
            "kv_pairs_wrong_value": self.pairs_wrong_value,
            "kv_pairs_invented_value": self.pairs_invented_value,
            "kv_blank_total": self.blank_total,
            "kv_blank_correct": self.blank_correct,
            "kv_judge_cost_usd": self.judge_cost_usd,
        }

    def as_detail(self) -> dict[str, Any]:
        return {
            "verdicts": [
                {
                    "path": v.path,
                    "expected": v.expected,
                    "verdict": v.verdict,
                    "is_blank": v.is_blank,
                }
                for v in self.verdicts
            ]
        }


def _reference_pairs(gt: GroundTruth) -> list[tuple[str, Any]]:
    """The flat label -> value pairs this document should yield.

    Array contents are excluded: a line-item table is not a key-value pair,
    and including it would score the same values twice — once here and once
    on the schema leg, where the array is what the schema asks for.
    """
    if gt.true_json is None:
        return []
    pairs: list[tuple[str, Any]] = []
    for f in walk_fields(gt.true_json, gt.json_schema):
        if f.in_array or not f.path:
            continue
        pairs.append((f.path, f.value))
    return pairs


def score_key_value(
    output_text: str,
    gt: GroundTruth,
    engine: str,
    structured_output: dict | None = None,
) -> KeyValueScore:
    """(engine output, ground truth) -> KeyValueScore.

    All engines are judged the same way: an LLM checks whether each GT
    scalar field value appears in the engine's output, regardless of format
    or field naming.
    """
    if not gt.has_gt_for("key_value"):
        return KeyValueScore(applicable=False, reason=gt.missing_gt_reason("key_value"))

    pairs = _reference_pairs(gt)
    if len(pairs) < 3:
        return KeyValueScore(
            applicable=False,
            reason="too few scalar fields for key-value evaluation"
            if pairs else "structured reference has no scalar fields",
        )

    document_stem = Path(gt.document_path).stem

    result: SchemaJudgeResult = judge_schema(
        true_json=gt.true_json,
        json_schema=gt.json_schema,
        engine_text=output_text or "",
        engine=f"{engine}__kv",
        document_stem=document_stem,
        structured_output=structured_output,
        scalar_only=True,
    )

    if result.error:
        return KeyValueScore(
            applicable=True,
            reason=f"judge error: {result.error}",
            field_accuracy=0.0,
            pairs_total=result.checklist_size,
            judge_cost_usd=result.cost_usd,
        )

    correct = sum(
        1 for i in result.items if i.verdict in ("correct", "blank_ok")
    )
    missing = sum(1 for i in result.items if i.verdict == "missing")
    wrong = sum(1 for i in result.items if i.verdict == "wrong")
    invented = sum(1 for i in result.items if i.verdict == "invented")
    blank_total = sum(1 for i in result.items if i.is_blank)
    blank_correct = sum(
        1 for i in result.items if i.is_blank and i.verdict == "blank_ok"
    )

    return KeyValueScore(
        applicable=True,
        field_accuracy=result.field_accuracy,
        pairs_total=len(result.items),
        pairs_correct=correct,
        pairs_missing=missing,
        pairs_wrong_value=wrong,
        pairs_invented_value=invented,
        blank_total=blank_total,
        blank_correct=blank_correct,
        judge_cost_usd=result.cost_usd,
        verdicts=result.items,
    )
