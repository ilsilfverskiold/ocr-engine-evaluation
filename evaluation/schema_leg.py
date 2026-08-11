"""Leg 5 — schema JSON (LLM-judged).

Every engine is evaluated the same way: an LLM judge checks whether each
ground-truth field value appears correctly in the engine's output, regardless
of the output format (JSON, FORMS key-value pairs, raw text) or field naming.

This replaced the deterministic key-matching scorer that required output keys
to match the schema exactly — which penalised engines with different naming
conventions and made native-KV engines like Textract unscorable without a
hand-written label mapping.

The checklist is built deterministically from ``true_json`` + ``json_schema``
(no LLM needed).  The judge (Gemini 2.5 Pro) checks each field value against
the engine output.  Exact substring matches are settled locally to save cost.

Score: field accuracy = (correct + blank_ok) / total fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.ground_truth import GroundTruth
from evaluation.schema_judge import (
    SchemaFieldVerdict,
    SchemaJudgeResult,
    judge_schema,
)


@dataclass
class SchemaScore:
    applicable: bool
    reason: str | None = None
    field_accuracy: float | None = None      # headline
    fields_total: int = 0
    fields_correct: int = 0
    fields_missing: int = 0
    fields_wrong_value: int = 0
    fields_invented_value: int = 0
    blank_total: int = 0
    blank_correct: int = 0
    judge_cost_usd: float = 0.0
    verdicts: list[SchemaFieldVerdict] = field(default_factory=list)

    @property
    def score(self) -> float | None:
        return self.field_accuracy

    def as_metrics(self) -> dict[str, Any]:
        return {
            "schema_field_accuracy": self.field_accuracy,
            "schema_fields_total": self.fields_total,
            "schema_fields_correct": self.fields_correct,
            "schema_fields_missing": self.fields_missing,
            "schema_fields_wrong_value": self.fields_wrong_value,
            "schema_fields_invented_value": self.fields_invented_value,
            "schema_blank_total": self.blank_total,
            "schema_blank_correct": self.blank_correct,
            "schema_judge_cost_usd": self.judge_cost_usd,
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


def score_schema(
    output_text: str,
    gt: GroundTruth,
    engine: str,
    structured_output: dict | None = None,
) -> SchemaScore:
    """(engine output, ground truth) -> SchemaScore.

    All engines are judged the same way: an LLM checks whether each GT field
    value appears in the engine's output, regardless of format or field naming.
    """
    if not gt.has_gt_for("schema"):
        return SchemaScore(applicable=False, reason=gt.missing_gt_reason("schema"))

    document_stem = Path(gt.document_path).stem

    result: SchemaJudgeResult = judge_schema(
        true_json=gt.true_json,
        json_schema=gt.json_schema,
        engine_text=output_text or "",
        engine=engine,
        document_stem=document_stem,
        structured_output=structured_output,
    )

    if result.error:
        return SchemaScore(
            applicable=True,
            reason=f"judge error: {result.error}",
            field_accuracy=0.0,
            fields_total=result.checklist_size,
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

    return SchemaScore(
        applicable=True,
        field_accuracy=result.field_accuracy,
        fields_total=len(result.items),
        fields_correct=correct,
        fields_missing=missing,
        fields_wrong_value=wrong,
        fields_invented_value=invented,
        blank_total=blank_total,
        blank_correct=blank_correct,
        judge_cost_usd=result.cost_usd,
        verdicts=result.items,
    )
