"""Leg 5 — schema JSON, scored by code.

A deterministic checklist is built from true_json + json_schema. The matcher
compares field names and values, aligning array rows before checking cells.
See kv_schema_code.py for normalization and matching rules. No LLM verdict
is requested. Score = (correct + blank_ok) / sampled fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.ground_truth import GroundTruth



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
    verdicts: list[Any] = field(default_factory=list)
    # code scoring (2026-10-01)
    engine_pairs: int = 0
    parsed_how: str = ""
    cleanup_log: list[str] = field(default_factory=list)
    scorer: str | None = None
    reference_repairs: dict[str, Any] = field(default_factory=dict)

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
            "schema_engine_pairs": self.engine_pairs,
            "schema_parsed_how": self.parsed_how,
            "schema_scorer": self.scorer,
            "schema_reference_repairs": self.reference_repairs,
            "schema_judge_cost_usd": self.judge_cost_usd,
        }

    def as_detail(self) -> dict[str, Any]:
        return {
            "cleanup": self.cleanup_log,
            "verdicts": [
                {
                    "path": v.path,
                    "expected": v.expected,
                    "verdict": v.verdict,
                    "is_blank": v.is_blank,
                    "key": getattr(v, "key", ""),
                    "value": getattr(v, "value", ""),
                    "how": getattr(v, "how", ""),
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
    """Compare parsed field pairs against the reference using code only."""
    if not gt.has_gt_for("schema"):
        return SchemaScore(applicable=False, reason=gt.missing_gt_reason("schema"))

    document_stem = Path(gt.document_path).stem

    # Code scoring (evaluation/kv_schema_code.py): the JSON answer parsed
    # (truncated JSON salvaged as far as it goes), keys matched by path or
    # its last segment, values matched exactly. No model.
    from evaluation.kv_schema_code import SCORER_VERSION, engine_pairs, score_fields
    from evaluation.schema_judge import build_schema_checklist

    items = build_schema_checklist(gt.true_json, gt.json_schema, document_stem,
                                   use_cache=False, scalar_only=False)
    pairs, parsed_how = engine_pairs(engine, output_text or "", structured_output)
    res = score_fields(items, pairs, "schema", None)
    return SchemaScore(
        applicable=True,
        field_accuracy=res.field_accuracy,
        fields_total=res.fields_total,
        fields_correct=res.correct,
        fields_missing=res.missing,
        fields_wrong_value=res.wrong,
        fields_invented_value=sum(1 for v in res.verdicts if v.verdict == "invented"),
        blank_total=res.blank_total,
        blank_correct=res.blank_correct,
        engine_pairs=res.pairs,
        parsed_how=parsed_how,
        cleanup_log=res.cleanup_log,
        scorer=SCORER_VERSION,
        reference_repairs=gt.metadata.get("structured_reference_repairs", {}),
        verdicts=res.verdicts,
    )
