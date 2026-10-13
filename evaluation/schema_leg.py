"""Score supplied-schema field values and extra keys without model judging."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    unpenalized_accuracy: float | None = None
    extra_fields: list[dict] = field(default_factory=list)
    compliance: dict[str, Any] = field(default_factory=dict)
    sample: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float | None:
        return self.field_accuracy

    def as_metrics(self) -> dict[str, Any]:
        return {
            "schema_field_accuracy": self.field_accuracy,
            "schema_sample": self.sample,
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
            "schema_unpenalized_accuracy": self.unpenalized_accuracy,
            "schema_extra_fields": len(self.extra_fields),
            "schema_valid_json": self.compliance.get("valid_json"),
            "schema_contract_valid": self.compliance.get("schema_valid"),
        }

    def as_detail(self) -> dict[str, Any]:
        return {
            "cleanup": self.cleanup_log,
            "extra_fields": self.extra_fields,
            "compliance": self.compliance,
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

    from evaluation.schema_contract import schema_scoring_reference
    gt = schema_scoring_reference(gt)

    from evaluation.kv_schema_code import SCORER_VERSION, engine_pairs, score_fields
    from evaluation.structured_metrics import is_blank, walk_fields

    reviewed = gt.metadata.get('structured_gold_reference')
    shared = reviewed is not None and reviewed.get('shared_field_selection') is True
    pairs, parsed_how = engine_pairs(engine, output_text or "", structured_output)
    if shared:
        from evaluation.shared_fields import VERSION, schema_compliance, score_shared
        res = score_shared(gt, pairs, "schema")
        return SchemaScore(
            applicable=res.total > 0, reason=None if res.total else "all shared reference fields excluded",
            field_accuracy=res.accuracy if res.total else None,
            unpenalized_accuracy=res.unpenalized_accuracy, extra_fields=res.extras,
            compliance=schema_compliance(output_text or "", gt.inference_schema),
            sample=gt.metadata.get("schema_sample", {}),
            fields_total=res.total, fields_correct=res.correct, fields_missing=res.missing,
            fields_wrong_value=res.wrong, fields_invented_value=res.invented,
            blank_total=res.blank_total, blank_correct=res.blank_correct, engine_pairs=len(pairs),
            parsed_how=parsed_how, scorer=VERSION, verdicts=res.verdicts,
            reference_repairs=gt.metadata.get("structured_reference_repairs", {}),
        )
    excluded = set(gt.metadata.get("structured_reference_repairs", {}).get("excluded_fields", []))
    items = [SchemaField(f.path, "" if is_blank(f.value) else str(f.value).strip(), is_blank(f.value))
             for f in walk_fields(gt.true_json, gt.json_schema) if f.path and f.path not in excluded]
    if len(items) > 80:
        step = len(items) / 80
        items = [items[int(i * step)] for i in range(80)]
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


@dataclass
class SchemaField:
    path: str
    value: str
    is_blank: bool = False
