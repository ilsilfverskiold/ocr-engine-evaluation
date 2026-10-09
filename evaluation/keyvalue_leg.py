"""KV evaluation: known keys first, value-assisted judge fallback, incremental cache.

All engines use the same reference fields as schema. Matching values nominate
candidates but do not earn credit until their field identity is confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.ground_truth import GroundTruth



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
    verdicts: list[Any] = field(default_factory=list)
    # code scoring (2026-10-01)
    fields_all: int = 0            # every scalar field; pairs_total is the denominator used
    engine_pairs: int = 0
    parsed_how: str = ""
    alias_missing: bool = False
    cleanup_log: list[str] = field(default_factory=list)
    scorer: str | None = None
    unresolved_keys: list[dict] = field(default_factory=list)
    synonym_reference_id: str | None = None
    pending_key_review: list[str] = field(default_factory=list)
    provisional_accuracy: float | None = None
    judge_errors: list[str] = field(default_factory=list)
    judge_reviews: list[dict] = field(default_factory=list)
    label_review: dict = field(default_factory=dict)
    pair_context: list = field(default_factory=list)
    native_matches: dict = field(default_factory=dict)

    @property
    def score(self) -> float | None:
        return self.field_accuracy

    def as_metrics(self) -> dict[str, Any]:
        return {
            "kv_field_accuracy": self.field_accuracy,
            "kv_pairs_total": self.pairs_total,
            "kv_fields_all": self.fields_all,
            "kv_pairs_correct": self.pairs_correct,
            "kv_pairs_missing": self.pairs_missing,
            "kv_pairs_wrong_value": self.pairs_wrong_value,
            "kv_pairs_invented_value": self.pairs_invented_value,
            "kv_blank_total": self.blank_total,
            "kv_blank_correct": self.blank_correct,
            "kv_populated_total": self.pairs_total - self.blank_total,
            "kv_populated_accuracy": (self.pairs_correct / (self.pairs_total - self.blank_total)
                                      if self.pairs_total > self.blank_total and not self.pending_key_review else None),
            "kv_blank_accuracy": self.blank_correct / self.blank_total if self.blank_total else None,
            "kv_engine_pairs": self.engine_pairs,
            "kv_parsed_how": self.parsed_how,
            "kv_alias_missing": self.alias_missing,
            "kv_scorer": self.scorer,
            "kv_judge_cost_usd": self.judge_cost_usd,
            "kv_unresolved_keys": len(self.unresolved_keys),
            "kv_synonym_reference_id": self.synonym_reference_id,
            "kv_pending_key_review": len(self.pending_key_review),
            "kv_provisional_accuracy": self.provisional_accuracy,
        }

    def as_detail(self) -> dict[str, Any]:
        return {
            "cleanup": self.cleanup_log,
            "pair_context": self.pair_context,
            "native_matches": self.native_matches,
            "label_review": self.label_review,
            "unresolved_keys": self.unresolved_keys,
            "synonym_reference_id": self.synonym_reference_id,
            "pending_key_review": self.pending_key_review,
            "judge_errors": self.judge_errors,
            "judge_reviews": self.judge_reviews,
            "verdicts": [
                {
                    "path": v.path,
                    "expected": v.expected,
                    "verdict": v.verdict,
                    "is_blank": v.is_blank,
                    "key": getattr(v, "key", ""),
                    "value": getattr(v, "value", ""),
                    "how": getattr(v, "how", ""),
                    "printed_pair": getattr(v, "printed_pair", None),
                }
                for v in self.verdicts
            ],
        }


def score_key_value(
    output_text: str,
    gt: GroundTruth,
    engine: str,
    structured_output: dict | None = None,
    use_judge: bool | None = None,
) -> KeyValueScore:
    """Shared live/stored-run path for every KV engine; no schema inference."""
    if not gt.has_gt_for("key_value"):
        return KeyValueScore(applicable=False, reason=gt.missing_gt_reason("key_value"))
    from evaluation.kv_schema_code import engine_pairs
    pair_context = []
    pairs, parsed_how = engine_pairs(engine, output_text or "", structured_output, pair_context)
    pair_context = pair_context if any(pair_context) else None
    from evaluation.kv_native_context import native_field_matches
    native_matches = native_field_matches(gt, pairs, pair_context or [])
    from evaluation.shared_fields import KV_VERSION, reference_id, score_shared, reference_fields
    from evaluation.kv_value_groups import resolve, relation
    resolution = resolve(gt, pairs, use_judge=use_judge, pair_context=pair_context, native_matches=native_matches)
    res = score_shared(gt, pairs, "key_value", resolution.synonyms, resolution.groups, native_matches, pair_context)
    pending = resolution.pending
    for item in res.unresolved:
        key, value = pairs[item['index']]
        item['value'] = value
        candidates = [f.path for f in reference_fields(gt, "key_value") if relation(f.value, value, f.path)]
        item['candidate_reference_fields'] = candidates
        item['decision'] = ('contextual value review pending' if any(p in candidates for p in pending)
                            else 'see cached contextual field reviews; no automatic value credit' if candidates
                            else 'no whole-value or complete-token containment candidate')
    return KeyValueScore(
        applicable=res.total > 0 and not pending,
        reason=("KV value-group judgment pending; provisional calculation is not a benchmark score" if pending
                else None if res.total else "all shared reference fields excluded"),
        field_accuracy=res.accuracy if res.total and not pending else None,
        provisional_accuracy=res.accuracy if pending else None, pending_key_review=pending,
        pairs_total=res.total, pairs_correct=res.correct, pairs_missing=res.missing,
        pairs_wrong_value=res.wrong, pairs_invented_value=res.invented,
        blank_total=res.blank_total, blank_correct=res.blank_correct,
        fields_all=res.total, engine_pairs=len(pairs), parsed_how=parsed_how,
        scorer=KV_VERSION, verdicts=res.verdicts, unresolved_keys=res.unresolved,
        synonym_reference_id=reference_id(gt),
        judge_cost_usd=resolution.cost_usd, judge_errors=resolution.errors,
        judge_reviews=resolution.decisions,
        label_review=gt.metadata.get("kv_label_review", {}),
        pair_context=pair_context or [], native_matches=native_matches,
    )
