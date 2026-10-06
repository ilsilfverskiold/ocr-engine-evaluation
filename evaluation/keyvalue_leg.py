"""Leg 4 — printed key-value pairs, scored by code.

Cached, engine-blind stage-1 aliases identify the printed label(s) for each
scalar GT field. The matcher parses the engine's pairs and compares keys and
values using the rules in kv_schema_code.py. Human gold may select any leaf
field, including fields stored inside arrays; its printed labels define KV.
No stage-1 cache or no printed pairs means not applicable; this scorer never
creates an alias cache or calls a judge. Score = (correct + blank_ok) / fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.ground_truth import GroundTruth

from evaluation.structured_metrics import walk_fields


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
            "unresolved_keys": self.unresolved_keys,
            "synonym_reference_id": self.synonym_reference_id,
            "pending_key_review": self.pending_key_review,
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
    """Compare parsed field pairs against the reference using code only."""
    if not gt.has_gt_for("key_value"):
        return KeyValueScore(applicable=False, reason=gt.missing_gt_reason("key_value"))

    reviewed = gt.metadata.get('structured_gold_reference')
    pairs = _reference_pairs(gt)
    if reviewed is None and len(pairs) < 3:
        return KeyValueScore(
            applicable=False,
            reason="too few scalar fields for key-value evaluation"
            if pairs else "structured reference has no scalar fields",
        )

    document_stem = Path(gt.document_path).stem

    # Code scoring (evaluation/kv_schema_code.py): pairs parsed from what the
    # engine returned, keys matched by field name or the page's printed labels
    # (stage 1, evaluation/alias_pass.py), values matched exactly. No model.
    from evaluation.kv_schema_code import (
        SCORER_VERSION, engine_pairs, load_aliases, score_fields,
    )
    from evaluation.schema_judge import SchemaChecklistItem, build_schema_checklist, _display_value
    from evaluation.structured_metrics import is_blank

    if reviewed is not None:
        aliases = reviewed.get('kv_aliases')
        excluded = set(reviewed.get('excluded_fields', []))
        # Human selection, not JSON storage shape or checklist sampling, is
        # authoritative. Keep every selected field, including array leaves.
        items = [SchemaChecklistItem(f.path, _display_value(f.value), is_blank(f.value))
                 for f in walk_fields(gt.true_json, gt.json_schema)
                 if f.path and f.path not in excluded
                 and (reviewed.get('shared_field_selection') is True
                      or (aliases or {}).get(f.path, {}).get('is_printed_pair'))]
        scorer = SCORER_VERSION + '+human-kv-fields-v1'
        if aliases is not None and not items:
            return KeyValueScore(applicable=False, scorer=scorer,
                                 reason='no fields selected as printed KV pairs in human gold')
    else:
        items = build_schema_checklist(gt.true_json, gt.json_schema, document_stem,
                                       use_cache=False, scalar_only=True)
        aliases = load_aliases(document_stem)
        scorer = SCORER_VERSION
    pairs, parsed_how = engine_pairs(engine, output_text or "", structured_output)
    if reviewed is not None and reviewed.get("shared_field_selection"):
        from evaluation.shared_fields import VERSION, reference_id, score_shared, load_synonym_decisions
        res = score_shared(gt, pairs, "key_value")
        decisions = load_synonym_decisions(gt)
        pending = sorted({r['key'] for r in res.unresolved if r['key'] not in decisions})
        return KeyValueScore(
            applicable=res.total > 0 and not pending,
            reason=("key-only synonym review pending; provisional calculation is not a benchmark score" if pending
                    else None if res.total else "all shared reference fields excluded"),
            field_accuracy=res.accuracy if res.total and not pending else None,
            provisional_accuracy=res.accuracy if pending else None, pending_key_review=pending,
            pairs_total=res.total, pairs_correct=res.correct, pairs_missing=res.missing,
            pairs_wrong_value=res.wrong, pairs_invented_value=res.invented,
            blank_total=res.blank_total, blank_correct=res.blank_correct,
            fields_all=res.total, engine_pairs=len(pairs), parsed_how=parsed_how,
            scorer=VERSION, verdicts=res.verdicts, unresolved_keys=res.unresolved,
            synonym_reference_id=reference_id(gt),
        )
    res = score_fields(items, pairs, "key_value", aliases, array_rows=reviewed is None)
    # The denominator is the set of fields the page prints as label+value
    # pairs, which only stage 1 can say. Without it, or when no scalar field
    # is a printed pair (chart_003: a page number, a footer, a value read off
    # a chart), the leg does not apply to the page — it is never "0 for
    # every engine".
    if res.alias_missing:
        return KeyValueScore(applicable=False, scorer=scorer, fields_all=res.fields_all,
                             reason="stage 1 (printed labels) not run for this page: key_value denominator undefined")
    if res.fields_total == 0:
        return KeyValueScore(applicable=False, scorer=scorer, fields_all=res.fields_all,
                             reason="no scalar field is printed as a label+value pair (stage 1)")
    return KeyValueScore(
        applicable=True,
        reason=None,
        field_accuracy=res.field_accuracy,
        pairs_total=res.fields_total,
        pairs_correct=res.correct,
        pairs_missing=res.missing,
        pairs_wrong_value=res.wrong,
        pairs_invented_value=sum(1 for v in res.verdicts if v.verdict == "invented"),
        blank_total=res.blank_total,
        blank_correct=res.blank_correct,
        fields_all=res.fields_all,
        engine_pairs=res.pairs,
        parsed_how=parsed_how,
        alias_missing=res.alias_missing,
        cleanup_log=res.cleanup_log,
        scorer=scorer,
        verdicts=res.verdicts,
    )
