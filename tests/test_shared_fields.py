import json

import pytest

from evaluation.ground_truth import GroundTruth
from evaluation.shared_fields import match_keys, reference_id, schema_compliance, score_shared
from evaluation.schema_leg import score_schema
from evaluation.keyvalue_leg import score_key_value


def gt(values=None, excluded=()):
    values = values or {"Total": "100", "Date": "2026-10-06"}
    schema = {"type": "object", "properties": {k: {"type": ["string", "null"]} for k in values},
              "required": list(values), "additionalProperties": False}
    return GroundTruth("documents/easy/invoice/x.png", "easy", "invoice", "", true_json=values,
                       json_schema=schema, metadata={"structured_gold_reference": {
                           "shared_field_selection": True, "excluded_fields": list(excluded),
                           "kv_aliases": {k: {"printed_labels": [k], "is_printed_pair": k not in excluded}
                                          for k in values}}})


def test_schema_extras_penalized_but_kv_completeness_not_assumed():
    ref = gt()
    pairs = [["Total", "100"], ["Date", "2026-10-06"], ["Invented", "value"]]
    assert score_shared(ref, pairs, "schema").accuracy == pytest.approx(2 / 3)
    assert score_shared(ref, pairs, "key_value").accuracy == 1


def test_excluded_and_contract_permitted_fields_are_not_extra():
    ref = gt({"Total": "100", "Uncertain": "maybe"}, excluded=["Uncertain"])
    ref.json_schema["properties"]["Optional"] = {"type": "string"}
    result = score_shared(ref, [["Total", "100"], ["Uncertain", "any"], ["Optional", "allowed"]], "schema")
    assert result.accuracy == 1 and not result.extras
    ref.json_schema["additionalProperties"] = True
    assert not score_shared(ref, [["Another", "allowed"]], "schema").extras


@pytest.mark.parametrize("expected,wrong", [("100", "1000"), ("12.34", "1234"), ("-100", "100")])
def test_wrong_values_cannot_find_credit_elsewhere(expected, wrong):
    ref = gt({"Total": expected})
    result = score_shared(ref, [["Subtotal", expected], ["Total", wrong]], "key_value")
    assert result.accuracy == 0 and result.wrong == 1


def test_missing_blank_is_not_an_explicit_null_answer():
    ref = gt({"Total": "100", "Empty": None})
    assert score_shared(ref, [["Total", "100"]], "schema").accuracy == .5
    assert score_shared(ref, [["Total", "100"], ["Empty", ""]], "schema").accuracy == 1


def test_synonym_matches_key_before_value_and_cannot_share_one_pair():
    ref = gt({"Total": "100", "Subtotal": "100"})
    assigned, unresolved, used = match_keys(ref, [["Amount due", "900"]], "key_value", {"Amount due": "Total"})
    assert set(assigned) == {"Total"} and used == {0}
    assert not unresolved
    result = score_shared(ref, [["Total", "100"]], "key_value")
    assert result.accuracy == .5


def test_duplicate_schema_keys_are_counted_and_do_not_choose_best_value():
    ref = gt({"Total": "100"})
    result = score_schema('{"Total":"900", "Total":"100"}', ref, "x")
    assert result.score == 0 and len(result.extra_fields) == 1
    assert result.compliance["valid_json"] is False


def test_schema_requires_supplied_keys_kv_can_use_printed_alias():
    ref = gt({"Invoice number": "0012"})
    ref.metadata["structured_gold_reference"]["kv_aliases"]["Invoice number"]["printed_labels"].append("Invoice #")
    text = '{"Invoice #":"0012"}'
    assert score_key_value(text, ref, "x").score == 1
    assert score_schema(text, ref, "x").score == 0


def test_reference_edit_invalidates_synonyms():
    ref = gt()
    before = reference_id(ref)
    ref.true_json["Total"] = "200"
    assert before != reference_id(ref)


def test_numbered_printed_label_matches_without_judge_or_value_search():
    ref = gt({"4b — Taxable amount": "100", "5b — Taxable amount": "200"})
    assigned, pending, _ = match_keys(ref, [("b Taxable amount 4b", "900")], "key_value")
    assert set(assigned) == {"4b — Taxable amount"} and not pending
    result = score_shared(ref, [("b Taxable amount 4b", "900")], "key_value")
    assert result.wrong == 1  # A matched key does not make its value correct.
    for key in ['a Taxable amount 4b', 'Taxable amount', 'b Taxable amount 6b', 'b Subtotal taxable amount 4b']:
        assert not match_keys(ref, [(key, "100")], "key_value")[0]
    assert not match_keys(ref, [("b Taxable amount 4b", "100")], "schema")[0]


def test_kv_waits_for_key_decisions_and_accepts_explicit_no_match(tmp_path, monkeypatch):
    from evaluation import shared_fields
    monkeypatch.setattr(shared_fields, "CACHE_ROOT", tmp_path)
    ref = gt({"Total": "100"})
    pending = score_key_value('{"Amount due":"100"}', ref, 'x')
    assert not pending.applicable and pending.score is None
    assert pending.pending_key_review == ['Amount due']
    record = {'reference_id': reference_id(ref), 'matches': [{'output_key': 'Amount due', 'reference_key': None}]}
    (tmp_path / 'x.json').write_text(json.dumps(record))
    done = score_key_value('{"Amount due":"100"}', ref, 'x')
    assert done.applicable and done.score == 0 and not done.pending_key_review
    record['matches'][0]['reference_key'] = 'Total'
    (tmp_path / 'x.json').write_text(json.dumps(record))
    assert score_key_value('{"Amount due":"100"}', ref, 'x').score == 1


def test_schema_usability_is_separate_from_recovered_values():
    ref = gt({"Total": "100"})
    result = score_schema('```json\n{"Total":"100"}\n```', ref, "x")
    assert result.score == 1 and result.compliance["valid_json"] is False
    assert schema_compliance('{"Total":100}', ref.inference_schema)["schema_valid"] is False
    assert schema_compliance('{"Total":"100"}', ref.inference_schema)["schema_valid"] is True
