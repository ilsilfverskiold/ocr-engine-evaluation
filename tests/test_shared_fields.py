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


def test_schema_usability_is_separate_from_recovered_values():
    ref = gt({"Total": "100"})
    result = score_schema('```json\n{"Total":"100"}\n```', ref, "x")
    assert result.score == 1 and result.compliance["valid_json"] is False
    assert schema_compliance('{"Total":100}', ref.inference_schema)["schema_valid"] is False
    assert schema_compliance('{"Total":"100"}', ref.inference_schema)["schema_valid"] is True
