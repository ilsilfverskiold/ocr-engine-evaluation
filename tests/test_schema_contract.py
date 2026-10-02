"""Regression cases for request/schema defects found in saved engine calls."""
import copy
from pathlib import Path

import pytest
from jsonschema import Draft7Validator
from jsonschema.exceptions import ValidationError

from evaluation.ground_truth import load_corpus
from evaluation.legs import instance_template
from evaluation.schema_contract import extraction_schema, repair_spanned_financial_rows


def test_all_reference_answers_satisfy_the_request_schema():
    corpus = [g for g in load_corpus() if g.json_schema]
    assert len(corpus) == 65
    errors = [(Path(g.document_path).stem, e.message)
              for g in corpus for e in Draft7Validator(g.inference_schema).iter_errors(g.true_json)]
    assert errors == []


def test_nullable_nested_types_keep_instance_template_structure():
    schema = {"type": "object", "properties": {
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "amount": {"type": "number"}}}},
        "state": {"type": "string", "enum": ["paid", "due"]}}}
    before = copy.deepcopy(schema)
    fixed = extraction_schema(schema)
    assert schema == before
    assert instance_template(fixed) == {"items": [{"amount": ""}], "state": ""}
    Draft7Validator(fixed).validate({"items": [{"amount": None}], "state": None})
    with pytest.raises(ValidationError):
        Draft7Validator(fixed).validate({"items": [{"amount": "invented"}], "state": "paid"})


def test_open_dictionary_cannot_silently_become_an_empty_extraction():
    with pytest.raises(ValueError, match="explicit target fields"):
        extraction_schema({"type": "object", "additionalProperties": {"type": "string"}})


def test_explicit_targets_survive_vendor_schema_sanitizer():
    from runners.llamaparse_runner import _sanitize_schema_for_llama
    docs = {Path(g.document_path).stem: g for g in load_corpus()}
    for stem in ("funsd_001", "funsd_005"):
        g = docs[stem]
        effective = _sanitize_schema_for_llama(g.inference_schema)
        assert set(effective["properties"]) == set(g.true_json)
        assert effective["additionalProperties"] is False
    g = docs["fintab_001"]
    effective = _sanitize_schema_for_llama(g.inference_schema)
    values = effective["properties"]["rows"]["items"]["properties"]["values"]
    assert set(values["properties"]) == {"31.7.17", "31.7.16"}
    assert values["additionalProperties"] is False


def test_merged_header_repair_changes_attribution_without_changing_amounts():
    reference = {"columns": ["", "2025", "2024"], "rows": [
        {"label": "Revenue", "values": {"2024": "70", "col_4": "336"}}]}
    before = copy.deepcopy(reference)
    html = '<table><tr><td></td><th colspan="2">2025</th><th colspan="2">2024</th></tr>' \
           '<tr><td>Revenue</td><td></td><td>70</td><td></td><td>336</td></tr></table>'
    fixed, changed = repair_spanned_financial_rows(reference, html)
    assert changed
    assert fixed["rows"] == [{"label": "Revenue", "values": {"2025": "70", "2024": "336"}}]
    assert reference == before
    with pytest.raises(ValueError, match="alter reference"):
        repair_spanned_financial_rows(reference, html.replace(">336<", ">337<"))


def test_financial_loader_omits_unrequested_csv_and_records_repairs():
    financial = [g for g in load_corpus() if g.doc_type == "financial_table"]
    assert len(financial) == 5
    assert all("csv" not in g.true_json for g in financial)
    repaired = [g for g in financial if g.metadata["structured_reference_repairs"]["spanning_headers_corrected"]]
    assert [Path(g.document_path).stem for g in repaired] == ["fintab_001"]
    row = next(r for r in repaired[0].true_json["rows"] if r["label"] == "Inspection")
    assert row["values"] == {"31.7.17": "70", "31.7.16": "336"}
