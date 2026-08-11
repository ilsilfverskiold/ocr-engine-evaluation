"""The registry is the one place an engine is declared — these hold it to that.

Each test here corresponds to a way the registry and the code that reads it can
drift apart without anything failing loudly: an engine entered in a leg that no
converter can actually read for it scores 0.000 and looks like a bad engine.
"""

from __future__ import annotations

import pytest

from engines import ENGINES, engines_for_leg
from evaluation.legs import ALL_LEGS, DEFAULT_PASS, LEGS, parse_legs, plan_passes
from evaluation.normalizers import MARKDOWN_ENGINES, SCHEMA_CAPABLE_ENGINES
from evaluation.table_extractor import extract_table_html

MARKDOWN_TABLE = "| a | b |\n| --- | --- |\n| 1 | 2 |\n"
HTML_TABLE = "<table><tr><td>a</td></tr></table>"
TSV_TABLE = "a\tb\n1\t2\n"

SAMPLE = {"markdown": MARKDOWN_TABLE, "html": HTML_TABLE, "tsv": TSV_TABLE}


def test_every_leg_has_a_spec():
    assert set(LEGS) == set(ALL_LEGS)


def test_prompted_legs_require_a_promptable_engine():
    for name, spec in ENGINES.items():
        if spec.prompted_legs:
            assert spec.accepts_task, f"{name} has prompted legs but accepts_task=False"


def test_engines_that_cannot_be_prompted_have_a_stated_reason():
    for name, spec in ENGINES.items():
        if not spec.accepts_task:
            assert spec.leg_notes, f"{name} sits out prompted legs with no reason given"


def test_table_leg_engines_can_actually_yield_a_table():
    """An engine in the tables leg must have a way to produce one.

    Either it declares a text format the converter understands, or it returns
    real markup in structured_output. Anything else scores 0.000 for a reason
    that has nothing to do with the engine.
    """
    from evaluation.normalizers import NATIVE_STRUCTURE_ENGINES

    for name in engines_for_leg("tables"):
        spec = ENGINES[name]
        if name in NATIVE_STRUCTURE_ENGINES:
            continue
        assert spec.table_format in SAMPLE, (
            f"{name} competes in the tables leg but declares "
            f"table_format={spec.table_format!r}"
        )
        recovered = extract_table_html(SAMPLE[spec.table_format], name)
        assert recovered and "<table" in recovered, (
            f"{name} declares table_format={spec.table_format!r} but the "
            f"converter recovered nothing from that format"
        )


def test_engines_outside_the_tables_leg_are_not_silently_convertible():
    for name in ENGINES:
        if ENGINES[name].answers("tables"):
            continue
        assert ENGINES[name].table_format is None, (
            f"{name} declares a table format but does not compete in the tables leg"
        )


def test_markdown_capability_matches_the_registry():
    assert MARKDOWN_ENGINES == frozenset(
        n for n, s in ENGINES.items() if "markdown" in s.native_legs
    )


def test_schema_capability_matches_the_registry():
    assert SCHEMA_CAPABLE_ENGINES == frozenset(
        n for n, s in ENGINES.items() if "schema" in s.prompted_legs
    )


def test_native_legs_share_one_pass():
    plans = plan_passes("docling", ["text", "markdown", "tables"])
    assert len(plans) == 1
    assert plans[0].name == DEFAULT_PASS
    assert plans[0].leg_for_task is None


def test_each_prompted_leg_adds_exactly_one_pass():
    plans = plan_passes("gemini_flash", ["text", "markdown", "tables", "key_value", "schema"])
    assert len(plans) == 3
    assert {p.name for p in plans} == {DEFAULT_PASS, "key_value", "schema"}


def test_an_engine_that_answers_nothing_plans_no_calls():
    assert plan_passes("tesseract", ["schema", "key_value"]) == []


def test_selecting_more_legs_never_costs_more_calls_than_the_legs_need():
    one = plan_passes("mistral_ocr_4", ["text"])
    three = plan_passes("mistral_ocr_4", ["text", "markdown", "tables"])
    assert len(one) == len(three) == 1


def test_schema_task_injects_the_documents_own_schema():
    schema = {"type": "object", "properties": {"invoice_no": {"type": "string"}}}
    task = LEGS["schema"].task_for(schema)
    assert "invoice_no" in task.prompt
    assert task.json_schema == schema
    assert task.wants_json is True


def test_schema_task_refuses_to_run_without_a_schema():
    with pytest.raises(ValueError):
        LEGS["schema"].task_for(None)


def test_text_leg_sends_no_task_so_leg_one_is_unchanged():
    assert LEGS["text"].task_for() is None


def test_parse_legs_round_trip_and_rejection():
    assert parse_legs("all") == list(ALL_LEGS)
    assert parse_legs("tables,text") == ["text", "tables"]
    with pytest.raises(ValueError):
        parse_legs("text,nonsense")


def test_registry_leg_membership_matches_the_scoring_design():
    # docs/scoring-design.md fixes these lists; drift here is drift from the spec.
    assert set(engines_for_leg("text")) == set(ENGINES)
    assert set(engines_for_leg("markdown")) == {
        "mistral_ocr_4", "llamaparse", "marker", "docling", "mineru",
        # glm_ocr answers markdown through its own call with the generic leg
        # ask, not by reusing Text Recognition output — see engines.py.
        "glm_ocr", "deepseek_ocr", "gemini_flash", "claude_sonnet_4_6", "qwen3_vl",
        # paddleocr_vl's raw_text is the pipeline's own markdown rendering,
        # with tables embedded as HTML (normalised by the scorer).
        "paddleocr_vl",
    }
    assert set(engines_for_leg("tables")) == {
        "mistral_ocr_4", "llamaparse", "marker", "docling", "mineru", "deepseek_ocr",
        "glm_ocr", "paddleocr_vl", "gemini_flash", "claude_sonnet_4_6",
        # qwen3_vl: general VLM, treated like gemini/claude — tables come from
        # its default markdown-preserving conversion (see engines.py).
        "qwen3_vl",
        "textract_structured",
    }
    assert set(engines_for_leg("schema")) == {
        # Every engine here can be prompted, so every one competes; a zero is
        # the model failing, not us declining to ask (see engines.py).
        "gemini_flash", "claude_sonnet_4_6", "qwen3_vl", "glm_ocr", "deepseek_ocr",
        "textract_structured",
        # via the vendors' own extraction features (Mistral Annotations,
        # LlamaExtract) — the documented structured path, not an invented prompt.
        "mistral_ocr_4", "llamaparse",
    }
