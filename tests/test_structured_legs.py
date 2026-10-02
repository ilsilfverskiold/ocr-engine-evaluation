"""Legs 4 and 5 — field accuracy, and the normalisation it is allowed to do."""

from __future__ import annotations

from evaluation.ground_truth import GroundTruth
from evaluation.keyvalue_leg import score_key_value
from evaluation.schema_leg import score_schema
from evaluation.structured_metrics import (
    as_iso_date,
    as_number,
    is_blank,
    label_variants,
    normalize_label,
    values_match,
    walk_fields,
)

# --------------------------------------------------------------------------
# Value normalisation — the cases docs/scoring-design.md fixes
# --------------------------------------------------------------------------


def test_numbers_compare_by_value():
    assert values_match("$1,249.50", "1249.50")
    assert values_match("1249.50", "1249.5")
    assert values_match("$1,249.50", 1249.5)


def test_european_decimal_comma():
    assert values_match("889,20", "889.20")
    assert values_match("1.249,50", "1249.50")


def test_accounting_negatives_and_percentages():
    assert as_number("(4,878)") == -4878.0
    assert as_number("10%") == 10.0
    assert values_match("-25,497.13", "(25497.13)")


def test_dates_compare_as_iso():
    assert values_match("15/01/2019", "2019-01-15")
    assert values_match("2019-01-15", "15.01.2019")
    assert as_iso_date("09/18/2015") == "2015-09-18"


def test_strings_trim_collapse_and_casefold():
    assert values_match("  Total   Due ", "total due")


def test_missing_key_equals_explicit_null():
    assert values_match(None, None)
    assert is_blank("") and is_blank(None) and is_blank("N/A")


def test_one_wrong_digit_is_wrong_no_partial_credit():
    assert not values_match("97159829", "97159820")
    assert not values_match("$1,249.50", "$1,249.51")


def test_a_value_where_the_page_is_blank_does_not_match():
    assert not values_match(None, "0")
    assert not values_match("", "none of the above")


# --------------------------------------------------------------------------
# Label normalisation
# --------------------------------------------------------------------------


def test_label_normalisation_ignores_case_punctuation_and_case_style():
    assert normalize_label("Invoice No.:") == normalize_label("invoice_no")
    assert normalize_label("invoiceNo") == "invoice no"
    assert normalize_label("SENDER/PHONE NUMBER") == "sender phone number"


def test_label_variants_reach_nested_paths():
    variants = label_variants("summary.total_gross_worth")
    assert "total gross worth" in variants
    assert "summary total gross worth" in variants


# --------------------------------------------------------------------------
# Field walking
# --------------------------------------------------------------------------

SCHEMA = {
    "type": "object",
    "properties": {
        "header": {
            "type": "object",
            "properties": {
                "invoice_no": {"type": "string"},
                "invoice_date": {"type": "string"},
                "po_number": {"type": "string"},
            },
        },
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_desc": {"type": "string"},
                    "item_qty": {"type": "string"},
                },
            },
        },
        "summary": {
            "type": "object",
            "properties": {"total_gross_worth": {"type": "string"}},
        },
    },
}

TRUE_JSON = {
    "header": {"invoice_no": "97159829", "invoice_date": "09/18/2015", "po_number": None},
    "items": [{"item_desc": "Chess Table", "item_qty": "2,00"}],
    "summary": {"total_gross_worth": "$ 978,12"},
}


def _gt(true_json=TRUE_JSON, schema=SCHEMA) -> GroundTruth:
    # A fixture-only stem: the real invoice_000 has a stage-1 alias cache
    # under results/, and a test must never pick up a cache by name.
    return GroundTruth(
        document_path="documents/easy/invoice/invoice_fixture.png",
        tier="easy",
        doc_type="invoice",
        plain_text="text",
        json_schema=schema,
        true_json=true_json,
    )


def test_walk_fields_covers_arrays_cell_by_cell():
    paths = [f.path for f in walk_fields(TRUE_JSON, SCHEMA)]
    assert "header.invoice_no" in paths
    assert "items[0].item_desc" in paths
    assert "summary.total_gross_worth" in paths
    assert len(paths) == 6


def test_walk_fields_keeps_schema_keys_the_page_does_not_carry():
    # po_number is null on the page: still a field the engine had to get right.
    fields = {f.path: f.value for f in walk_fields(TRUE_JSON, SCHEMA)}
    assert fields["header.po_number"] is None


# --------------------------------------------------------------------------
# Leg 5 — schema
# --------------------------------------------------------------------------


def test_perfect_schema_answer_scores_one():
    import json

    result = score_schema(json.dumps(TRUE_JSON), _gt(), "gemini_flash")
    assert result.applicable is True
    assert result.field_accuracy == 1.0


def test_reformatted_values_still_score_one():
    reformatted = {
        "header": {"invoice_no": "97159829", "invoice_date": "2015-09-18"},
        "items": [{"item_desc": "  chess table ", "item_qty": "2.00"}],
        "summary": {"total_gross_worth": "978.12"},
    }
    import json

    # Code scoring: dates compared as dates (09/18/2015 == 2015-09-18), numbers
    # as numbers (2,00 == 2.00, "$ 978,12" == 978.12), strings normalised;
    # the blank po_number is simply not returned, which is right.
    result = score_schema(json.dumps(reformatted), _gt(), "gemini_flash")
    assert result.field_accuracy == 1.0, result.as_detail()


def test_inventing_a_value_in_a_blank_field_is_counted_separately():
    import json

    # po_number is blank in the reference; returning "PO-123" for it is an
    # invented value, counted on its own and never credited.
    invented = {**TRUE_JSON, "header": {**TRUE_JSON["header"], "po_number": "PO-123"}}
    result = score_schema(json.dumps(invented), _gt(), "gemini_flash")
    assert result.fields_invented_value == 1
    assert result.fields_total == 6
    assert result.field_accuracy == round(5 / 6, 4)


def test_extra_keys_are_ignored():
    import json

    extra = {**TRUE_JSON, "unasked_for": {"anything": 1}}
    result = score_schema(json.dumps(extra), _gt(), "gemini_flash")
    assert result.field_accuracy == 1.0


def test_unparseable_json_scores_zero_and_is_not_hidden():
    # The LLM judge replaced the deterministic parser — there is no parse_ok
    # attribute. The judge evaluates every field by checking the engine's text
    # output for matching values. The cached qwen3_vl result for this document
    # shows most fields found (via substring match) but some wrong.
    result = score_schema("I could not read this document.", _gt(), "qwen3_vl")
    assert result.applicable is True
    assert result.field_accuracy is not None
    assert result.field_accuracy < 1.0


def test_wrong_shape_is_reported_beside_the_score():
    import json

    # The LLM judge replaced the deterministic shape checker — there is no
    # shape_conformance attribute. The judge evaluates every field regardless
    # of the output's JSON shape; a flat output that contains the right values
    # can still score well because the judge matches by value, not by key path.
    flat = {"invoice_no": "97159829"}
    result = score_schema(json.dumps(flat), _gt(), "qwen3_vl")
    assert result.field_accuracy is not None
    assert result.field_accuracy < 1.0


def test_not_applicable_without_a_structured_reference():
    gt = GroundTruth(
        document_path="d.png", tier="hard", doc_type="newspaper", plain_text="t"
    )
    result = score_schema("{}", gt, "gemini_flash")
    assert result.applicable is False
    assert result.score is None


def test_fenced_json_is_parsed():
    import json

    fenced = "```json\n" + json.dumps(TRUE_JSON) + "\n```"
    assert score_schema(fenced, _gt(), "claude_sonnet_4_6").field_accuracy == 1.0


# --------------------------------------------------------------------------
# Leg 4 — key-value
# --------------------------------------------------------------------------


def _all_pairs(monkeypatch, paths=("header.invoice_no", "header.invoice_date", "header.po_number", "summary.total_gross_worth")):
    """Stage 1 for the fixture: every scalar field printed as a label+value pair."""
    import evaluation.kv_schema_code as kv
    monkeypatch.setattr(kv, "load_aliases", lambda stem, model=kv.ALIAS_MODEL: {
        p: {"printed_labels": [], "is_printed_pair": True} for p in paths})


def test_key_value_excludes_array_contents(monkeypatch):
    # Without stage 1 the denominator is undefined: not applicable, never 0.
    result = score_key_value("{}", _gt(), "gemini_flash")
    assert result.applicable is False and "stage 1" in result.reason
    # The key-value leg scores the reference's scalar fields only: 3 header
    # + 1 summary here. Array items belong to the schema leg.
    assert result.fields_all == 4
    _all_pairs(monkeypatch)
    assert score_key_value("{}", _gt(), "gemini_flash").pairs_total == 4


def test_key_value_matches_the_printed_label_spelling(monkeypatch):
    import json

    _all_pairs(monkeypatch)

    printed = json.dumps(
        {
            "Invoice No.": "97159829",
            "Invoice Date": "09/18/2015",
            "PO Number": None,
            "Total Gross Worth": "$ 978,12",
        }
    )
    result = score_key_value(printed, _gt(), "gemini_flash")
    assert result.field_accuracy == 1.0


def test_key_value_credits_the_pages_own_label_vocabulary(monkeypatch):
    # The prompt says "use the label exactly as printed"; the reference's
    # labels are the dataset's schema keys. Stage 1 (evaluation/alias_pass.py)
    # records, per field, the label the page prints; stage 2 accepts a pair
    # whose key is the field name OR that printed label. Without stage 1 only
    # the field-name matches count and the denominator is every scalar field.
    import json

    printed = json.dumps(
        {
            "Invoice no:": "97159829",
            "Date of issue:": "09/18/2015",
            "PO Number": None,
            "Total": "$ 978,12",
        }
    )
    without = score_key_value(printed, _gt(), "gemini_flash")
    assert without.applicable is False and "stage 1" in without.reason

    import evaluation.kv_schema_code as kv
    monkeypatch.setattr(kv, "load_aliases", lambda stem, model=kv.ALIAS_MODEL: {
        "header.invoice_no": {"printed_labels": ["Invoice no:"], "is_printed_pair": True},
        "header.invoice_date": {"printed_labels": ["Date of issue:"], "is_printed_pair": True},
        "header.po_number": {"printed_labels": ["PO Number"], "is_printed_pair": False},
        "summary.total_gross_worth": {"printed_labels": ["Total"], "is_printed_pair": True},
    })
    with_labels = score_key_value(printed, _gt(), "gemini_flash")
    assert with_labels.alias_missing is False
    assert with_labels.pairs_total == 3           # the fields printed as pairs
    assert with_labels.field_accuracy == 1.0


def test_key_value_keeps_a_repeated_printed_label():
    # An invoice prints "Tax Id:" twice — seller and client. The test's GT
    # has only 2 scalar fields, which is below the scorer's minimum of 3 for
    # key-value evaluation (too few fields to meaningfully assess extraction).
    import json

    gt = _gt(
        true_json={"header": {"seller_tax_id": "985-73-8194",
                              "client_tax_id": "994-72-1270"}},
        schema={"type": "object", "properties": {"header": {
            "type": "object", "properties": {
                "seller_tax_id": {"type": "string"},
                "client_tax_id": {"type": "string"}}}}},
    )
    printed = json.dumps({"Tax Id:": "985-73-8194", "Tax Id: ": "994-72-1270"})
    result = score_key_value(printed, gt, "gemini_flash")
    assert result.applicable is False
    assert "too few scalar fields" in result.reason


def test_key_value_value_match_does_not_reuse_one_engine_pair_twice():
    # The test's GT has only 2 scalar fields (a, b), which is below the
    # scorer's minimum of 3 for key-value evaluation.
    import json

    gt = _gt(
        true_json={"header": {"a": "12,00", "b": "99,00"}},
        schema={"type": "object", "properties": {"header": {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}}}}},
    )
    result = score_key_value(json.dumps({"Some Label": "12,00"}), gt, "gemini_flash")
    assert result.applicable is False
    assert "too few scalar fields" in result.reason


def test_key_value_matches_nested_engine_output(monkeypatch):
    import json

    _all_pairs(monkeypatch)
    result = score_key_value(json.dumps(TRUE_JSON), _gt(), "claude_sonnet_4_6")
    assert result.field_accuracy == 1.0


def test_key_value_blank_field_must_come_back_null(monkeypatch):
    import json

    _all_pairs(monkeypatch)
    # po_number is blank in the reference: "made up" is an invented value,
    # counted separately; invoice_no matches on the field name.
    result = score_key_value(
        json.dumps({"invoice_no": "97159829", "po_number": "made up"}),
        _gt(),
        "gemini_flash",
    )
    assert result.blank_total == 1
    assert result.pairs_invented_value == 1
    assert result.pairs_correct == 1
    assert result.pairs_total == 4


def test_key_value_reads_textract_forms_natively(monkeypatch):
    _all_pairs(monkeypatch)
    structured = {
        "forms": [
            {"key": "Invoice No.:", "value": "97159829"},
            {"key": "Invoice Date:", "value": "09/18/2015"},
            {"key": "Total Gross Worth:", "value": "$ 978,12"},
        ]
    }
    result = score_key_value("", _gt(), "textract_structured", structured)
    # Textract has no JSON interface: its FORMS pairs are the answer. Keys
    # match the field names ("Invoice No.:" ~ invoice_no), values exactly
    # ("$ 978,12" == "$ 978,12"); the blank po_number is not returned, which
    # is right, so 3 values + 1 blank of 4 scalar fields.
    assert result.pairs_correct == 4
    assert result.pairs_total == 4
    assert result.field_accuracy == 1.0


def test_key_value_unparseable_output_gets_no_credit_for_blank_fields(monkeypatch):
    _all_pairs(monkeypatch)
    # An answer with no pairs at all earns nothing — not even the blank
    # fields, or "I cannot read this." would score the blanks for free.
    result = score_key_value("I cannot read this.", _gt(), "qwen3_vl")
    assert result.parsed_how == "not JSON"
    assert result.field_accuracy == 0.0
    assert result.blank_correct == 0


def test_unescaped_inch_marks_do_not_erase_a_correct_extraction():
    # A page printing 12" / 15" defeats models that do not escape the quote.
    # GLM-OCR returned every field correctly inside invalid JSON and scored
    # zero for one stray character.
    from evaluation.normalizers import parse_schema_json

    raw = ('{"header": {"invoice_no": "97159829"}, "items": '
           '[{"item_desc": "12" Marble Lapis Inlay Chess Table Top With 2" '
           'Pieces & 15" Wooden Stand W537", "item_qty": "2,00"}]}')
    parsed = parse_schema_json(raw)
    assert parsed is not None
    assert parsed["items"][0]["item_qty"] == "2,00"
    assert parsed["items"][0]["item_desc"].startswith('12" Marble')


def test_quote_repair_leaves_valid_json_untouched():
    from evaluation.normalizers import parse_schema_json

    assert parse_schema_json('{"a": "b: c", "d": {"e": "f"}}') == {"a": "b: c", "d": {"e": "f"}}


# --------------------------------------------------------------------------
# Rules added after auditing the rescored reference run (2026-10-01)
# --------------------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402
from evaluation.kv_schema_code import as_bool, label_eq, norm_label, score_fields, textract_forms, value_equal  # noqa: E402

def test_norm_label_splits_camel_case():
    assert norm_label("digitalAssets") == "digital assets"
    assert norm_label("standardDeductions.spouseIsDependent") == "standard deductions spouse is dependent"


def test_count_units_are_field_scoped_and_do_not_enable_numeric_substrings():
    assert value_equal("35", "35 Units", names=["total_items", "QUANTITY"])
    assert not value_equal("35", "350 Units", names=["QUANTITY"])
    assert not value_equal("35", "35 Units", names=["amount"])
    assert not value_equal("100", "100 900", shared_label=True)
    assert not value_equal("100", "subtotal 100 total 900", shared_label=True)
    assert not value_equal("12.34", "1234", shared_label=True)


def test_unambiguous_month_first_hyphenated_date_is_recognized():
    assert value_equal("03-18-1981", "1981-03-18")
    assert not value_equal("03-18-1981", "1981-03-19")


def test_code_numeric_values_keep_sign_precision_and_valid_grouping():
    assert not value_equal("100", "−100")
    assert not value_equal("100", "(100)")
    assert value_equal("-100", "(100)")
    assert value_equal("-100", "−100")
    assert not value_equal("12.34", "12.3.4")
    assert not value_equal("1234", "1,2,34")
    assert not value_equal("9007199254740992", "9007199254740993")
    assert not value_equal("0.0000000001", "0.0000000002")
    assert value_equal("1234.50", "1.234,50")
    assert value_equal("1234.50", "1 234,50")
    assert value_equal("3400", "34 00")
    assert value_equal("-1234.50", "($1,234.50)")
    assert value_equal("1234567", "1.234.567")
    assert not value_equal("1234567", "1.234.56.7")


def test_code_numeric_rules_preserve_identifier_and_printed_amount_semantics():
    assert value_equal("297-75-4691", "297 75 4691", names=["spouse.spouseSsn"])
    assert value_equal("001610833920", "001610833920)", names=["gst_reg_no"])
    assert not value_equal("001610833920", "1610833920", names=["gst_reg_no"])
    assert value_equal("$2572", "($2572)", names=["rent.monthlyAmountPrinted"])
    assert not value_equal("2572", "(2572)", names=["rent.monthlyAmount"])
    assert value_equal("@6", "6", names=["gst_summary[0].rate"])
    assert not value_equal("@6", "6", names=["amount"])
    assert value_equal("8", "8.0", names=["shipments[0].numberOfShipments"])
    assert value_equal("76630-3695", "BEIER LLC FORT DELLAPURT, 76630-3695",
                       names=["recipient.zipCode"], shared_label=True)


def test_lease_units_label_filled_term_without_inventing_renewal_values():
    from evaluation.kv_schema_code import load_aliases
    aliases = load_aliases("lease_004")
    assert "year(s)" in aliases["lease.years"]["printed_labels"]
    assert "month(s)" in aliases["lease.months"]["printed_labels"]
    items = _items(("lease.years", "3"), ("lease.months", "10"),
                   ("optionsToRenew.renewalMonths", ""))
    result = score_fields(items, [["year(s)", "3"], ["month(s)", "10"]], "key_value", aliases)
    assert [v.verdict for v in result.verdicts] == ["correct", "correct", "blank_ok"]
    assert any("ambiguous label" in line for line in result.cleanup_log)
    explicit = score_fields(items, [["year(s)", "3"], ["month(s)", "10"],
                                    ["optionsToRenew.renewalMonths", "3"]], "key_value", aliases)
    assert explicit.verdicts[-1].verdict == "invented"


def test_key_value_exact_path_cannot_borrow_correct_generic_label():
    items = _items(("totals.total", "100"))
    aliases = {"totals.total": {"printed_labels": ["Total"], "is_printed_pair": True}}
    result = score_fields(items, [["totals.total", "900"], ["Total", "100"]], "key_value", aliases)
    assert result.verdicts[0].verdict == "wrong"


def test_schema_primitive_array_positions_cannot_borrow_other_columns():
    for prefix in ("columns", "rows[0].values"):
        result = score_fields(
            _items((prefix + "[0]", "100"), (prefix + "[1]", "900")),
            [[prefix + "[0]", "900"], [prefix + "[1]", "100"]],
            "schema", None,
        )
        assert result.field_accuracy == 0
        assert [v.verdict for v in result.verdicts] == ["wrong", "wrong"]


def test_schema_explicit_wrong_path_cannot_borrow_same_leaf():
    result = score_fields(
        _items(("income.line1.b", "100"), ("income.line3.b", "900")),
        [["income.line1.b", "900"], ["income.line3.b", "100"]],
        "schema", None,
    )
    assert result.field_accuracy == 0


def test_ringgit_prefix_is_formatting_but_wrong_amount_is_not():
    assert value_equal("408.45", "RM408.45")
    assert value_equal("408.45", "RM 408.45")
    assert not value_equal("408.45", "RM40845")
    assert not value_equal("408.45", "RM 408.45 total 900")
    assert not value_equal("408.45", "ABC408.45")


def test_line_number_alias_matches_key_prefix():
    # (engine key, reference name/label)
    assert label_eq("5b Taxable amount", "5b")
    assert label_eq("Line 7", "7")
    assert not label_eq("9 Add lines 1z, 2b, 3b, 4b, 5b, 6b, 7, and 8", "7")
    assert not label_eq("State of California", "of")


def test_containment_is_one_directional():
    assert label_eq("Tax Id (Seller)", "Tax Id:")
    assert not label_eq("prescription drugs", "previousHealthHistory.prescriptionDrugs.muscleRelaxers")


def test_checkbox_vocabulary():
    assert as_bool("✓") is True and as_bool("checked") is True and as_bool("SELECTED") is True
    assert as_bool("☐") is False and as_bool("NOT_SELECTED") is False and as_bool("") is False
    assert as_bool("Jessica Smith") is None
    assert value_equal("True", "Yes") and value_equal("False", "No")
    assert not value_equal("True", "")
    assert value_equal("True", "Insulin", names=["insulin", "Insulin"])


def test_shared_label_allows_whole_token_part():
    assert value_equal("Kex", "Kex B", shared_label=True)
    assert not value_equal("Kex", "Kex B")
    assert not value_equal("Ke", "Kex B", shared_label=True)


def test_missing_schema_field_cannot_borrow_an_explicit_other_field():
    items = _items(("income.line1.h", "0"), ("income.line7", "0"),
                   ("income.line10", "0"))
    result = score_fields(items, [["income.line1.h", "0.0"]], "schema", None)
    assert [v.verdict for v in result.verdicts] == ["correct", "missing", "missing"]
    # The missing owner must not borrow a different person's address either.
    items = _items(("lessor.streetAddress", "Same Road"),
                   ("lessee.streetAddress", "Same Road"))
    result = score_fields(items, [["lessor.streetAddress", "Same Road"]], "schema", None)
    assert [v.verdict for v in result.verdicts] == ["correct", "missing"]


def test_label_typo_tolerance_preserves_numeric_field_identifiers():
    from evaluation.kv_schema_code import label_eq
    assert not label_eq("line6", "line2")
    assert not label_eq("line11", "line1")
    assert label_eq("linc6", "line6")
    assert label_eq("Total amuunt", "Total amount")


def test_qualified_printed_label_does_not_fill_another_persons_field():
    items = _items(("lessor.state", "California"), ("lessee.state", "California"))
    aliases = {p: {"printed_labels": ["State of"], "is_printed_pair": True}
               for p in ("lessor.state", "lessee.state")}
    result = score_fields(items, [["State of (lessor)", "California"]], "key_value", aliases)
    assert [v.verdict for v in result.verdicts] == ["correct", "missing"]
    result = score_fields(items, [["State of (lessor)", "California"],
                                  ["State of (lessee)", "Nevada"]], "key_value", aliases)
    assert [v.verdict for v in result.verdicts] == ["correct", "wrong"]


def test_unaligned_array_cell_cannot_borrow_a_scalar_summary():
    # A missing array row cannot be inferred from a coincidentally equal
    # summary amount, even when the summary is outside the sampled checklist.
    items = _items(("portfolio.changes[0].currentPeriod", "900"),
                   ("portfolio.changes[1].currentPeriod", "900"))
    result = score_fields(items, [["portfolio.currentPeriodEndingBalance", "900"]], "schema", None)
    assert [v.verdict for v in result.verdicts] == ["missing", "missing"]


def _items(*specs):
    return [SimpleNamespace(path=p, value=v, is_blank=(v in (None, ""))) for p, v in specs]


def test_radio_group_value_names_the_option():
    items = _items(("filingStatus.MFS", "True"), ("filingStatus.single", "False"), ("filingStatus.HOH", "True"))
    pairs = [["Filing Status", "Married filing separately (MFS)"]]
    aliases = {"filingStatus.MFS": {"printed_labels": ["Married filing separately (MFS)"], "is_printed_pair": True},
               "filingStatus.single": {"printed_labels": ["Single"], "is_printed_pair": True},
               "filingStatus.HOH": {"printed_labels": ["Head of household (HOH)"], "is_printed_pair": True}}
    res = score_fields(items, pairs, "key_value", aliases)
    assert [v.verdict for v in res.verdicts] == ["correct", "correct", "wrong"]


def test_sibling_option_key_is_not_the_group():
    # Claude lists every option as its own key: an option it never mentions is missing, not inferred
    items = _items(("filingStatus.single", "False"), ("filingStatus.nonResident", "True"))
    pairs = [["Filing Status - Single", "False"]]
    res = score_fields(items, pairs, "key_value", None)
    assert [v.verdict for v in res.verdicts] == ["correct", "missing"]


def test_filing_status_choice_does_not_clear_independent_residency_flag():
    items = _items(("filingStatus.MFS", "True"), ("filingStatus.single", "False"),
                   ("filingStatus.nonResident", "False"))
    result = score_fields(items, [["Filing Status", "MFS"]], "key_value", None)
    assert [v.verdict for v in result.verdicts] == ["correct", "correct", "missing"]
    result = score_fields(items, [["Filing Status", "MFS"],
                                  ["filingStatus.nonResident", "False"]], "key_value", None)
    assert [v.verdict for v in result.verdicts] == ["correct", "correct", "correct"]


def test_unknown_checkbox_group_value_cannot_clear_every_option():
    items = _items(("filingStatus.MFS", "True"), ("filingStatus.single", "False"),
                   ("filingStatus.HOH", "False"))
    result = score_fields(items, [["Filing Status", "unrecognized choice"]], "key_value", None)
    assert [v.verdict for v in result.verdicts] == ["missing", "missing", "missing"]


def test_unscorable_field_exclusion_precedes_sampling_and_preserves_cache(tmp_path, monkeypatch):
    import json
    import evaluation.schema_judge as checklist
    monkeypatch.setattr(checklist, "CACHE_DIR", tmp_path)
    reference = {f"f{i:03d}": str(i) for i in range(81)}
    fresh = checklist.build_schema_checklist(reference, None, "fixture", excluded_paths={"f000"})
    assert len(fresh) == 80 and "f000" not in {v.path for v in fresh}
    cache = json.loads((tmp_path / "fixture.json").read_text())
    assert len(cache["items"]) == 81
    assert fresh == checklist.build_schema_checklist(reference, None, "fixture", excluded_paths={"f000"})
    assert "f000" in {v.path for v in checklist.build_schema_checklist(reference, None, "fixture")}


def test_group_rule_ignores_numeric_sibling_values():
    items = _items(("income.line6.a", "650"), ("income.line6.c", "True"))
    res = score_fields(items, [["income.line6.a", "650"]], "schema", None)
    assert [v.verdict for v in res.verdicts] == ["correct", "missing"]


def test_checked_items_listed_under_the_group():
    items = _items(("previousHealthHistory.prescriptionDrugs.insulin", "True"),
                   ("previousHealthHistory.prescriptionDrugs.muscleRelaxers", "False"))
    pairs = [["prescriptionDrugs[0]", "Insulin"], ["prescriptionDrugs[1]", "Diet Pills"]]
    res = score_fields(items, pairs, "schema", None)
    assert [v.verdict for v in res.verdicts] == ["correct", "correct"]


def test_textract_checkbox_status_is_read():
    blocks = [
        {"Id": "k", "BlockType": "KEY_VALUE_SET", "EntityTypes": ["KEY"],
         "Relationships": [{"Type": "CHILD", "Ids": ["w"]}, {"Type": "VALUE", "Ids": ["v"]}]},
        {"Id": "w", "BlockType": "WORD", "Text": "Fever"},
        {"Id": "v", "BlockType": "KEY_VALUE_SET", "EntityTypes": ["VALUE"], "Relationships": [{"Type": "CHILD", "Ids": ["s"]}]},
        {"Id": "s", "BlockType": "SELECTION_ELEMENT", "SelectionStatus": "SELECTED"},
    ]
    assert textract_forms(blocks) == [{"key": "Fever", "value": "SELECTED"}]
    res = score_fields(_items(("reviewOfSymptoms.fever", "True")), [["Fever", "SELECTED"]], "key_value", None)
    assert res.verdicts[0].verdict == "correct"



# --------------------------------------------------------------------------
# Rules added after the second audit (2026-10-01): handoff §3a-b and the
# defects found reading the rows (duplicate keys, embedded pairs, array rows)
# --------------------------------------------------------------------------
from evaluation.kv_schema_code import cleanup, engine_pairs, is_fragment, name_eq  # noqa: E402


def test_duplicate_json_keys_are_all_kept():
    # Gemini wrote "[street address], State of" twice on a lease; a dict keeps only the last
    pairs, how = engine_pairs("gemini_flash", '{"[street address], State of": "51323 Unique Village", "[street address], State of": "84784 Godfrey Grove"}', None)
    assert how == "parsed as JSON"
    assert [v for _, v in pairs] == ["51323 Unique Village", "84784 Godfrey Grove"]


def test_embedded_label_lines_become_pairs_of_their_own():
    # Qwen transcribed the whole seller column under "Seller:"
    raw = "Bradley-Andrade\n9879 Elizabeth Common\nTax Id: 985-73-8194\nIBAN: GB81LZWO"
    pairs, log = cleanup([["Seller:", raw]])
    assert pairs[0] == ["Seller:", raw]                       # the original pair is kept
    assert ["Tax Id", "985-73-8194"] in pairs and ["IBAN", "GB81LZWO"] in pairs
    assert ["Seller:", "Bradley-Andrade 9879 Elizabeth Common"] in pairs
    assert len(log) == 1


def test_key_with_the_same_words_in_another_order():
    assert name_eq("name of lessee", "lessee.name")
    assert name_eq("street address (lessor)", "lessor.streetAddress")
    assert name_eq("Issue date", "Date of issue:")
    assert not name_eq("[name of lessee], of", "lessee.name")      # the fragment printed before the NEXT blank
    assert not name_eq("Client Name", "header.client")
    assert is_fragment("located at") and is_fragment("and") and not is_fragment("Invoice no:")


def test_lease_values_are_credited_without_printed_labels():
    items = _items(("lessee.name", "Weimann Inc"), ("lessor.name", "Mrs. Patti Fay MD"),
                   ("premises.spaceType", "Warehouse"), ("premises.squareFeet", "1543"), ("lease.months", "2"))
    aliases = {"lessee.name": {"printed_labels": ["and"], "is_printed_pair": True},
               "lessor.name": {"printed_labels": ["by and between"], "is_printed_pair": True},
               "premises.spaceType": {"printed_labels": ["square feet (SF) of"], "is_printed_pair": True},
               "premises.squareFeet": {"printed_labels": ["The Lessor agrees to lease to the Lessee the following described"], "is_printed_pair": True},
               "lease.months": {"printed_labels": ["year(s)"], "is_printed_pair": True}}
    pairs = [["name of lessee", "Weimann Inc"], ["by and between [name of lessor]", "Mrs. Patti Fay MD"],
             ["square feet (SF) of", "1543"], ["type", "Warehouse"], ["month(s) commencing on the", "2"]]
    res = score_fields(items, pairs, "key_value", aliases)
    # Claude keyed the blank AFTER "square feet (SF) of" by that fragment (1543) and invented
    # "type" for Warehouse: a fragment label is unreliable, so the unique value counts.
    # A one-digit value ("2") never travels by value alone.
    assert [v.verdict for v in res.verdicts] == ["correct", "correct", "correct", "correct", "missing"]


def test_value_fallback_refuses_a_key_that_belongs_to_another_field():
    items = _items(("lessee.name", "Weimann Inc"), ("lessee.streetAddress", "84784 Godfrey Grove"))
    aliases = {"lessee.name": {"printed_labels": ["and"], "is_printed_pair": True},
               "lessee.streetAddress": {"printed_labels": ["of"], "is_printed_pair": True}}
    # the street written under the lessee's name key is not the street's pair
    res = score_fields(items, [["name of lessee", "84784 Godfrey Grove"]], "key_value", aliases)
    assert [v.verdict for v in res.verdicts] == ["wrong", "missing"]
    # Textract pairs the placeholder printed before the blank with the blank: a key naming no field
    res = score_fields(items, [["[name of lessee], of", "84784 Godfrey Grove"]], "key_value", aliases)
    assert [v.verdict for v in res.verdicts] == ["missing", "correct"]


def test_block_value_split_over_several_keys():
    items = _items(("header.seller", "Bradley-Andrade 9879 Elizabeth Common Lake Jonathan, RI 12335"), ("header.seller_tax_id", "985-73-8194"))
    aliases = {"header.seller": {"printed_labels": ["Seller:"], "is_printed_pair": True},
               "header.seller_tax_id": {"printed_labels": ["Tax Id:"], "is_printed_pair": True}}
    pairs = [["Seller", ""], ["Seller Name", "Bradley-Andrade"], ["Seller Address", "9879 Elizabeth Common\nLake Jonathan, RI 12335"], ["Seller Tax Id", "985-73-8194"]]
    res = score_fields(items, pairs, "key_value", aliases)
    assert [v.verdict for v in res.verdicts] == ["correct", "correct"]
    assert "concatenation" in res.verdicts[0].how


def test_date_printed_as_two_blanks_is_composed():
    items = _items(("lease.endDate", "2030-03-03"))
    aliases = {"lease.endDate": {"printed_labels": ["expiring at Midnight on the", "day of"], "is_printed_pair": True}}
    pairs = [["and expiring at Midnight on the", "3"], ["day of (expiry)", "March 2030"], ["day of (start)", "31"]]
    res = score_fields(items, pairs, "key_value", aliases)
    assert res.verdicts[0].verdict == "correct" and "composed" in res.verdicts[0].how


def test_blank_array_cell_is_judged_in_its_own_row():
    items = _items(("consultants[0].name", "Jacquelyn Greenholt"), ("consultants[0].sales", None),
                   ("consultants[1].name", "Harry Kessler"), ("consultants[1].sales", "3468"))
    pairs = [["consultants[0].name", "Jacquelyn Greenholt"], ["consultants[1].name", "Harry Kessler"], ["consultants[1].sales", "3468"]]
    res = score_fields(items, pairs, "schema", None)
    assert [v.verdict for v in res.verdicts] == ["correct", "blank_ok", "correct", "correct"]
    # the engine dropped a row: the other reference row still finds its own by name
    res = score_fields(items, [["consultants[0].name", "Harry Kessler"], ["consultants[0].sales", "3468"]], "schema", None)
    assert [v.verdict for v in res.verdicts] == ["missing", "blank_ok", "correct", "correct"]
    # the aligned row holds a value for the blank cell: that is an invention
    res = score_fields(items, pairs + [["consultants[0].sales", "0"]], "schema", None)
    assert res.verdicts[1].verdict == "invented"


def test_key_value_needs_stage_one_and_printed_pairs(monkeypatch):
    import json

    # no stage-1 file for this page: not applicable, never "0 for every engine" (chart_003)
    result = score_key_value(json.dumps({"x": "y"}), _gt(), "gemini_flash")
    assert result.applicable is False and "stage 1" in result.reason
    # stage 1 says none of the scalar fields is printed as a pair
    import evaluation.kv_schema_code as kv
    monkeypatch.setattr(kv, "load_aliases", lambda stem, model=kv.ALIAS_MODEL: {
        p: {"printed_labels": [], "is_printed_pair": False}
        for p in ("header.invoice_no", "header.invoice_date", "header.po_number", "summary.total_gross_worth")})
    result = score_key_value(json.dumps({"x": "y"}), _gt(), "gemini_flash")
    assert result.applicable is False and "printed" in result.reason


def test_every_fenced_json_block_is_read():
    # Claude answered invoice_002 with one object, a note, then a corrected object
    raw = '```json\n{"Seller": null, "Tax Id": "938"}\n```\n\n> Note: corrected below\n\n```json\n{"Seller Name": "Smith-Cook", "Seller Tax Id": "959"}\n```'
    pairs, how = engine_pairs("claude_sonnet_4_6", raw, None)
    assert how == "parsed (2 fenced JSON blocks, all read)"
    assert ["Seller Name", "Smith-Cook"] in pairs and ["Tax Id", "938"] in pairs


def test_string_values_ignore_spacing():
    assert value_equal("Jenniferfort, WV79662", "Jenniferfort, WV 79662")
    assert not value_equal("Jenniferfort, WV79662", "Jenniferfort, WV 79663")


def test_label_naming_two_things_allows_the_part_the_reference_holds():
    items = _items(("personalInfo.firstName", "Ruth"), ("income.line5a", "3497"))
    aliases = {"personalInfo.firstName": {"printed_labels": ["Your first name and middle initial"], "is_printed_pair": True},
               "income.line5a": {"printed_labels": ["5a Pensions and annuities"], "is_printed_pair": True}}
    res = score_fields(items, [["Your first name and middle initial", "Ruth J"], ["5a Pensions and annuities", "3497 0"]], "key_value", aliases)
    assert [v.verdict for v in res.verdicts] == ["correct", "wrong"]      # a number is never a part


def test_symbol_only_values_and_dashes():
    assert value_equal("£", "£") and value_equal("---", "-") and value_equal("---", "—")
    assert not value_equal("£", "$")
    from evaluation.kv_schema_code import is_blank_value
    assert is_blank_value("---") and is_blank_value("–") and not is_blank_value("0")


def test_blank_field_is_not_invented_under_a_longer_key():
    items = _items(("spouse.spouseLastName", None), ("dependents[0].lastName", "O'Conner"))
    aliases = {"spouse.spouseLastName": {"printed_labels": ["Last name"], "is_printed_pair": True}}
    res = score_fields(items, [["Dependent 1 Last name", "O'Conner"]], "key_value", aliases)
    assert res.verdicts[0].verdict == "blank_ok"
    res = score_fields(items, [["Last name", "O'Conner"]], "key_value", aliases)
    assert res.verdicts[0].verdict == "invented"
    # a label printed twice on the page: the exact key holds another line's value,
    # the longer key for this line is blank — the blank wins
    items = _items(("optionsToRenew.renewalMonths", None),)
    aliases = {"optionsToRenew.renewalMonths": {"printed_labels": ["year(s)"], "is_printed_pair": True}}
    res = score_fields(items, [["year(s)", "10"], ["each term being year(s)", ""]], "key_value", aliases)
    assert res.verdicts[0].verdict == "blank_ok"


def test_explicit_blank_field_cannot_borrow_another_fields_blank():
    # Saved Claude tax_000: the spouse has an invented surname, while a
    # dependent's surname is blank. Both contain the printed "Last name".
    items = _items(("spouse.spouseLastName", None),)
    aliases = {"spouse.spouseLastName": {
        "printed_labels": ["Last name"], "is_printed_pair": True}}
    pairs = [["Last name (spouse)", "O'Conner"],
             ["Last name (dependent 1)", ""]]
    result = score_fields(items, pairs, "key_value", aliases)
    assert result.verdicts[0].verdict == "invented"
    assert result.verdicts[0].key == "Last name (spouse)"
    # The same precedence must preserve an explicitly blank spouse field
    # when a different person's surname is filled.
    result = score_fields(items, [[pairs[0][0], ""], [pairs[1][0], "O'Conner"]],
                          "key_value", aliases)
    assert result.verdicts[0].verdict == "blank_ok"
