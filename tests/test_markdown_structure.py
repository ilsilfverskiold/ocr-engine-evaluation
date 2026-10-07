"""Formatting and content are separate, and output blocks cannot be reused."""
import pytest

from evaluation.markdown_structure import blocks, match_items
from evaluation.code_judges import score_markdown_code


@pytest.mark.parametrize("text", ["# Totals", "Totals\n======", "<h2>Totals</h2>"])
def test_real_heading_formats_are_accepted(text):
    assert score_markdown_code(text, [{"type": "heading", "content": "Totals"}]).element_accuracy == 1


def test_bold_is_paragraph_and_never_half_credit_for_wrong_heading_type():
    result = score_markdown_code("**Totals**", [{"type": "heading", "content": "Totals"}])
    assert result.element_accuracy == 0 and result.wrong_type == 1


def test_one_output_block_cannot_satisfy_two_identical_reference_blocks():
    items = [{"type": "paragraph", "content": "The repeated paragraph."}] * 2
    result = score_markdown_code("The repeated paragraph.", items)
    assert result.correct == 1 and result.absent == 1


def test_split_paragraph_run_cannot_be_reused():
    items = [{"type": "paragraph", "content": "Seller Acme Company Street Address Town"},
             {"type": "paragraph", "content": "Seller Acme Company Street Address Town"}]
    matches, _ = match_items("Seller Acme Company\n\nStreet Address Town", items)
    assert sum(m.verdict == "correct" for m in matches) == 1


def test_fake_table_syntax_is_not_structural_table_output():
    assert not any(kind == "table_row" for kind, _, _ in blocks("| A | B |\n| 1 | 2 |"))
    assert [kind for kind, _, _ in blocks("| A | B |\n| --- | --- |\n| 1 | 2 |")] == ["table_header", "table_row"]


def test_code_sample_cannot_create_headings():
    assert blocks("```python\n# Totals\n```")[0][0] == "code"


def test_nested_list_text_is_not_duplicated_into_parent():
    assert [(k, t) for k, t, _ in blocks("<ul><li>Parent<ul><li>Child</li></ul></li></ul>")] == [
        ("list_item", "Parent"), ("list_item", "Child")]


def test_markdown_cer_wer_use_actual_markdown_answer(monkeypatch):
    from evaluation.ground_truth import GroundTruth
    from evaluation.markdown_leg import score_markdown
    import evaluation.code_judges as code
    monkeypatch.setattr(code, "load_markdown_checklist", lambda _: [{"type": "heading", "content": "Total 100"}])
    gt = GroundTruth("documents/easy/invoice/x.png", "easy", "invoice", "Total 100", markdown="# Total 100")
    exact = score_markdown("**Total 100**", gt)
    wrong = score_markdown("# Total 1000", gt)
    assert exact.cer == exact.wer == 0 and exact.score == 0
    assert wrong.cer > 0 and wrong.wer > 0


def test_distinct_fields_share_a_paragraph_without_competing():
    fields = ["Tax Id: 959-84-2124", "IBAN: GB20BAKH22085364527355"]
    matches, _ = match_items("\n".join(fields), [{"type": "key_value", "content": v} for v in fields])
    assert [m.verdict for m in matches] == ["correct", "correct"]
    a, b = [m.evidence[0] for m in matches]
    assert a["token_end"] <= b["token_start"]


def test_receipt_header_and_payment_fields_in_shared_paragraphs():
    fields = ["Receipt #: CS00082258", "Date: 09/02/2018", "Cashier: USER", "Time: 08:32:00",
              "Sub Total (RM): 411.50", "Discount (RM): 0.00", "Total GST (RM): 24.69"]
    matches, _ = match_items("\n".join(fields), [{"type": "key_value", "content": v} for v in fields])
    assert all(m.verdict == "correct" for m in matches)


def test_embedded_heading_is_present_but_wrong_type():
    text = "Form 1040 Department of the Treasury **U.S. Individual Income Tax Return** 2024 OMB No. 1545-0074"
    m, _ = match_items(text, [{"type": "heading", "content": "U.S. Individual Income Tax Return"}])
    assert m[0].verdict == "wrong_type" and m[0].similarity == 1


@pytest.mark.parametrize("text", [
    "| Description | Qty |\n|---|---|\n| Pen | 3 |",
    "<table><thead><tr><th>Description</th><th>Qty</th></tr></thead><tr><td>Pen</td><td>3</td></tr></table>",
])
def test_actual_table_headers_and_legacy_generic_rows(text):
    for kind in ("table_header", "table_row"):
        m, _ = match_items(text, [{"type": kind, "content": "Description | Qty"}])
        assert m[0].verdict == "correct"


def test_data_row_with_row_header_is_not_a_column_header():
    text = "<table><tr><th>Pen</th><td>3</td></tr></table>"
    m, _ = match_items(text, [{"type": "table_header", "content": "Pen | 3"}])
    assert m[0].verdict == "wrong_type"


def test_label_value_table_and_two_fields_in_one_cell():
    text = "| Field | Value |\n|---|---|\n| Qty: 1 Total Exclude GST: | 193.00 |"
    m, _ = match_items(text, [{"type": "key_value", "content": v}
                            for v in ("Qty: 1", "Total Exclude GST: 193.00")])
    assert [x.verdict for x in m] == ["correct", "correct"]
    owned = [(e['block'], n) for x in m for e in x.evidence for n in range(e['token_start'], e['token_end'])]
    assert len(owned) == len(set(owned))


def test_does_not_borrow_a_value_from_another_table_row():
    text = "| Label | Value |\n|---|---|\n| Total | 900 |\n| Subtotal | 100 |"
    m, _ = match_items(text, [{"type": "key_value", "content": "Total: 100"}])
    assert m[0].verdict != "correct"


@pytest.mark.parametrize("returned", ["1000", "-100", "10.0", "1,00"])
def test_changed_number_cannot_get_full_credit(returned):
    m, _ = match_items(f"Total: {returned}", [{"type": "key_value", "content": "Total: 100"}])
    assert m[0].verdict != "correct"


def test_numbered_legacy_table_reference_does_not_lose_first_cell():
    text = "| No. | Item |\n|---|---|\n| 1. | Pen |"
    m, _ = match_items(text, [{"type": "table_row", "content": "1. | Pen", "content_format": "markdown"}])
    assert m[0].verdict == "correct"


def test_literal_asterisks_are_not_reference_emphasis():
    text = r"\*\*\*\*Thank You. Please Come Again.\*\*\*\*"
    m, _ = match_items(text, [{"type": "paragraph", "content": "****Thank You. Please Come Again.****"}])
    assert m[0].verdict == "correct"


def test_content_match_precedes_type_preference():
    text = "An unrelated paragraph about annual tax.\n\n| Name | Value |\n|---|---|\n| Annual tax | 250 |"
    m, _ = match_items(text, [{"type": "heading", "content": "Annual tax"}])
    assert m[0].verdict == "wrong_type" and m[0].similarity == 1


def test_no_credit_for_accidental_character_resemblance():
    m, _ = match_items("separate instructions", [{"type": "paragraph", "content": "IRA distributions"}])
    assert m[0].verdict == "absent"


def test_table_cell_reordering_or_missing_description_is_not_correct():
    for row in ("Pen | 20 | 2", "Pencil | 2 | 20", " | 2 | 20"):
        m, _ = match_items("| Item | Qty | Price |\n|---|---|---|\n| " + row + " |",
                           [{"type": "table_row", "content": "Pen | 2 | 20"}])
        assert m[0].verdict != "correct"


def test_split_item_rows_are_not_silently_reconstructed():
    text = "| Item | Qty | Price |\n|---|---|---|\n| 123 | 2 | 20 |\n| Blue pen | | |"
    m, _ = match_items(text, [{"type": "table_row", "content": "123 Blue pen | 2 | 20"}])
    assert m[0].verdict == "partial"
    assert "ordered-cell similarity" in m[0].how


def test_two_occurrences_can_satisfy_two_references():
    m, _ = match_items("Alpha beta. Alpha beta.", [{"type": "paragraph", "content": "Alpha beta"}] * 2)
    assert [x.verdict for x in m] == ["correct", "correct"]


def test_empty_padding_column_is_not_a_missing_markdown_header():
    text = "| | VAT | Amount |\n|---|---|---|\n| | 10% | 100 |"
    m, _ = match_items(text, [{"type": "table_header", "content": "VAT | Amount"},
                             {"type": "table_row", "content": "10% | 100"}])
    assert all(x.verdict == "correct" for x in m)


def test_invoice_fields_together_but_total_paragraphs_are_not_a_table_row():
    text = "Invoice no: 97159829\nDate of issue: 09/18/2015\n\nTotal: $ 889,20\n\nVAT: $ 88,92\n\nGross worth: $ 978,12"
    m, _ = match_items(text, [
        {"type": "key_value", "content": "Invoice no: 97159829"},
        {"type": "key_value", "content": "Date of issue: 09/18/2015"},
        {"type": "table_row", "content": "Total | $ 889,20 | $ 88,92 | $ 978,12"}])
    assert [x.verdict for x in m] == ["correct", "correct", "wrong_type"]


@pytest.mark.parametrize("text", ["# RECEIPT C13", "- RECEIPT C13", "RECEIPT C13"])
def test_key_value_is_semantic_and_can_use_valid_heading_or_list(text):
    m, _ = match_items(text, [{"type": "key_value", "content": "RECEIPT C13"}])
    assert m[0].verdict == "correct"


def test_parking_receipt_fields_in_header_and_body_table_rows():
    text = "| ENTRY TIME: | 23.03.18 14:59 |\n|---|---|\n| EXIT TIME: | 23.03.18 15:27 |\n| KIND OF PAYMENT: | CASH |"
    m, _ = match_items(text, [{"type": "key_value", "content": s} for s in
                            ("ENTRY TIME: 23.03.18 14:59", "EXIT TIME: 23.03.18 15:27", "KIND OF PAYMENT: CASH")])
    assert all(x.verdict == "correct" for x in m)


def test_catalogue_spacing_error_does_not_hide_gst_number():
    reference = [{"type": "key_value", "content": "GSTNo.000296652800"}]
    for text in ("GST No. 000296652800", "Company\nAddress\nGST No. 000296652800\nTEL No: 1300881698"):
        m, _ = match_items(text, reference)
        assert m[0].verdict == "correct"
    m, _ = match_items("GST No. 000296652801", reference)
    assert m[0].verdict != "correct"


def test_long_alphanumeric_identifier_is_compared_without_backtracking():
    identifier = "A" * 200 + "1" * 200
    m, _ = match_items("ID: " + identifier, [{"type": "key_value", "content": "ID: " + identifier}])
    assert m[0].verdict == "correct"


def test_present_text_in_oversized_heading_is_partial_not_absent():
    m, _ = match_items("# TAX INVOICE CELERY STICKS CUT 300G", [{"type": "heading", "content": "TAX INVOICE"}])
    assert m[0].verdict == "partial" and m[0].similarity == 1


def test_item_in_combined_list_entry_is_partial_not_absent():
    text = "- Options to Renew: Lessee may not renew the Lease. The Lessee may have the right to renew with a total of four renewal periods."
    m, _ = match_items(text, [{"type": "list_item", "content": "Lessee may not renew the Lease."}])
    assert m[0].verdict == "partial"


def test_present_reference_cells_in_wider_row_are_partial_not_absent():
    text = "| Name | ID | Code | Gross | Tax | Net |\n|---|---|---|---|---|---|\n| Homer Goyette | 1200040391 | 715373 | R1140.00 | R21.00 | R1119.00 |"
    m, _ = match_items(text, [{"type": "table_row", "content": "Homer Goyette | 1200040391 | 715373"}])
    assert m[0].verdict == "partial" and m[0].similarity == 1


def test_single_record_table_preserves_column_label_value_relationships():
    text = "| First name | Tax ID |\n|---|---|\n| Ruth | 054-30-3132 |"
    items = [{"type": "key_value", "content": s} for s in ("First name: Ruth", "Tax ID: 054-30-3132")]
    m, _ = match_items(text, items)
    assert [x.verdict for x in m] == ["correct", "correct"]
    wrong, _ = match_items("| First name | Tax ID |\n|---|---|\n| 054-30-3132 | Ruth |", items)
    assert not any(x.verdict == "correct" for x in wrong)


@pytest.mark.parametrize("returned", ["-$100", "−100", ".100", "(100)"])
def test_numeric_signs_and_leading_decimal_are_not_lost(returned):
    m, _ = match_items("Total: " + returned, [{"type": "key_value", "content": "Total: 100"}])
    assert m[0].verdict != "correct"


def test_repeated_date_amount_columns_can_reflow_without_losing_associations():
    items = [{"type": "table_row", "content": "Date | Amount | Date | Amount"},
             {"type": "table_row", "content": "2023-12-29 | $707,431.00 | 2023-12-21 | $706,832.00"}]
    text = "| Date | Amount |\n|---|---|\n| 2023-12-29 | $707,431.00 |\n| 2023-12-21 | $706,832.00 |"
    m, _ = match_items(text, items)
    assert [v.verdict for v in m] == ["correct", "correct"]
    assert "reflowed repeated-column groups" in m[1].how
    swapped = text.replace('$707,431.00', '$TEMP').replace('$706,832.00', '$707,431.00').replace('$TEMP', '$706,832.00')
    m, _ = match_items(swapped, items)
    assert m[1].verdict != "correct"


def test_reflow_does_not_borrow_from_separate_or_duplicate_date_rows():
    items = [{"type": "table_row", "content": "Date | Amount | Date | Amount"},
             {"type": "table_row", "content": "2023-12-29 | $100 | 2023-12-21 | $200"}]
    text = "| Date | Amount |\n|---|---|\n| 2023-12-29 | $100 |\n\n| Date | Amount |\n|---|---|\n| 2023-12-21 | $200 |"
    m, _ = match_items(text, items)
    assert m[1].verdict != "correct"

    duplicate = "| Date | Amount |\n|---|---|\n| 2023-12-29 | $100 |\n| 2023-12-29 | $900 |\n| 2023-12-21 | $200 |"
    m, _ = match_items(duplicate, items)
    assert m[1].verdict != "correct"


def test_complete_row_with_blank_cell_owns_all_credited_content():
    text = "| Account | Volume | Stores |\n|---|---|---|\n| Kroger | | 21 |"
    m, _ = match_items(text, [{"type": "table_row", "content": "Kroger | | 21"},
                              {"type": "paragraph", "content": "Kroger"}])
    assert m[0].verdict == "correct"
    assert m[0].evidence[0]["text"] == "kroger | | 21"
    assert m[1].verdict != "correct"
