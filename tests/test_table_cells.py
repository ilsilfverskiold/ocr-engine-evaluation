"""Gold cell scoring invariants: association, multiplicity, blanks and spans."""
from html import escape

import pytest

from evaluation.ground_truth import GroundTruth
from evaluation.table_cells import parse_tables, compare_cells
from evaluation.table_leg import score_tables


def html(rows):
    return "<table>" + "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>"
                               for row in rows) + "</table>"


def score(ref, pred, **kwargs):
    gt = GroundTruth("documents/easy/invoice/x.png", "easy", "invoice", "", table_html=ref)
    return score_tables(pred, gt, "unknown", **kwargs)


ROWS = [["Item", "2025", "2024"], ["Apples", "100", "900"], ["Pears", "200", "800"]]


def test_gold_cells_need_no_model_checklist():
    result = score(html(ROWS), html(ROWS))
    assert result.score == result.cell_precision == result.cell_recall == 1
    assert result.reference_cells == result.output_cells == 9
    assert len(result.code_verdicts) == 9


def test_wrong_column_values_do_not_move_to_find_credit():
    pred = [ROWS[0], ["Apples", "900", "100"], ["Pears", "800", "200"]]
    assert score(html(ROWS), html(pred)).score == pytest.approx(5 / 9)


def test_reordered_complete_columns_and_named_rows_preserve_associations():
    pred = [["Item", "2024", "2025"], ["Pears", "800", "200"], ["Apples", "900", "100"]]
    assert score(html(ROWS), html(pred)).score == 1


@pytest.mark.parametrize("pred,correct", [
    ([["Name", "Value"], ["A", ""]], 4),
    ([["Name", "Value"], ["A", "invented"]], 3),
    ([["Name", "Value"], ["A"]], 3),
])
def test_blank_must_be_a_real_returned_cell(pred, correct):
    result = score(html([["Name", "Value"], ["A", ""]]), html(pred))
    assert result.cell_counts["correct"] == correct
    assert result.output_cells == sum(map(len, pred))


def test_markdown_parser_does_not_pad_missing_cells_or_drop_extra_ones():
    ref = html([["Name", "Value"], ["A", ""]])
    missing = score(ref, "| Name | Value |\n| --- | --- |\n| A |")
    extra = score(ref, "| Name | Value |\n| --- | --- |\n| A | | extra |")
    assert missing.output_cells == 3 and missing.cell_counts["correct_blank_cells"] == 0
    assert extra.output_cells == 5 and extra.cell_precision == .8


def test_duplicate_rows_and_tables_reduce_precision():
    ref = html(ROWS)
    row = score(ref, html(ROWS + [ROWS[1]]))
    tables = score(ref, ref + ref)
    assert row.cell_recall == 1 and row.cell_precision == .75
    assert tables.cell_recall == 1 and tables.cell_precision == .5
    assert tables.structure_similarity < 1
    assert row.score == tables.score == 1


def test_extra_columns_do_not_shift_labelled_values_or_reduce_headline():
    result = score(html(ROWS), html([["Item", "Extra", "2024", "2025"],
                                  ["Apples", "x", "900", "100"], ["Pears", "y", "800", "200"]]))
    assert result.score == 1 and result.cell_counts['extra_cells'] == 3


def test_transpose_and_table_chunks_keep_label_value_associations():
    transposed = [list(row) for row in zip(*ROWS)]
    assert score(html(ROWS), html(transposed)).score == 1
    assert score(html(ROWS), html(ROWS[:2]) + html([ROWS[0], ROWS[2]])).score == 1


def test_repeated_columns_may_reflow_but_not_swap_amounts():
    ref = [["Date", "Amount", "Date", "Amount"], ["2023-12-29", "$100", "2023-12-21", "$200"]]
    pred = [["Date", "Amount"], ["2023-12-21", "200"], ["2023-12-29", "100"]]
    assert score(html(ref), html(pred)).score == 1
    pred[1][1], pred[2][1] = '100', '200'
    assert score(html(ref), html(pred)).score < 1


def test_partial_text_is_fractional_and_wrong_numbers_do_not_get_substring_credit():
    result = score(html([["Description", "Price"], ["Red ceramic coffee cup", "$12.34"]]),
                   html([["Description", "Price"], ["Red ceramic cup", "1234"]]))
    assert result.cell_counts['partial'] == 1
    assert .5 < result.score < .75
    assert result.code_verdicts[-1]['credit'] == 0


def test_two_reference_tables_cannot_reuse_one_output_table():
    ref = html(ROWS)
    result = score(ref + ref, ref)
    assert result.cell_precision == 1 and result.cell_recall == .5
    assert result.structure_similarity < 1


def test_native_and_markdown_are_one_representation_not_duplicates():
    ref = html(ROWS)
    result = score(ref, ref, structured_output={"tables": [{"html": ref}]})
    assert result.score == 1 and result.output_cells == 9


def test_merged_cell_is_counted_once_and_span_changes_affect_structure():
    ref = '<table><tr><td colspan="2">Heading</td></tr><tr><td>A</td><td>B</td></tr></table>'
    assert len(parse_tables(ref)[0].cells) == 3
    same = score(ref, ref)
    wrong_span = score(ref, ref.replace('colspan="2"', 'colspan="3"'))
    assert same.score == same.structure_similarity == 1
    assert wrong_span.structure_similarity < 1


def test_a_large_merged_text_cell_cannot_satisfy_two_reference_cells():
    ref = html([["Name"], ["Apples"], ["Pears"]])
    result = score(ref, html([["Name"], ["Apples Pears"]]))
    assert result.cell_counts["correct"] == 1


def test_numeric_punctuation_and_digit_changes_are_errors():
    assert score(html([["Total"], ["12.34"]]), html([["Total"], ["1234"]])).score == .5
    assert score(html([["Total"], ["100"]]), html([["Total"], ["1000"]])).score == .5


def test_shipping_chooses_one_complete_reference_and_records_both():
    gold, original = html(ROWS), html([["Name", "Price"], ["Orange", "17.2"]])
    gt = GroundTruth("documents/hard/shipping/x.png", "hard", "shipping", "", table_html=gold,
                     metadata={"table_gold_reference": {}, "table_original_reference": {"table_html": original}})
    # A real saved gold has a nonempty provenance record.
    gt.metadata["table_gold_reference"] = {"version": 1}
    result = score_tables(original, gt, "unknown")
    assert result.score == 1 and result.selected_reference == "original"
    assert set(result.reference_comparisons) == {"gold", "original"}
    assert result.reference_cells == 4


def test_tsv_keeps_all_tables_and_all_explicit_cells():
    from evaluation.table_extractor import extract_all_tables_html
    tables = extract_all_tables_html("Name\tValue\nA\t\n\nAnother\tHeader\nB\t2\textra", "mineru")
    assert len(tables) == 2
    assert [len(t.cells) for markup in tables for t in parse_tables(markup)] == [4, 5]


def test_blank_header_column_is_not_discarded_by_label_alignment():
    html = "<table><tr><td></td><td>Net</td><td>VAT</td></tr><tr><td>Total</td><td>10</td><td>2</td></tr></table>"
    tables = parse_tables(html)
    result = compare_cells(tables, tables)
    assert result["score"] == 1
    assert result["extra_cells"] == 0
