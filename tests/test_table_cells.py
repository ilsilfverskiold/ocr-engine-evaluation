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


def test_item_can_span_adjacent_rows_without_losing_its_amounts():
    ref = [['', 'RM', 'Code'], ['PPD 4MM DENLIME 1.000 PC x', '1.887', 'SR 2.00'],
           ['6023# GARDEN 1.000 PC x', '5.000', 'SR 5.30']]
    pred = [ref[0], ['PPD 4MM DENLIME', '', ''], ['', '1.887', 'SR'], ['1.000 PC x', '', '2.00'],
            ['6023# GARDEN', '', ''], ['', '5.000', 'SR'], ['1.000 PC x', '', '5.30'],
            ['Subtotal', '', '7.30']]
    result = score(html(ref), html(pred))
    assert result.score == 1
    assert result.cell_counts['extra_cells'] > 0
    values = [v for v in result.code_verdicts if v['reference'] and v['reference']['text'] == 'SR 2.00']
    assert values[0]['matched_text'] == 'sr 2.00'
    assert len(values[0]['output_components']) == 2
    # The same numbers in the wrong logical item cannot rescue it.
    pred[2][1], pred[5][1] = pred[5][1], pred[2][1]
    assert score(html(ref), html(pred)).score == pytest.approx(7/9)


def test_continuation_stops_at_unknown_item_or_subtotal():
    ref = [['Item', 'Price'], ['Apple pack', '10']]
    pred = [ref[0], ['Apple', ''], ['Unrelated item', '10']]
    result = score(html(ref), html(pred))
    assert next(v for v in result.code_verdicts if v['reference'] and v['reference']['text']=='10')['credit'] == 0
    pred[-1][0] = 'Subtotal'
    assert score(html(ref), html(pred)).score == result.score


def test_merged_columns_use_ordered_disjoint_spans_not_value_search():
    ref = [['Item', 'Qty', 'Price'], ['Apples', '2', '100']]
    pred = [['Item', 'Qty Price'], ['Apples', '2 100']]
    result = score(html(ref), html(pred))
    assert result.score == 1
    parts = [p for v in result.code_verdicts if v['reference'] for p in v['output_components']
             if p['row']==1 and p['col']==1]
    assert [(p['start'],p['end']) for p in parts] == [(0,1),(2,5)]
    assert score(html(ref), html([pred[0], ['Apples','100 2']])).score == pytest.approx(4/6)
    assert score(html(ref), html([pred[0], ['Apples','2 1000']])).score == pytest.approx(5/6)


def test_split_columns_combine_only_when_headers_establish_identity():
    ref = [['Item', 'Tax code amount'], ['Apples', 'SR 100']]
    pred = [['Item', 'Tax code', 'amount'], ['Apples', 'SR', '100']]
    assert score(html(ref), html(pred)).score == 1
    pred[0][-1] = 'Other item amount'
    assert score(html(ref), html(pred)).score < 1


def test_html_colspan_values_are_partitioned_in_column_order():
    ref = html([['Item','Qty','Price'],['Apples','2','100']])
    pred = '<table><tr><td>Item</td><td>Qty</td><td>Price</td></tr><tr><td>Apples</td><td colspan="2">2 100</td></tr></table>'
    assert score(ref, pred).score == 1
    assert score(ref, pred.replace('2 100','100 2')).score < 1


def test_complete_item_does_not_absorb_unlabeled_numeric_record():
    rows = [['Name','Current','Previous'], ['Fees','10','20'], ['', '30','40']]
    assert score(html(rows),html(rows)).score == 1


def test_header_can_follow_metadata_rows():
    ref = [['GST Summary','Amount','Tax'],['SR @ 6','6.89','0.41']]
    pred = [['Total items','2','']] + ref
    assert score(html(ref),html(pred)).score == 1


def test_crowded_header_does_not_merge_distinct_data_columns():
    ref = [['Item','Note','2025','2024'],['Apples','4','245','618']]
    pred = [['Item','Note 2025 2024','','',''],['Apples','4','245','618','']]
    r=score(html(ref),html(pred))
    assert all(v['credit']==1 for v in r.code_verdicts if v['reference'] and v['reference']['row']==1)


def test_blank_record_after_partial_item_stays_a_separate_record():
    ref = [['Item','Price'],['Red apple box','100'],['','']]
    pred = [['Item','Price'],['Red apple','100'],['','']]
    r=score(html(ref),html(pred))
    assert r.cell_counts['correct_blank_cells']==2


def test_description_after_item_code_and_spanning_columns_is_one_item():
    ref = html([['Item','Qty','Price'],['6783 SR CERAMIC CAP','5','3.50']])
    pred = '<table><tr><th>Item</th><th>Qty</th><th>Price</th></tr><tr><td>6783</td><td>5</td><td>3.50</td></tr><tr><td colspan="3">SR CERAMIC CAP</td></tr></table>'
    assert score(ref,pred).score == 1


def test_section_label_spanning_columns_is_not_split_into_numeric_fields():
    ref = html([['Name','2025','2024'],['CAPITAL AND RESERVES','','']])
    pred = '<table><tr><th>Name</th><th>2025</th><th>2024</th></tr><tr><td colspan="3">CAPITAL AND RESERVES</td></tr></table>'
    r=score(ref,pred)
    assert next(v for v in r.code_verdicts if v['reference'] and v['reference']['text']=='CAPITAL AND RESERVES')['credit']==1


def test_missing_value_in_combined_cell_does_not_consume_other_values():
    ref = html([['Item','Qty','Price'],['Apples','2','100']])
    pred = html([['Item','Qty Price'],['Apples','100']])
    assert score(ref,pred).score == pytest.approx(5/6)


def test_missing_header_does_not_discard_first_returned_item():
    ref = [['Item','Price'],['Apples','10'],['Pears','20']]
    assert score(html(ref),html(ref[1:])).score == pytest.approx(4/6)


def test_section_label_with_missing_number_does_not_absorb_units_row():
    ref = [['Name','Year'],['1. Capital',''],['',"£'000"],['Apples','10']]
    pred = [ref[0],['Capital',''],ref[2],ref[3]]
    r=score(html(ref),html(pred))
    assert next(v for v in r.code_verdicts if v['reference'] and v['reference']['text']=="£'000")['credit']==1


def test_shipment_identifier_takes_priority_over_repeated_fee_description():
    ref = [['Shipment','Description','Amount'],['123456','Delivery fee handling fee','10'],['654321','Delivery fee','20']]
    pred = [ref[0],['123456','Delivery fee','10'],['','handling fee',''],['654321','Delivery fee','20']]
    assert score(html(ref),html(pred)).score == 1
