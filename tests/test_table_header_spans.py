import json
from pathlib import Path

import pytest

from evaluation.ground_truth import GroundTruthLoader
from evaluation.table_cells import (
    _flex_columns, _header_identity, compare_cells, parse_tables,
)
from evaluation.table_leg import score_tables


def table(headers, rows):
    return parse_tables('<table><tr>' + ''.join(f'<th>{h}</th>' for h in headers)
                        + '</tr>' + ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row)
                                            + '</tr>' for row in rows) + '</table>')


def verdict(result, row, col):
    return next(v for v in result['verdicts'] if v['reference']
                and (v['reference']['row'], v['reference']['col']) == (row, col))


def test_reordered_currency_headers_identify_reordered_columns():
    ref = table(['Item', 'Qty', '(RM) RSP', '(RM) Amount'], [['Cap', '5', '3.50', '17.50']])
    pred = table(['Amount (RM)', 'Item', 'RSP (RM)', 'Qty'], [['17.50', 'Cap', '3.50', '5']])
    assert _flex_columns(ref[0], pred[0]) == {0: 1, 1: 3, 2: 2, 3: 0}
    assert compare_cells(ref, pred)['score'] == 1


@pytest.mark.parametrize('values', [['Cap', '5', '17.50', '3.50'], ['Cap', '5', '3.51', '17.51']])
def test_wrong_or_swapped_amounts_do_not_gain_credit(values):
    ref = table(['Item', 'Qty', '(RM) RSP', '(RM) Amount'], [['Cap', '5', '3.50', '17.50']])
    pred = table(['Item', 'Qty', 'RSP (RM)', 'Amount (RM)'], [values])
    result = compare_cells(ref, pred)
    assert verdict(result, 1, 2)['credit'] == 0
    assert verdict(result, 1, 3)['credit'] == 0


def test_units_and_field_word_order_remain_significant():
    assert _header_identity('(RM) Amount') == _header_identity('Amount (RM)')
    assert _header_identity('Amount (RM)') != _header_identity('Amount (USD)')
    assert _header_identity('Amount (%)') != _header_identity('Amount')
    assert _header_identity('Start to end (days)') != _header_identity('End to start (days)')
    ref = table(['Item', 'Amount (RM)', 'Amount (USD)'], [['Cap', '17', '4']])
    pred = table(['Item', '(USD) Amount', '(RM) Amount'], [['Cap', '4', '17']])
    assert _flex_columns(ref[0], pred[0]) == {0: 0, 1: 2, 2: 1}
    assert compare_cells(ref, pred)['score'] == 1


@pytest.mark.parametrize('inside,outside,correct', [('45', '411.50', True),
                                                    ('44', '45', False)])
def test_reference_span_uses_only_covered_output_columns(inside, outside, correct):
    ref = parse_tables('<table><tr><th>Item</th><th>Qty</th><th>Rate</th><th>Amount</th></tr>'
                       '<tr><td colspan="3">TOT QTY: 45</td><td>411.50</td></tr></table>')
    pred = table(['Item', 'Qty', 'Rate', 'Amount'], [['TOT QTY:', inside, '', outside]])
    result = compare_cells(ref, pred)
    total = verdict(result, 1, 0)
    assert (total['credit'] == 1) == correct
    assert {c['col'] for c in total['output_components']} == {0, 1}
    if correct:
        assert result['score'] == 1


def test_saved_gemini_receipt_scores_all_thirteen_returned_amounts():
    root = Path(__file__).resolve().parents[1]
    source = root / 'results/ocr_outputs/reconciled_outputs/easy__receipt__sroie_001/gemini_flash/tables.call.json'
    before = source.read_bytes()
    call = json.loads(before)
    gt = GroundTruthLoader().load(root / 'documents/easy/receipt/sroie_001.jpg')
    result = score_tables(call['raw_text'], gt, 'gemini_flash', call.get('structured_output'), use_judge=False)
    amounts = [v for v in result.code_verdicts if v['reference_table'] == 0
               and v['reference']['row'] > 0 and v['reference']['col'] in (2, 3)]
    assert len(amounts) == 13
    assert all(v['credit'] == 1 for v in amounts)
    total = next(v for v in result.code_verdicts if v['reference_table'] == 0
                 and v['reference']['row'] == 7 and v['reference']['col'] == 0)
    assert total['credit'] == 1
    assert result.score == pytest.approx(0.9975579975579976)
    assert source.read_bytes() == before


def test_spanning_total_does_not_absorb_following_subtotal_labels():
    ref = parse_tables('<table><tr><th>Item</th><th>Qty</th><th>Rate</th><th>Amount</th></tr>'
                       '<tr><td colspan="3">TOT QTY: 45</td><td>411.50</td></tr></table>')
    # Even with the amount missing, subsequent records are not continuations.
    pred = table(['Item', 'Qty', 'Rate', 'Amount'], [
        ['TOT QTY:', '45', '', ''], ['', 'Subtotal', '', '411.50'],
        ['', '', 'Discount', '0.00'],
    ])
    result = compare_cells(ref, pred)
    total = verdict(result, 1, 0)
    assert total['credit'] == 1
    assert {c['row'] for c in total['output_components']} == {1}
    assert verdict(result, 1, 3)['credit'] == 0


def test_diff_report_indexes_reference_and_extra_table_cells_separately():
    from scripts.build_result_diff import verdict_identity

    ref = {'reference_table': 0, 'reference': {'row': 7, 'col': 0}}
    extra = {'reference': None, 'output_table': 0, 'output': {'row': 7, 'col': 0}}
    assert verdict_identity(ref, 1) == 'reference table 0 r7 c0'
    assert verdict_identity(extra, 2) == 'output table 0 r7 c0'
    assert verdict_identity({'path': 'total'}, 3) == 'total'
