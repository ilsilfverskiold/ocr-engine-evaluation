"""Leg 3 — TEDS, and the dialect normalisation it depends on."""

from __future__ import annotations

from evaluation.ground_truth import GroundTruth
from evaluation.table_leg import (
    compute_teds,
    html_to_tree,
    normalize_table_html,
    score_tables,
)

PLAIN = """
<table>
  <tr><td>Item</td><td>Qty</td><td>Total</td></tr>
  <tr><td>Widget</td><td>2</td><td>19.99</td></tr>
</table>
"""

# Byte-different, structurally identical: thead/tbody wrappers, th cells, and
# the presentation attributes a markdown renderer adds. This exact difference
# once scored 0.846.
THEAD_DIALECT = """
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;"><th>Item</th><th>Qty</th><th>Total</th></tr>
  </thead>
  <tbody>
    <tr><td align="left">Widget</td><td>2</td><td>19.99</td></tr>
  </tbody>
</table>
"""


def test_thead_and_th_dialect_scores_exactly_one():
    assert compute_teds(THEAD_DIALECT, PLAIN) == 1.000


def test_normalisation_is_symmetric():
    assert compute_teds(PLAIN, THEAD_DIALECT) == 1.000


def test_normalize_unwraps_and_downgrades():
    normalised = normalize_table_html(THEAD_DIALECT)
    assert "thead" not in normalised
    assert "tbody" not in normalised
    assert "<th" not in normalised
    assert "border" not in normalised
    assert "text-align" not in normalised


def test_identical_table_scores_one():
    assert compute_teds(PLAIN, PLAIN) == 1.000


def test_one_wrong_cell_costs_less_than_a_missing_row():
    wrong_cell = PLAIN.replace("19.99", "19.98")
    missing_row = """
    <table><tr><td>Item</td><td>Qty</td><td>Total</td></tr></table>
    """
    assert compute_teds(wrong_cell, PLAIN) > compute_teds(missing_row, PLAIN)


def test_cell_text_with_braces_does_not_break_the_tree():
    # The previous implementation encoded the tree as a bracket string and was
    # corrupted by cell text containing braces.
    braced = PLAIN.replace("Widget", "Widget {a} }{")
    assert 0.0 <= compute_teds(braced, PLAIN) < 1.0


def test_entities_and_whitespace_are_normalised():
    a = "<table><tr><td>A &amp; B</td></tr></table>"
    b = "<table><tr><td>A &   B</td></tr></table>"
    assert compute_teds(a, b) == 1.000


def test_case_differences_do_not_count_as_structure():
    a = "<table><tr><td>Total</td></tr></table>"
    b = "<table><tr><td>TOTAL</td></tr></table>"
    assert compute_teds(a, b) == 1.000


def test_no_table_in_output_scores_zero():
    assert compute_teds("no table here", PLAIN) == 0.0


def test_html_to_tree_shape():
    tree = html_to_tree(PLAIN)
    assert tree is not None
    assert len(tree.children) == 2
    assert len(tree.children[0].children) == 3
    assert tree.size() == 1 + 2 + 6


def _gt(table_html: str | None, source: str | None = None) -> GroundTruth:
    return GroundTruth(
        document_path="documents/easy/invoice/x.png",
        tier="easy",
        doc_type="invoice",
        plain_text="text",
        table_html=table_html,
        table_html_source=source,
    )


def test_scorer_returns_not_applicable_without_a_reference():
    result = score_tables("| a | b |", _gt(None), "gemini_flash", use_judge=False)
    assert result.applicable is False
    assert result.score is None
    assert "no table reference" in result.reason


def test_scorer_scores_markdown_output_against_html_reference():
    markdown = (
        "| Item | Qty | Total |\n"
        "| --- | --- | --- |\n"
        "| Widget | 2 | 19.99 |\n"
    )
    result = score_tables(markdown, _gt(PLAIN), "gemini_flash", use_judge=False)
    assert result.applicable is True
    assert result.teds == 1.000
    assert result.reference_rows == 2
    assert result.reference_cols == 3


def test_engine_with_no_table_structure_scores_zero_not_not_applicable():
    result = score_tables("flat text with no table", _gt(PLAIN), "tesseract", use_judge=False)
    assert result.applicable is True
    assert result.teds == 0.0
    assert "no recoverable table structure" in result.reason


def test_weak_reference_is_flagged():
    result = score_tables(
        "| Item |\n| --- |\n| Widget |",
        _gt(PLAIN, "schema GT record arrays (converted)"),
        "gemini_flash",
        use_judge=False,
    )
    assert result.reference_is_weak is True


def test_structured_output_is_preferred_over_text():
    structured = {"tables": [{"html": PLAIN}]}
    result = score_tables("", _gt(PLAIN), "textract_structured", structured, use_judge=False)
    assert result.teds == 1.000
