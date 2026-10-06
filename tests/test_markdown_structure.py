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
    assert [kind for kind, _, _ in blocks("| A | B |\n| --- | --- |\n| 1 | 2 |")] == ["table_row", "table_row"]


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
