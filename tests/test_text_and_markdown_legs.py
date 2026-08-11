"""Legs 1 and 2 — text coverage, and format-tolerant markdown structure."""

from __future__ import annotations

from evaluation.ground_truth import GroundTruth
from evaluation.markdown_leg import parse_blocks, score_markdown
from evaluation.text_leg import score_text

REFERENCE_MD = """\
## Invoice

**invoice_no:** 97159829

## Items

| item_desc | item_qty |
| --- | --- |
| Chess Table | 2,00 |

## Summary

- total_net_worth: 889,20
- total_vat: 88,92
"""


def _gt(**kwargs) -> GroundTruth:
    base = dict(
        document_path="documents/easy/invoice/invoice_000.png",
        tier="easy",
        doc_type="invoice",
        plain_text="Invoice 97159829 Chess Table",
    )
    base.update(kwargs)
    return GroundTruth(**base)


# --------------------------------------------------------------------------
# Leg 1
# --------------------------------------------------------------------------


def test_coverage_is_order_blind():
    gt = _gt(plain_text="alpha beta gamma")
    assert score_text("gamma beta alpha", gt).coverage == 1.0


def test_coverage_ignores_extra_words_the_engine_adds():
    gt = _gt(plain_text="alpha beta")
    assert score_text("Header: alpha | beta | footer", gt).coverage == 1.0


def test_missing_words_are_listed_for_audit():
    gt = _gt(plain_text="alpha beta gamma")
    result = score_text("alpha beta", gt)
    assert result.coverage < 1.0
    assert "gamma" in result.missed_words


def test_repetition_is_flagged_because_coverage_cannot_see_it():
    gt = _gt(plain_text="alpha beta")
    result = score_text("alpha beta " * 40, gt)
    assert result.coverage == 1.0
    assert result.repetition_suspected is True
    assert result.length_ratio > 3


def test_silver_ground_truth_is_not_applicable_rather_than_zero():
    gt = _gt(doc_type="newspaper", gt_source="silver_tesseract")
    result = score_text("anything", gt)
    assert result.applicable is False
    assert result.score is None
    assert "silver" in result.reason


def test_supporting_metrics_are_recorded():
    gt = _gt(plain_text="alpha beta gamma")
    metrics = score_text("alpha beta gamma", gt).as_metrics()
    assert metrics["cer"] == 0.0
    assert metrics["word_f1"] == 1.0


# --------------------------------------------------------------------------
# Leg 2
# --------------------------------------------------------------------------


def test_identical_markdown_scores_one():
    result = score_markdown(REFERENCE_MD, _gt(markdown=REFERENCE_MD))
    assert result.applicable is True
    assert result.markdown_score == 1.0


def test_heading_depth_is_ignored():
    variant = REFERENCE_MD.replace("## ", "# ")
    assert score_markdown(variant, _gt(markdown=REFERENCE_MD)).element_f1 == 1.0


def test_bold_line_is_the_same_heading_as_a_hash_heading():
    reference = "## Totals\n\nsome text\n"
    variant = "**Totals**\n\nsome text\n"
    assert score_markdown(variant, _gt(markdown=reference)).element_f1 == 1.0


def test_bullet_style_is_ignored():
    reference = "- one\n- two\n"
    variant = "* one\n+ two\n"
    assert score_markdown(variant, _gt(markdown=reference)).element_f1 == 1.0


def test_table_alignment_row_carries_no_content():
    reference = "| a | b |\n| --- | --- |\n| 1 | 2 |\n"
    variant = "| a | b |\n|:---:|----:|\n| 1 | 2 |\n"
    assert score_markdown(variant, _gt(markdown=reference)).element_f1 == 1.0


def test_emphasis_and_trailing_punctuation_are_ignored():
    reference = "Total due.\n"
    variant = "*Total due*\n"
    assert score_markdown(variant, _gt(markdown=reference)).element_f1 == 1.0


def test_reordered_blocks_lose_order_not_elements():
    reference = "# A\n\n# B\n\n# C\n"
    scrambled = "# C\n\n# B\n\n# A\n"
    result = score_markdown(scrambled, _gt(markdown=reference))
    assert result.element_f1 == 1.0
    assert result.order_score < 0.5
    assert result.markdown_score < 1.0


def test_missing_blocks_lower_recall_and_are_listed():
    reference = "# A\n\n# B\n\n# C\n"
    result = score_markdown("# A\n", _gt(markdown=reference))
    assert result.element_recall < 1.0
    assert any(text == "b" for _, text in result.missed)


def test_split_paragraphs_still_match_the_annotators_grouping():
    # The annotator grouped the seller fields into one paragraph; mistral
    # emitted each on its own line with blank lines between. Same content,
    # different segmentation — formatting, not structure, and it once cost
    # a markdown-native engine most of its element F1.
    reference = "Seller: Bradley-Andrade 9879 Elizabeth Common Lake Jonathan\n"
    split = "Seller:\n\nBradley-Andrade\n\n9879 Elizabeth Common\n\nLake Jonathan\n"
    result = score_markdown(split, _gt(markdown=reference))
    assert result.element_recall == 1.0
    assert result.element_precision == 1.0
    assert result.markdown_score == 1.0


def test_per_type_recall_is_broken_out():
    result = score_markdown(REFERENCE_MD, _gt(markdown=REFERENCE_MD))
    assert result.per_type_recall["heading"] == 1.0
    assert result.per_type_recall["table_row"] == 1.0
    assert result.per_type_recall["list_item"] == 1.0


def test_not_applicable_without_a_markdown_reference():
    result = score_markdown("# anything", _gt())
    assert result.applicable is False
    assert result.score is None


def test_engine_that_emits_no_markdown_is_not_applicable_not_zero():
    result = score_markdown(None, _gt(markdown=REFERENCE_MD))
    assert result.applicable is False
    assert "no markdown" in result.reason


def test_parse_blocks_types():
    kinds = [b.kind for b in parse_blocks(REFERENCE_MD)]
    assert kinds.count("heading") == 3
    assert kinds.count("table_row") == 2
    assert kinds.count("list_item") == 2


def test_an_answer_wrapped_in_one_code_fence_is_unwrapped():
    # qwen3_vl returns the whole document inside ```markdown ... ```. Parsed as
    # a code block, every heading and table inside is invisible and a correct
    # conversion scores as if it returned nothing.
    reference = "# Invoice\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n"
    fenced = "```markdown\n" + reference + "```"
    assert score_markdown(fenced, _gt(markdown=reference)).markdown_score == 1.0


def test_a_document_containing_a_code_sample_is_not_unwrapped():
    reference = "# Title\n\nSome text\n"
    with_code = "# Title\n\n```python\nx = 1\n```\n\nSome text\n"
    result = score_markdown(with_code, _gt(markdown=reference))
    assert result.per_type_recall.get("heading") == 1.0


def test_tab_separated_tables_count_as_table_rows():
    # MinerU renders tables tab-separated. Without this the leg saw no table
    # at all and scored 0.200 on a page it had parsed correctly.
    reference = "| No. | Description |\n| --- | --- |\n| 1. | Chess Table |\n"
    tsv = "No.\tDescription\n1.\tChess Table\n"
    result = score_markdown(tsv, _gt(markdown=reference))
    assert result.per_type_recall.get("table_row") == 1.0


def test_matching_nothing_scores_zero_not_the_order_weight():
    # Kendall tau of an empty sequence is 1.0, so an engine that recovered no
    # blocks was collecting the full order weight — a 0.2 floor that ten
    # engines shared as their minimum in the first full run.
    reference = "# A\n\n# B\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n"
    result = score_markdown("nothing here matches", _gt(markdown=reference))
    assert result.matched_blocks == 0
    assert result.markdown_score == 0.0


def test_order_still_counts_once_there_is_an_order_to_judge():
    reference = "# A\n\n# B\n\n# C\n"
    assert score_markdown(reference, _gt(markdown=reference)).markdown_score == 1.0
    scrambled = score_markdown("# C\n\n# B\n\n# A\n", _gt(markdown=reference))
    assert scrambled.element_f1 == 1.0
    assert scrambled.markdown_score < 1.0
