"""Leg 6 — geometry: located words (mutual center containment) over reference words."""

from __future__ import annotations

from evaluation.ground_truth import GroundTruth
from evaluation.geometry_leg import (
    center_hit,
    iou,
    merge_words_into_lines,
    parse_word_boxes,
    score_geometry,
)

REFERENCE = [
    {"text": "Invoice", "bbox": [0.10, 0.10, 0.30, 0.15]},
    {"text": "97159829", "bbox": [0.35, 0.10, 0.55, 0.15]},
    {"text": "Total", "bbox": [0.10, 0.80, 0.25, 0.85]},
]


def test_iou_of_identical_boxes_is_one():
    assert iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0


def test_iou_of_disjoint_boxes_is_zero():
    assert iou((0, 0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6)) == 0.0


def test_iou_half_overlap():
    assert abs(iou((0, 0, 1, 1), (0.5, 0, 1.5, 1)) - 1 / 3) < 1e-9


def test_perfect_prediction_scores_one():
    result = score_geometry(REFERENCE, reference=REFERENCE)
    assert result.applicable is True
    assert result.geometry_score == 1.0
    assert result.match_rate == 1.0


def test_unmatched_words_count_as_misses():
    result = score_geometry(REFERENCE[:2], reference=REFERENCE)
    assert result.matched_words == 2
    assert abs(result.geometry_score - 2 / 3) < 1e-3
    assert result.mean_iou_matched == 1.0


def test_right_text_wrong_place_scores_the_overlap_not_the_match():
    shifted = [{"text": w["text"], "bbox": [0.9, 0.9, 0.95, 0.95]} for w in REFERENCE]
    result = score_geometry(shifted, reference=REFERENCE)
    assert result.match_rate == 1.0
    assert result.geometry_score == 0.0


def test_text_must_match_before_geometry_counts():
    wrong_text = [{"text": "WRONG", "bbox": w["bbox"]} for w in REFERENCE]
    result = score_geometry(wrong_text, reference=REFERENCE)
    assert result.matched_words == 0
    assert result.geometry_score == 0.0


def test_text_matching_is_punctuation_and_case_insensitive():
    noisy = [{"text": w["text"].upper() + ",", "bbox": w["bbox"]} for w in REFERENCE]
    assert score_geometry(noisy, reference=REFERENCE).geometry_score == 1.0


def test_json_string_output_is_parsed():
    import json

    result = score_geometry(json.dumps(REFERENCE), reference=REFERENCE)
    assert result.geometry_score == 1.0


def test_fenced_json_is_parsed():
    import json

    fenced = "```json\n" + json.dumps(REFERENCE) + "\n```"
    assert score_geometry(fenced, reference=REFERENCE).geometry_score == 1.0


def test_unparseable_output_scores_zero_and_records_why():
    result = score_geometry("I cannot provide bounding boxes.", reference=REFERENCE)
    assert result.applicable is True
    assert result.parse_ok is False
    assert result.geometry_score == 0.0


def test_parse_word_boxes_rejects_malformed_entries():
    assert parse_word_boxes([{"text": "a", "bbox": [1, 2]}]) is None


def test_box_drawing_convention_is_not_penalised():
    # The tesseract-vs-SynFinTabs case: same center, box twice as tall. Raw
    # IoU called this 0.45 for a perfectly placed word; the headline must not.
    tall = [{"text": w["text"],
             "bbox": [w["bbox"][0], w["bbox"][1] - 0.025,
                      w["bbox"][2], w["bbox"][3] + 0.025]} for w in REFERENCE]
    result = score_geometry(tall, reference=REFERENCE)
    assert result.geometry_score == 1.0
    assert result.mean_iou_matched < 0.6  # the diagnostic still shows the gap


def test_center_hit_requires_mutual_containment():
    assert center_hit((0.1, 0.1, 0.3, 0.15), (0.1, 0.05, 0.3, 0.20))
    assert not center_hit((0.1, 0.1, 0.3, 0.15), (0.35, 0.1, 0.55, 0.15))


def test_words_merge_into_lines_left_to_right():
    words = [
        {"text": "SDN", "bbox": [0.4, 0.10, 0.5, 0.14]},
        {"text": "OJC", "bbox": [0.1, 0.10, 0.2, 0.14]},
        {"text": "BHD", "bbox": [0.6, 0.11, 0.7, 0.15]},
        {"text": "TOTAL", "bbox": [0.1, 0.80, 0.3, 0.84]},
    ]
    lines = merge_words_into_lines(words)
    assert [l["text"] for l in lines] == ["OJC SDN BHD", "TOTAL"]
    assert lines[0]["bbox"] == (0.1, 0.10, 0.7, 0.15)


def test_line_level_reference_matches_word_level_output():
    # The SROIE case: no word engine can text-match "OJC MARKETING SDN BHD"
    # word-by-word; with line granularity it must score, misread char and all.
    reference = [{"text": "OJC MARKETING SDN BHD", "bbox": [0.1, 0.1, 0.7, 0.15]},
                 {"text": "TOTAL 12.30", "bbox": [0.1, 0.8, 0.4, 0.85]}]
    predicted = [
        {"text": "OJC", "bbox": [0.10, 0.10, 0.20, 0.15]},
        {"text": "MARKET1NG", "bbox": [0.22, 0.10, 0.42, 0.15]},  # misread char
        {"text": "SDN", "bbox": [0.44, 0.10, 0.54, 0.15]},
        {"text": "BHD", "bbox": [0.56, 0.10, 0.70, 0.15]},
        {"text": "TOTAL", "bbox": [0.10, 0.80, 0.25, 0.85]},
        {"text": "12.30", "bbox": [0.27, 0.80, 0.40, 0.85]},
    ]
    result = score_geometry(predicted, reference=reference, granularity="line")
    assert result.geometry_score == 1.0
    assert result.matched_words == 2


def test_column_segment_references_match_within_a_visual_line():
    # SROIE's "lines" are column segments: QTY, PRICE, AMOUNT are separate
    # entries on one visual line. Whole-line merging capped every engine at
    # the same ceiling; window matching must locate each segment.
    reference = [{"text": "QTY", "bbox": [0.10, 0.30, 0.20, 0.33]},
                 {"text": "PRICE", "bbox": [0.40, 0.30, 0.52, 0.33]},
                 {"text": "AMOUNT", "bbox": [0.70, 0.30, 0.85, 0.33]}]
    predicted = [{"text": "QTY", "bbox": [0.10, 0.30, 0.20, 0.33]},
                 {"text": "PRICE", "bbox": [0.40, 0.30, 0.52, 0.33]},
                 {"text": "AMOUNT", "bbox": [0.70, 0.30, 0.85, 0.33]}]
    result = score_geometry(predicted, reference=reference, granularity="line")
    assert result.geometry_score == 1.0


def test_granularity_flows_from_the_ground_truth_sidecar():
    gt = GroundTruth(
        document_path="documents/easy/receipt/sroie_000.jpg",
        tier="easy",
        doc_type="receipt",
        plain_text="text",
        word_boxes=[{"text": "TWO WORDS", "bbox": [0.1, 0.1, 0.5, 0.15]}],
        word_boxes_granularity="line",
    )
    predicted = [{"text": "TWO", "bbox": [0.1, 0.1, 0.28, 0.15]},
                 {"text": "WORDS", "bbox": [0.32, 0.1, 0.5, 0.15]}]
    assert score_geometry(predicted, gt).geometry_score == 1.0


def test_gemini_box_2d_convention_is_understood():
    # box_2d is documented as [ymin, xmin, ymax, xmax] in 0-1000. Both the
    # axis order and the scale must be undone, not scored as misplacement.
    b2d = [{"text": w["text"],
            "box_2d": [w["bbox"][1] * 1000, w["bbox"][0] * 1000,
                       w["bbox"][3] * 1000, w["bbox"][2] * 1000]}
           for w in REFERENCE]
    result = score_geometry(b2d, reference=REFERENCE)
    assert result.geometry_score == 1.0
    assert result.rescaled_from_thousand is True


def test_named_corner_dict_boxes_are_understood():
    named = [{"text": w["text"],
              "bbox": {"xmin": w["bbox"][0], "ymin": w["bbox"][1],
                       "xmax": w["bbox"][2], "ymax": w["bbox"][3]}}
             for w in REFERENCE]
    assert score_geometry(named, reference=REFERENCE).geometry_score == 1.0


def test_one_mangled_item_does_not_zero_the_page():
    import json

    good = [json.dumps(w) for w in REFERENCE]
    broken = '{"text": "junk", "bbox": {"ymin": 1, "xmax": 5]}'  # gemini did this
    blob = "[" + ",".join(good[:2] + [broken] + good[2:]) + "]"
    result = score_geometry(blob, reference=REFERENCE)
    assert result.parse_ok is True
    assert result.geometry_score == 1.0  # the three well-formed words survive


def test_qwen_native_boxes_survive_a_malformed_and_truncated_answer():
    raw = '''```json
    [{"bbox_2d": [100, 200, 300, 220], "text_content": "Total"},
     {"bbox_2d": [400, 200, 500, 220], "broken"},
     {"bbox_2d": [600, 200, 700, 220], "text_content": "100"},
     {"bbox_2d": [800,'''
    reference = [{"text": "Total", "bbox": [.1, .2, .3, .22]},
                 {"text": "100", "bbox": [.6, .2, .7, .22]}]
    result = score_geometry(raw, reference=reference)
    assert result.parse_ok and result.rescaled_from_thousand
    assert result.predicted_words == 2
    assert result.located_words == 2
    assert not result.axes_swapped


def test_swapped_axes_are_detected_and_undone():
    # gemini's `bounding_box` key carries [ymin, xmin, ymax, xmax] with no
    # documentation to swap on — the page-level portrait/landscape guard
    # catches it: words are wider than tall, so majority-portrait output
    # against a landscape reference means the axes came swapped.
    swapped = [{"text": w["text"],
                "bounding_box": [w["bbox"][1], w["bbox"][0],
                                 w["bbox"][3], w["bbox"][2]]}
               for w in REFERENCE]
    result = score_geometry(swapped, reference=REFERENCE)
    assert result.axes_swapped is True
    assert result.geometry_score == 1.0
    assert score_geometry(REFERENCE, reference=REFERENCE).axes_swapped is False


def test_thousand_scale_output_is_rescaled_not_zeroed():
    # Grounding-trained VLMs emit 0-1000 despite the 0-1 ask. Without the
    # guard every box scores as misplaced and the engine gets a fake zero.
    scaled = [{"text": w["text"], "bbox": [c * 1000 for c in w["bbox"]]}
              for w in REFERENCE]
    result = score_geometry(scaled, reference=REFERENCE)
    assert result.geometry_score == 1.0
    assert result.rescaled_from_thousand is True
    assert score_geometry(REFERENCE, reference=REFERENCE).rescaled_from_thousand is False


def test_corpus_documents_are_not_applicable_for_this_leg():
    gt = GroundTruth(
        document_path="documents/easy/receipt/sroie_000.jpg",
        tier="easy",
        doc_type="receipt",
        plain_text="text",
    )
    result = score_geometry(REFERENCE, gt)
    assert result.applicable is False
    assert result.score is None
    assert "no word-box reference" in result.reason


def test_repeated_values_pair_by_position_not_by_first_exact_match():
    # A receipt prints the same amount several times. An engine that misread
    # one instance ("1 193.00" glued into "1193.00") used to lose the whole
    # column: the first reference took the exact "193.00" from another row,
    # and every later reference shifted one instance down. Among windows
    # whose text is close enough, the one overlapping the reference wins.
    reference = [{"text": "193.00", "bbox": [0.47, 0.61, 0.58, 0.63]},
                 {"text": "193.00", "bbox": [0.65, 0.69, 0.77, 0.71]},
                 {"text": "193.00", "bbox": [0.65, 0.78, 0.77, 0.80]}]
    predicted = [{"text": "1193.00", "bbox": [0.44, 0.61, 0.58, 0.63]},
                 {"text": "193.00", "bbox": [0.66, 0.69, 0.76, 0.71]},
                 {"text": "193.00", "bbox": [0.66, 0.78, 0.76, 0.80]}]
    result = score_geometry(predicted, reference=reference, granularity="line")
    assert result.located_words == 3, result.worst_matches
