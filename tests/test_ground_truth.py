"""The loader is the only way to read ground truth — these check what it reports.

Every assertion here runs through `GroundTruthLoader`, never by opening a
sidecar. A repair once landed on disk but not in the object the benchmark
scores against, and the script that checked it read the file, so it confirmed a
fix scoring never saw. Tests that read files would repeat that mistake.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.ground_truth import DOCUMENTS_ROOT, GroundTruthLoader, load_corpus

CORPUS = load_corpus()


def test_every_document_loads():
    assert len(CORPUS) == 93
    assert all(gt.plain_text for gt in CORPUS)


def test_tier_and_doc_type_come_from_the_layout():
    for gt in CORPUS:
        assert gt.tier in ("easy", "medium", "hard")
        assert gt.doc_type != "unknown"


def test_provenance_classification_totals():
    counts: dict[str, int] = {}
    for gt in CORPUS:
        counts[gt.gt_source] = counts.get(gt.gt_source, 0) + 1
    # Matches documents/gt_limitations.json: 11 silver, 19 model-transcribed/reviewed or partially repaired.
    assert counts["silver_tesseract"] == 11
    assert counts["model_verified"] == 19
    assert counts["human"] == 63


def test_model_verified_documents_are_the_ones_provenance_names():
    from evaluation.ground_truth import MODEL_VERIFIED_GT

    assert MODEL_VERIFIED_GT, "provenance file parsed to nothing"
    loaded = {Path(gt.document_path).stem for gt in CORPUS if gt.gt_source == "model_verified"}
    assert loaded == set(MODEL_VERIFIED_GT)


def test_per_leg_reference_counts():
    def n(leg: str) -> int:
        return sum(1 for gt in CORPUS if gt.has_gt_for(leg))

    assert n("markdown") == 57
    assert n("tables") == 42        # a human marked the page as a table page
    assert n("schema") == 65
    # One rule for key_value, shared by runner and scorer: at least 3 scalar
    # (non-array) fields. 9 chart/table pages have structured GT but fewer.
    assert n("key_value") == 56
    # 17 sidecars built 2026-08 from the source datasets' own boxes: SROIE 6
    # (pixel coords / image size), FUNSD 6 (LayoutLM 0-1000 -> /1000 — dividing
    # by image size corrupted x-coords once), SynFinTabs 5 (cell word boxes).
    assert n("geometry") == 17     # no reference exists in this corpus
    assert n("text") == 82        # 93 minus the 11 silver documents


def test_markdown_reference_comes_from_a_sidecar_or_a_markdown_native_dataset():
    sources = {gt.markdown_source for gt in CORPUS if gt.markdown}
    assert sources == {"sidecar", "gt_txt_is_markdown"}
    for gt in CORPUS:
        if gt.markdown_source == "gt_txt_is_markdown":
            assert gt.dataset == "getomni-ai/ocr-benchmark"


def test_structured_reference_carries_schema_and_values():
    with_schema = [gt for gt in CORPUS if gt.has_gt_for("schema")]
    assert all(gt.json_schema for gt in with_schema)
    assert all(gt.true_json is not None for gt in with_schema)


def test_silver_documents_are_not_scorable_on_text():
    for gt in CORPUS:
        if gt.gt_source == "silver_tesseract":
            assert gt.has_gt_for("text") is False
            assert "silver" in gt.missing_gt_reason("text")


def test_missing_reference_always_has_a_stated_reason():
    for gt in CORPUS:
        for leg in ("text", "markdown", "tables", "key_value", "schema", "geometry"):
            if not gt.has_gt_for(leg):
                assert gt.missing_gt_reason(leg)


def test_unknown_leg_is_an_error_not_a_silent_false():
    with pytest.raises(ValueError):
        CORPUS[0].has_gt_for("not_a_leg")


def test_disagreeing_text_sidecars_raise(tmp_path):
    doc = tmp_path / "documents" / "easy" / "invoice" / "x.png"
    doc.parent.mkdir(parents=True)
    doc.write_bytes(b"not really an image")
    (doc.parent / "x_gt.json").write_text('{"plain_text": "the repaired text"}')
    (doc.parent / "x_gt.txt").write_text("the stale text")

    with pytest.raises(ValueError, match="sidecars disagree"):
        GroundTruthLoader().load(doc)


def test_missing_sidecar_raises_file_not_found(tmp_path):
    doc = tmp_path / "documents" / "easy" / "invoice" / "y.png"
    doc.parent.mkdir(parents=True)
    doc.write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        GroundTruthLoader().load(doc)


def test_corpus_documents_are_never_mutated_by_loading():
    before = {p: p.stat().st_mtime_ns for p in DOCUMENTS_ROOT.rglob("*") if p.is_file()}
    load_corpus()
    after = {p: p.stat().st_mtime_ns for p in DOCUMENTS_ROOT.rglob("*") if p.is_file()}
    assert before == after


def test_reviewed_text_references_only_require_printed_content():
    import json
    from evaluation.text_leg import score_text

    chart = next(g for g in CORPUS if Path(g.document_path).stem == "chart_003")
    assert "5,500" not in chart.plain_text and "12,000" in chart.plain_text
    assert "F=Female M=Male" in chart.plain_text and "0–24" in chart.plain_text
    assert "Our COVID-19 Response" in chart.plain_text
    assert "|" not in chart.plain_text.split("Our COVID-19 Response")[0]
    assert chart.markdown == chart.plain_text
    for key in ("employmentLevelBreakdown", "ageAndGenderBreakdown", "workforceByGender"):
        assert key not in chart.true_json
        assert key not in chart.inference_schema["properties"]
        assert key not in chart.inference_schema["required"]
    assert chart.true_json["topWorkforceSector"] == "Operations"
    # The archived annotation and its estimated series are still available.
    raw = json.loads(Path(chart.document_path).with_name("chart_003_structured_gt.json").read_text())
    assert "employmentLevelBreakdown" in raw["true_json"]
    for gt in CORPUS:
        if Path(gt.document_path).stem.startswith("scanned_"):
            assert "Source: " not in gt.plain_text
            assert "removed_unprinted_source_citation" in score_text(gt.plain_text, gt).as_metrics()["text_reference_repairs"]
    # Genuine page URLs and identifiers beside removed citations survive.
    scanned = {Path(g.document_path).stem: g for g in CORPUS if g.doc_type == "scanned_legacy"}
    assert "MNKOI 0003439151" in scanned["scanned_001"].plain_text
    assert "http://www.fda.gov/" in scanned["scanned_002"].plain_text


def test_citation_repair_does_not_remove_other_pages_printed_source(tmp_path):
    doc = tmp_path / "documents" / "easy" / "invoice" / "other.png"
    doc.parent.mkdir(parents=True)
    doc.write_bytes(b"fixture")
    text = "Source: https://www.industrydocuments.ucsf.edu/docs/hqcx0242"
    doc.with_name("other_gt.txt").write_text(text)
    gt = GroundTruthLoader().load(doc)
    assert gt.plain_text == text and not gt.metadata


def test_invoice_text_references_match_reviewed_printed_content():
    import json
    from evaluation.metrics import normalize_text
    from evaluation.schema_leg import score_schema
    invoices = {Path(g.document_path).stem: g for g in CORPUS if g.doc_type == "invoice"}
    assert len(invoices) == 6
    for gt in invoices.values():
        assert gt.metadata["text_reference_repairs"]["version"].startswith("invoice-printed-text-v1")
        assert "1." in gt.plain_text.split()
    text = normalize_text(invoices["invoice_003"].plain_text)
    assert "26343874." not in text and "26343874" in text
    assert "sunshine." not in text and "sunshine" in text
    assert "108" not in text.split() and "1 108,80" in text
    assert "3 172,99" in text and "1 369,50" in text
    assert '15"' in invoices["invoice_000"].plain_text
    assert "50 GB" in invoices["invoice_004"].plain_text
    for stem, excluded in (("invoice_004", "items[2].item_desc"),
                           ("invoice_005", "items[0].item_desc")):
        gt = invoices[stem]
        assert "00000000" not in gt.plain_text and "00OO" not in gt.plain_text
        assert gt.metadata["structured_reference_repairs"]["excluded_fields"] == [excluded]
        score = score_schema(json.dumps(gt.true_json), gt, "gemini_flash")
        assert excluded not in {v.path for v in score.verdicts}
        assert score.score == 1
        raw = json.loads(Path(gt.document_path).with_name(stem + "_structured_gt.json").read_text())
        assert raw["true_json"] != gt.true_json
    assert 'White " / TESTED' in invoices["invoice_005"].true_json["items"][4]["item_desc"]


def test_receipt_repairs_preserve_boxes_and_disclose_exclusions():
    import json
    from evaluation.geometry_leg import score_geometry

    receipts = {Path(g.document_path).stem: g for g in CORPUS if g.doc_type == "receipt"}
    expected = {
        "sroie_000": "NO 2 & 4, JALAN BAYU 4,",
        "sroie_002": "KG BATU 30, ULU YAM LAMA",
        "sroie_003": "SEKYSEN 4, BANDAR UTAMA, 44300",
        "sroie_004": "BATU 3, JALAN IPOH",
        "sroie_005": "22A-1, JALAN 17/54,",
    }
    for stem, address in expected.items():
        gt = receipts[stem]
        assert address in gt.plain_text
        assert address in [w["text"] for w in gt.word_boxes]
        original = json.loads(Path(gt.document_path).with_name(stem + "_geometry_gt.json").read_text())["words"]
        repair = gt.metadata["geometry_reference_repairs"]
        excluded = {w["source_index"] for w in repair["excluded_boxes"]}
        assert [w["bbox"] for w in gt.word_boxes] == [w["bbox"] for i, w in enumerate(original) if i not in excluded]
        assert score_geometry(None, gt).as_metrics()["geometry_reference_repairs"] == repair
        assert address not in [w["text"] for w in original]
    assert len(receipts["sroie_002"].word_boxes) == 53
    assert len(receipts["sroie_003"].word_boxes) == 35
    assert [w["text"] for w in receipts["sroie_003"].word_boxes].count("TOTAL QTY :") == 1
    assert "INCLUSLVE" not in receipts["sroie_002"].plain_text
    assert receipts["sroie_000"].true_json["items"][0]["description"] == "KINGS SAFETY SHOES KWD B05"
    # The receipt with uncertain registration digits is not silently edited.
    assert "text_reference_repairs" not in receipts["sroie_001"].metadata


def test_receipt_repair_rejects_changed_annotation():
    import copy

    gt = copy.deepcopy(next(g for g in CORPUS if Path(g.document_path).stem == "sroie_004"))
    gt.plain_text = "a replacement reference that needs a fresh review"
    with pytest.raises(ValueError, match="reference changed"):
        GroundTruthLoader._repair_receipt_references(Path(gt.document_path), gt)
