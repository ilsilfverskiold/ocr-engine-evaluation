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
    # Matches documents/gt_limitations.json: 11 silver, 8 model-transcribed.
    assert counts["silver_tesseract"] == 11
    assert counts["model_verified"] == 8
    assert counts["human"] == 74


def test_model_verified_documents_are_the_ones_provenance_names():
    from evaluation.ground_truth import MODEL_VERIFIED_GT

    assert MODEL_VERIFIED_GT, "provenance file parsed to nothing"
    loaded = {Path(gt.document_path).stem for gt in CORPUS if gt.gt_source == "model_verified"}
    assert loaded == set(MODEL_VERIFIED_GT)


def test_per_leg_reference_counts():
    def n(leg: str) -> int:
        return sum(1 for gt in CORPUS if gt.has_gt_for(leg))

    assert n("markdown") == 57
    assert n("tables") == 42
    assert n("schema") == 65
    assert n("key_value") == 65
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
