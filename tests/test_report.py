"""Aggregation, the Modal cost audit gate, and the report's own arithmetic."""

from __future__ import annotations

import json
import logging

import pytest

import run_benchmark
from run_benchmark import aggregate_legs, engine_leg_means, engine_operational


def _leg(engine, tier, doc_type, doc, leg, score, applicable=True):
    return {
        "engine_name": engine, "tier": tier, "doc_type": doc_type,
        "document_path": doc, "gt_source": "human", "leg": leg,
        "pass_name": "default", "applicable": applicable, "score": score,
        "reason": None, "metrics": {}, "detail": {}, "error": None,
    }


def test_aggregation_keeps_document_types_apart():
    records = [
        _leg("e", "easy", "invoice", "a.png", "schema", 1.0),
        _leg("e", "easy", "invoice", "b.png", "schema", 0.0),
        _leg("e", "medium", "tax_form", "c.png", "schema", 0.5),
    ]
    agg = aggregate_legs(records)
    assert agg[("e", "easy", "invoice", "schema")]["score"] == 0.5
    assert agg[("e", "medium", "tax_form", "schema")]["score"] == 0.5
    assert len(agg) == 2


@pytest.mark.parametrize("selection,expected", [
    ("key_value,schema", ["key_value", "schema"]),
    ("all", list(run_benchmark.ALL_LEGS)),
])
def test_skip_judge_keeps_code_scored_legs_in_cli(monkeypatch, tmp_path, selection, expected):
    import sys
    from types import SimpleNamespace
    from evaluation.ground_truth import GroundTruthLoader

    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--engines", "claude_sonnet_4_6",
                                    "--legs", selection, "--skip-judge", "--dry-run"])
    monkeypatch.setattr(run_benchmark, "setup_logging", lambda *args: logging.getLogger("test"))
    monkeypatch.setattr(run_benchmark, "discover_documents", lambda *args: [tmp_path / "a.png"])
    monkeypatch.setattr(GroundTruthLoader, "load", lambda *args: SimpleNamespace(has_gt_for=lambda leg: True))
    captured = []
    monkeypatch.setattr(run_benchmark, "_print_dry_run", lambda engines, legs, *args: captured.extend(legs))
    run_benchmark.main()
    assert captured == expected


def test_report_method_uses_saved_scorer_not_todays_registry():
    old = _leg("e", "easy", "invoice", "a.png", "tables", 0.9)
    old["metrics"] = {"cell_accuracy": 0.9, "judge_model": "saved-model"}
    current = dict(old, metrics={"cell_accuracy": 0.8, "scorer": "code-v3"})
    assert "LLM verdicts (saved-model)" in run_benchmark._recorded_scoring_method("tables", [old])[1]
    assert "code verdicts" in run_benchmark._recorded_scoring_method("tables", [current])[1]
    mixed = run_benchmark._recorded_scoring_method("tables", [old, current])[1]
    assert "LLM verdicts" in mixed and "code verdicts" in mixed


def test_estimate_does_not_charge_for_removed_table_judge(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace
    from evaluation.ground_truth import GroundTruthLoader

    monkeypatch.setenv("GOOGLE_API_KEY", "fake-for-estimate-only")
    monkeypatch.setattr(GroundTruthLoader, "load", lambda *args: SimpleNamespace(has_gt_for=lambda leg: True))
    monkeypatch.setattr(run_benchmark, "_measured_modal_costs", lambda: {})
    run_benchmark.print_estimate(["claude_sonnet_4_6"], ["tables"], [tmp_path / "a.png"], True, 0)
    output = capsys.readouterr().out
    assert "LLM judge (tables)" not in output
    assert "Grand total (no judge)" in output


def test_engine_mean_is_a_mean_of_doc_type_means_not_a_pooled_mean():
    # Two invoices at 1.0 and one tax form at 0.0. Pooled would be 0.667;
    # the mean of doc-type means is 0.5, and that is the published figure.
    records = [
        _leg("e", "easy", "invoice", "a.png", "schema", 1.0),
        _leg("e", "easy", "invoice", "b.png", "schema", 1.0),
        _leg("e", "medium", "tax_form", "c.png", "schema", 0.0),
    ]
    means = engine_leg_means(aggregate_legs(records))
    assert means[("e", "schema")] == 0.5


def test_not_applicable_is_excluded_from_the_mean_not_counted_as_zero():
    records = [
        _leg("e", "easy", "invoice", "a.png", "tables", 1.0),
        _leg("e", "easy", "invoice", "b.png", "tables", None, applicable=False),
    ]
    row = aggregate_legs(records)[("e", "easy", "invoice", "tables")]
    assert row["score"] == 1.0
    assert row["n"] == 1
    assert row["n_documents"] == 2
    assert row["n_not_applicable"] == 1


def test_cost_per_page_sums_every_pass_for_a_document():
    inference = [
        {"engine_name": "e", "document_path": "a.png", "pass_name": "default",
         "steady_cost_per_page": 0.001, "steady_latency_seconds": 1.0},
        {"engine_name": "e", "document_path": "a.png", "pass_name": "schema",
         "steady_cost_per_page": 0.002, "steady_latency_seconds": 2.0},
    ]
    stats = engine_operational(inference)["e"]
    # cost_per_page is the ONE-CALL figure (the pass serving the text leg),
    # so engines are compared on the same basis however many legs the
    # benchmark asked each one; the full spend is cost_all_passes.
    assert stats["cost_per_page"] == pytest.approx(0.001)
    assert stats["latency_s"] == pytest.approx(1.0)
    assert stats["cost_all_passes"] == pytest.approx(0.003)
    assert stats["latency_all_passes"] == pytest.approx(3.0)
    assert stats["passes"] == 2


def test_errored_calls_do_not_enter_cost():
    inference = [
        {"engine_name": "e", "document_path": "a.png", "pass_name": "default",
         "steady_cost_per_page": 0.001, "steady_latency_seconds": 1.0},
        {"engine_name": "e", "document_path": "b.png", "pass_name": "default",
         "cost_per_page": None, "error": "boom"},
    ]
    assert engine_operational(inference)["e"]["cost_per_page"] == pytest.approx(0.001)


# --------------------------------------------------------------------------
# The Modal cost audit gate
# --------------------------------------------------------------------------

def _modal_inference(**overrides):
    record = {
        "engine_name": "glm_ocr", "tier": "easy", "doc_type": "invoice",
        "document_path": "a.png", "gt_source": "human", "pass_name": "default",
        "legs": ["text"], "measurement_scope": "modal_batch", "batch_id": "b1",
        "batch_gpu_seconds": 2.0, "batch_size": 1,
        "effective_gpu_seconds_per_document": 2.0,
        "latency_seconds": 2.0, "steady_latency_seconds": 2.0,
        "cost_per_page": 2.0 * 0.000306, "steady_cost_per_page": 2.0 * 0.000306,
        "startup_excluded": True, "gpu_type": "A10",
        "gpu_rate_usd_per_second": 0.000306, "error": None,
    }
    record.update(overrides)
    return record


def _write_run(tmp_path, monkeypatch, inference, legs):
    monkeypatch.setattr(run_benchmark, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(run_benchmark, "SUMMARIES_DIR", tmp_path / "summaries")
    monkeypatch.setattr(run_benchmark, "LOGS_DIR", tmp_path / "logs")
    (tmp_path / "runs").mkdir(parents=True)
    ts = "20260101T000000Z"
    with open(tmp_path / "runs" / f"{ts}_inference.jsonl", "w") as f:
        for r in inference:
            f.write(json.dumps(r) + "\n")
    with open(tmp_path / "runs" / f"{ts}_results.jsonl", "w") as f:
        for r in legs:
            f.write(json.dumps(r) + "\n")
    return ts


def test_report_aborts_when_a_modal_cost_cannot_be_re_derived(tmp_path, monkeypatch):
    # cost_per_page no longer equals gpu_seconds x the GPU rate.
    bad = _modal_inference(cost_per_page=0.99, steady_cost_per_page=0.99)
    ts = _write_run(
        tmp_path, monkeypatch, [bad],
        [_leg("glm_ocr", "easy", "invoice", "a.png", "text", 0.9)],
    )
    with pytest.raises(ValueError, match="Modal timing/cost audit failed"):
        run_benchmark.report(ts, logging.getLogger("test"))


def test_report_aborts_when_startup_was_not_excluded(tmp_path, monkeypatch):
    ts = _write_run(
        tmp_path, monkeypatch, [_modal_inference(startup_excluded=False)],
        [_leg("glm_ocr", "easy", "invoice", "a.png", "text", 0.9)],
    )
    with pytest.raises(ValueError, match="Modal timing/cost audit failed"):
        run_benchmark.report(ts, logging.getLogger("test"))


def test_report_runs_when_modal_records_are_consistent(tmp_path, monkeypatch):
    ts = _write_run(
        tmp_path, monkeypatch, [_modal_inference()],
        [_leg("glm_ocr", "easy", "invoice", "a.png", "text", 0.9)],
    )
    run_benchmark.report(ts, logging.getLogger("test"), legs=["text"])
    assert (tmp_path / "summaries" / f"{ts}_per_doc.csv").exists()
    assert (tmp_path / "summaries" / f"{ts}_engines.csv").exists()


def test_diagnose_names_the_failure_from_measured_metrics():
    assert "repetition" in run_benchmark._diagnose({
        "leg": "text", "metrics": {"repetition_suspected": True, "length_ratio": 12.0},
    })
    # Field verdicts replaced the old parse/shape checks; schema
    # failures now report invented values or fall through to "fields missing
    # or wrong".
    assert "invented" in run_benchmark._diagnose({
        "leg": "schema", "metrics": {"schema_fields_invented_value": 3},
    })
    assert "fields missing or wrong" in run_benchmark._diagnose({
        "leg": "schema", "metrics": {},
    })
    assert "reading order" in run_benchmark._diagnose({
        "leg": "markdown", "metrics": {"element_recall": 0.9, "order_score": 0.2},
    })
