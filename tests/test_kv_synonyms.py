import json

import pytest

from evaluation.ground_truth import GroundTruth
from evaluation.kv_synonyms import prepare_page, validate_reply, freeze_page
from evaluation import shared_fields


def reference():
    return GroundTruth("x.png", "easy", "invoice", "", true_json={"Total": "SECRET_VALUE", "Date": "SECRET_DATE"},
                       json_schema={"type": "object", "properties": {"Total": {"type": "string"}, "Date": {"type": "string"}}},
                       metadata={"structured_gold_reference": {"shared_field_selection": True}})


def test_plan_is_key_only_deduplicated_and_deterministic():
    gt = reference()
    page = prepare_page(gt, ["Amount due", "Amount due", "Total", "Bill date"])
    assert page == prepare_page(gt, ["Bill date", "Total", "Amount due"])
    assert page["unresolved_keys"] == 2
    request, = page["requests"]
    assert "SECRET" not in json.dumps(request)
    assert request["output_keys"] == ["Amount due", "Bill date"]


def test_freeze_null_decisions_and_reference_invalidation(tmp_path, monkeypatch):
    gt = reference()
    page = prepare_page(gt, ["Amount due", "Unknown"])
    request, = page["requests"]
    reply = {"matches": [{"output_id": 1, "reference_id": 1, "reason": "same meaning"},
                         {"output_id": 2, "reference_id": 0, "reason": "ambiguous"}]}
    freeze_page(page, {request["id"]: {"request_id": request["id"], "reply": reply}}, tmp_path)
    monkeypatch.setattr(shared_fields, "CACHE_ROOT", tmp_path)
    assert shared_fields.load_synonyms(gt) == {"Amount due": "Total"}
    gt.true_json["Total"] = "edited"
    assert shared_fields.load_synonyms(gt) == {}


@pytest.mark.parametrize("rows", [[], [{"output_id": 2, "reference_id": 1, "reason": "bad"}],
    [{"output_id": 1, "reference_id": 99, "reason": "bad"}],
    [{"output_id": True, "reference_id": 1, "reason": "bad"}],
    [{"output_id": 1, "reference_id": 1, "reason": "a"}, {"output_id": 1, "reference_id": 2, "reason": "b"}]])
def test_incomplete_duplicate_or_invented_ids_rejected(rows):
    request, = prepare_page(reference(), ["Unknown"])["requests"]
    with pytest.raises(ValueError):
        validate_reply(request, {"matches": rows})


def test_no_partial_cache_published(tmp_path):
    page = prepare_page(reference(), ["A", "B"], chunk_size=1)
    with pytest.raises(KeyError):
        freeze_page(page, {}, tmp_path)
    assert not list(tmp_path.glob("*.json"))


def test_execution_keeps_failed_paid_request_and_does_not_retry(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from google import genai
    from scripts import build_kv_synonyms as cli
    from evaluation import kv_synonyms as ks
    gt = reference()
    page = prepare_page(gt, ["Amount due"])
    plan = {"version": ks.VERSION, "model": ks.MODEL, "pricing": ks.PRICES,
            "pages": [page], "planning_reserve_usd": .10}
    plan["id"] = ks.digest(plan)
    monkeypatch.setattr(cli, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(cli, "load_corpus", lambda: [gt])
    monkeypatch.setenv("GOOGLE_API_KEY", "offline-test")
    attempts = []
    def fail(**kwargs):
        attempts.append(kwargs)
        raise RuntimeError("network outcome unknown")
    monkeypatch.setattr(genai, "Client", lambda **kw: SimpleNamespace(models=SimpleNamespace(generate_content=fail)))
    with pytest.raises(RuntimeError):
        cli.execute(plan, 1)
    assert len(attempts) == 1
    assert not (tmp_path / "x.json").exists()
    with pytest.raises(ValueError, match="Unfinished paid request"):
        cli.execute(plan, 1)
    assert len(attempts) == 1
    with pytest.raises(ValueError, match="Budget below"):
        cli.execute(plan, .01)
    gt.true_json["Total"] = "changed"
    with pytest.raises(ValueError, match="Reference changed"):
        cli.execute(plan, 1)


def test_successful_execution_reuses_saved_reply(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from google import genai
    from scripts import build_kv_synonyms as cli
    from evaluation import kv_synonyms as ks
    gt = reference()
    page = prepare_page(gt, ["Amount due"])
    plan = {"version": ks.VERSION, "model": ks.MODEL, "pricing": ks.PRICES,
            "pages": [page], "planning_reserve_usd": .10}
    plan["id"] = ks.digest(plan)
    monkeypatch.setattr(cli, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(cli, "load_corpus", lambda: [gt])
    monkeypatch.setenv("GOOGLE_API_KEY", "offline-test")
    attempts = []
    reply = {"matches": [{"output_id": 1, "reference_id": 1, "reason": "same field"}]}
    def success(**kwargs):
        attempts.append(kwargs)
        return SimpleNamespace(text=json.dumps(reply), usage_metadata=None,
                               model_dump=lambda **kwargs: {"fixture": True})
    monkeypatch.setattr(genai, "Client", lambda **kw: SimpleNamespace(models=SimpleNamespace(generate_content=success)))
    cli.execute(plan, 1)
    cli.execute(plan, 1)
    assert len(attempts) == 1
    assert not (tmp_path / "x.json").exists()
    record = json.loads((tmp_path / "calls" / plan["id"] / (page["requests"][0]["id"] + ".json")).read_text())
    assert record["reply"]["matches"][0]["reference_id"] == 1
    assert "SECRET" not in attempts[0]["contents"]


def test_invalid_reply_does_not_skip_other_approved_requests_or_get_retried(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from google import genai
    from scripts import build_kv_synonyms as cli
    from evaluation import kv_synonyms as ks
    gt = reference()
    page = prepare_page(gt, ["A bad", "B good"], chunk_size=1)
    plan = {"version": ks.VERSION, "model": ks.MODEL, "pricing": ks.PRICES,
            "pages": [page], "planning_reserve_usd": .10}
    plan["id"] = ks.digest(plan)
    monkeypatch.setattr(cli, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(cli, "load_corpus", lambda: [gt])
    monkeypatch.setenv("GOOGLE_API_KEY", "offline-test")
    attempts = []
    def response(**kwargs):
        attempts.append(kwargs)
        text = '{"matches":[' if '"key": "A bad"' in kwargs['contents'] else json.dumps({
            "matches": [{"output_id": 1, "reference_id": 1, "reason": "same field"}]})
        return SimpleNamespace(text=text, usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=10),
                               model_dump=lambda **kwargs: {"fixture": True})
    monkeypatch.setattr(genai, "Client", lambda **kw: SimpleNamespace(models=SimpleNamespace(generate_content=response)))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="1 requests need review"):
            cli.execute(plan, 1)
    assert len(attempts) == 2  # Both approved requests once, zero retries.
    assert not (tmp_path / "x.json").exists()
    summary = json.loads((tmp_path / "calls" / plan["id"] / "execution.json").read_text())
    assert summary["valid_requests"] == 1 and len(summary["failures"]) == 1
    assert summary["actual_saved_cost_usd"] == pytest.approx(.000056)


def control_plan():
    from evaluation import kv_synonyms as ks
    source = {"pages": [prepare_page(reference(), ["Amount due", "Unknown"])]}
    source["id"] = ks.digest(source)
    return ks.prepare_controls(source, {"x": {"Amount due": "Total", "Unknown": None}})


def test_controls_keep_answers_and_values_out_of_request():
    from evaluation import kv_synonyms as ks
    from google.genai import types
    plan = control_plan()
    request = plan["pages"][0]["requests"][0]
    assert "expected_matches" not in json.dumps(request)
    assert "SECRET" not in json.dumps(request)
    assert request["config"]["thinking_budget"] == 1024
    assert request["reply_format"] == ks.COMPACT_FORMAT
    # Validate schema through the installed SDK without a provider call.
    config = types.GenerateContentConfig(response_schema=request["response_schema"])
    assert config.response_schema is not None
    assert validate_reply(request, {"1": 1, "2": 0})[0]["reference_key"] == "Total"


@pytest.mark.parametrize("reply", [{"1": 1}, {"1": 1, "2": 0, "0": 0},
                                    {"1": True, "2": 0}, {"1": 3, "2": 0},
                                    {"1": -1, "2": 0}, {"1": "1", "2": 0}])
def test_compact_replies_reject_missing_extra_and_invalid_ids(reply):
    request = control_plan()["pages"][0]["requests"][0]
    with pytest.raises(ValueError):
        validate_reply(request, reply)


def test_duplicate_json_properties_cannot_silently_change_decisions():
    from evaluation import kv_synonyms as ks
    with pytest.raises(ValueError, match="Duplicate JSON"):
        ks.decode_reply('{"1": 1, "1": 0, "2": 0}')


def test_controls_distinguish_wrong_missing_and_unmatched():
    from evaluation import kv_synonyms as ks
    plan = control_plan()
    request = plan["pages"][0]["requests"][0]
    missing = ks.grade_controls(plan, {})
    assert missing["correct"] == 0 and missing["missing_decisions"] == 2
    assert not missing["passed"]
    wrong = ks.grade_controls(plan, {request["id"]: {"reply": {"1": 0, "2": 1}}})
    assert wrong["false_matches"] == 1 and wrong["missed_matches"] == 1
    right = ks.grade_controls(plan, {request["id"]: {"reply": {"1": 1, "2": 0}}})
    assert right["passed"] and right["correct"] == 2


def test_control_subset_cannot_be_frozen_as_full_page_cache(tmp_path):
    page = control_plan()["pages"][0]
    with pytest.raises(ValueError, match="Control cases"):
        freeze_page(page, {}, tmp_path)
    assert not list(tmp_path.iterdir())


def test_controls_reject_invented_keys_and_one_sided_cases():
    from evaluation import kv_synonyms as ks
    source = {"pages": [prepare_page(reference(), ["Amount due", "Unknown"])]}
    source["id"] = ks.digest(source)
    for cases in ({"Invented key": "Total", "Unknown": None},
                  {"Amount due": "Absent reference", "Unknown": None},
                  {"Amount due": "Total"}, {"Unknown": None}):
        with pytest.raises(ValueError):
            ks.prepare_controls(source, {"x": cases})


@pytest.mark.parametrize("correct", [False, True])
def test_control_execution_grades_without_publishing(tmp_path, monkeypatch, correct):
    from types import SimpleNamespace
    from google import genai
    from scripts import build_kv_synonyms as cli
    plan = control_plan()
    monkeypatch.setattr(cli, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(cli, "load_corpus", lambda: [reference()])
    monkeypatch.setenv("GOOGLE_API_KEY", "offline-test")
    attempts = []
    def response(**kwargs):
        attempts.append(kwargs)
        return SimpleNamespace(text=json.dumps({"1": 1, "2": 0 if correct else 1}), usage_metadata=None,
                               model_dump=lambda **kwargs: {"fixture": True})
    monkeypatch.setattr(genai, "Client", lambda **kw: SimpleNamespace(models=SimpleNamespace(generate_content=response)))
    if correct:
        cli.execute(plan, 1)
    else:
        with pytest.raises(RuntimeError, match="controls failed"):
            cli.execute(plan, 1)
    assert len(attempts) == 1
    assert not list(tmp_path.glob("*.json"))
    summary = json.loads((tmp_path / "calls" / plan["id"] / "execution.json").read_text())
    assert summary["controls"]["passed"] == correct
    assert summary["cache_status"] == "held_for_semantic_review"
