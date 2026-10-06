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
    record = json.loads((tmp_path / "x.json").read_text())
    assert record["matches"][0]["reference_key"] == "Total"
    assert "SECRET" not in attempts[0]["contents"]
