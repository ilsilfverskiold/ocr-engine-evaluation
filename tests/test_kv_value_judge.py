"""Legacy cache protocol and shared journal/budget compatibility. Current evaluation is tested in test_kv_value_groups.py."""
import json
from types import SimpleNamespace

import pytest

from evaluation import kv_value_judge as j
from evaluation.ground_truth import GroundTruth
from evaluation.keyvalue_leg import score_key_value
from evaluation.schema_leg import score_schema


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(j, 'CACHE_ROOT', tmp_path)
    monkeypatch.setattr(j, '_reserved_usd', 0)


def gt(values):
    return GroundTruth('test.png', 'easy', 'invoice', '', true_json=values,
        json_schema={'type': 'object', 'properties': {k: {'type': ['string', 'null']} for k in values}},
        metadata={'structured_gold_reference': {'shared_field_selection': True}})


def judge(monkeypatch, decide):
    monkeypatch.setenv('GOOGLE_API_KEY', 'offline-fixture')
    calls = []
    def ask(request):
        calls.append(request)
        return j.validate(request, decide(request)), .001
    monkeypatch.setattr(j, '_ask', ask)
    return calls


def test_request_validation_and_repeated_pairs():
    ref = gt({'Total': '12.34'})
    _, requests, local = j.requests_for(ref, [('Amount', '12.34'), ('Amount', '12.34'), ('Bad', '1234')])
    assert len(requests) == 1 and len(requests[0]['output_pairs']) == 1 and len(local) == 1
    assert '12.34' in requests[0]['prompt']
    with pytest.raises(ValueError): j.validate(requests[0], {})
    with pytest.raises(ValueError): j.decode_reply('{"1":{},"1":{}}')



def test_sdk_attempt_is_journaled_and_cached_without_repaying(monkeypatch):
    from google import genai
    monkeypatch.setenv('GOOGLE_API_KEY', 'offline-fixture')
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text='{"1":{"reference_id":1,"synonym":true}}', usage_metadata=None,
                               model_dump=lambda **kwargs: {'fixture': True})
    monkeypatch.setattr(genai, 'Client', lambda **kw: SimpleNamespace(
        models=SimpleNamespace(generate_content=generate), close=lambda: None))
    ref = gt({'Total': '100'})
    _, requests, _ = j.requests_for(ref, [('Amount', '100')])
    assert j._ask(requests[0])[0][0]['reference_key'] == 'Total'
    assert j._ask(requests[0])[1] == 0
    assert len(calls) == 1



def test_budget_refuses_before_sdk_submission(monkeypatch):
    monkeypatch.setenv('OCR_KV_JUDGE_BUDGET_USD', '0.000001')
    _, requests, _ = j.requests_for(gt({'Total': '100'}), [('Amount', '100')])
    with pytest.raises(RuntimeError, match='exceed'):
        j._ask(requests[0])
    assert not list(j.CACHE_ROOT.glob('calls/*.json'))



def test_later_batches_are_replanned_after_an_alias_is_learned(monkeypatch):
    monkeypatch.setattr(j, 'BATCH_SIZE', 1)
    calls = judge(monkeypatch, lambda r: {'1': {'reference_id': 1, 'synonym': True}})
    ref = gt({'Total': '100', 'Subtotal': '900'})
    result = j.resolve(ref, [('Amount due', '100'), ('Amount due', '900')])
    assert len(calls) == 1 and result.synonyms['Amount due'] == 'Total'



def test_saved_reply_recovers_without_repaying_after_batch_changes(monkeypatch):
    ref = gt({'Total': '100', 'Date': '2026-10-06'})
    _, requests, _ = j.requests_for(ref, [('Amount', '100'), ('Issued', '2026-10-06')])
    path = j.CACHE_ROOT / 'calls' / (requests[0]['id'] + '.json')
    path.parent.mkdir()
    path.write_text(json.dumps({'request': requests[0], 'error': 'old strict validation',
        'raw_text': '{"1":{"reference_id":2,"synonym":true},"2":{"reference_id":2,"synonym":true}}'}))
    calls = judge(monkeypatch, lambda r: pytest.fail('already paid for this pair'))
    # Only one of the two original pairs now appears: the request hash differs.
    result = score_key_value('{"Issued":"2026-10-06"}', ref, 'x')
    assert result.score == .5 and result.judge_cost_usd == 0 and not calls
    assert j._read(ref)['decisions'][0]['rejected_value_proposal'] == 'Date'
