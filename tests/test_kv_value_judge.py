"""Shared KV evaluator: spend only on needed decisions, then reuse confirmations."""
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


def test_known_keys_and_clearly_wrong_values_never_call_judge(monkeypatch):
    calls = judge(monkeypatch, lambda r: pytest.fail('unnecessary judge'))
    ref = gt({'Total': '100'})
    assert score_key_value('{"Total":"100"}', ref, 'x').score == 1
    assert score_key_value('{"Total":"1000"}', ref, 'x').score == 0
    assert score_key_value('{"Amount due":"1000"}', ref, 'x').score == 0
    assert not calls


def test_confirmation_cached_for_next_engine_and_values_still_checked(monkeypatch):
    calls = judge(monkeypatch, lambda r: {'1': {'reference_id': 1, 'synonym': True}})
    ref = gt({'Total': '100'})
    first = score_key_value('{"Amount due":"100"}', ref, 'engine_a')
    assert first.score == 1 and first.judge_cost_usd == .001
    assert score_key_value('{"Amount due":"100"}', ref, 'engine_b').score == 1
    assert score_key_value('{"Amount due":"900"}', ref, 'engine_c').score == 0
    assert len(calls) == 1
    assert score_schema('{"Amount due":"100"}', ref, 'engine_a').score == 0


def test_value_context_does_not_promote_generic_key_to_global_alias(monkeypatch):
    calls = judge(monkeypatch, lambda r: {'1': {'reference_id': r['output_pairs'][0]['candidate_ids'][0], 'synonym': False}})
    ref = gt({'Subtotal': '100', 'Total': '200'})
    assert score_key_value('{"Amount":"100"}', ref, 'a').score == .5
    assert score_key_value('{"Amount":"200"}', ref, 'b').score == .5
    assert len(calls) == 2
    result = score_key_value('{"Amount":"100"}', ref, 'c', use_judge=False)
    assert result.score == .5 and len(calls) == 2


def test_repeated_value_needs_judgment_and_rejection_is_cached(monkeypatch):
    calls = judge(monkeypatch, lambda r: {'1': {'reference_id': 0, 'synonym': False}})
    ref = gt({'Subtotal': '100', 'Total': '100'})
    result = score_key_value('{"Amount":"100"}', ref, 'a')
    assert result.score == 0 and calls[0]['output_pairs'][0]['candidate_ids'] == [1, 2]
    assert score_key_value('{"Amount":"100"}', ref, 'b').score == 0
    assert len(calls) == 1


def test_matching_value_never_bypasses_contradictory_known_key(monkeypatch):
    calls = judge(monkeypatch, lambda r: pytest.fail('must not remap a matched key'))
    ref = gt({'Subtotal': '100', 'Total': '900'})
    result = score_key_value('{"Subtotal":"900","Total":"100"}', ref, 'a')
    assert result.score == 0 and not calls


def test_skip_judge_leaves_only_plausible_unmatched_pairs_pending(monkeypatch):
    calls = judge(monkeypatch, lambda r: pytest.fail('skip judge'))
    ref = gt({'Total': '100'})
    result = score_key_value('{"Amount":"100","Unknown":"1000"}', ref, 'x', use_judge=False)
    assert result.score is None and result.pending_key_review == ['Amount'] and not calls


def test_reference_edit_invalidates_value_dependent_cache(monkeypatch):
    calls = judge(monkeypatch, lambda r: {'1': {'reference_id': 1, 'synonym': True}})
    ref = gt({'Total': '100'})
    assert score_key_value('{"Amount":"100"}', ref, 'x').score == 1
    ref.true_json['Total'] = '200'
    assert score_key_value('{"Amount":"200"}', ref, 'x').score == 1
    assert len(calls) == 2


def test_invalid_and_incomplete_replies_do_not_activate_cache(monkeypatch):
    calls = judge(monkeypatch, lambda r: {'1': {'reference_id': 999, 'synonym': True}})
    ref = gt({'Total': '100', 'Date': '2026-10-06'})
    result = score_key_value('{"Amount":"100"}', ref, 'x')
    assert result.score is None and result.judge_errors and len(calls) == 1
    assert not j._read(ref)['decisions']


def test_wrong_value_proposal_is_rejected_without_losing_other_decisions(monkeypatch):
    calls = judge(monkeypatch, lambda r: {'1': {'reference_id': 2, 'synonym': True},
                                         '2': {'reference_id': 2, 'synonym': True}})
    ref = gt({'Total': '100', 'Date': '2026-10-06'})
    result = score_key_value('{"Amount":"100","Issued":"2026-10-06"}', ref, 'x')
    assert result.score == .5 and not result.judge_errors and len(calls) == 1
    entries = j._read(ref)['decisions']
    assert entries[0]['reference_key'] is None and entries[0]['rejected_value_proposal'] == 'Date'
    assert entries[1]['reference_key'] == 'Date'


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


def test_parallel_engines_share_one_page_confirmation(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    calls = judge(monkeypatch, lambda r: {'1': {'reference_id': 1, 'synonym': True}})
    ref = gt({'Total': '100'})
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda engine: score_key_value('{"Amount due":"100"}', ref, engine), ['a','b','c','d']))
    assert all(r.score == 1 for r in results) and len(calls) == 1


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


def test_repeated_reference_labels_cannot_become_unconditional_aliases(monkeypatch):
    calls = judge(monkeypatch, lambda r: {'1': {'reference_id': r['output_pairs'][0]['candidate_ids'][0], 'synonym': True}})
    ref = gt({'Shipment 1 — Weight': '100', 'Shipment 2 — Weight': '200'})
    assert score_key_value('{"Weight":"100"}', ref, 'a').score == .5
    assert score_key_value('{"Weight":"200"}', ref, 'b').score == .5
    assert len(calls) == 2
    assert not any(d['synonym'] for d in j._read(ref)['decisions'])
