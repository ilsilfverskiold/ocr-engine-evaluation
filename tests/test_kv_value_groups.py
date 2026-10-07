import json
import pytest

from evaluation import kv_value_groups as groups, kv_value_judge as legacy
from evaluation.ground_truth import GroundTruth
from evaluation.keyvalue_leg import score_key_value


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(legacy, 'CACHE_ROOT', tmp_path)
    monkeypatch.setattr(legacy, '_reserved_usd', 0)


def gt(values):
    return GroundTruth('test.png', 'easy', 'invoice', '', true_json=values,
                       json_schema={'type': 'object', 'properties': {k: {'type': 'string'} for k in values}},
                       metadata={'structured_gold_reference': {'shared_field_selection': True}})


def fake_judge(monkeypatch, decide):
    monkeypatch.setenv('GOOGLE_API_KEY', 'offline-test')
    calls = []
    def ask(request):
        calls.append(request)
        return groups.validate(request, decide(request)), .001
    monkeypatch.setattr(legacy, '_ask', ask)
    return calls


def verdict(ids, synonym=False, reason='correct group'):
    return {'output_ids': ids, 'synonym': synonym, 'reason': reason}


def test_partial_values_generate_judge_items_without_automatic_credit():
    ref = gt({'Seller': 'Example Ltd 10 Main Street'})
    pairs = [('Seller', ''), ('organisation', 'Example Ltd'), ('delivery location', '10 Main Street')]
    _, _, _, approved, requests = groups.prepare(ref, pairs)
    assert not approved and len(requests) == 1
    assert requests[0]['review_items'][0]['candidates'] == [
        {'output_id': 2, 'overlap': 'output_fragment'}, {'output_id': 3, 'overlap': 'output_fragment'}]
    assert requests[0]['returned_pairs'][0]['value'] == ''
    result = score_key_value(json.dumps(dict(pairs)), ref, 'any-engine', use_judge=False)
    assert result.score is None and result.pending_key_review == ['Seller']
    assert result.pairs_correct == 0


def test_judge_can_confirm_split_values_and_cache_exact_context(monkeypatch):
    calls = fake_judge(monkeypatch, lambda r: {'1': verdict([2, 3])})
    ref = gt({'Party': 'Example Ltd 10 Main Street'})
    text = json.dumps({'Party': '', 'organisation': 'Example Ltd', 'location': '10 Main Street'})
    result = score_key_value(text, ref, 'a')
    assert result.score == 1 and result.judge_cost_usd == .001
    assert result.verdicts[0].key == 'organisation + location'
    assert 'judge-confirmed' in result.verdicts[0].how
    assert score_key_value(text, ref, 'b', use_judge=False).score == 1 and len(calls) == 1
    assert not legacy._read(ref)['decisions']  # Group is not a fabricated synonym.


def test_partial_parent_value_also_goes_to_judge(monkeypatch):
    calls = fake_judge(monkeypatch, lambda r: {'1': verdict([1,2])})
    ref = gt({'Party':'Example Ltd Main Street'})
    text = '{"Party":"Example Ltd","Location":"Main Street"}'
    assert score_key_value(text,ref,'x',use_judge=False).score is None
    assert score_key_value(text,ref,'x').score == 1
    assert calls[0]['review_items'][0]['repair_output_id'] == 1


def test_wrong_owner_group_rejected_and_cached(monkeypatch):
    calls = fake_judge(monkeypatch, lambda r: {'1': verdict([], reason='wrong owner')})
    ref = gt({'Seller': 'Example Ltd 10 Main Street'})
    text = json.dumps({'Buyer name': 'Example Ltd', 'Buyer address': '10 Main Street'})
    first = score_key_value(text, ref, 'x')
    assert first.score == 0 and first.judge_reviews[0]['reason'] == 'wrong owner'
    assert score_key_value(text, ref, 'y').score == 0 and len(calls) == 1


def test_old_rejections_do_not_suppress_changed_review_items():
    ref = gt({'Party': 'Example Ltd Main Street', 'Item — Net': '8.50', 'Summary — Net': '8.50'})
    record = legacy._read(ref)
    record['decisions'] = [
        {'key': 'Name', 'value': 'Example Ltd', 'reference_key': None, 'synonym': False,
         'reason': 'no complete reference value matches', 'request_id': None},
        {'key': 'Net', 'value': '8.50', 'reference_key': None, 'synonym': False, 'request_id': 'old-paid'}]
    legacy._save(ref, record)
    pairs = [('Name', 'Example Ltd'), ('Address', 'Main Street'), ('Net', '8.50')]
    request = groups.prepare(ref, pairs)[-1][0]
    assert len(request['review_items']) == 3
    assert len(request['returned_pairs']) == 3
    assert score_key_value(json.dumps(dict(pairs)), ref, 'x', use_judge=False).score is None


def test_new_context_invalidates_group_rejection(monkeypatch):
    fake_judge(monkeypatch, lambda r: {'1': verdict([], reason='ambiguous')})
    ref = gt({'Party': 'Example Ltd Main Street'})
    text = json.dumps({'Name': 'Example Ltd', 'Address': 'Main Street'})
    assert score_key_value(text, ref, 'x').score == 0
    assert score_key_value(json.dumps({'Seller Name': 'Example Ltd', 'Seller Address': 'Main Street'}),
                           ref, 'x', use_judge=False).score is None


def test_judge_cannot_approve_incomplete_group(monkeypatch):
    fake_judge(monkeypatch, lambda r: {'1': verdict([1])})
    ref = gt({'Party': 'Example Ltd Main Street'})
    result = score_key_value('{"Name":"Example Ltd"}', ref, 'x')
    assert result.score == 0
    assert result.judge_reviews[0]['validation_error'] == 'incomplete value coverage'


def test_one_pair_cannot_be_credited_to_two_repeated_fields(monkeypatch):
    fake_judge(monkeypatch, lambda r: {'1': verdict([1]), '2': verdict([1])})
    result = score_key_value('{"Net":"8.50"}', gt({'Item — Net':'8.50','Summary — Net':'8.50'}), 'x')
    assert result.score == .5
    assert result.judge_reviews[1]['validation_error'] == 'output reused'


def test_group_candidates_do_not_bypass_nonempty_known_wrong_answer():
    ref = gt({'Seller': 'Example Ltd Main Street'})
    pairs = [('Seller', 'Wrong Company'), ('Name', 'Example Ltd'), ('Address', 'Main Street')]
    assert not groups.prepare(ref, pairs)[-1]
    assert score_key_value(json.dumps(dict(pairs)), ref, 'x', use_judge=False).score == 0


def test_single_safe_wording_alias_reused_with_new_context_and_values_checked(monkeypatch):
    calls = fake_judge(monkeypatch, lambda r: {'1': verdict([1], synonym=True)})
    ref = gt({'Total': '8.50'})
    assert score_key_value('{"Amount due":"8.50"}', ref, 'a').score == 1
    assert score_key_value('{"Amount due":"8.50","Unused":"text"}', ref, 'b').score == 1
    assert score_key_value('{"Amount due":"900"}', ref, 'c').score == 0
    assert len(calls) == 1


@pytest.mark.parametrize('value', ['$ 8,50', '€8.50', '8,50 €', '8.50$'])
def test_currency_forms_generate_complete_candidates(value):
    assert groups.relation('8.50', value, 'Total') == 'complete'


@pytest.mark.parametrize('expected,value', [('100','1000'), ('12.34','1234'), ('-100','100'),
                                           ('100','total 1000 dollars'), ('0012','12')])
def test_containment_never_uses_numeric_substrings(expected, value):
    assert groups.relation(expected, value, 'Identifier' if expected == '0012' else 'Total') is None


def test_reference_inside_longer_value_still_needs_judgment(monkeypatch):
    ref = gt({'Street': '10 Main Street'})
    text = '{"Address":"Example Ltd 10 Main Street London"}'
    assert score_key_value(text, ref, 'x', use_judge=False).score is None
    fake_judge(monkeypatch, lambda r: {'1': verdict([1])})
    assert score_key_value(text, ref, 'x').score == 1


def test_parallel_engines_share_one_context_review(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    calls = fake_judge(monkeypatch, lambda r: {'1': verdict([1, 2])})
    ref = gt({'Party': 'Example Ltd Main Street'})
    with ThreadPoolExecutor(max_workers=4) as pool:
        scores = list(pool.map(lambda e: score_key_value('{"Name":"Example Ltd","Address":"Main Street"}', ref, e).score,
                               ['a','b','c','d']))
    assert scores == [1] * 4 and len(calls) == 1


def test_batches_replan_with_claimed_outputs_and_never_call_twice(monkeypatch):
    monkeypatch.setattr(groups, 'BATCH_SIZE', 1)
    calls = fake_judge(monkeypatch, lambda r: {str(r['review_items'][0]['id']): verdict([1])})
    ref = gt({'Item — Net':'8.50','Summary — Net':'8.50'})
    assert score_key_value('{"Net":"8.50"}', ref, 'x').score == .5
    assert len(calls) == 1  # Second field has no unused candidate after first approval.


def test_context_sdk_journal_recovered_without_repaying(monkeypatch):
    from google import genai
    from types import SimpleNamespace
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=json.dumps({'1':verdict([1,2])}), usage_metadata=None,
                               model_dump=lambda **kw: {'test': True})
    monkeypatch.setenv('GOOGLE_API_KEY', 'offline-test')
    monkeypatch.setattr(genai, 'Client', lambda **kw: SimpleNamespace(
        models=SimpleNamespace(generate_content=generate), close=lambda: None))
    ref = gt({'Party':'Example Ltd Main Street'})
    pairs = [('Name','Example Ltd'),('Address','Main Street')]
    request = groups.prepare(ref,pairs)[-1][0]
    assert legacy._ask(request)[0][0]['indices'] == [0,1]
    # Simulate interruption after response journal, before context-cache save.
    assert groups.resolve(ref,pairs,use_judge=False).groups[0]['indices'] == [0,1]
    assert len(calls) == 1


def test_malformed_response_stays_pending_without_retry(monkeypatch):
    calls = fake_judge(monkeypatch, lambda r: {'1':verdict([999])})
    result = score_key_value('{"Name":"Example Ltd"}',gt({'Party':'Example Ltd Main Street'}),'x')
    assert result.score is None and result.judge_errors and len(calls) == 1


def test_ineligible_existing_component_rejects_only_its_field(monkeypatch):
    # Native OCR sometimes puts address text in a key with an empty value.
    # The judge cannot use that empty value as a component, but its unrelated
    # valid verdict must survive and must not require another paid request.
    calls = fake_judge(monkeypatch, lambda r: {'1': verdict([1,2]), '2': verdict([3])})
    ref = gt({'Address':'Main Street London', 'Amount due':'8.50'})
    text = '{"Address line":"Main Street","London":"","Total":"8.50"}'
    result = score_key_value(text,ref,'x')
    assert result.score == .5
    assert result.judge_reviews[0]['validation_error'] == 'selected output is not a value candidate'
    assert result.judge_reviews[1]['indices'] == [2]
    assert score_key_value(text,ref,'x',use_judge=False).score == .5
    assert len(calls) == 1


def test_journal_recovery_reuses_paid_fields_when_aliases_change_batch(monkeypatch):
    ref = gt({'Party':'Example Ltd Main Street', 'Total':'8.50'})
    pairs = [('Name','Example Ltd'),('Address','Main Street'),('Amount due','8.50')]
    request = groups.prepare(ref,pairs)[-1][0]
    saved = {'request':request,'raw_text':json.dumps({'1':verdict([1,2]),'2':verdict([3],True)})}
    calls_dir=legacy.CACHE_ROOT/'calls';calls_dir.mkdir()
    (calls_dir/(request['id']+'.json')).write_text(json.dumps(saved))
    # A later alias changes batching but must not cause another paid call.
    record=legacy._read(ref)
    record['decisions'].append({'key':'Amount due','value':'8.50','reference_key':'Total',
                                'synonym':True,'request_id':'earlier-valid-alias'})
    legacy._save(ref,record)
    assert len(groups.prepare(ref,pairs)[-1][0]['review_items'])==1
    monkeypatch.setattr(legacy,'_ask',lambda r: pytest.fail('must reuse journal without calling'))
    result=score_key_value(json.dumps(dict(pairs)),ref,'x',use_judge=False)
    assert result.score==1 and not result.pending_key_review


def test_gold_edit_invalidates_context_review(monkeypatch):
    fake_judge(monkeypatch,lambda r:{'1':verdict([1,2])})
    ref = gt({'Party':'Example Ltd Main Street'})
    text = '{"Name":"Example Ltd","Address":"Main Street"}'
    assert score_key_value(text,ref,'x').score == 1
    ref.true_json['Party'] = 'Example Ltd New Street'
    assert score_key_value(text,ref,'x',use_judge=False).score is None
