"""Human structured gold stays separate from table gold and reaches both field legs."""
import hashlib
import json
from pathlib import Path

import pytest

from evaluation.ground_truth import GroundTruthLoader
from evaluation.keyvalue_leg import score_key_value
from evaluation.schema_leg import score_schema
from evaluation.structured_references import field_patterns, validate_structured_record
from scripts.structured_reference_editor import StructuredGoldStore
from scripts.table_reference_editor import Conflict


@pytest.fixture
def reference(tmp_path):
    doc = tmp_path / 'documents/easy/sample/page.png'
    doc.parent.mkdir(parents=True)
    doc.write_bytes(b'source image')
    doc.with_name('page_gt.json').write_text(json.dumps({'plain_text': 'Original text', 'table_html': '<table><tr><td>original table</td></tr></table>'}))
    schema = {'type': 'object', 'properties': {k: {'type': ['string', 'null']} for k in ['name', 'total', 'note']}}
    doc.with_name('page_structured_gt.json').write_text(json.dumps({'true_json': {'name': 'Original', 'total': '10', 'note': None}, 'json_schema': schema}))
    versions = tmp_path / 'documents/structured_references'
    (versions / 'old').mkdir(parents=True)
    (versions / 'manifest.json').write_text(json.dumps({'dataset_id': 'fixture', 'pages': ['page']}))
    image_hash = hashlib.sha256(doc.read_bytes()).hexdigest()
    (versions / 'old/page.json').write_text(json.dumps({'image_sha256': image_hash}))
    record = {'version': 1, 'reference_kind': 'structured', 'page': 'page', 'dataset_id': 'fixture', 'image_sha256': image_hash,
              'true_json': {'name': 'Reviewed', 'total': '20', 'note': None}, 'json_schema': schema,
              'kv_aliases': {'name': {'printed_labels': ['Customer'], 'is_printed_pair': True},
                             'total': {'printed_labels': ['Amount'], 'is_printed_pair': True},
                             'note': {'printed_labels': ['Note'], 'is_printed_pair': False}}, 'excluded_fields': []}
    return doc, versions, record


def test_gold_reaches_schema_and_kv_without_changing_table_gold(reference, monkeypatch):
    doc, versions, record = reference
    before = doc.with_name('page_structured_gt.json').read_bytes()
    table = doc.parents[2] / 'table_references/gold/page.json'
    table.parent.mkdir(parents=True)
    table.write_text(json.dumps({'version': 1, 'page': 'page', 'image_sha256': record['image_sha256'], 'table_html': '<table><tr><td>human table</td></tr></table>'}))
    table_before = table.read_bytes()
    (versions / 'patched').mkdir()
    (versions / 'patched/page.json').write_text(json.dumps(record))
    assert GroundTruthLoader().load(doc).true_json['name'] == 'Original'
    store = StructuredGoldStore(versions, versions / 'gold')
    store.save('page', record, None)
    gt = GroundTruthLoader().load(doc)
    assert gt.true_json['name'] == 'Reviewed'
    assert gt.plain_text == 'Original text'
    assert 'human table' in gt.table_html
    assert table.read_bytes() == table_before
    assert doc.with_name('page_structured_gt.json').read_bytes() == before
    monkeypatch.setattr('evaluation.kv_schema_code.load_aliases', lambda *_: pytest.fail('Gold KV labels must replace the cache'))
    assert gt.printed_pair_count() == 2
    assert score_schema(json.dumps(record['true_json']), gt, 'fixture').field_accuracy == 1
    kv = score_key_value('{"Customer":"Reviewed","Amount":"20"}', gt, 'fixture')
    assert kv.field_accuracy == 2 / 3
    assert kv.pairs_total == 3
    (versions / 'gold/page.json').unlink()
    assert GroundTruthLoader().load(doc).true_json['name'] == 'Original'


def test_structured_save_backup_conflict_and_image_identity(reference):
    doc, versions, record = reference
    store = StructuredGoldStore(versions, versions / 'gold')
    first = store.save('page', record, None)
    old_bytes = (store.gold / 'page.json').read_bytes()
    record['notes'] = 'Second review'
    store.save('page', record, first['revision'])
    assert next((store.gold / 'history').glob('*.json')).read_bytes() == old_bytes
    with pytest.raises(Conflict):
        store.save('page', record, first['revision'])
    doc.write_bytes(b'other page')
    with pytest.raises(ValueError, match='source image changed'):
        GroundTruthLoader().load(doc)


@pytest.mark.parametrize('change', [
    {'reference_kind': 'table'}, {'json_schema': {'type': 'not_a_type'}},
    {'kv_aliases': {'unknown': {'printed_labels': ['X'], 'is_printed_pair': True}}},
    {'excluded_fields': ['unknown']}, {'true_json': {'name': float('nan')}},
])
def test_invalid_gold_is_rejected(reference, change):
    _, versions, record = reference
    store = StructuredGoldStore(versions, versions / 'gold')
    with pytest.raises(ValueError):
        store.save('page', {**record, **change}, None)
    assert not (store.gold / 'page.json').exists()


def test_gold_exclusions_apply_to_both_legs(reference):
    doc, versions, record = reference
    record['excluded_fields'] = ['total']
    StructuredGoldStore(versions, versions / 'gold').save('page', record, None)
    gt = GroundTruthLoader().load(doc)
    assert score_schema('{"name":"Reviewed","total":"wrong"}', gt, 'fixture').field_accuracy == 1
    kv = score_key_value('{"Customer":"Reviewed","Amount":"wrong"}', gt, 'fixture')
    assert kv.pairs_total == 2
    assert kv.field_accuracy == .5


def test_unreviewed_gold_aliases_do_not_reuse_stale_cache(reference, monkeypatch):
    doc, versions, record = reference
    record['kv_aliases'] = None
    StructuredGoldStore(versions, versions / 'gold').save('page', record, None)
    monkeypatch.setattr('evaluation.kv_schema_code.load_aliases', lambda *_: pytest.fail('Do not reuse stale aliases'))
    gt = GroundTruthLoader().load(doc)
    assert gt.printed_pair_count() is None
    assert score_key_value('{}', gt, 'fixture', use_judge=False).score == 0


def test_nullable_parent_keeps_declared_fields_and_aliases(reference):
    _, _, record = reference
    record['json_schema']['properties']['spouse'] = {'type': ['object', 'null'], 'properties': {'name': {'type': ['string', 'null']}}}
    record['true_json']['spouse'] = None
    record['kv_aliases']['spouse.name'] = {'printed_labels': ['Spouse name'], 'is_printed_pair': True}
    validate_structured_record(record)
    from evaluation.structured_metrics import walk_fields
    fields = {f.path: f.value for f in walk_fields(record['true_json'], record['json_schema'])}
    assert fields['spouse.name'] is None
    assert 'spouse' not in fields


def test_new_schema_targets_recorded_from_server_baseline(reference):
    _, versions, record = reference
    original = json.loads(json.dumps(record))
    (versions / 'old/page.json').write_text(json.dumps(original))
    record['true_json']['account_id'] = '1234'
    record['json_schema']['properties']['account_id'] = {'type': 'string'}
    record['kv_aliases']['account_id'] = {'printed_labels': ['Account'], 'is_printed_pair': False}
    record['historical_reference_schema'] = {'properties': {'account_id': {}}}
    record['new_schema_field_patterns'] = []
    saved = StructuredGoldStore(versions, versions / 'gold').save('page', record, None)['record']
    assert saved['historical_reference_schema'] == original['json_schema']
    assert saved['new_schema_field_patterns'] == ['account_id']
    assert saved['kv_aliases']['account_id']['is_printed_pair'] is False
    assert record['new_schema_field_patterns'] == []  # caller is not mutated


def test_schema_scope_compares_columns_not_row_indices():
    original = {'true_json': {'items': []}, 'json_schema': {'type': 'object', 'properties': {
        'items': {'type': 'array', 'items': {'type': 'object', 'properties': {'name': {'type': 'string'}}}}}}}
    expanded = json.loads(json.dumps(original))
    expanded['true_json']['items'] = [{'name': 'A', 'unit': 'each'}, {'name': 'B', 'unit': 'box'}]
    expanded['json_schema']['properties']['items']['items']['properties']['unit'] = {'type': 'string'}
    assert field_patterns(expanded) - field_patterns(original) == {'items[*].unit'}


def test_legacy_kv_flags_do_not_remove_shared_array_fields(reference):
    doc, versions, record = reference
    record['true_json'] = {'contacts': [{'phone': '123456', 'extension': None}]}
    record['json_schema'] = {'type': 'object', 'properties': {'contacts': {
        'type': 'array', 'items': {'type': 'object', 'properties': {
            'phone': {'type': 'string'}, 'extension': {'type': ['string', 'null']}}}}}}
    record['kv_aliases'] = {
        'contacts[0].phone': {'printed_labels': ['Business phone'], 'is_printed_pair': True},
        'contacts[0].extension': {'printed_labels': ['Extension'], 'is_printed_pair': True}}
    store = StructuredGoldStore(versions, versions / 'gold')
    saved = store.save('page', record, None)
    gt = GroundTruthLoader().load(doc)
    assert gt.has_gt_for('key_value')
    assert gt.printed_pair_count() == 2
    result = score_key_value('{"Business phone":"123456","Extension":null}', gt, 'fixture')
    assert result.field_accuracy == 1
    assert result.pairs_total == 2
    wrong = score_key_value('{"Business phone":"999999","Extension":"42"}', gt, 'fixture')
    assert wrong.field_accuracy == 0
    assert wrong.pairs_invented_value == 1
    assert score_schema(json.dumps(record['true_json']), gt, 'fixture').field_accuracy == 1
    record['kv_aliases']['contacts[0].phone']['is_printed_pair'] = False
    saved = store.save('page', record, saved['revision'])
    gt = GroundTruthLoader().load(doc)
    assert gt.true_json['contacts'][0]['phone'] == '123456'
    assert gt.printed_pair_count() == 1
    assert score_key_value('{"Extension":null}', gt, 'fixture').field_accuracy == .5
    record['kv_aliases']['contacts[0].extension']['is_printed_pair'] = False
    store.save('page', record, saved['revision'])
    gt = GroundTruthLoader().load(doc)
    assert gt.printed_pair_count() == 0
    assert score_key_value('{"Extension":null}', gt, 'fixture').field_accuracy == .5


def test_human_selected_fields_are_not_sampled_out(reference):
    doc, versions, record = reference
    record['true_json'] = {f'field{i}': str(i) for i in range(85)}
    record['json_schema'] = {'type': 'object', 'properties': {
        k: {'type': 'string'} for k in record['true_json']}}
    record['kv_aliases'] = {k: {'printed_labels': [k], 'is_printed_pair': True}
                            for k in record['true_json']}
    record['shared_field_selection'] = True
    StructuredGoldStore(versions, versions / 'gold').save('page', record, None)
    result = score_key_value(json.dumps(record['true_json']), GroundTruthLoader().load(doc), 'fixture')
    assert result.pairs_total == 85
    assert result.field_accuracy == 1
    schema = score_schema(json.dumps(record['true_json']), GroundTruthLoader().load(doc), 'fixture')
    assert schema.fields_total == result.pairs_total == 85
    assert {v.path for v in schema.verdicts} == {v.path for v in result.verdicts}


def test_shared_reference_array_inclusion_is_identical_for_both_legs(reference):
    doc, versions, record = reference
    record['shared_field_selection'] = True
    record['true_json']['rows'] = [{'amount': '17'}, {'amount': '42'}]
    record['json_schema']['properties']['rows'] = {'type': 'array', 'items': {
        'type': 'object', 'properties': {'amount': {'type': 'string'}}}}
    from evaluation.structured_metrics import walk_fields
    record['excluded_fields'] = ['rows[1].amount']
    record['kv_aliases'] = {f.path: {'printed_labels': [], 'is_printed_pair': f.path not in record['excluded_fields']}
                            for f in walk_fields(record['true_json'], record['json_schema'])}
    StructuredGoldStore(versions, versions / 'gold').save('page', record, None)
    gt = GroundTruthLoader().load(doc)
    output = json.dumps(record['true_json'])
    kv = score_key_value(output, gt, 'fixture')
    schema = score_schema(output, gt, 'fixture')
    assert kv.field_accuracy == schema.field_accuracy == 1
    assert kv.pairs_total == schema.fields_total == 4
    assert {v.path: v.expected for v in kv.verdicts} == {v.path: v.expected for v in schema.verdicts}
    assert gt.printed_pair_count() == 4
    record['kv_aliases']['rows[0].amount']['is_printed_pair'] = False
    with pytest.raises(ValueError, match='inclusion must agree'):
        validate_structured_record(record)
