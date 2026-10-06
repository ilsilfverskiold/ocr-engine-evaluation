"""Actual filesystem saves, backups and conflicts; no benchmark calls."""
import json

import pytest
from scripts.table_reference_editor import Conflict, GoldStore


@pytest.fixture
def store(tmp_path):
    versions = tmp_path / 'versions'
    (versions / 'old').mkdir(parents=True)
    (versions / 'manifest.json').write_text(json.dumps({'dataset_id': 'fixture', 'pages': ['page_000']}))
    (versions / 'old/page_000.json').write_text(json.dumps({'image_sha256': 'image'}))
    return GoldStore(versions, versions / 'gold')


def record(text='old'):
    return {'version': 1, 'page': 'page_000', 'dataset_id': 'fixture', 'image_sha256': 'image',
            'table_html': f'<table><tr><td>{text}</td></tr></table>', 'selection': 'old'}


def test_save_reload_backup_and_conflict(store):
    first = store.save('page_000', record(), None)
    assert store.read('page_000') == (record(), first['revision'])
    before = (store.gold / 'page_000.json').read_bytes()
    second = store.save('page_000', record('edited'), first['revision'])
    assert second['backed_up']
    assert next((store.gold / 'history').glob('*.json')).read_bytes() == before
    with pytest.raises(Conflict):
        store.save('page_000', record('stale'), first['revision'])
    assert store.read('page_000')[0] == record('edited')


@pytest.mark.parametrize('page,change', [('../escape', {}), ('page_000', {'image_sha256': 'wrong'}),
                                      ('page_000', {'table_html': '<script>alert(1)</script>'}),
                                      ('page_000', {'table_html': '<table onclick="x"><tr><td>x</td></tr></table>'})])
def test_invalid_save_does_not_create_gold(store, page, change):
    with pytest.raises(ValueError):
        store.save(page, {**record(), **change}, None)
    assert list(store.gold.iterdir()) == []
