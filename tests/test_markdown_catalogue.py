"""Missing catalogue generation is separate from reviewed reference activation."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from PIL import Image
from evaluation import markdown_catalogue as c
from evaluation.ground_truth import GroundTruth, GroundTruthLoader
from evaluation.markdown_leg import score_markdown


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'CACHE', tmp_path / 'results/markdown_judge_cache')
    page = tmp_path / 'documents/easy/receipt/receipt.png'
    page.parent.mkdir(parents=True)
    Image.new('RGB', (8, 8), 'white').save(page)
    page.with_name('receipt_gt.txt').write_text('Receipt')
    (tmp_path / 'documents/manifest.json').write_text(json.dumps([
        {'file': str(page.relative_to(tmp_path)), 'sha256': c.source_hash(page)}]))
    return page


def items():
    return [{'type': 'heading', 'content': 'Receipt', 'confident': True, 'notes': ''},
            {'type': 'paragraph', 'content': 'unreadable', 'confident': False, 'notes': 'uncertain'}]


def proposal(page):
    request = c.prepare()['requests'][0]
    record = {**request, 'items': items(), 'review_status': 'pending'}
    dest = c.CACHE / 'proposals' / (page.stem + '.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record))
    return record


def test_plan_only_uses_source_image_and_skips_existing_catalogue(corpus):
    corpus.with_suffix('.txt').write_text('ENGINE ANSWER MUST NEVER BE SENT')
    plan = c.prepare()
    assert plan['calls'] == 1 and 'ENGINE ANSWER' not in json.dumps(plan)
    assert plan['requests'][0]['image_sha256'] == c.source_hash(corpus)
    c.CACHE.mkdir(parents=True)
    (c.CACHE / 'receipt.json').write_text('{}')
    assert c.prepare()['calls'] == 0


def test_image_drift_blocks_preparation_and_execution(corpus):
    plan = c.prepare()
    corpus.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Source image changed'): c.prepare()
    with pytest.raises(ValueError, match='Source image differs'): c.execute(plan, 10)


def test_budget_and_prompt_drift_block_before_submission(corpus, monkeypatch):
    plan = c.prepare()
    monkeypatch.setattr(c, '_one', lambda r: pytest.fail('must not call'))
    with pytest.raises(ValueError, match='exceeds'): c.execute(plan, .001)
    plan['requests'][0]['prompt'] += ' changed'
    with pytest.raises(ValueError, match='changed'): c.execute(plan, 10)


def test_proposals_are_ineligible_until_source_review_and_uncertain_items_excluded(corpus):
    proposal(corpus)
    assert c.reviewed_items(corpus) is None
    gt = GroundTruthLoader().load(corpus)
    assert not gt.has_gt_for('markdown')
    with pytest.raises(ValueError, match='Record who'): c.accept(corpus.stem, '', '')
    c.accept(corpus.stem, 'reviewer', 'Checked heading and excluded uncertain text against source')
    assert c.reviewed_items(corpus) == items()[:1]
    gt = GroundTruthLoader().load(corpus)
    assert gt.has_gt_for('markdown') and gt.markdown is None
    assert score_markdown('# Receipt', gt).score == 1
    assert score_markdown('**Receipt**', gt).score == 0
    with pytest.raises(FileExistsError): c.accept(corpus.stem, 'reviewer', 'Reviewed')


def test_sampled_catalogue_does_not_become_full_text_or_full_block_reference():
    gt = GroundTruth('receipt.png', 'easy', 'receipt', 'Receipt alpha',
        metadata={'reviewed_markdown_catalogue': items()[:1]})
    scored = score_markdown('# Receipt', gt)
    assert scored.score == 1 and scored.cer > 0 and scored.wer > 0
    assert scored.element_f1 is None and scored.order_score is None
    gt.gt_source = 'silver_tesseract'
    scored = score_markdown('# Receipt', gt)
    assert scored.score == 1 and scored.cer is None and scored.wer is None


def test_accepted_catalogue_checks_current_image(corpus):
    proposal(corpus)
    c.accept(corpus.stem, 'reviewer', 'Checked image')
    corpus.write_bytes(b'new image')
    with pytest.raises(ValueError, match='image mismatch'): c.reviewed_items(corpus)


def test_paid_reply_is_journaled_once_and_not_automatically_activated(corpus, monkeypatch):
    from google import genai
    monkeypatch.setenv('GOOGLE_API_KEY', 'offline-fixture')
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=json.dumps({'items': items()}), usage_metadata=SimpleNamespace(
            prompt_token_count=100, candidates_token_count=200, thoughts_token_count=30),
            model_dump=lambda **kw: {'fixture': True})
    monkeypatch.setattr(genai, 'Client', lambda **kw: SimpleNamespace(
        models=SimpleNamespace(generate_content=generate), close=lambda: None))
    plan = c.prepare()
    result = c.execute(plan, 1)
    assert result[0]['cost_usd'] == pytest.approx(.002425)
    assert (c.CACHE / 'proposals/receipt.json').exists()
    assert not (c.CACHE / 'receipt.json').exists()
    c.execute(plan, 1)
    assert len(calls) == 1


def test_pending_catalogue_in_active_directory_cannot_score(corpus, monkeypatch):
    from evaluation import code_judges
    record = proposal(corpus)
    (c.CACHE / 'receipt.json').write_text(json.dumps(record))
    monkeypatch.setattr(code_judges, 'MARKDOWN_CACHE_DIR', c.CACHE)
    assert code_judges.load_markdown_checklist('receipt') is None
    assert not GroundTruthLoader().load(corpus).has_gt_for('markdown')
