"""Regression checks for published source identity and offline rescoring."""
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys
import unittest
from unittest.mock import patch

import run_benchmark as rb
from evaluation import artifacts


class PublishedRunTest(unittest.TestCase):
    def test_cli_rescore_is_offline_even_with_credentials_present(self):
        with patch.object(sys, 'argv', ['run_benchmark.py', '--rescore', 'current']), \
             patch.dict(os.environ, {'GOOGLE_API_KEY': 'not-a-real-key'}), \
             patch.object(rb, '_resolve_timestamp', return_value='published'), \
             patch.object(rb, 'setup_logging', return_value=logging.getLogger('test')), \
             patch.object(rb, 'rescore_runs') as replay, patch.object(rb, 'report'):
            rb.main()
            self.assertTrue(replay.call_args.kwargs['skip_judge'])

    def test_current_is_newest_complete_scored_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'runs').mkdir()
            for name in ('20261001T000000Z', '20261002T000000Z'):
                for suffix in ('results.jsonl', 'inference.jsonl'):
                    (root / 'runs' / f'{name}_{suffix}').write_text('')
                (root / 'runs' / f'{name}_run.json').write_text('{"scored":true}')
            (root / 'runs/20261003T000000Z_results.jsonl').write_text('')
            (root / 'runs/20261003T000000Z_run.json').write_text('{}')
            with patch.object(artifacts, 'RESULTS_DIR', root):
                self.assertEqual(artifacts.current_timestamp(), '20261002T000000Z')
                (root / 'runs/20261002T000000Z_run.json').unlink()
                self.assertEqual(artifacts.current_timestamp(), '20261001T000000Z')
                (root / 'runs/20261001T000000Z_run.json').unlink()
                with self.assertRaises(FileNotFoundError):
                    artifacts.current_timestamp()


class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runs = self.root / 'runs'
        self.runs.mkdir()
        self.outputs = self.root / 'outputs'
        self.doc = rb.ROOT / 'documents/easy/invoice/invoice_000.png'
        self.gt = SimpleNamespace(tier='easy', doc_type='invoice', gt_source='silver_tesseract')
        self.common = {'engine_name': 'llamaparse', 'document_path': str(self.doc),
                       'pass_name': 'default', 'timestamp': 'source'}
        records = [{**self.common, 'legs': ['text'], 'source_timestamp': 'plain'},
                   {**self.common, 'legs': ['tables'], 'source_timestamp': 'table'}]
        (self.runs / 'source_inference.jsonl').write_text(
            ''.join(json.dumps(r) + '\n' for r in records))
        judged = {**self.common, 'leg': 'judge', 'score': .75, 'applicable': True,
                  'metrics': {}, 'error': None}
        (self.runs / 'source_results.jsonl').write_text(json.dumps(judged) + '\n')
        for source, text in [('plain', 'Invoice plain text'), ('table', '| Item | Price |')]:
            path = self.outputs / source / 'easy__invoice__invoice_000/llamaparse/default.call.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'raw_text': text}))

    def tearDown(self):
        self.temp.cleanup()

    def replay(self, scorer):
        with patch.object(rb, 'RUNS_DIR', self.runs), \
             patch.object(rb, 'OCR_OUTPUTS_DIR', self.outputs), \
             patch('evaluation.ground_truth.GroundTruthLoader.load', return_value=self.gt), \
             patch.object(rb, '_score_pass_legs', side_effect=scorer), \
             patch.object(rb, '_judge_silver_document', side_effect=AssertionError('unexpected judge')) as judge:
            # Enabling KV decisions must never also refresh text judgments.
            rb.rescore_runs(['source'], 'target', logging.getLogger('test'), skip_judge=False)
            judge.assert_not_called()

    def test_distinct_source_views_and_one_historical_verdict(self):
        seen = []
        def score(engine, doc, gt, base, pass_name, legs, text, *args, **kwargs):
            seen.append((legs, text))
            return []
        self.replay(score)
        self.assertEqual(seen, [(['text'], 'Invoice plain text'), (['tables'], '| Item | Price |')])
        rows = rb.read_jsonl(self.runs / 'target_results.jsonl')
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]['leg'], rows[0]['score']), ('judge', .75))
        self.assertTrue(rows[0]['metrics']['historical_verdicts'])

    def test_missing_source_fails_before_completed_manifest(self):
        path = self.outputs / 'table/easy__invoice__invoice_000/llamaparse/default.call.json'
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.replay(lambda *a, **k: [])
        self.assertFalse((self.runs / 'target_run.json').exists())

    def test_missing_run_does_not_borrow_a_published_answer(self):
        record = {**self.common, 'legs': ['text']}
        (self.runs / 'source_inference.jsonl').write_text(json.dumps(record) + '\n')
        unrelated = self.outputs / 'reconciled_outputs/easy__invoice__invoice_000/llamaparse/default.call.json'
        unrelated.parent.mkdir(parents=True)
        unrelated.write_text('{"raw_text":"unrelated published answer"}')
        with self.assertRaises(FileNotFoundError):
            self.replay(lambda *a, **k: self.fail('Wrong source reached the scorer'))

    def test_two_views_in_one_folder_reach_their_own_scoring_legs(self):
        folder = self.outputs / 'reconciled_outputs/easy__invoice__invoice_000/llamaparse'
        folder.mkdir(parents=True)
        (self.outputs / 'plain/easy__invoice__invoice_000/llamaparse/default.call.json').rename(folder / 'default.call.json')
        (self.outputs / 'table/easy__invoice__invoice_000/llamaparse/default.call.json').rename(folder / 'default.tables.call.json')
        records = [{**self.common, 'legs': ['text'], 'source_timestamp': 'reconciled_outputs'},
                   {**self.common, 'legs': ['tables'], 'source_timestamp': 'reconciled_outputs',
                    'source_pass_name': 'default.tables'}]
        (self.runs / 'source_inference.jsonl').write_text(
            ''.join(json.dumps(r) + '\n' for r in records))
        self.assertEqual(len(rb.unique_inference_calls(records)), 2)
        seen = []
        def score(engine, doc, gt, base, pass_name, legs, text, *args, **kwargs):
            seen.append((pass_name, legs, text))
            return []
        self.replay(score)
        self.assertEqual(seen, [('default', ['text'], 'Invoice plain text'),
                                ('default', ['tables'], '| Item | Price |')])

    def test_rescore_recomputes_instead_of_copying_previous_numeric_score(self):
        from evaluation.ground_truth import GroundTruth
        gt = GroundTruth(str(self.doc), 'easy', 'invoice', 'alpha beta')
        record = {**self.common, 'legs': ['text'], 'source_timestamp': 'plain'}
        (self.runs / 'source_inference.jsonl').write_text(json.dumps(record) + '\n')
        previous = {**self.common, 'leg': 'text', 'score': 0.999, 'applicable': True,
                    'metrics': {'gt_coverage': 0.999}, 'detail': {}, 'error': None}
        original = json.dumps(previous) + '\n'
        (self.runs / 'source_results.jsonl').write_text(original)
        response = self.outputs / 'plain/easy__invoice__invoice_000/llamaparse/default.call.json'
        response.write_text(json.dumps({'raw_text': 'alpha'}))
        with patch.object(rb, 'RUNS_DIR', self.runs), \
             patch.object(rb, 'OCR_OUTPUTS_DIR', self.outputs), \
             patch('evaluation.ground_truth.GroundTruthLoader.load', return_value=gt):
            rb.rescore_runs(['source'], 'target', logging.getLogger('test'), skip_judge=True)
        rows = rb.read_jsonl(self.runs / 'target_results.jsonl')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['score'], 0.5)
        self.assertEqual(rows[0]['metrics']['gt_coverage'], 0.5)
        self.assertEqual(rows[0]['detail']['missed_words'], ['beta'])
        self.assertEqual((self.runs / 'source_results.jsonl').read_text(), original)

    def test_rescore_rejects_missing_inference_ledger(self):
        with patch.object(rb, 'RUNS_DIR', self.runs):
            with self.assertRaisesRegex(FileNotFoundError, 'No inference records'):
                rb.rescore_runs(['absent'], 'target', logging.getLogger('test'), skip_judge=True)
        self.assertFalse((self.runs / 'target_run.json').exists())

    def test_score_missing_output_cannot_mark_run_complete(self):
        (self.outputs / 'plain/easy__invoice__invoice_000/llamaparse/default.call.json').unlink()
        manifest = self.runs / 'source_run.json'
        manifest.write_text(json.dumps({'scored': True, 'scored_at': 'previous-attempt'}))
        with patch.object(rb, 'RUNS_DIR', self.runs), \
             patch.object(rb, 'OCR_OUTPUTS_DIR', self.outputs), \
             patch('evaluation.ground_truth.GroundTruthLoader.load', return_value=self.gt):
            with self.assertRaises(FileNotFoundError):
                rb.score_stored_run('source', logging.getLogger('test'), skip_judge=True)
        saved = json.loads(manifest.read_text())
        self.assertFalse(saved['scored'])
        self.assertNotIn('scored_at', saved)

    def test_score_missing_reference_cannot_mark_run_complete(self):
        manifest = self.runs / 'source_run.json'
        manifest.write_text('{"scored": false}')
        with patch.object(rb, 'RUNS_DIR', self.runs), \
             patch('evaluation.ground_truth.GroundTruthLoader.load', side_effect=FileNotFoundError('GT')):
            with self.assertRaisesRegex(RuntimeError, 'cannot load reference'):
                rb.score_stored_run('source', logging.getLogger('test'), skip_judge=True)
        self.assertFalse(json.loads(manifest.read_text())['scored'])

    def test_score_new_collected_response_uses_real_scorer(self):
        from evaluation.ground_truth import GroundTruth
        gt = GroundTruth(str(self.doc), 'easy', 'invoice', 'alpha beta')
        record = {**self.common, 'legs': ['text'], 'source_timestamp': 'plain'}
        (self.runs / 'source_inference.jsonl').write_text(json.dumps(record) + '\n')
        (self.runs / 'source_results.jsonl').write_text('')
        (self.runs / 'source_run.json').write_text('{"scored": false, "collect_only": true}')
        response = self.outputs / 'plain/easy__invoice__invoice_000/llamaparse/default.call.json'
        response.write_text(json.dumps({'raw_text': 'alpha'}))
        with patch.object(rb, 'RUNS_DIR', self.runs), \
             patch.object(rb, 'OCR_OUTPUTS_DIR', self.outputs), \
             patch('evaluation.ground_truth.GroundTruthLoader.load', return_value=gt):
            rb.score_stored_run('source', logging.getLogger('test'), skip_judge=True)
        rows = rb.read_jsonl(self.runs / 'source_results.jsonl')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['score'], 0.5)
        self.assertEqual(rows[0]['detail']['missed_words'], ['beta'])
        self.assertTrue(json.loads((self.runs / 'source_run.json').read_text())['scored'])


if __name__ == '__main__':
    unittest.main()
