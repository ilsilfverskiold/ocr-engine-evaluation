"""Offline replay audit; writes a separate audit run, never the published run.

Checks all row identities, scores, applicability and failures. Scorer updates
may change values; every difference is saved, not silently accepted. Network
connections and new judge requests are blocked even if API keys are present.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def key(row):
    return row['engine_name'], rb._repo_path(row['document_path']), row['leg']


def main():
    global rb
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=None,
                        help='Defaults to current, or the original baseline with --historical')
    parser.add_argument('--legs', default='all')
    parser.add_argument('--historical', action='store_true',
                        help='Use the published evaluator and saved judgment caches from Git')
    parser.add_argument('--allow-kv-changes', action='store_true',
                        help='Allow current KV scorer changes; still record each difference')
    parser.add_argument('--blank-only', action='store_true',
                        help='Require scored KV changes to be solely omitted blank reference fields')
    args = parser.parse_args()
    historical = None
    if args.historical:
        # Keep the original evaluator and its caches isolated. Current source
        # files and current cached rulings are not changed by this audit.
        from repair_reconciled_sources import ARCHIVE, ARCHIVE_SHA256
        archive = subprocess.check_output(['git', 'show', ARCHIVE], cwd=ROOT)
        if hashlib.sha256(archive).hexdigest() != ARCHIVE_SHA256:
            raise RuntimeError('Historical archive checksum mismatch')
        historical = tempfile.TemporaryDirectory(prefix='ocr-published-evaluator-')
        snapshot = Path(historical.name)
        revision = ARCHIVE.split(':')[0]
        code = subprocess.check_output(['git', 'archive', revision, 'evaluation'], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(code)) as tar:
            tar.extractall(snapshot, filter='data')
        (snapshot / 'documents').symlink_to(ROOT / 'documents', target_is_directory=True)
        with zipfile.ZipFile(io.BytesIO(archive)) as saved:
            for name in saved.namelist():
                if name.startswith(('results/kv_synonyms/', 'results/markdown_judge_cache/',
                                    'results/judge_checklists/')):
                    path = snapshot / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(saved.read(name))
        sys.path.insert(0, str(snapshot))
    import run_benchmark as rb
    from evaluation import kv_value_judge
    source_name = args.source or ('20261017T065717Z' if args.historical else 'current')
    source = rb._resolve_timestamp(source_name)
    if args.historical and source != '20261017T065717Z':
        raise ValueError('--historical is pinned to published run 20261017T065717Z')
    selected_legs = None if args.legs == 'all' else rb.parse_legs(args.legs)
    target = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_offline_audit'
    destination = rb.RESULTS_DIR / 'audits' / target
    destination.mkdir(parents=True, exist_ok=False)
    for suffix in ('_inference.jsonl', '_results.jsonl', '_run.json'):
        shutil.copyfile(rb.RUNS_DIR / (source + suffix), destination / (source + suffix))
    rb.RUNS_DIR = destination
    log = logging.getLogger('offline-replay')
    log.setLevel(logging.INFO)
    log.addHandler(logging.FileHandler(destination / 'replay.log'))
    forbidden = AssertionError('Offline replay attempted a network or judge call')
    with patch.object(socket.socket, 'connect', side_effect=forbidden), \
         patch.object(socket, 'create_connection', side_effect=forbidden), \
         patch.object(kv_value_judge, '_ask', side_effect=forbidden) as ask, \
         patch.object(rb, '_judge_silver_document', side_effect=forbidden) as judge:
        rb.rescore_runs([source], target, log, skip_judge=True, selected_legs=selected_legs)
        assert ask.call_count == judge.call_count == 0
    before = rb.read_jsonl(destination / (source + '_results.jsonl'))
    after = rb.read_jsonl(destination / (target + '_results.jsonl'))
    if selected_legs is not None:
        included = set(selected_legs) | ({'judge'} if 'text' in selected_legs else set())
        before = [r for r in before if r['leg'] in included]
    old = {key(r): r for r in before}
    new = {key(r): r for r in after}
    diffs = []
    blank_rule_failures = []
    pending_field_differences = []
    for identity in sorted(old.keys() | new.keys()):
        a, b = old.get(identity), new.get(identity)
        if args.blank_only and identity[2] == 'key_value' and a is not None and b is not None:
            av = {v['path']: v for v in a.get('detail', {}).get('verdicts', [])}
            bv = {v['path']: v for v in b.get('detail', {}).get('verdicts', [])}
            omitted = sum(v['is_blank'] and v['verdict'] == 'missing' for v in av.values())
            expected = a.get('score')
            if expected is not None and av:
                counted = len(av) - omitted
                expected = (sum(v['verdict'] in ('correct', 'blank_ok') for v in av.values()) / counted
                            if counted else None)
            bad_fields = [path for path in av.keys() | bv.keys()
                          if path not in av or path not in bv or
                          bv[path]['verdict'] != ('blank_not_returned' if av[path]['is_blank']
                                                 and av[path]['verdict'] == 'missing' else av[path]['verdict'])]
            if a.get('score') is None and b.get('score') is None and bad_fields:
                pending_field_differences.append({'identity': identity, 'fields': bad_fields})
                bad_fields = []  # Unscored diagnostics are reported separately, never given a score.
            if b.get('score') != expected or bad_fields or a.get('error') != b.get('error'):
                blank_rule_failures.append({'identity': identity, 'expected_score': expected,
                                            'actual_score': b.get('score'), 'fields': bad_fields})
        fields = ('score', 'applicable', 'error')
        if a is None or b is None or any(a.get(f) != b.get(f) for f in fields):
            diffs.append({'identity': identity,
                          'before': {f: a.get(f) for f in fields} if a else None,
                          'after': {f: b.get(f) for f in fields} if b else None})
    summary = {'source': source, 'replay': target,
               'evaluator': 'published b80c0c82' if args.historical else 'working tree',
               'before_rows': len(before),
               'after_rows': len(after), 'duplicate_rows': len(after) - len(new),
               'network_and_new_judging': 'blocked',
               'differences_by_leg': dict(Counter(d['identity'][2] for d in diffs)),
               'differences': diffs}
    if args.blank_only:
        summary['blank_rule_failures'] = blank_rule_failures
        summary['pending_field_differences'] = pending_field_differences
    path = destination / 'comparison.json'
    path.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'differences'}, indent=2))
    print(f'Full audit: {path}')
    if historical is not None:
        historical.cleanup()
    if summary['duplicate_rows'] or blank_rule_failures or any(
            not (args.allow_kv_changes or args.blank_only) or d['identity'][2] != 'key_value' for d in diffs):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
