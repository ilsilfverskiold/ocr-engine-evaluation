"""Publish one scored run and its exact source calls as results/current.zip.

Offline only. Existing working files are not removed by this command.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_benchmark as rb
from evaluation import artifacts
from evaluation.ground_truth import GroundTruthLoader

def publish(timestamp):
    timestamp = rb._resolve_timestamp(timestamp)
    rows = rb.read_jsonl(rb.RUNS_DIR / f'{timestamp}_results.jsonl')
    inference = rb.read_jsonl(rb.RUNS_DIR / f'{timestamp}_inference.jsonl')
    if not rows or not inference:
        raise ValueError('Both scored results and inference records are required')
    sources = {}
    def add(path):
        key = path.resolve().relative_to(rb.ROOT.resolve()).as_posix()
        sources[key] = artifacts.read_text(path).encode('utf-8')
    for suffix in ('results.jsonl', 'inference.jsonl', 'run.json'):
        add(rb.RUNS_DIR / f'{timestamp}_{suffix}')
    loader = GroundTruthLoader()
    missing = []
    ground_truth = {}
    for record in inference:
        doc = Path(record['document_path'])
        if doc not in ground_truth:
            ground_truth[doc] = loader.load(doc)
        gt = ground_truth[doc]
        source = record.get('source_timestamp') or record['timestamp']
        path = rb.OCR_OUTPUTS_DIR / source / f'{gt.tier}__{gt.doc_type}__{doc.stem}' / record['engine_name'] / ((record.get('pass_name') or 'default') + '.call.json')
        try:
            add(path)
        except FileNotFoundError:
            if not record.get('error'):
                missing.append(str(path))
    if missing:
        raise ValueError('Missing successful source calls:\n' + '\n'.join(missing))
    # Retain paid decisions and shared catalogues for offline scoring.
    for folder in ('kv_synonyms', 'markdown_judge_cache', 'judge_checklists'):
        prefix = 'results/' + folder + '/'
        if artifacts.BUNDLE.exists():
            for name in artifacts.archive().namelist():
                if name.startswith(prefix):
                    sources[name] = artifacts.archive().read(name)
        for path in (rb.RESULTS_DIR / folder).rglob('*.json'):
            add(path)
    call_count = sum(name.startswith('results/ocr_outputs/') for name in sources)
    manifest = {'format': 1, 'run': timestamp, 'scores': len(rows), 'inference_records': len(inference),
                'files': {key: hashlib.sha256(data).hexdigest() for key, data in sorted(sources.items())}}
    target = artifacts.BUNDLE
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.zip.tmp')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
            output.writestr('manifest.json', json.dumps(manifest, indent=2))
            for key, data in sorted(sources.items()):
                output.writestr(key, data)
        with zipfile.ZipFile(temporary) as check:
            for key, digest in manifest['files'].items():
                if hashlib.sha256(check.read(key)).hexdigest() != digest:
                    raise ValueError(f'Archive verification failed: {key}')
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'{target}: {len(rows)} scores, {len(inference)} inference records, {call_count} source calls; {target.stat().st_size/1024**2:.1f} MiB. SHA-256 verified.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', help='Scored run timestamp, current, or latest')
    publish(parser.parse_args().run)
