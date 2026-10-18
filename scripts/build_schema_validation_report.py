"""Show every field from a fresh schema validation beside the published output."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.ground_truth import GroundTruthLoader
from evaluation.schema_leg import score_schema
from evaluation.reporting import mean_metric
import run_benchmark as rb


def read_run(stamp, suffix):
    return [json.loads(line) for line in (ROOT / 'results/runs' / f'{stamp}_{suffix}.jsonl').read_text().splitlines()]


def source_file(record):
    doc = '__'.join((record['tier'], record['doc_type'], Path(record['document_path']).stem))
    name = record.get('source_pass_name') or record.get('pass_name') or 'default'
    return ROOT / 'results/ocr_outputs' / record['source_timestamp'] / doc / record['engine_name'] / (name + '.call.json')


def provider_text(call):
    return ''.join(p.get('text', '') for c in call['raw_response'].get('candidates', [])
                   for p in c.get('content', {}).get('parts', []) if not p.get('thought'))


def sources_for(stamp):
    records = read_run(stamp, 'inference')
    return {(records[i]['engine_name'], rb._repo_path(records[i]['document_path']), leg): records[i]
            for (_, i), legs in rb._rescore_selection([records]).items() for leg in legs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('published')
    parser.add_argument('validation')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/reports/gemini-schema-validation.html')
    args = parser.parse_args()
    old = [r for r in read_run(args.published, 'results') if r['engine_name'] == 'gemini_flash' and r['leg'] == 'schema']
    new = read_run(args.validation, 'results')
    assert len(new) == 3 and all(r['engine_name'] == 'gemini_flash' and r['leg'] == 'schema' for r in new)
    assert all(r['score'] is not None and not r.get('error') for r in new)
    old_sources = sources_for(args.published)
    new_sources = sources_for(args.validation)
    scored = [r for r in old if r['score'] is not None]
    direct = 0
    for row in scored:
        source = old_sources[('gemini_flash', rb._repo_path(row['document_path']), 'schema')]
        call = json.loads(source_file(source).read_text())
        assert provider_text(call) == call['raw_text']
        direct += 1
    cases = []
    for b in new:
        a = next(r for r in old if r['document_path'] == b['document_path'])
        key = ('gemini_flash', rb._repo_path(b['document_path']), 'schema')
        paths = [source_file(sources[key]) for sources in (old_sources, new_sources)]
        calls = [json.loads(p.read_text()) for p in paths]
        assert all(provider_text(c) == c['raw_text'] for c in calls)
        assert calls[0]['prompt_sent'] == calls[1]['prompt_sent']
        assert calls[0]['json_schema'] == calls[1]['json_schema']
        assert calls[0]['model_version'] == calls[1]['model_version']
        gt = GroundTruthLoader().load(ROOT / b['document_path'])
        assert score_schema(calls[1]['raw_text'], gt, 'gemini_flash').score == b['score']
        schema = calls[1]['json_schema']
        nulls = score_schema(json.dumps({k: None for k in schema['properties']}), gt, 'gemini_flash')
        wrong = score_schema(json.dumps({k: '__deliberately_wrong_value__' for k in schema['properties']}), gt, 'gemini_flash')
        assert wrong.score == 0
        av = {v['path']: v for v in a['detail']['verdicts']}
        fields = [{'field': v['path'], 'expected': v['expected'], 'old': av[v['path']], 'new': v}
                  for v in b['detail']['verdicts']]
        cases.append(dict(page=Path(b['document_path']).stem, document=b['document_path'],
                          old_score=a['score'], new_score=b['score'], fields=fields,
                          old_call=paths[0].relative_to(ROOT).as_posix(), new_call=paths[1].relative_to(ROOT).as_posix(),
                          call_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
                          identical_prompt_and_schema=True, provider_text_matches=True,
                          model=calls[1]['model_version'], cost_usd=b['cost_per_page'],
                          old_raw=calls[0]['raw_text'], new_raw=calls[1]['raw_text'],
                          null_only_control=nulls.score, wrong_value_control=wrong.score,
                          old_blank_fields=a['metrics']['schema_blank_total'],
                          changed_fields=sum(f['old']['verdict'] != f['new']['verdict'] for f in fields)))
    payload = dict(published_run=args.published, validation_run=args.validation, cases=cases,
                   published_score=mean_metric(scored), populated_accuracy=
                   sum(r['metrics']['schema_fields_correct'] for r in scored) /
                   sum(r['metrics']['schema_fields_total'] - r['metrics']['schema_blank_total'] for r in scored),
                   saved_provider_text_verified=direct, calls=3, judge_calls=0,
                   cost_usd=sum(c['cost_usd'] for c in cases), validation_replaces_published_results=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(payload, indent=2) + '\n')
    esc = lambda x: html.escape(str(x))
    link = lambda p: esc(os.path.relpath(ROOT / p, args.output.parent))
    out = ['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gemini schema — three fresh calls</title><style>
body{font:15px/1.5 system-ui;color:#182a3a;background:#f5f7fa;margin:0}main{max-width:1450px;margin:auto;padding:24px}section{background:white;padding:20px;margin:20px 0;border:1px solid #ccd6df;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid #dde3ea;overflow-wrap:anywhere}th{background:#eef3f7}td:first-child{max-width:480px}.changed{background:#fff0df}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:550px;overflow:auto;background:#f5f7fa;padding:12px}.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}img{max-width:100%;max-height:1000px}a{color:#155b93}.scroll{overflow:auto}@media(max-width:700px){.two{grid-template-columns:1fr}}
</style><main><h1>Gemini schema: published output vs three fresh calls</h1>''']
    out.append('<p>Three targeted checks of previously perfect pages, using the identical saved prompt, schema and model, '
               'with temperature 0. These checks are separate from the published benchmark. '
               'Every field is shown below; no outputs, references or scores were patched.</p>')
    out.append(f'<p>Model: gemini-3.1-flash-lite · Validation run: {esc(args.validation)} · Cost: ${payload["cost_usd"]:.6f} · No judge calls.</p>')
    out.append(f'<p>Published benchmark: {esc(args.published)}. The fresh validation remains separate.</p>')
    out.append('<p><a href="ocr-benchmark-final.html">Current benchmark</a> · <a href="gemini-schema-validation.json">Complete validation data</a></p>')
    out.append('<section><h2>All three results</h2><table><tr><th>Page</th><th>Published</th><th>Fresh</th><th>Fields</th><th>Reference blanks</th><th>Changed verdicts</th></tr>')
    for c in cases:
        out.append(f'<tr><td><a href="#{c["page"]}">{c["page"]}</a></td><td>{c["old_score"]:.4f}</td><td>{c["new_score"]:.4f}</td><td>{len(c["fields"])}</td><td>{c["old_blank_fields"]}</td><td>{c["changed_fields"]}</td></tr>')
    out.append('</table></section><section><h2>What this validates</h2>')
    out.append(f'<p>All {direct} scored saved schema answers match their stored provider-response text. '
               f'The published family-weighted score is {payload["published_score"]:.4f}; pooled populated-field accuracy is {payload["populated_accuracy"]:.4f}.</p>')
    out.append('<p>Schema supplies exact target keys, section names and row identities. Some identity values, such as consultant names, '
               'are already present in those keys. This is guided value extraction, and every participating engine receives the same frozen field sample. '
               'The three schemas contain no value defaults, enums, constants or examples. Correct blanks also receive credit.</p>')
    out.append('<p>On funsd_003, the fresh answer skips McKeesport Candy Co. and shifts subsequent table rows. '
               'The eight resulting errors are visible against the source image below. The saved answer aligns those rows correctly. '
               'Two perfect scores reproduced; the third did not. Three targeted calls do not establish whole-corpus repeatability.</p>')
    out.append('<p>Offline controls: replacing every value with a wrong string scores zero on all three pages. '
               'A null-only answer earns blank credit; the existing scorer also treats null as an unchecked checkbox. '
               'These controls are calculated in memory and never replace provider outputs.</p><table><tr><th>Page</th><th>Null-only control</th><th>Wrong-value control</th></tr>')
    for c in cases:
        out.append(f'<tr><td>{c["page"]}</td><td>{c["null_only_control"]:.4f}</td><td>{c["wrong_value_control"]:.4f}</td></tr>')
    out.append('</table></section>')
    for c in cases:
        out.append(f'<section id="{c["page"]}"><h2>{c["page"]}: {c["old_score"]:.4f} → {c["new_score"]:.4f}</h2>')
        out.append(f'<p><a href="{link(c["old_call"])}">Original call JSON</a> · <a href="{link(c["new_call"])}">Fresh call JSON</a> · <a href="{link(c["document"])}">Source image</a></p>')
        out.append(f'<details><summary>View source image</summary><img loading="lazy" src="{link(c["document"])}" alt="{c["page"]}"></details>')
        out.append('<details><summary>Compare complete raw engine answers</summary><div class="two"><div><h3>Published</h3><pre>'+esc(c['old_raw'])+'</pre></div><div><h3>Fresh</h3><pre>'+esc(c['new_raw'])+'</pre></div></div></details>')
        out.append('<h3>Every scored field</h3><div class="scroll"><table><tr><th>Field</th><th>Reference</th><th>Published value</th><th>Fresh value</th><th>Published verdict</th><th>Fresh verdict</th></tr>')
        for f in c['fields']:
            out.append('<tr'+(' class="changed"' if f['old']['verdict'] != f['new']['verdict'] else '')+'>'
                       + ''.join('<td>'+esc(v)+'</td>' for v in (f['field'], f['expected'], f['old']['value'],
                                  f['new']['value'], f['old']['verdict'], f['new']['verdict'])) + '</tr>')
        out.append('</table></div></section>')
    out.append('</main></html>')
    args.output.write_text('\n'.join(out))
    print(json.dumps({k: v for k, v in payload.items() if k != 'cases'}, indent=2))
    print(args.output)


if __name__ == '__main__':
    main()
