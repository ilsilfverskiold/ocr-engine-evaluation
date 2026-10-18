"""Build an offline old/new review from two completed saved result files."""
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.reporting import mean_metric, valid

LEGS = ['text', 'markdown', 'tables', 'key_value', 'schema', 'geometry', 'judge']
LABELS = dict(zip(LEGS, ['Text', 'Markdown', 'Tables', 'KV', 'Schema', 'Geometry', 'Historical judge']))


def identity(row):
    parts = Path(row['document_path']).parts
    document = Path(*parts[parts.index('documents'):]).as_posix()
    return row['engine_name'], document, row['leg']


def verdict_identity(verdict, index):
    if 'path' in verdict:
        return verdict['path']
    for side in ('reference', 'output'):
        cell = verdict.get(side)
        if isinstance(cell, dict) and 'row' in cell and 'col' in cell:
            return f"{side} table {verdict.get(side + '_table')} r{cell['row']} c{cell['col']}"
    return f'verdict {index}'


def state(row):
    if row is None:
        return 'Absent'
    if row.get('error'):
        return 'Error'
    if valid(row):
        return 'Scored'
    return 'Pending' if 'pending' in str(row.get('reason', '')).lower() else 'N/A'


def load(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    indexed = {identity(row): row for row in rows}
    if len(rows) != len(indexed):
        raise ValueError(f'Duplicate engine/document/leg rows in {path}')
    manifest = path.with_name(path.name.replace('_results.jsonl', '_run.json'))
    if not manifest.exists() or not json.loads(manifest.read_text()).get('scored'):
        raise ValueError(f'Refusing incomplete result file: {path}')
    return rows, indexed, json.loads(manifest.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('old', type=Path)
    parser.add_argument('new', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results/reports/old-vs-new.html'))
    args = parser.parse_args()
    old_rows, old, old_manifest = load(args.old)
    new_rows, new, new_manifest = load(args.new)
    records = []
    for key in sorted(old.keys() | new.keys()):
        a, b = old.get(key), new.get(key)
        changed = a is None or b is None or any(
            a.get(k) != b.get(k) for k in ('score', 'applicable', 'error'))
        av, bv = (a or {}).get('score'), (b or {}).get('score')
        record = dict(engine=key[0], document=key[1], page=Path(key[1]).stem,
                      leg=key[2], family=(b or a)['doc_type'], old=av, new=bv,
                      delta=bv-av if av is not None and bv is not None else None,
                      old_status=state(a), new_status=state(b), changed=changed,
                      old_reason=(a or {}).get('reason'), new_reason=(b or {}).get('reason'),
                      old_error=(a or {}).get('error'), new_error=(b or {}).get('error'))
        if changed:
            record['old_metrics'] = (a or {}).get('metrics', {})
            record['new_metrics'] = (b or {}).get('metrics', {})
            avs = {verdict_identity(v, i): v for i, v in enumerate((a or {}).get('detail', {}).get('verdicts', []))}
            bvs = {verdict_identity(v, i): v for i, v in enumerate((b or {}).get('detail', {}).get('verdicts', []))}
            record['field_changes'] = [{'field': path, 'old': avs.get(path), 'new': bvs.get(path)}
                                       for path in sorted(avs.keys() | bvs.keys())
                                       if avs.get(path) != bvs.get(path)]
        records.append(record)
    engines = sorted({r['engine'] for r in records})
    summary = []
    for engine in engines:
        for leg in LEGS:
            ar = [r for r in old_rows if r['engine_name'] == engine and r['leg'] == leg]
            br = [r for r in new_rows if r['engine_name'] == engine and r['leg'] == leg]
            if not ar and not br:
                continue
            av, bv = mean_metric(ar), mean_metric(br)
            summary.append(dict(engine=engine, leg=leg, old=av, new=bv,
                                delta=bv-av if av is not None and bv is not None else None,
                                old_n=sum(map(valid, ar)), new_n=sum(map(valid, br)),
                                changed=sum(r['changed'] for r in records if r['engine']==engine and r['leg']==leg)))
    changed = [r for r in records if r['changed']]
    from evaluation.artifacts import current_timestamp
    published = current_timestamp() == new_manifest['timestamp']
    metadata = dict(old_run=old_manifest['timestamp'], new_run=new_manifest['timestamp'],
                    old_file=str(args.old), new_file=str(args.new),
                    old_sha256=hashlib.sha256(args.old.read_bytes()).hexdigest(),
                    new_sha256=hashlib.sha256(args.new.read_bytes()).hexdigest(),
                    old_scorers=old_manifest.get('scorers'), new_scorers=new_manifest.get('scorers'),
                    total_rows=len(records), changed_rows=len(changed),
                    changes_by_leg=dict(Counter(r['leg'] for r in changed)),
                    rescored_from_saved_outputs=bool(new_manifest.get('rescore_of')),
                    kv_judge_cost_usd=sum(float((r.get('metrics') or {}).get('kv_judge_cost_usd') or 0)
                                          for r in new_rows),
                    publication_changed=published)
    payload = dict(metadata=metadata, summary=summary, records=records, labels=LABELS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(payload, indent=2) + '\n')
    columns = ['engine','document','leg','old','new','delta','old_status','new_status','changed','old_reason','new_reason','old_error','new_error']
    with args.output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(records)
    with args.output.with_name(args.output.stem + '-engines.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    encoded = json.dumps(payload, separators=(',', ':')).replace('<', '\\u003c')
    fmt = lambda x: 'N/A' if x is None else f'{x:.4f}'
    initial = ''.join('<tr><td>'+html.escape(r['engine'])+'</td><td>'+LABELS[r['leg']]+'</td>'
                      +''.join('<td>'+fmt(r[k])+'</td>' for k in ('old','new','delta'))
                      +f"<td>{r['old_n']} → {r['new_n']}</td><td>{r['changed']}</td></tr>"
                      for r in summary)
    by_engine_leg = {(r['engine'], r['leg']): r for r in summary}
    matrix = ('<table id="all-engine-scores"><thead><tr><th rowspan="2">Engine</th>'
              + ''.join('<th colspan="2">'+LABELS[leg]+'</th>' for leg in LEGS)
              + '</tr><tr>' + '<th>Old</th><th>New</th>' * len(LEGS)
              + '</tr></thead><tbody>')
    for engine in engines:
        matrix += '<tr><td>'+html.escape(engine)+'</td>'
        for leg in LEGS:
            row = by_engine_leg.get((engine, leg), {})
            for version in ('old', 'new'):
                matrix += '<td>'+fmt(row.get(version))+'</td>'
        matrix += '</tr>'
    matrix += '</tbody></table>'
    document_rows = ''.join(
        '<tr'+(' class="changed"' if r['changed'] else '')+'>'
        + ''.join('<td>'+html.escape(str(value))+'</td>' for value in
                  (r['engine'], r['page'], LABELS[r['leg']], fmt(r['old']),
                   fmt(r['new']), fmt(r['delta']), r['old_status'], r['new_status']))
        + ('<td><button data-index="'+str(i)+'">Compare fields</button></td>'
           if r['changed'] else '<td>Unchanged</td>') + '</tr>'
        for i, r in sorted(enumerate(records), key=lambda item: item[1]['page']))
    template = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OCR results — old vs new</title><style>
body{font:15px/1.5 system-ui;margin:0;background:#f5f7fa;color:#182a3a}main{max-width:1450px;margin:auto;padding:28px}h1{margin:0;font-size:30px}h2{font-size:21px}section{background:white;border:1px solid #d6dfe7;border-radius:10px;padding:20px;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}td,th{padding:9px;border-bottom:1px solid #e1e6ed;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}th{background:#eef3f7}.scroll{overflow:auto}.controls{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin:14px 0}select,input,button{font:inherit;padding:7px;border:1px solid #bdcad6;border-radius:5px;background:white}a{color:#155b93}.muted{color:#536475;font-size:13px}.note{border-left:4px solid #c28c36;padding:10px 16px;background:#fff9ec}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:420px;overflow:auto;background:#f5f7fa;padding:12px}details{margin-top:8px}tr.changed td{background:#fff9ec}.down{color:#a12b25;font-weight:700}.up{color:#1d6247}summary{cursor:pointer}#detail{border:2px solid #aabecb}.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:700px){main{padding:12px}.two{grid-template-columns:1fr}}
</style><main><h1>Old vs new OCR benchmark results</h1>
<p class="note">__PUBLICATION_NOTE__ The comparison includes every saved result, including unchanged rows, errors and pending judgments. Selected responses live in reconciled_outputs; the inference ledger identifies the exact file for each engine, document and leg. The three separate Gemini schema validation calls are excluded from published scores.</p>
<p id="provenance" class="muted"></p><p><a href="old-vs-new.csv">Download every document comparison (CSV)</a> · <a href="old-vs-new-engines.csv">Engine summary (CSV)</a> · <a href="old-vs-new.json">Complete comparison data</a> · <a href="ocr-benchmark-final.html">Published report</a></p>
<section><h2>All engines — old and new scores</h2><p>Every engine and every leg. Each pair of columns shows the old published score and the new rescore. Unchanged results are included. N/A means no scored result.</p><div class="scroll">__MATRIX__</div></section>
<section><h2>What changed</h2><p id="counts"></p><p>The tables below compare every saved score and status in the two runs. Expand a changed result to inspect its old and new metrics and field or cell verdicts. Missing judgments and failed responses remain visible.</p><div id="leg-summary" class="scroll"></div></section>
<section><h2>Engine averages</h2><p class="muted">Scores use the benchmark’s equal weighting of document families, on a 0–1 scale. Δ = new − old. Pages reports scored coverage; a coverage change affects comparability.</p>
<div class="controls"><label>Leg <select id="leg"><option value="">All legs</option><option value="key_value">KV</option></select></label><label>Engine <select id="engine"><option value="">All engines</option></select></label></div>
<div class="scroll"><table><thead><tr><th>Engine</th><th>Leg</th><th>Old</th><th>New</th><th>Δ</th><th>Scored pages</th><th>Changed rows</th></tr></thead><tbody id="engine-rows">__INITIAL__</tbody></table></div></section>
<section><h2>Document results</h2><div class="controls"><label><input id="only-changes" type="checkbox"> Changes only</label><label>Find page or family <input id="search" placeholder="e.g. bank_002"></label><label>Order <select id="sort"><option value="page">Page name</option><option value="magnitude">Largest change</option><option value="decrease">Decreases first</option></select></label></div><p id="row-count" class="muted"></p>
<div class="scroll"><table><thead><tr><th>Engine</th><th>Page</th><th>Leg</th><th>Old</th><th>New</th><th>Δ</th><th>Old status</th><th>New status</th><th>Review</th></tr></thead><tbody id="document-rows">__DOCUMENTS__</tbody></table></div></section>
<section id="detail" hidden><h2 id="detail-title"></h2><p id="detail-scores"></p><a id="evidence">Inspect the saved OCR answer and current published verdicts</a><p class="muted">The evidence link shows the current publication. Old and new verdicts and metrics are compared below.</p><div id="fields"></div><div class="two"><div><h3>Old metrics</h3><pre id="old-metrics"></pre></div><div><h3>New metrics</h3><pre id="new-metrics"></pre></div></div></section>
<noscript>Enable JavaScript to filter all legs and inspect per-document changes. All engine averages and all document results are visible above without JavaScript.</noscript></main>
<script type="application/json" id="data">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=x=>x==null?'N/A':Number(x).toFixed(4),delta=x=>x==null?'—':(x>0?'+':'')+Number(x).toFixed(4),cls=x=>x<0?'down':x>0?'up':'';
$('provenance').textContent='Old: '+D.metadata.old_run+' · New: '+D.metadata.new_run+' · '+(D.metadata.rescored_from_saved_outputs?'Scores recalculated from saved engine responses. ':'')+'New KV judge spend recorded: $'+Number(D.metadata.kv_judge_cost_usd||0).toFixed(6)+'.';
$('counts').textContent=D.metadata.total_rows.toLocaleString()+' rows compared · '+D.metadata.changed_rows+' changed · '+(D.metadata.total_rows-D.metadata.changed_rows).toLocaleString()+' unchanged.';
for(const [k,v] of Object.entries(D.labels))if(k!=='key_value')$('leg').add(new Option(v,k));for(const e of [...new Set(D.records.map(r=>r.engine))].sort())$('engine').add(new Option(e,e));
$('leg-summary').innerHTML='<table><thead><tr><th>Leg</th><th>Rows compared</th><th>Changed</th><th>Unchanged</th></tr></thead><tbody>'+Object.entries(D.labels).map(([k,v])=>{const n=D.records.filter(r=>r.leg===k).length,c=D.metadata.changes_by_leg[k]||0;return '<tr><td>'+esc(v)+'</td><td>'+n+'</td><td>'+c+'</td><td>'+(n-c)+'</td></tr>'}).join('')+'</tbody></table>';
function matching(r){return (!$('leg').value||r.leg===$('leg').value)&&(!$('engine').value||r.engine===$('engine').value)}
function render(){
$('engine-rows').innerHTML=D.summary.filter(matching).map(r=>'<tr><td>'+esc(r.engine)+'</td><td>'+esc(D.labels[r.leg])+'</td><td>'+fmt(r.old)+'</td><td>'+fmt(r.new)+'</td><td class="'+cls(r.delta)+'">'+delta(r.delta)+'</td><td>'+r.old_n+' → '+r.new_n+'</td><td>'+r.changed+'</td></tr>').join('');
const q=$('search').value.toLowerCase();let rows=D.records.map((r,i)=>({...r,index:i})).filter(r=>matching(r)&&(!$('only-changes').checked||r.changed)&&(!q||(r.page+' '+r.family).toLowerCase().includes(q)));
rows.sort((a,b)=>$('sort').value==='page'?a.page.localeCompare(b.page):$('sort').value==='decrease'?(a.delta??Infinity)-(b.delta??Infinity):Math.abs(b.delta??Infinity)-Math.abs(a.delta??Infinity));
$('row-count').textContent=rows.length+' of '+D.records.length+' document results shown. N/A → scored means new coverage, not a numeric improvement from zero.';
$('document-rows').innerHTML=rows.map(r=>'<tr class="'+(r.changed?'changed':'')+'"><td>'+esc(r.engine)+'</td><td>'+esc(r.page)+'</td><td>'+esc(D.labels[r.leg])+'</td><td>'+fmt(r.old)+'</td><td>'+fmt(r.new)+'</td><td class="'+cls(r.delta)+'">'+delta(r.delta)+'</td><td>'+r.old_status+'</td><td>'+r.new_status+'</td><td>'+(r.changed?'<button data-index="'+r.index+'">Compare fields</button>':'Unchanged')+'</td></tr>').join('');
}
$('document-rows').onclick=e=>{const button=e.target.closest('button[data-index]');if(!button)return;const r=D.records[Number(button.dataset.index)];$('detail').hidden=false;$('detail-title').textContent=r.engine+' · '+r.page+' · '+D.labels[r.leg];$('detail-scores').textContent=fmt(r.old)+' → '+fmt(r.new)+' (Δ '+delta(r.delta)+') · '+(r.old_reason||'')+' → '+(r.new_reason||'');$('evidence').href='ocr-benchmark-evidence.html#'+new URLSearchParams({engine:r.engine,leg:r.leg,page:r.page});$('old-metrics').textContent=JSON.stringify(r.old_metrics,null,2);$('new-metrics').textContent=JSON.stringify(r.new_metrics,null,2);$('fields').innerHTML='<h3>Changed field verdicts or matching evidence</h3>'+(r.field_changes||[]).map(f=>'<details><summary>'+esc(f.field)+' · '+esc(f.old?.verdict)+' → '+esc(f.new?.verdict)+'</summary><div class="two"><pre>'+esc(JSON.stringify(f.old,null,2))+'</pre><pre>'+esc(JSON.stringify(f.new,null,2))+'</pre></div></details>').join('');$('detail').scrollIntoView({behavior:'smooth'});};
for(const id of ['leg','engine','only-changes','search','sort'])$(id).oninput=render;render();
</script></html>'''
    publication_note = ('<strong>Published.</strong> “Old” is the previous publication; “New” is now the current benchmark.'
                        if published else '<strong>For review.</strong> “Old” remains the current published benchmark.')
    args.output.write_text(template.replace('__DATA__', encoded).replace('__INITIAL__', initial).replace('__MATRIX__', matrix).replace('__DOCUMENTS__', document_rows).replace('__PUBLICATION_NOTE__', publication_note))
    print(json.dumps(metadata, indent=2))
    print(args.output.resolve())


if __name__ == '__main__':
    main()
