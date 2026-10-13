"""Small, offline engine-comparison report; detailed payloads live separately."""
import html
import json
from collections import defaultdict
from pathlib import Path

from evaluation.reporting import (METRICS, EXTRA_METRICS, METHOD, comparison,
                                  mean_metric, metric_value, valid, case_key, _operations)

LEGS = ['text', 'markdown', 'tables', 'key_value', 'schema', 'geometry']
LABELS = dict(zip(LEGS, ['Text', 'Markdown', 'Tables', 'Key–value', 'Schema', 'Geometry']))


def build(rows, inference, run, output, evidence_name='ocr-benchmark-evidence.html'):
    judged = [r for r in rows if r['leg'] == 'judge']
    rows = [r for r in rows if r['leg'] in LEGS]
    analysis = comparison(rows)
    excluded = {(c['document_path'], c['leg']) for c in analysis['cases']}
    tiers = {r['document_path']: r['tier'] for r in rows}
    for case in analysis['cases']:
        case['tier'] = tiers[case['document_path']]
    scopes = defaultdict(list)
    for r in rows:
        for tier, family in [('', ''), (r['tier'], ''), ('', r['doc_type']), (r['tier'], r['doc_type'])]:
            scopes[tier + '|' + family].append(r)
    data = {}
    metrics = {leg: METRICS[leg] + EXTRA_METRICS.get(leg, []) for leg in LEGS}
    for scope, records in scopes.items():
        result = {}
        for leg in LEGS:
            group = [r for r in records if r['leg'] == leg]
            ops = _operations(group, inference)
            engines = []
            for engine in sorted({r['engine_name'] for r in group}):
                rs = [r for r in group if r['engine_name'] == engine]
                kept = [r for r in rs if case_key(r) not in excluded]
                if not any(valid(r) or r.get('error') for r in rs):
                    continue
                counts = {key: sum(valid(r) and isinstance(metric_value(r, key), (float, int)) for r in rs)
                          for _, key, _ in metrics[leg]}
                engines.append(dict(engine=engine, full=mean_metric(rs), trimmed=mean_metric(kept),
                    n=sum(valid(r) for r in rs), kept=sum(valid(r) for r in kept),
                    errors=sum(bool(r.get('error')) for r in rs),
                    na=sum(not valid(r) and not r.get('error') for r in rs),
                    metrics={key: mean_metric(rs, key) for _, key, _ in metrics[leg]}, counts=counts,
                    **ops[engine]))
            result[leg] = engines
        data[scope] = result
    sonnet_geometry = [r for r in inference if r['engine_name'] == 'claude_sonnet_4_6'
                       and r.get('pass_name') == 'geometry' and not r.get('error')]
    pixel_calls = sum('absolute pixel coordinates in this supplied image' in (r.get('prompt_used') or '')
                      for r in sonnet_geometry)
    if pixel_calls:
        geometry_note = (f"Sonnet geometry uses the pixel-coordinate prompt on {pixel_calls}/{len(sonnet_geometry)} "
                         "successful saved calls. Coordinates are normalized using each recorded sent-image frame. "
                         "Prompt improvements do not guarantee correct boxes; inspect the overlays and stop reasons.")
    else:
        geometry_note = ("Sonnet geometry still uses the saved normalized-coordinate requests; "
                         "the prepared pixel-coordinate rerun is not represented in these results.")
    judge_scopes = {}
    for scope in data:
        tier, family = scope.split('|')
        group = [r for r in judged if (not tier or r['tier'] == tier)
                 and (not family or r['doc_type'] == family)]
        judge_scopes[scope] = [dict(engine=e, full=mean_metric(rs),
            n=sum(valid(r) for r in rs), errors=sum(bool(r.get('error')) for r in rs))
            for e in sorted({r['engine_name'] for r in rows if r['leg'] == 'text'})
            for rs in [[r for r in group if r['engine_name'] == e]]]
    judge_note = (f"Saved model-judged checklist accuracy on {len({r['document_path'] for r in judged})} pages. "
                  "Separate from text coverage and CER/WER. Engines with no saved verdict are N/A, not zero.")
    if any((r.get('metrics') or {}).get('historical_verdicts') for r in judged):
        judge_note += " Historical verdicts are restored unchanged; the older substring shortcut may inflate these scores."
    payload = dict(scopes=data, judge_scopes=judge_scopes, judge_note=judge_note, legs=LEGS, labels=LABELS, metrics=metrics, methods=METHOD,
                   evidence=evidence_name, cases=analysis['cases'])
    encoded = json.dumps(payload, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    esc = html.escape
    def fmt(value):
        return 'N/A' if value is None else f'{value:.4f}'
    # Immediate, readable initial overview even before JavaScript runs.
    overall = data['|']
    all_engines = sorted({r['engine'] for rs in overall.values() for r in rs})
    cells = {(r['engine'], leg): r['full'] for leg, rs in overall.items() for r in rs}
    initial = '<table><thead><tr><th>Engine</th>' + ''.join('<th>'+LABELS[l]+' ↑</th>' for l in LEGS) + '</tr></thead><tbody>'
    for engine in sorted(all_engines, key=lambda e: (cells.get((e, 'text')) is None, -(cells.get((e, 'text')) or 0), e)):
        initial += '<tr><td>'+esc(engine)+'</td>'+''.join('<td>'+fmt(cells.get((engine,l)))+'</td>' for l in LEGS)+'</tr>'
    initial += '</tbody></table>'
    template = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OCR benchmark — engine comparison</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:#182a3a;font:15px/1.5 system-ui}main{max-width:1560px;margin:auto;padding:28px}h1{font-size:30px;margin:0}h2{font-size:21px;margin:0 0 10px}p{margin:8px 0 14px}a{color:#155b93}header{margin-bottom:20px}section{background:white;border:1px solid #d6dfe7;border-radius:10px;padding:20px;margin:18px 0}.muted{color:#536475;font-size:13px}.controls{display:flex;flex-wrap:wrap;gap:18px;align-items:center;margin:18px 0}select,button{font:inherit;border:1px solid #bdcad6;border-radius:5px;background:white;color:inherit;padding:7px 11px;cursor:pointer}.tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}.tabs button[aria-selected=true]{background:#1d526f;color:white}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:9px 12px;text-align:right;border-bottom:1px solid #e1e6ed;white-space:nowrap}th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:white}th{background:#eef3f7;cursor:pointer}th:first-child{background:#eef3f7}td.best{font-weight:750;background:#e9f4ed;color:#174e31}td.na{color:#88939d}tr:hover td{background:#f0f6fa}summary{cursor:pointer;font-weight:600}details{margin:15px 0}.note{border-left:3px solid #c28c36;padding-left:12px}.pill{font-size:12px;color:#617284;margin-left:6px}@media(max-width:650px){main{padding:12px}section{padding:12px}h1{font-size:25px}}
</style><main><header><h1>OCR benchmark</h1><p>Engine comparison · __RUN__ · __ENGINES__ engines/configurations · __PAGES__ recorded pages</p><a href="__EVIDENCE__">Open supporting evidence and individual document results →</a></header>
<div class="controls"><label>Difficulty <select id="tier"><option value="">All difficulties</option><option>easy</option><option>medium</option><option>hard</option></select></label><label>Document family <select id="family"><option value="">All document families</option></select></label><button id="reset">Reset filters</button></div>
<section><h2>Scores by engine and leg</h2><p class="muted">Full scores; higher is better. Best values are highlighted. Click any column to rank engines; click a score for its evidence. Scores from different legs are not blended.</p><div id="overview" class="scroll">__INITIAL__</div></section>
<section><div id="tabs" class="tabs" role="tablist" aria-label="Evaluation leg"></div><h2 id="leg-title"></h2><p id="method"></p><p id="scope-note" class="muted"></p><div id="ranking" class="scroll"></div><p class="muted">Full = all scored pages. Without outliers = the same scores with the common low-page exclusions below. Pages shows Full → retained. Supporting metrics use Full pages; parentheses show fewer available metric pages. Click any metric to sort; click again to reverse. N/A stays last.</p><details><summary>Other metrics for this leg</summary><div id="extra" class="scroll"></div></details><div id="judged-text"><h2>Text on pages without trusted references</h2><p id="judged-note"></p><div id="judged-ranking" class="scroll"></div><p><a href="__EVIDENCE__#leg=judge">Inspect judged-text pages and every saved item verdict →</a></p></div><h2>Cost and latency</h2><div id="operations" class="scroll"></div><p class="muted">Actual saved pass serving each scored page; steady-state where recorded. Lowest cost first initially. Shared calls are paid once: do not sum costs across legs. Excludes judge, startup and failed-call spend. Local $0 excludes machine expense.</p><details><summary>Outlier exclusions for this leg</summary><p>At least three engines scored the page, all ≤0.25, with no failed or pending participant. Every engine uses the same excluded page/leg cases. This is a sensitivity comparison, not proof the references are wrong.</p><div id="exclusions"></div></details><p><a id="leg-evidence" href="__EVIDENCE__">Inspect this leg’s document results and evidence →</a></p></section>
<details><summary>Aggregation and current limitations</summary><p>Scores average pages within each difficulty/document-family group, then weight those groups equally, matching the terminal report. Errors and N/A are excluded from means and shown separately. Different page coverage can affect rankings. Removing a whole group can change the average in either direction.</p><p>Difficulty filters also work for small legs, but the default compares all their pages together. Cost and latency are arithmetic page means and include their own coverage counts.</p><p class="note">__GEOMETRY_NOTE__ Exact requests, model versions and stop reasons are available in the evidence.</p></details><noscript><p>JavaScript enables filters, metric tables and sorting. The overall scores above remain visible.</p></noscript></main>
<script id="comparison-data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('comparison-data').textContent),$=id=>document.getElementById(id);let leg='text';const sort={};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(v,n=4)=>v==null?'N/A':Number(v).toFixed(n);
function link(engine='',l=leg){return D.evidence+'#'+new URLSearchParams({engine,leg:l,tier:$('tier').value,doctype:$('family').value})}
function table(id,cols,rows,defaultKey,defaultUp){let state=sort[id]||[defaultKey,defaultUp],idx=cols.findIndex(c=>c.key===state[0]);if(idx<0){state=[defaultKey,defaultUp];idx=cols.findIndex(c=>c.key===state[0])}const column=cols[idx],value=r=>column.get(r);rows=[...rows].sort((a,b)=>{const x=value(a),y=value(b);if(x==null||y==null)return x==null?(y==null?0:1):-1;return (typeof x==='number'?x-y:String(x).localeCompare(String(y)))*(state[1]?1:-1)});const best=cols.map(c=>{const vs=rows.map(c.get).filter(v=>typeof v==='number');return vs.length?(c.high?Math.max(...vs):Math.min(...vs)):null});$(id).innerHTML='<table><thead><tr>'+cols.map((c,i)=>'<th tabindex="0" aria-sort="'+(i===idx?(state[1]?'ascending':'descending'):'none')+'">'+esc(c.label)+(i===idx?(state[1]?' ▴':' ▾'):'')+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+cols.map((c,i)=>{const v=c.get(r);return '<td class="'+(v==null?'na':c.highlight&&v===best[i]?'best':'')+'">'+(c.render?c.render(r):esc(typeof v==='number'?fmt(v):v??'N/A'))+'</td>'}).join('')+'</tr>').join('')+'</tbody></table>';$(id).querySelectorAll('th').forEach((th,i)=>{const change=()=>{sort[id]=[cols[i].key,state[0]===cols[i].key?!state[1]:!cols[i].high];render()};th.onclick=change;th.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();change()}}})}
const engineCol={key:'engine',label:'Engine',get:r=>r.engine,render:r=>'<a href="'+esc(link(r.engine))+'">'+esc(r.engine)+'</a>'};
function metricCol([label,key,high]){return {key,label:label+(high?' ↑':' ↓'),high,highlight:true,get:r=>r.metrics[key],render:r=>fmt(r.metrics[key])+(r.counts[key]&&r.counts[key]<r.n?' <span class="pill">('+r.counts[key]+')</span>':'')}}
function render(){const scope=D.scopes[$('tier').value+'|'+$('family').value]||{},rs=scope[leg]||[],engines=[...new Set(Object.values(scope).flat().map(r=>r.engine))].sort(),lookup=Object.fromEntries(D.legs.map(l=>[l,Object.fromEntries((scope[l]||[]).map(r=>[r.engine,r]))]));
table('overview',[{...engineCol,render:r=>esc(r.engine)},...D.legs.map(l=>({key:l,label:D.labels[l]+' ↑',high:true,highlight:true,get:r=>lookup[l][r.engine]?.full,render:r=>'<a href="'+esc(link(r.engine,l))+'">'+fmt(lookup[l][r.engine]?.full)+'</a>'}))],engines.map(engine=>({engine})),'text',false);
$('leg-title').textContent=D.labels[leg];$('method').textContent=D.methods[leg];$('scope-note').textContent=($('tier').value||'All difficulties')+' · '+($('family').value||'All document families')+' · '+rs.length+' scored/failed engine configurations';
const headline=[engineCol,{key:'full',label:'Full ↑',high:true,highlight:true,get:r=>r.full},{key:'trimmed',label:'Without outliers ↑',high:true,highlight:true,get:r=>r.trimmed},{key:'n',label:'Pages → kept',high:true,get:r=>r.n,render:r=>r.n+' → '+r.kept}];
const metrics=D.metrics[leg],mainN={text:3,markdown:4,tables:3,geometry:4,key_value:3,schema:4}[leg];
table('ranking',[...headline,...metrics.slice(0,mainN).map(metricCol),{key:'errors',label:'Errors ↓',get:r=>r.errors,render:r=>String(r.errors)},{key:'na',label:'N/A',get:r=>r.na,render:r=>String(r.na)}],rs,'full',false);
table('extra',[engineCol,...metrics.map(metricCol)],rs,metrics[0][1],!metrics[0][2]);
$('judged-text').hidden=leg!=='text';$('judged-note').textContent=D.judge_note;
table('judged-ranking',[{...engineCol,render:r=>'<a href="'+esc(link(r.engine,'judge'))+'">'+esc(r.engine)+'</a>'},{key:'full',label:'Judged item accuracy ↑',high:true,highlight:true,get:r=>r.full},{key:'n',label:'Pages',get:r=>r.n,render:r=>String(r.n)},{key:'errors',label:'Errors',get:r=>r.errors,render:r=>String(r.errors)}],D.judge_scopes[$('tier').value+'|'+$('family').value]||[],'full',false);
table('operations',[engineCol,{key:'cost',label:'$/page ↓',get:r=>r.cost,highlight:true,render:r=>fmt(r.cost,6)},{key:'latency',label:'Seconds/page ↓',get:r=>r.latency,highlight:true,render:r=>fmt(r.latency,2)},{key:'cost_n',label:'Cost / time pages',get:r=>r.cost_n,render:r=>r.cost_n+' / '+r.latency_n}],rs,'cost',true);
const cs=D.cases.filter(c=>c.leg===leg&&(!$('family').value||c.doc_type===$('family').value)&&(!$('tier').value||c.tier===$('tier').value));$('exclusions').innerHTML=cs.length?cs.map(c=>'<p>'+esc(c.page)+' · '+esc(c.doc_type)+' · '+c.engines+' engines · '+fmt(c.min)+'–'+fmt(c.max)+'</p>').join(''):'<p>No exclusions in this selection.</p>';
$('leg-evidence').href=link();$('tabs').querySelectorAll('button').forEach(b=>b.setAttribute('aria-selected',b.dataset.leg===leg));}
function families(){const prev=$('family').value,tier=$('tier').value;$('family').innerHTML='<option value="">All document families</option>'+Object.keys(D.scopes).filter(k=>k.startsWith(tier+'|')&&k.split('|')[1]).map(k=>k.split('|')[1]).sort().map(v=>'<option value="'+esc(v)+'">'+esc(v)+'</option>').join('');if([...$('family').options].some(o=>o.value===prev))$('family').value=prev}
$('tabs').innerHTML=D.legs.map(l=>'<button role="tab" data-leg="'+l+'">'+D.labels[l]+'</button>').join('');$('tabs').querySelectorAll('button').forEach(b=>b.onclick=()=>{leg=b.dataset.leg;render()});$('tier').onchange=()=>{families();render()};$('family').onchange=render;$('reset').onclick=()=>{$('tier').value='';$('family').value='';families();render()};families();render();
</script></html>'''
    replacements = {'__RUN__': esc(run), '__ENGINES__': str(len(all_engines)),
                    '__PAGES__': str(len({r['document_path'] for r in rows})),
                    '__EVIDENCE__': esc(evidence_name), '__INITIAL__': initial, '__DATA__': encoded,
                    '__GEOMETRY_NOTE__': esc(geometry_note)}
    for key, value in replacements.items():
        template = template.replace(key, value)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template)
    return output
