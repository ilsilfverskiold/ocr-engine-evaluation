"""Self-contained saved-result viewer. No model calls or score changes."""
import argparse,base64,collections,hashlib,html,json,mimetypes,sys,io
from PIL import Image
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_benchmark as rb
from evaluation.ground_truth import GroundTruthLoader
from evaluation.kv_schema_code import engine_pairs
from evaluation.normalizers import markdown_for_scoring
from evaluation.code_judges import engine_grids
from evaluation.metrics import normalize_text
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('run',help='Saved scored run timestamp')
parser.add_argument('--output',type=Path,default=Path('results/reports/ocr-benchmark-final.html'))
parser.add_argument('--compare-with',help='Optional saved baseline timestamp')
parser.add_argument('--include-historical-judge',action='store_true',help='Also show unchanged historical silver-document estimates, separate from the six legs')
parser.add_argument('--runs-dir',type=Path,default=rb.RUNS_DIR,help='Read a provisional run outside the repository')
parser.add_argument('--status',default='Work in progress: remaining calls and reference coverage are listed below.',help='Visible run status; do not label an incomplete run final')
args=parser.parse_args(); TS=args.run
rb.RUNS_DIR=args.runs_dir
rows=rb.read_jsonl(rb.RUNS_DIR/f'{TS}_results.jsonl'); inf=rb.read_jsonl(rb.RUNS_DIR/f'{TS}_inference.jsonl')
if not args.include_historical_judge:
 rows=[r for r in rows if r['leg']!='judge']
lookup={(r['engine_name'],r['document_path'],r.get('pass_name') or 'default'):r for r in inf}
old=rb.read_jsonl(rb.RUNS_DIR/f'{args.compare_with}_results.jsonl') if args.compare_with else []; old={(r['engine_name'],r['document_path'],r['leg']):r for r in old}
loader=GroundTruthLoader(); calls={}; images={}; gts={}; evidence=[]
for r in rows:
 d=r['document_path']; e=r['engine_name']; p=r.get('pass_name') or 'default'; key=e+'|'+d+'|'+p
 if d not in images:
  path=Path(d)
  with Image.open(path) as source:
   preview=source.convert('RGB'); preview.thumbnail((1700,2200)); buffer=io.BytesIO(); preview.save(buffer,format='JPEG',quality=82)
  images[d]='data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode();gts[d]=loader.load(path)
 if key not in calls:
  rec=lookup.get((e,d,p),{}); src=rec.get('source_timestamp') or rec.get('timestamp'); call=rb._read_stored_call(src,e,Path(d),gts[d],p) if src else None
  call=call or {}; raw=call.get('raw_text') or '';so=call.get('structured_output') or {};prompt=call.get('prompt_sent');prompt_note='Saved per-call prompt.'
  if not prompt and e in ('claude_sonnet_4_6','gemini_flash') and p=='default':
   from runners.claude_runner import ClaudeRunner
   from runners.gemini_runner import GeminiRunner
   prompt=(ClaudeRunner if e=='claude_sonnet_4_6' else GeminiRunner)._PROMPT;prompt_note='Recovered from the unchanged historical runner default; omitted by the original call logger.'
  if not prompt:prompt_note='No free-form prompt persisted for this native pipeline pass.'
  finish=[]
  def find_stops(v):
   if isinstance(v,dict):
    for k,x in v.items():
     if k in ('finish_reason','finishReason','stop_reason'):finish.append(str(x))
     elif isinstance(x,(dict,list)):find_stops(x)
   elif isinstance(v,list):
    for x in v:find_stops(x)
  find_stops(call.get('raw_response',{}))
  calls[key]={'raw':raw,'structured':so,'model':call.get('model_version') or rec.get('model_version'),'prompt':prompt,'prompt_note':prompt_note,'source':src,'pass':p,'stop_reasons':sorted(set(finish)),'path':str(rb.OCR_OUTPUTS_DIR/str(src)/f'{gts[d].tier}__{gts[d].doc_type}__{Path(d).stem}'/e/(p+'.call.json')),'raw_sha256':hashlib.sha256(raw.encode()).hexdigest(),'json_schema':call.get('json_schema')}
 item={**r,'call_key':key,'page':Path(d).stem,'observation':rb._diagnose(r) if r['leg']!='judge' else 'Historical model-judged estimate, copied unchanged.'}
 c=calls[key]
 if r['leg']=='markdown':
  item['emitted_markdown']=markdown_for_scoring(c['raw'],e,c['structured'])
  item['markdown_payload_source']='structured_output.markdown' if c['structured'].get('markdown') else 'raw_text'
 if r['leg']=='markdown' and r.get('applicable') and not r.get('error') and not str((r.get('metrics') or {}).get('scorer','')).startswith('markdown-structure-'):
  if e=='claude_sonnet_4_6' and Path(d).stem=='invoice_000':
   item['case_note']='Known scoring flaw: all four heading checks (Seller, Client, ITEMS, SUMMARY) received credit from bold emphasis rather than heading markup. Both address paragraphs also matched the same interleaved output paragraph. The saved 16/16 is not evidence of correct document structure.'
 if r['leg']=='text' and r.get('applicable') and not r.get('error'):
  reference=normalize_text(gts[d].plain_text); output=normalize_text(c['raw'])
  ref_tokens=set(reference.split()); out_tokens=set(output.split())
  reproduced=round(len(ref_tokens & out_tokens)/len(ref_tokens),4) if ref_tokens else 1.0
  if reproduced != r['score']: raise ValueError(f'Current text normalization does not reproduce saved score: {key}')
  item.update(normalized_output=output,normalized_reference=reference,unique_reference_tokens=len(ref_tokens),matched_reference_tokens=len(ref_tokens & out_tokens))
  if e=='claude_sonnet_4_6' and Path(d).stem=='invoice_000':
   item['case_note']=f"The raw answer contains {c['raw'].count('&nbsp;')} &nbsp; spacing entities, decoded by the scorer. Seller/client text is interleaved across columns, unlike the reference order; this contributes to CER/WER. Coverage does not assess layout or reading order."
 if r['leg'] in ('schema','key_value'):
  pairs,how=engine_pairs(e,c['raw'],c['structured']);item.update(pairs=pairs,pairs_how=how)
 if r['leg']=='tables':
  from evaluation.table_cells import parse_tables
  table_html=(r.get('detail') or {}).get('output_tables')
  item['grids']=engine_grids(c['raw'],e,c['structured']) if table_html is None else []
  item['logical_tables']=[[vars(cell) for cell in t.cells] for markup in (table_html or []) for t in parse_tables(markup)]
  item['reference_tables']={name:[[vars(cell) for cell in t.cells] for t in parse_tables(comp.get('reference_html',''))] for name,comp in (r.get('detail') or {}).get('reference_comparisons',{}).items()}
 if r['leg']=='markdown':
  item['engine_note']='Structure and text fidelity are separate. Inspect the matched block type and content; bold text cannot satisfy a heading.'
  if e=='gemini_flash' and p=='default':
   item['engine_note']='Historical default transcription request did not explicitly ask for Markdown. This is a prompt-fairness limitation; a dedicated Markdown control is deferred.'
  elif e=='claude_sonnet_4_6' and p=='default':
   item['engine_note']='The historical default request asked to preserve structure but was not a dedicated Markdown-only control. The exact recovered request is shown below.'
 if r['leg']=='tables':
  item['engine_note']='Compare the returned logical cells with the selected reference and inspect missing, wrong and extra cells. A table-only control has not been run for historical general-model default answers.'
 if r['leg']=='geometry':
  item['engine_note']='Location is evaluated from the returned text/box pairs displayed here. Transcription from another pass is not evidence for geometry.'
 if r['leg']=='text':
  item['engine_note']='Coverage ignores order and added/repeated words. CER/WER measure sequence differences; compare the normalized output and reference below before attributing those differences to recognition.'
 if any(stop in ('length','max_tokens','MAX_TOKENS') for stop in c['stop_reasons']):
  item['engine_note']=item.get('engine_note','')+' The provider recorded an output-limit stop for this answer.'
 item['method']=rb._recorded_scoring_method(r['leg'],[r])[1] if r['leg'] in rb.ALL_LEGS else 'Separate model-judged estimate; check saved version and historical status.'
 if r['leg']=='markdown' and str((r.get('metrics') or {}).get('scorer','')).startswith('markdown-structure-'):
  item['method']='Actual Markdown/HTML types and one-to-one block matching. Full-string content similarity: ≥0.85 correct; 0.50–0.85 partial; wrong type scores zero. Bold is not a heading. CER/WER measure text separately.'
 if r['leg']=='markdown' and (r.get('metrics') or {}).get('scorer')in ('markdown-structure-v2-spans','markdown-structure-v3-spans-reflow'):
  item['method']='Locate full/partial content first, then check actual Markdown/HTML structure. Distinct spans may share a paragraph; credited text cannot be reused. Table rows compare ordered nonempty cells. Empty padding is ignored here (the table leg scores blanks). Correct + half partial, divided by reference items. CER/WER remain separate.'
  if r['metrics']['scorer']=='markdown-structure-v3-spans-reflow':
   item['method']+=' Repeated column groups can reflow under matching headers with unique row anchors.'
   if e=='llamaparse' and Path(d).stem=='bank_000':
    item['case_note']='Source-image review: the returned Account Transfers Out row contains the opening balances, and ETF numeric values shift onto the following funds. Consultant/address/preference text is omitted. Daily Date/Amount pairs may legitimately reflow into a narrow table; this score now accepts that layout while checking each date’s own amount. Three cached reference typos were corrected against the image and accepted gold. See docs/follow-up-2026-10-07.md for exact comparisons.'
 if r['leg']=='geometry':
  item['method']='One-to-one text/box matching and mutual centre containment. Recall is the headline; precision/F1 and box-text CER/WER are separate. Scope and line/word units are recorded below.'
 if r['leg']=='key_value' and (r.get('metrics') or {}).get('kv_scorer')=='shared-fields-v5-judged-value-groups':
  item['method']='Shared accepted KV/schema fields. Known keys and valid cached matches first; value containment nominates candidates for contextual judge review, never automatic credit. Judge-confirmed groups require complete value coverage and no reuse. Unfinished reviews remain pending.'
  item['engine_note']='Inspect returned pairs, selected components and contextual judge decisions below. Old isolated-pair rejections do not settle changed partial-value/context reviews.'
 if r['leg']=='schema' and c.get('json_schema') == gts[d].inference_schema:
  item['engine_note']='Current shared-schema request. Failed provider calls remain errors; successful answers are compared with the exact supplied keys shown below.'
 if r['leg']=='schema' and (r.get('metrics') or {}).get('schema_sample'):
  sample=r['metrics']['schema_sample']
  item['method']='Correct values under exact requested keys / (sampled reference fields + unexpected or duplicate output fields). Same frozen schema sample for every engine; no judge.'
  item['engine_note']=f"Current schema run: {sample['selected']} of {sample['available']} eligible fields requested on this page. Full gold is unchanged. The returned answer is scored only against this common sample."
 if r['leg']=='schema' and (r.get('metrics') or {}).get('schema_rerun_required'):
  item['method']='Rerun required: the saved call did not receive the current accepted schema. Its old answer is retained as evidence, not scored against new keys.'
 before=old.get((e,d,r['leg']))
 if before:item['previous_score']=before.get('score')
 evidence.append(item)
valid=lambda r:r.get('applicable') and r.get('score') is not None and not r.get('error')
means=rb.engine_leg_means(rb.aggregate_legs(rows));engines=sorted({r['engine_name'] for r in rows});legs=list(rb.ALL_LEGS)
changes=collections.Counter(r['leg'] for r in evidence if 'previous_score' in r and r['score']!=r['previous_score'])
manifest=json.loads((rb.RUNS_DIR/f'{TS}_run.json').read_text())
if not changes: changes=manifest.get('audit_score_changes',{})
payload={'run':TS,'rows':evidence,'calls':calls,'images':images,'engines':engines,'legs':legs,'manifest':manifest,'changes':dict(changes),'means':{e:{l:means.get((e,l)) for l in legs} for e in engines},'status':args.status}

methods={'text':'Unique normalized GT-token coverage, not character accuracy. CER/WER support it. Sensitive to spacing/punctuation, blind to order and extra words; no judge.','markdown':'Saved code checklist score: (correct + 0.5 × partial) / elements. Known flaw: the scorer promotes bold text to headings and can reuse output blocks. This is not strict Markdown structure accuracy. Heading syntax and block separation are not reliably validated by this saved score.','tables':'Code verdicts on nonempty sampled cells in returned grids, with row/column association. Cached model-authored reference checklist. TEDS is supporting/fallback.','key_value':'Returned scalar pairs versus fields marked as printed label/value pairs by cached aliases. (correct + blank_ok) / printed fields. Code verdicts; no new judge.','schema':'Parsed fields and aligned array rows versus the supplied schema reference. (correct + blank_ok) / sampled fields. Not full JSON equality; extra rows are not independently penalized.','geometry':'Text-associated boxes and mutual-center containment. SROIE uses line fragments; FUNSD/SynFinTabs use words. Line/coarse-block diagnostics stay separate. Errors are excluded from means and shown.','judge':'Old model-judged estimates for silver-GT pages, copied unchanged. Separate from the six legs; no judge rerun.'}
payload['methods']=methods
html_template=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OCR benchmark — audited saved results</title><style>
body{font:15px/1.5 system-ui;margin:0;color:#172b35;background:#f3f5f6}main{max-width:1450px;margin:auto;padding:22px}section,header{background:white;border:1px solid #ccd6dc;border-radius:8px;padding:20px;margin-bottom:16px}h1,h2,h3{line-height:1.2}h1{font-size:28px}h2{font-size:23px}select,input,button{font:inherit;padding:8px;max-width:100%}label{display:inline-block;margin:5px 16px 5px 0}button,summary{cursor:pointer}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:7px 9px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}th{background:#edf1f4;position:sticky;top:0}td{overflow-wrap:anywhere}.scroll{overflow:auto;max-height:500px}.short{max-height:320px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#f2f4f6;padding:12px;max-height:500px;overflow:auto}details{margin:12px 0}summary{font-weight:600}.warn{border-left:5px solid #c88013;padding:8px 14px;background:#fff6e7}.muted{color:#53636e}.cols{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.3fr);gap:18px}img,svg{width:100%;height:auto}#inventory tr{cursor:pointer}#inventory tr:hover{background:#eef6fa}.pill{display:inline-block;padding:3px 8px;background:#e7f0f5;margin:3px;border-radius:4px}a{color:#065ba3}@media(max-width:850px){.cols{grid-template-columns:1fr}main{padding:10px}}
</style><main><header><h1>OCR benchmark: results and evidence</h1><p id="intro"></p><p class="warn" id="status"></p><p>Scores describe the saved configurations. N/A, pending work and failed calls remain visible and do not become zero scores. Inspect the exact prompt and output before assigning a cause.</p></header>
<section><h2>Compare results</h2><label>Engine <select id="engine"></select></label><label>Document type <select id="doctype"></select></label><label>Leg <select id="leg"></select></label><label>Page <input id="search" type="search" placeholder="e.g. invoice_000"></label><p>Means average pages within each document type, then types equally. Counts show scored rows / selected rows. Filters apply to the overview and evidence list.</p><div id="overview" class="scroll"></div><details><summary>Run provenance and changed scores</summary><div id="changes"></div><pre id="manifest"></pre></details><p id="count"></p><div id="inventory" class="scroll short"></div></section>
<section id="case"><h2 id="title"></h2><p id="method"></p><p id="observation"></p><details><summary>Configuration and interpretation for this leg</summary><p id="engine-note"></p></details><p id="provenance"></p><div id="evidence"></div><details><summary>Exact prompt and source provenance</summary><div id="prompt"></div></details><details><summary>Full saved transcription / structured payload (supporting raw data)</summary><div id="raw"></div></details><details><summary>All stored metrics, verdicts and reference repairs</summary><div id="metrics"></div></details></section></main>
<script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);let selected=null;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pretty=o=>'<pre>'+esc(typeof o==='string'?o:JSON.stringify(o,null,2))+'</pre>';
const tbl=(heads,rows)=>'<table><thead><tr>'+heads.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(x=>'<td>'+esc(x)+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
const fmt=x=>x==null?'N/A':Number(x).toFixed(4);
$('status').textContent=D.status;
$('intro').textContent='Run '+D.run+' · '+D.engines.length+' engines/configurations · all saved results included · current scorer versions recorded below.';
$('overview').innerHTML=tbl(['Engine',...D.legs],D.engines.map(e=>[e,...D.legs.map(l=>fmt(D.means[e][l]))]));$('manifest').textContent=JSON.stringify(D.manifest,null,2);$('changes').innerHTML='<p>Scores changed relative to the superseded provisional rescore (changes overlap prior audit batches):</p>'+tbl(['Leg','Changed rows'],Object.entries(D.changes));
function options(id,vals){$(id).innerHTML='<option value="">All</option>'+vals.map(x=>'<option value="'+esc(x)+'">'+esc(x)+'</option>').join('')}
options('engine',D.engines);options('doctype',[...new Set(D.rows.map(r=>r.doc_type))].sort());options('leg',D.legs.concat(D.rows.some(r=>r.leg==='judge')?['judge']:[]));
function list(){let q=$('search').value.toLowerCase(),rs=D.rows.filter(r=>(!$('engine').value||r.engine_name===$('engine').value)&&(!$('doctype').value||r.doc_type===$('doctype').value)&&(!$('leg').value||r.leg===$('leg').value)&&r.page.toLowerCase().includes(q));
const ls=$('leg').value?[$('leg').value]:D.legs,es=[...new Set(rs.map(r=>r.engine_name))].sort();
function mean(e,l){let all=rs.filter(r=>r.engine_name===e&&r.leg===l),valid=all.filter(r=>r.applicable&&r.score!=null&&!r.error),groups={};for(const r of valid)(groups[r.doc_type]??=[]).push(r.score);let types=Object.values(groups).map(a=>a.reduce((x,y)=>x+y,0)/a.length);return (types.length?fmt(types.reduce((a,b)=>a+b,0)/types.length):'N/A')+' ('+valid.length+'/'+all.length+')'}
const extras={text:[['CER ↓','cer'],['WER ↓','wer']],markdown:[['Text CER ↓','markdown_cer'],['Text WER ↓','markdown_wer']],geometry:[['Precision ↑','geometry_precision'],['F1 ↑','geometry_f1'],['Box CER ↓','geometry_cer'],['Box WER ↓','geometry_wer']],tables:[['Precision ↑','cell_precision'],['Recall ↑','cell_recall'],['Structure ↑','structure_similarity']]}[$('leg').value]||[];
function support(e,key){let groups={},n=0;for(const r of rs){const v=r.metrics?.[key];if(r.engine_name===e&&r.applicable&&!r.error&&typeof v==='number'&&Number.isFinite(v)){(groups[r.doc_type]??=[]).push(v);n++}}let types=Object.values(groups).map(a=>a.reduce((x,y)=>x+y,0)/a.length);return types.length?fmt(types.reduce((x,y)=>x+y,0)/types.length)+' (n='+n+')':'N/A'}
$('overview').innerHTML=tbl(['Engine',...ls,...extras.map(x=>x[0])],es.map(e=>[e,...ls.map(l=>mean(e,l)),...extras.map(x=>support(e,x[1]))]));
$('count').textContent=rs.length+' results, including '+rs.filter(r=>r.error).length+' errors and '+rs.filter(r=>!r.error&&(!r.applicable||r.score==null)).length+' pending/N/A. Click a row; use a page filter to compare engines on the same page.';
$('inventory').innerHTML=tbl(['Engine','Leg','Page','Type','Score','Status'],rs.map(r=>[r.engine_name,r.leg,r.page,r.doc_type,fmt(r.score),r.error||r.reason||'scored']));$('inventory').querySelectorAll('tbody tr').forEach((tr,i)=>tr.onclick=()=>show(rs[i]));}

function tablePreview(cells){let covered=new Set(),starts=new Map(),rows=0,cols=0;for(const c of cells){rows=Math.max(rows,c.row+c.rowspan);cols=Math.max(cols,c.col+c.colspan);starts.set(c.row+','+c.col,c);for(let r=c.row;r<c.row+c.rowspan;r++)for(let k=c.col;k<c.col+c.colspan;k++)covered.add(r+','+k)}let out='<div class="scroll"><table aria-label="Table cells and spans"><tbody>';for(let r=0;r<rows;r++){out+='<tr>';for(let k=0;k<cols;k++){let c=starts.get(r+','+k);if(c)out+='<td style="border:1px solid #b9c5ce" rowspan="'+c.rowspan+'" colspan="'+c.colspan+'">'+esc(c.text)+'</td>';else if(!covered.has(r+','+k))out+='<td title="No returned cell at this position; not an explicit blank" style="background:#eee">∅</td>'}out+='</tr>'}return out+'</tbody></table></div>'}
function refs(r){return (r.detail?.matches||[]).map(m=>({text:m.text,bbox:m.reference_bbox})).filter(x=>x.bbox)}
function boxes(r){return r.detail?.scored_boxes||[]}
function geometry(r){const refs_=refs(r),pred=boxes(r),m=r.metrics||{},outside=pred.filter(w=>w.bbox.some(c=>c<0||c>1)).length;return '<p><strong>'+m.geometry_located_words+'/'+m.geometry_reference_words+' reference units located</strong>; '+m.geometry_matched_words+' text-matched. '+pred.length+' returned boxes after the recorded coordinate conversion; '+outside+' extend outside the page. Line diagnostic: '+fmt(m.geometry_line_score)+'.</p>'+tbl(['Metric','Value'],[['Located precision',fmt(m.geometry_precision)],['Located recall',fmt(m.geometry_recall??m.geometry_score)],['Located F1',fmt(m.geometry_f1)],['Box-text CER',fmt(m.geometry_cer)],['Box-text WER',fmt(m.geometry_wer)],['Scoring unit',m.geometry_scoring_unit],['Invalid boxes',m.geometry_invalid_boxes],['Outside annotated scope',m.geometry_outside_scope_predictions]])+'<details><summary>Annotated scope and excluded predictions</summary>'+pretty(r.detail?.scope||m.geometry_scope||{})+pretty(r.detail?.excluded_predictions||[])+'</details><label><input id="ref-toggle" type="checkbox" checked> Reference green</label><label><input id="pred-toggle" type="checkbox" checked> Returned red</label><label><input id="extent-toggle" type="checkbox"> Fit all returned boxes, including outside page</label><p class="muted">Preview image is compressed for display; original full-resolution page is linked above. Hover a box for its attached word and coordinates. Red means engine output, not necessarily wrong. Matching may combine native words into SROIE line fragments; full native boxes remain shown.</p><div id="overlay"></div><details><summary>Every reference/returned match and placement verdict</summary>'+tbl(['Reference text','Returned text','Located','Reference box','Matched output box'],(r.detail?.matches||[]).map(x=>[x.text,x.predicted_text,x.hit,JSON.stringify(x.reference_bbox),JSON.stringify(x.predicted_bbox)]))+'</details><details><summary>All returned boxes, including unmatched and out-of-frame</summary>'+pretty(pred)+'</details>'}
function overlay(){const r=selected,img=new Image();img.onload=()=>{if(selected!==r)return;let ratio=img.naturalHeight/img.naturalWidth;let ws=boxes(r),rs=refs(r),ext=$('extent-toggle').checked?ws.flatMap(w=>[[w.bbox[0],w.bbox[1]],[w.bbox[2],w.bbox[3]]]):[];let x0=Math.min(0,...ext.map(x=>x[0])),x1=Math.max(1,...ext.map(x=>x[0])),y0=Math.min(0,...ext.map(x=>x[1])),y1=Math.max(1,...ext.map(x=>x[1]));let s='<svg viewBox="'+[x0*1000,y0*1000*ratio,(x1-x0)*1000,(y1-y0)*1000*ratio].join(' ')+'" role="img" aria-label="Actual returned boxes and scored reference boxes"><image href="'+D.images[r.document_path]+'" width="1000" height="'+1000*ratio+'"/>';
for(const [arr,color,on] of [[rs,'#169346',$('ref-toggle').checked],[ws,'#d52b40',$('pred-toggle').checked]])if(on)for(const b of arr){let [a,c,d,e]=b.bbox;s+='<rect x="'+Math.min(a,d)*1000+'" y="'+Math.min(c,e)*1000*ratio+'" width="'+Math.abs(d-a)*1000+'" height="'+Math.abs(e-c)*1000*ratio+'" fill="none" stroke="'+color+'" stroke-width="1.4"><title>'+esc(b.text+' '+JSON.stringify(b.bbox))+'</title></rect>'}$('overlay').innerHTML=s+'</svg>'};img.src=D.images[r.document_path]}
function show(r){selected=r;const c=D.calls[r.call_key],m=r.metrics||{},v=r.detail?.verdicts||[];$('title').textContent=r.engine_name+' · '+(r.leg==='text'?'text coverage':r.leg)+' · '+r.page+' · '+fmt(r.score);$('method').textContent=r.method||D.methods[r.leg]||'';$('observation').textContent=r.leg==='text'&&r.normalized_output!==undefined ? r.matched_reference_tokens+'/'+r.unique_reference_tokens+' distinct normalized reference tokens found. This measures token coverage, not perfect transcription. '+(r.case_note||'') : (r.case_note ? r.case_note+' ' : '')+'Scorer observation: '+r.observation;$('engine-note').textContent=r.engine_note||'See the saved request, output and per-item verdicts below.';$('provenance').textContent='Pass: '+r.pass_name+' · recorded model: '+c.model+' · stop reason: '+(c.stop_reasons.join(', ')||'not recorded')+' · source run: '+c.source;$('provenance').innerHTML+=' · <a href="'+esc('file://'+r.document_path)+'" target="_blank">Open original full-resolution page</a>';
let body='';if(r.error){body='<p class="warn">'+esc(r.error)+'</p><p>This is an error, not a scored zero.</p>'}else if(r.leg==='geometry'){body=boxes(r).length||r.detail?.matches?geometry(r):'<p>No scored word-box payload. Native structured output:</p>'+pretty(c.structured)}else if(r.leg==='key_value'||r.leg==='schema'){let vs=v;body='<h3>Returned '+(r.leg==='key_value'?'field pairs':'structured answer')+'</h3>'+(r.leg==='schema'?pretty(c.raw):'')+'<p>'+esc(r.pairs_how)+'</p><div class="scroll">'+tbl(['Returned key','Returned value'],r.pairs||[])+'</div><h3>Field-by-field scoring (provisional when pending)</h3><div class="scroll">'+tbl(['Reference field','Expected','Matched key','Returned value','Verdict','Reason'],vs.map(x=>[x.path,x.expected,x.key,x.value,x.verdict,x.how]))+'</div><details><summary>Contextual judge reviews</summary>'+pretty(r.detail?.judge_reviews||[])+'</details><details><summary>Unmatched-pair decisions / unexpected schema fields</summary>'+pretty(r.leg==='key_value'?(r.detail?.unresolved_keys||[]):(r.detail?.extra_fields||[]))+'</details>'}else if(r.leg==='tables'){body='<h3>Table metrics</h3>'+tbl(['Metric','Value'],[['Cell F1',fmt(m.cell_f1)],['Precision',fmt(m.cell_precision)],['Recall',fmt(m.cell_recall)],['Correct / reference / output cells',[m.correct,m.reference_cells,m.output_cells].join(' / ')],['Correct / reference blank cells',[m.correct_blank_cells,m.reference_blank_cells].join(' / ')],['Span-aware structure similarity',fmt(m.structure_similarity)],['Selected reference',m.selected_reference]])+'<h3>Returned logical cells used by the scorer</h3>';for(let i=0;i<(r.logical_tables||[]).length;i++)body+='<h4>Table '+(i+1)+'</h4>'+tablePreview(r.logical_tables[i])+'<details><summary>Logical cells and spans</summary>'+tbl(['Row','Column','Rowspan','Colspan','Text'],r.logical_tables[i].map(c=>[c.row+1,c.col+1,c.rowspan,c.colspan,c.text]))+'</details>';if(!r.logical_tables?.length)for(const g of r.grids||[])body+=tbl(['Row','Cells'],g.map((x,i)=>[i+1,JSON.stringify(x)]));for(const [name,tables] of Object.entries(r.reference_tables||{}))body+='<details><summary>Reference: '+esc(name)+(name===m.selected_reference?' (selected)':'')+'</summary>'+tables.map(tablePreview).join('')+'</details>';body+='<h3>Cell-by-cell verdicts</h3>'+tbl(['Reference table / row / column','Expected','Output table / row / column','Returned','Verdict'],v.map(x=>[x.reference?[x.reference_table+1,x.reference.row+1,x.reference.col+1].join(' / '):'extra',x.reference?.text??x.value,x.output?[x.output_table+1,x.output.row+1,x.output.col+1].join(' / '):'missing',x.output?.text,x.verdict]))+'<details><summary>Reference comparisons (both versions for shipping)</summary>'+tbl(['Version','F1','Precision','Recall'],Object.entries(r.detail?.reference_comparisons||{}).map(([k,x])=>[k,fmt(x.f1),fmt(x.precision),fmt(x.recall)]))+pretty(r.detail?.reference_comparisons||{})+'</details>'}
else if(r.leg==='markdown'){const modern=String(m.scorer||'').startsWith('markdown-structure-');body=(modern||r.score==null?'':'<p class="warn">Historical scorer: bold-as-heading and block-reuse defects may affect this saved score.</p>')+'<h3>Structure and text metrics</h3>'+tbl(['Metric','Value'],[['Headline',fmt(r.score)],['Block precision',fmt(m.element_precision)],['Block recall',fmt(m.element_recall)],['Block F1',fmt(m.element_f1)],['Reading order',fmt(m.order_score)],['Markdown-text CER',fmt(m.markdown_cer)],['Markdown-text WER',fmt(m.markdown_wer)],['Text reference',m.markdown_text_reference_source]])+'<h3>Exact returned Markdown</h3><p>Scored payload: '+esc(r.markdown_payload_source)+'</p>'+pretty(r.emitted_markdown)+'<h3>Structural-element verdicts</h3>'+tbl(['Type','Expected content','Verdict','Matched content (normalized)','Reason'],v.map(x=>[x.type,x.content,x.verdict,(x.evidence||[]).map(s=>'Block '+s.block+' ['+s.token_start+':'+s.token_end+']: '+s.text).join('\n'),x.how]));if(!v.length)body+='<p>No catalogue verdicts available; see the recorded reason above.</p>'}
else if(r.leg==='text'){body='<h3>Text metrics</h3>'+tbl(['Metric','Saved value'],[['Distinct-token coverage',fmt(m.gt_coverage)],['Character error rate (lower is better)',fmt(m.cer)],['Word error rate (lower is better)',fmt(m.wer)],['Word precision',fmt(m.word_precision)],['Word recall',fmt(m.word_recall)],['Word F1',fmt(m.word_f1)],['Output/reference token count',m.output_words+' / '+m.gt_words]])+'<p>CER/WER compare sequences, so reading order and reference differences can raise them without misrecognized words. Coverage ignores order, extra words and repetition.</p><h3>Normalized output used for text comparison</h3><p>Same Markdown/HTML conversion, whitespace collapse and lowercasing as the scorer. Exact raw output remains below.</p>'+pretty(r.normalized_output)+'<details><summary>Normalized reference used for comparison</summary>'+pretty(r.normalized_reference)+'</details><details><summary>Exact saved output, including formatting markup</summary>'+pretty(c.raw)+'</details><h3>Unmatched reference tokens</h3><p>Differences may reflect recognition, tokenization or reference errors. Saved list is capped at 50 tokens.</p>'+pretty(r.detail?.missed_words||[])}else{body=pretty(r.detail||{})}
if(r.applicable===false&&!r.error)body='<p class="warn"><strong>Pending / not scored:</strong> '+esc(r.reason||'No reason recorded')+'</p><p>Excluded from means. The saved answer below is evidence, not a scored zero.</p>'+body;
$('evidence').innerHTML=r.leg==='geometry'&&r.applicable&&!r.error?body:'<div class="cols"><div><h3>Source page</h3><img alt="Source page '+esc(r.page)+'" src="'+D.images[r.document_path]+'"></div><div>'+body+'</div></div>';
$('prompt').innerHTML='<p>'+esc(c.prompt_note)+'</p>'+pretty(c.prompt||'')+(c.json_schema?'<h3>Exact supplied schema</h3>'+pretty(c.json_schema):'')+pretty({source_call:c.path,raw_text_sha256:c.raw_sha256});$('raw').innerHTML=pretty(c.raw)+pretty(c.structured);$('metrics').innerHTML=pretty(m)+pretty(r.detail||{});
if(r.leg==='geometry'&&!r.error&&$('ref-toggle')){for(const id of ['ref-toggle','pred-toggle','extent-toggle'])$(id).onchange=overlay;overlay()} }
for(const id of ['engine','doctype','leg','search'])$(id).oninput=list;list();show(D.rows.find(r=>r.engine_name==='textract_text'&&r.leg==='geometry'&&r.page==='funsd_000')||D.rows[0]);
</script></html>'''
page=html_template.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029'))
path=args.output;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(page)
print(path.resolve(),path.stat().st_size,'bytes;',len(evidence),'results;',len(calls),'source calls;',len(images),'source pages',flush=True)
