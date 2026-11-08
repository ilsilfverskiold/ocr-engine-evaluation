// Interactive viewer for the published run. It reads the saved benchmark files when it opens; nothing is built
// or embedded. Serve the repository directory (`python -m http.server 8000` in ocr-benchmark/) and open
// http://localhost:8000/viewer/viewer.html: browsers do not let a file:// page read other local files.
const HERE = new URL(".", document.currentScript.src);
const ROOT = new URL("../", HERE);                           // the repository: results/, documents/, engines.py
const LEG_TITLES = {text: "Text", markdown: "Markdown", tables: "Tables", key_value: "Key-value",
  schema: "Schema", geometry: "Geometry"};
const D = {legs: [], docs: {}, baseline: null, manifest: null, ledger: new Map(),
  // Slim copies of every recorded row and ledger entry, for the Results view (results.js).
  results: {rows: [], judge: [], ledger: []}};
let docs = [], engines = [];
const state = {view: "page", engine: null, doc: null, leg: "text", filter: "all",
  layers: {ref: true, pred: true, unmatched: true}, geoLevel: "word"};

const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const clean = html => window.DOMPurify ? DOMPurify.sanitize(html) : esc(html);
const md = src => window.marked ? clean(marked.parse(src || "")) : `<pre>${esc(src)}</pre>`;
// The scorer removes one outer Markdown fence as transport packaging; render the same.
const unfence = src => { const m = /^\s*```(?:markdown|md)?\s*\n([\s\S]*?)\n```\s*$/i.exec(src || ""); return m ? m[1] : (src || ""); };
// Fixed-point text as Python prints it: an exact decimal tie rounds to even (toFixed rounds it up); everything else
// agrees, both being correctly rounded from the binary value. A double is an exact tie at the n-th decimal only
// when x·2^(n+1) is an odd integer, and scaling by a power of two is exact.
function pyFixed(x, n) {
  const y = x * 2 ** (n + 1);
  if (Number.isInteger(y) && Math.abs(y) % 2 === 1) {
    const k = (Math.abs(x) * 2 * 10 ** n - 1) / 2;
    return (Math.sign(x) * (k % 2 === 0 ? k : k + 1) / 10 ** n).toFixed(n);
  }
  return x.toFixed(n);
}
const fmt = v => v == null ? "—" : typeof v === "number" ? (Math.abs(v) <= 1 && !Number.isInteger(v) ? pyFixed(v, 3) : String(+pyFixed(v, 4))) : esc(v);
const el = id => document.getElementById(id);

// ---------- light / dark ----------
// Follows the system until the reader picks one; the pick is remembered in this browser only.
const THEME_KEY = "ocr-viewer-theme";
const effectiveTheme = () => document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
function setTheme(theme) {
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
  const button = el("theme-toggle");
  if (button) button.textContent = effectiveTheme() === "dark" ? "☀ Light mode" : "☾ Dark mode";
}
try { setTheme(localStorage.getItem(THEME_KEY)); } catch (e) { setTheme(null); }
matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => setTheme(null));
el("theme-toggle")?.addEventListener("click", () => {
  const next = effectiveTheme() === "dark" ? "light" : "dark";
  setTheme(next);
  try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* storage blocked: the choice lasts this visit */ }
});
const isEmpty = v => v == null || (typeof v === "object" && !Object.keys(v).length);

// ---------- reading files ----------
const fetched = new Map();
function once(url, kind) {
  const key = `${kind} ${url}`;
  if (!fetched.has(key)) fetched.set(key, fetch(url, {cache: "no-cache"}).then(r => r.ok ? (kind === "json" ? r.json() : r.text()) : null));
  return fetched.get(key);
}
const getJSON = url => once(String(url), "json");   // null when the file does not exist
const getText = url => once(String(url), "text");
// A served URL as a path relative to the repository.
const rel = url => decodeURIComponent(new URL(url).pathname.slice(ROOT.pathname.length));
const stemOf = name => name.split("/").pop().replace(/\.[^.]*$/, "");
const docDir = path => { const p = path.split("/"); return `${p.at(-3)}__${p.at(-2)}__${stemOf(p.at(-1))}`; };

// A saved path, relative to the repository or absolute on the machine that scored it, as a served URL.
function fileURL(path) {
  if (!path.startsWith("/")) return new URL(path, ROOT);
  const i = path.lastIndexOf("/documents/");
  return i >= 0 ? new URL(path.slice(i + 1), ROOT) : null;
}

// The server's directory listing, as file names.
async function listDir(url) {
  const html = await getText(url);
  if (html == null) return null;
  return [...new DOMParser().parseFromString(html, "text/html").querySelectorAll("a[href]")]
    .map(a => decodeURIComponent(a.getAttribute("href")));
}

async function eachLine(url, fn) {
  // Always revalidate: the local server sets no Cache-Control, and a ledger rewritten on disk
  // was otherwise served from the browser cache (heuristic freshness) on reload.
  const r = await fetch(url, {cache: "no-cache"});
  if (!r.ok) throw new Error(`${rel(url)}: HTTP ${r.status}`);
  // The score ledger may be committed gzipped (GitHub's file-size limit); the browser inflates it here.
  const body = url.pathname.endsWith(".gz") ? r.body.pipeThrough(new DecompressionStream("gzip")) : r.body;
  const reader = body.pipeThrough(new TextDecoderStream()).getReader();
  let rest = "";
  for (;;) {
    const {value, done} = await reader.read();
    if (done) break;
    const lines = (rest + value).split("\n");
    rest = lines.pop();
    for (const line of lines) if (line.trim()) fn(line);
  }
  if (rest.trim()) fn(rest);
}

const progress = text => { el("content").innerHTML = `<div class="empty">${esc(text)}</div>`; };

// Scalars only: the Results view aggregates scores, metrics, cost and time; it never reads detail or replies.
const CALL_FIELDS = ["engine_name", "pass_name", "legs", "latency_seconds", "steady_latency_seconds",
  "cost_per_page", "steady_cost_per_page", "measurement_scope", "batch_id", "error", "source_timestamp", "source_pass_name",
  "timestamp", "gpu_type", "is_cold_start", "document_path", "input_tokens", "output_tokens"];
function slimCall(r) {
  const c = {};
  for (const k of CALL_FIELDS) if (k in r) c[k] = r[k];
  c.stem = stemOf(r.document_path);
  return c;
}
function slimRow(r) {
  const m = {};
  for (const [k, v] of Object.entries(r.metrics || {})) if (v == null || typeof v !== "object") m[k] = v;
  return {engine_name: r.engine_name, tier: r.tier, doc_type: r.doc_type, stem: stemOf(r.document_path), leg: r.leg,
    pass_name: r.pass_name, applicable: r.applicable, score: r.score, error: r.error, reason: r.reason, metrics: m,
    unscored: r.scoring_status === "unscored", pending: !!m.kv_pending_key_review};
}

// ---------- loading the run ----------
const HOW_TO_OPEN = `This page reads the saved results files, and a browser only allows that through a local server
started in the repository directory (the folder that holds results/ and documents/).
In a terminal: cd ${location.protocol === "file:" ? decodeURIComponent(ROOT.pathname) : "<the repository directory>"} && python3 -m http.server 8000
Then open http://localhost:8000/viewer/viewer.html`;

async function load() {
  if (location.protocol === "file:") throw new Error(HOW_TO_OPEN);
  // The newest complete, scored run, as `--report current` picks it (evaluation/artifacts.current_timestamp).
  const runsURL = new URL("results/runs/", ROOT);
  const names = await listDir(runsURL).catch(() => null);
  if (!names) throw new Error(`Cannot list ${rel(runsURL)}.\n${HOW_TO_OPEN}`);
  for (const ts of names.filter(n => n.endsWith("_run.json")).map(n => n.slice(0, -"_run.json".length)).sort().reverse()) {
    const results = [`${ts}_results.jsonl`, `${ts}_results.jsonl.gz`].find(n => names.includes(n));
    if (!results || !names.includes(`${ts}_inference.jsonl`)) continue;
    const manifest = await getJSON(new URL(`${ts}_run.json`, runsURL));
    if (manifest?.scored) { D.baseline = ts; D.manifest = manifest; D.resultsFile = results; break; }
  }
  if (!D.baseline) throw new Error(`No complete, scored run in ${rel(runsURL)}`);
  D.legs = Object.keys(LEG_TITLES).filter(l => D.manifest.legs.includes(l));

  progress(`Reading the ledger ${D.baseline}_inference.jsonl…`);
  await eachLine(new URL(`${D.baseline}_inference.jsonl`, runsURL), line => {
    const rec = JSON.parse(line);
    D.results.ledger.push(slimCall(rec));
    for (const leg of rec.legs || [])
      D.ledger.set([rec.engine_name, docDir(rec.document_path), rec.pass_name, leg].join("|"),
        [rec.source_timestamp || D.baseline, rec.source_pass_name || rec.pass_name]);
  });
  // A row's detail (verdicts, boxes, missed words) is the bulk of the file; the saved line is kept and parsed
  // only when that page is opened.
  progress(`Reading the scores ${D.resultsFile}…`);
  const rows = [];
  await eachLine(new URL(D.resultsFile, runsURL), line => {
    const r = JSON.parse(line);
    if (r.leg === "judge") { D.results.judge.push(slimRow(r)); return; }
    if (!D.legs.includes(r.leg)) return;
    D.results.rows.push(slimRow(r));
    delete r.detail;
    rows.push({...r, _line: line});
  });

  for (const row of rows) {
    const path = row.document_path, stem = stemOf(path), p = path.split("/");
    const doc = D.docs[stem] ||= {stem, path: p.slice(p.lastIndexOf("documents")).join("/"), tier: p.at(-3),
      doc_type: p.at(-2), image: fileURL(path), engines: {}};
    const entry = doc.engines[row.engine_name] ||= {legs: {}, calls: {}};
    const loc = callLocation(row);
    const call = entry.calls[loc.key] ||= {...loc, pass_name: row.pass_name, scored_legs: []};
    if (!call.scored_legs.includes(row.leg)) call.scored_legs.push(row.leg);
    const line = row._line;
    let parsed;
    entry.legs[row.leg] = {
      score: row.score, applicable: row.applicable, reason: row.reason, error: row.error,
      metrics: row.metrics, call: loc.key, document_path: path, pass_name: row.pass_name,
      scoring_status: (row.metrics || {}).kv_pending_key_review ? "pending_review" : row.scoring_status || "scored",
      get detail() { return parsed ??= (JSON.parse(line).detail ?? {}); },
    };
  }
  docs = Object.keys(D.docs).sort();
  engines = [...new Set(rows.map(r => r.engine_name))].sort();
  D.limits = parseLimits((await getText(new URL("engines.py", ROOT))) || "");
  Results.init();
}

// Each engine's default_limits as written in engines.py: a JSON-style dict, or the name of a shared one.
function parseLimits(text) {
  const literal = from => {
    const start = text.indexOf("{", from);
    for (let j = start, depth = 0, inStr = false; start >= 0 && j < text.length; j++) {
      const ch = text[j];
      if (inStr) { if (ch === "\\") j++; else if (ch === '"') inStr = false; continue; }
      if (ch === '"') inStr = true; else if (ch === "{") depth++; else if (ch === "}" && --depth === 0) return JSON.parse(text.slice(start, j + 1));
    }
    return null;
  };
  const out = {};
  for (const m of text.matchAll(/^    "([\w.-]+)": EngineSpec\(/gm)) {
    const end = text.indexOf("\n    ),", m.index), block = text.slice(m.index, end < 0 ? undefined : end);
    const v = /default_limits=(\{|[A-Za-z_]\w*)/.exec(block);
    if (!v) continue;
    try {
      const at = v[1] === "{" ? m.index + v.index : new RegExp(`^${v[1]} = \\{`, "m").exec(text)?.index;
      out[m[1]] = at == null ? null : literal(at);
    } catch { out[m[1]] = null; }
  }
  return out;
}

// Where a row's call is saved: the ledger names the folder (source_timestamp) and file (source_pass_name).
function callLocation(row) {
  const dir = docDir(row.source_document_path || row.document_path), engine = row.engine_name;
  const named = D.ledger.get([engine, dir, row.pass_name, row.leg].join("|"));
  if (!named) return {key: `unlisted/${row.pass_name}/${row.leg}`,
    missing: `${D.baseline}_inference.jsonl has no entry for this leg's call`};
  const [source, file] = named;
  return {key: `${source}/${file}`, url: new URL(`results/ocr_outputs/${source}/${dir}/${engine}/${file}.call.json`, ROOT)};
}

async function loadCall(info) {
  if (!info || info.missing) return null;
  return getJSON(info.url);
}

// The reference file the score row names.
async function loadReference(doc, leg, row) {
  const sidecar = suffix => doc.path.replace(/\.[^.]*$/, suffix);
  if (leg === "text") {
    const json = await getJSON(new URL(sidecar("_gt.json"), ROOT));
    if (json) return {file: sidecar("_gt.json"), text: json.plain_text ?? ""};
    const text = await getText(new URL(sidecar("_gt.txt"), ROOT));
    return text == null ? null : {file: sidecar("_gt.txt"), text};
  }
  if (leg === "markdown") {
    const text = await getText(new URL(sidecar("_markdown_gt.md"), ROOT));
    return text == null ? null : {file: sidecar("_markdown_gt.md"), text};
  }
  if (leg === "tables") {
    if (row.detail?.selected_reference !== "gold") return null;
    const file = `documents/table_references/gold/${doc.stem}.json`;
    const json = await getJSON(new URL(file, ROOT));
    return json ? {file, html: json.table_html} : null;
  }
  if (leg === "key_value" || leg === "schema") {
    const path = leg === "schema" ? row.metrics?.schema_reference_repairs?.gold_path : row.detail?.label_review?.gold_path;
    const url = path && fileURL(path);
    const json = url && await getJSON(url);
    return json ? {file: rel(url), json: json.true_json} : null;
  }
  return null;
}

// The engine's EngineSpec exactly as written in engines.py.
async function registryBlock(engine) {
  const lines = ((await getText(new URL("engines.py", ROOT))) || "").split("\n");
  const start = lines.findIndex(l => l.startsWith(`    "${engine}": EngineSpec(`));
  if (start < 0) return null;
  const end = lines.findIndex((l, i) => i > start && l.startsWith("    ),"));
  return {file: "engines.py", text: lines.slice(start, end + 1).join("\n")};
}
const writtenField = (block, name) => (block?.text.match(new RegExp(`\\b${name}=("[^"]*"|\\([^)]*\\)|[\\w.]+)`)) || [])[1];

// ---------- state helpers ----------
const legRow = (engine, doc, leg) => D.docs[doc]?.engines[engine]?.legs[leg];
function legStatus(engine, doc, leg) {
  const row = legRow(engine, doc, leg);
  if (!row) return "none";
  if (row.error) return "err";
  if (row.scoring_status === "unscored") return "unscored";
  if (row.scoring_status === "pending_review") return "pending";
  if (row.score == null) return row.applicable === false ? "na" : "pending";
  return "s";
}
const STATUS_TEXT = {err: "error", none: "not recorded", unscored: "unscored", pending: "pending", na: "n/a"};
function scoreColor(s) {
  return s == null ? "var(--na)" : s >= .8 ? "var(--ok)" : s >= .5 ? "var(--part)" : "var(--bad)";
}

// ---------- sidebar / tabs ----------
function renderSidebar() {
  const results = state.view === "results";
  let html = `<div class="eng results ${results ? "active" : ""}"><span>Results</span><span class="muted" style="font-size:11px">tables · charts</span></div>`;
  html += `<div class="group-title">Published run ${esc(D.baseline)}</div>`;
  for (const e of engines) {
    // In the Results view each dot is the engine's mean for that leg under the current filter; on a page it is that page's score.
    const dots = results ? Results.dots(e).map(d => `<span class="dot ${d.score == null ? "none" : "s"} ${d.dim ? "dim" : ""}" ${d.score == null ? "" : `style="background:${scoreColor(d.score)}"`} data-v="${d.score ?? ""}" title="${LEG_TITLES[d.leg]}: ${d.score == null ? "no scored page in this selection" : `${Results.f4(d.score)} · ${d.n} page${d.n === 1 ? "" : "s"}`}${d.errors ? ` · ${d.errors} error${d.errors === 1 ? "" : "s"}` : ""}"></span>`).join("")
      : D.legs.map(l => {
        const st = legStatus(e, state.doc, l), row = legRow(e, state.doc, l);
        const bg = st === "s" ? `style="background:${scoreColor(row.score)}"` : "";
        return `<span class="dot ${st}" ${bg} title="${LEG_TITLES[l]}: ${st === "s" ? fmt(row.score) : STATUS_TEXT[st]}"></span>`;
      }).join("");
    html += `<div class="eng ${!results && e === state.engine ? "active" : ""}" data-e="${e}"><span>${esc(e)}</span><span class="dots">${dots}</span></div>`;
  }
  el("engine-list").innerHTML = html;
  el("engine-list").querySelectorAll(".eng[data-e]").forEach(n => n.onclick = () =>
    go({engine: n.dataset.e, leg: results && Results.rf.leg !== "all" ? Results.rf.leg : state.leg}));
  el("engine-list").querySelector(".eng.results").onclick = () => go({view: "results"});
}
// The documents the engine actually has rows for.
const availableDocs = engine => docs.filter(d => D.docs[d].engines[engine]);

function renderTabs() {
  // One dropdown: the selected engine's documents, grouped by family.
  const avail = availableDocs(state.engine);
  const byType = new Map();
  for (const d of avail) {
    const t = D.docs[d].doc_type;
    if (!byType.has(t)) byType.set(t, []);
    byType.get(t).push(d);
  }
  const groups = [...byType.keys()].sort().map(t => `<optgroup label="${esc(t)}">` +
    byType.get(t).map(d => `<option value="${esc(d)}" ${d === state.doc ? "selected" : ""}>${esc(d)} · ${esc(D.docs[d].tier)}</option>`).join("") +
    `</optgroup>`).join("");
  el("doc-tabs").innerHTML = `<label>Document <select id="doc-pick">${groups}</select></label>
    <span class="sc muted">${avail.length} page${avail.length === 1 ? "" : "s"} for ${esc(state.engine)} · ←/→ steps through them</span>`;
  el("doc-pick").onchange = ev => go({doc: ev.target.value});
  el("leg-tabs").innerHTML = D.legs.map((l, i) => {
    const row = legRow(state.engine, state.doc, l), st = legStatus(state.engine, state.doc, l);
    return `<span class="tab ${l === state.leg ? "active" : ""}" data-l="${l}">${i + 1} ${LEG_TITLES[l]}<span class="sc">${st === "s" ? fmt(row.score) : STATUS_TEXT[st]}</span></span>`;
  }).join("");
  el("leg-tabs").querySelectorAll(".tab").forEach(n => n.onclick = () => go({leg: n.dataset.l}));
}

// ---------- engine header: how it's built and how we use it ----------
// Structured view of any JSON value: objects/arrays expand, long text collapses.
function tree(v, depth = 0) {
  if (v === null || v === undefined) return '<span class="muted">null</span>';
  if (typeof v === "string") {
    if (v.length <= 160 && !v.includes("\n")) return esc(v);
    return `<details><summary>${esc(v.slice(0, 80).replace(/\n/g, " "))}… <span class="muted">(${v.length} chars)</span></summary><pre>${esc(v)}</pre></details>`;
  }
  if (typeof v !== "object") return esc(v);
  const entries = Array.isArray(v) ? v.map((x, i) => [i, x]) : Object.entries(v);
  if (!entries.length) return `<span class="muted">${Array.isArray(v) ? "[]" : "{}"}</span>`;
  const rows = entries.map(([k, x]) => `<tr><th>${esc(k)}</th><td>${tree(x, depth + 1)}</td></tr>`).join("");
  const table = `<table class="v tree">${rows}</table>`;
  if (depth === 0) return table;
  return `<details><summary class="muted">${Array.isArray(v) ? `list, ${entries.length} items` : `${entries.length} fields: ${esc(entries.slice(0, 4).map(e => e[0]).join(", "))}${entries.length > 4 ? "…" : ""}`}</summary>${table}</details>`;
}
const MAX_TREE = 200_000;  // characters of a provider reply shown as a tree; longer replies are shown as text
function providerReply(raw) {
  if (isEmpty(raw)) return '<p class="muted">Not retained in this call file.</p>';
  const text = JSON.stringify(raw, null, 1);
  return text.length <= MAX_TREE ? tree(raw)
    : `<p class="muted">Shown as text (${text.length.toLocaleString()} characters; first 200,000 here, the call file has all of it).</p><pre>${esc(text.slice(0, MAX_TREE))}</pre>`;
}

// The reply, formatted by what it is: JSON as fields, Markdown/HTML rendered, plain text readable.
function formatReply(text) {
  const body = unfence(text || "");
  if (!body.trim()) return '<p class="muted">Empty reply.</p>';
  try {
    const parsed = JSON.parse(body);
    if (parsed && typeof parsed === "object") return tree(parsed);
  } catch (e) { /* not JSON */ }
  if (/^\s*\|.*\|\s*$/m.test(body) || /<table[\s>]/i.test(body) || /^#{1,6}\s/m.test(body)) {
    return `<div class="rendered">${md(body)}</div><details><summary>source</summary><pre>${esc(text)}</pre></details>`;
  }
  return `<div class="plain">${esc(body)}</div>`;
}

const sentRecord = call => call.sent || (call.metadata || {}).sent || null;
const servedLegs = info => D.legs.filter(l => (info.scored_legs || []).includes(l));
// Recorded markers in the saved reply that output was cut off. A Modal app's
// finish_reason is kept once, in structured_output, so search both.
function truncationHints(call) {
  const blob = JSON.stringify([call.raw_response ?? {}, call.structured_output ?? {}]);
  return ['"finish_reason":"length"', '"stop_reason":"max_tokens"', '"finishReason":"MAX_TOKENS"', '"truncated":true']
    .filter(needle => blob.includes(needle));
}

function callBlock(info, call, leg) {
  const sent = sentRecord(call) || {};
  const meta = call.metadata || {};
  const amortized = meta.measurement_scope === "modal_batch" && !meta.client_wall_seconds;
  const timeLabel = amortized ? "amortized compute s/item; not latency" : "observed completion seconds";
  const hints = truncationHints(call);
  const recorded = {
    "legs this call serves": servedLegs(info).join(", "),
    "model (recorded)": call.model_version,
    [timeLabel]: call.latency_seconds, "recorded inference cost estimate (USD)": call.cost_per_page,
    "truncation markers in the reply": hints.length ? hints.join(", ") : null,
    "generation": (call.structured_output || {}).generation,
    ...meta,
  };
  delete recorded.sent;
  delete recorded.prompt_used;  // shown under "Prompt sent"
  for (const k of Object.keys(recorded)) if (recorded[k] == null) delete recorded[k];
  // The whole request content: prompt text, the schema object where one was sent.
  const schema = call.json_schema_sent ? call.json_schema : null;
  const where = /document_annotation_format/.test(sent.constraint || "") ? "document_annotation_format"
    : /data_schema/.test(sent.constraint || "") ? "LlamaExtract data_schema"
    : /Extract schema/.test(sent.constraint || "") ? "Datalab Extract schema"
    : /response_format/.test(sent.constraint || "") ? "response_format (json_schema)" : null;
  const promptSent = (sentRecord(call)
    ? ((sent.text ?? call.prompt_sent) ? `<pre>${esc((sent.text ?? call.prompt_sent))}</pre>`
      : `<p><b>${/^(none|no instruction|converter selector)/.test(sent.source || "") ? "No prompt text was sent." : "Prompt text not recorded."}</b> ${esc(sent.source)}</p>`)
    : call.prompt_sent ? `<p class="muted">prompt_sent, as recorded:</p><pre>${esc(call.prompt_sent)}</pre>`
    : '<p><b>No prompt.</b> <span class="muted">prompt_sent is null in this call file.</span></p>')
    + (schema && where ? `<h3>Schema sent as ${esc(where)} (complete)</h3><pre>${esc(JSON.stringify(schema, null, 2))}</pre>` : "")
    + (schema && !where ? `<p class="muted">The schema is inside the prompt text above.</p>` : "");
  // The leg panel below already shows raw_text for these legs; don't repeat it here.
  const replyBelow = ["text", "key_value", "schema"].includes(leg) || (leg === "markdown" && (call.structured_output || {}).markdown == null);
  return `<div class="panel prompt"><h3>How we asked: call "${esc(call.pass_name)}" → ${esc(servedLegs(info).join(", "))}</h3>
    ${sentRecord(call) ? `<table class="v tree"><tr><th>Prompt</th><td>${esc(sent.source)}</td></tr>
      <tr><th>Output constraint</th><td>${esc(sent.constraint)}</td></tr>
      <tr><th>Schema sent separately</th><td>${esc(sent.schema_separately)}</td></tr></table>`
      : `<table class="v tree"><tr><th>Prompt (prompt_sent)</th><td>${call.prompt_sent ? esc(call.prompt_sent.slice(0, 160)) + (call.prompt_sent.length > 160 ? "… (full text under Prompt sent)" : "") : "none (prompt_sent is null)"}</td></tr>
      <tr><th>response_format</th><td>${esc(call.response_format ?? "null")}</td></tr>
      <tr><th>json_schema_sent</th><td>${esc(call.json_schema_sent)}</td></tr></table>`}
    ${call.error ? `<pre class="warn">${esc(call.error)}</pre>` : ""}
    <details><summary>Prompt sent</summary>${promptSent}</details>
    <details><summary>Settings and usage recorded in this call</summary>${meta.measurement_scope === "modal_batch" ? '<p class="muted">Cost is an equal share of measured batch work. GPU costs exclude CPU/RAM; CPU costs are allocation floors. Startup, idle and unmetered failures are excluded.</p>' : ""}${tree(recorded)}</details>
    ${replyBelow ? "" : `<details><summary>Reply (raw_text, ${(call.raw_text || "").length} chars)</summary>${formatReply(call.raw_text)}</details>`}
    <details><summary>Full provider reply</summary>${providerReply(call.raw_response)}</details>
    <div class="muted" style="font-size:11px;margin-top:4px">${esc(rel(info.url))}</div></div>`;
}

// What each leg's call received, from the saved call.
function promptByLeg(legCalls) {
  const rows = legCalls.map(([leg, row, info, call]) => {
    const head = `<th>${LEG_TITLES[leg]}</th>`;
    if (!row) return `<tr>${head}<td colspan="3" class="muted">no call recorded for this leg on this page</td></tr>`;
    if (!call) return `<tr>${head}<td colspan="3" class="warn">call file not found: ${esc(info.missing || rel(info.url))}</td></tr>`;
    const sent = sentRecord(call);
    const fold = text => text.length <= 90 && !text.includes("\n") ? `<div>${esc(text)}</div>`
      : `<details><summary>${esc(text.slice(0, 90).replace(/\n/g, " "))}…</summary><pre>${esc(text)}</pre></details>`;
    if (!sent) return `<tr>${head}<td>${esc(call.pass_name)}</td><td>${call.prompt_sent ? `prompt_sent${fold(call.prompt_sent)}` : "none (prompt_sent is null)"}</td><td>response_format: ${esc(call.response_format ?? "null")}</td></tr>`;
    return `<tr>${head}<td>${esc(call.pass_name)}</td><td>${esc(sent.source)}${(sent.text ?? call.prompt_sent) ? fold((sent.text ?? call.prompt_sent)) : ""}</td><td>${esc(sent.constraint)}</td></tr>`;
  }).join("");
  return `<table class="v tree"><tr><th>Leg</th><th>Call</th><th>Prompt received</th><th>Output constraint</th></tr>${rows}</table>`;
}

let headSeq = 0, contentSeq = 0;
async function renderHead() {
  const seq = ++headSeq, engine = state.engine, doc = state.doc;
  const entry = D.docs[doc]?.engines[engine];
  if (!entry) { el("engine-head").innerHTML = `<h1>${esc(engine)}</h1><div class="muted">No row recorded for ${esc(engine)} on ${esc(doc)}.</div>`; return; }
  const legCalls = await Promise.all(D.legs.map(async leg => {
    const row = legRow(engine, doc, leg), info = row && entry.calls[row.call];
    return [leg, row, info, await loadCall(info)];
  }));
  const block = await registryBlock(engine);
  if (seq !== headSeq || state.view !== "page") return;
  const written = ["kind", "gpu_type", "model_size", "table_format", "accepts_task", "native_legs", "prompted_legs"]
    .map(f => [f, writtenField(block, f)]).filter(([, v]) => v != null);
  const shownCalls = [...new Map(legCalls.filter(([, row, info]) => info).map(([, , info]) => [info.key, info])).values()];
  // Named by the saved file, so two views of one call (default, default.tables) stay apart.
  const chips = shownCalls.map(i => `<span class="chip">${esc(i.key.split("/").pop())} → ${esc(servedLegs(i).join(", "))}</span>`).join("");
  el("engine-head").innerHTML = `
    <h1>${esc(engine)}</h1>
    <div class="meta">${block ? `${written.map(([k, v]) => `<span>${esc(k)}=${esc(v)}</span>`).join("")}<span>(as written in ${esc(block.file)})</span>` : `<span>No EngineSpec entry for ${esc(engine)} in engines.py.</span>`}</div>
    <div style="margin-top:6px">Calls on this page: ${chips || '<span class="muted">none recorded</span>'}</div>
    <details class="panel" id="how" ${state.howOpen ? "open" : ""}><summary>How it's built and how we use it</summary>
      <h3 style="margin-top:12px">Prompt by leg on ${esc(doc)}</h3>${promptByLeg(legCalls)}
      <p class="muted">Per-call prompt, recorded settings and replies are under "How we asked" in each leg.</p>
      ${block ? `<details><summary>Registry entry in ${esc(block.file)}</summary><pre>${esc(block.text)}</pre></details>` : ""}
    </details>`;
  el("how").ontoggle = ev => { state.howOpen = ev.target.open; };
}

// ---------- pieces ----------
// The geometry levels the scorer recorded for a row: Word always, Line and Block when their evidence is saved.
function geoLevels(row) {
  const d = row?.detail || {}, m = row?.metrics || {};
  return [["word", "Word", row?.score], ...(d.line_matches ? [["line", "Line", m.geometry_line_score]] : []),
    ...(d.block_matches ? [["block", "Block cov", m.geometry_block_score]] : [])];
}
const geoLevel = row => geoLevels(row).some(([k]) => k === state.geoLevel) ? state.geoLevel : "word";
const clip = (t, n = 200) => { t = String(t ?? ""); return t.length > n ? t.slice(0, n) + "…" : t; };

function imagePanel(withBoxes) {
  const doc = D.docs[state.doc];
  let svg = "", controls = "";
  if (withBoxes) {
    const row = legRow(state.engine, state.doc, "geometry"), d = row?.detail || {}, level = geoLevel(row);
    const rect = (b, color, title, dash) => b && b.length === 4 ?
      `<rect x="${b[0]}" y="${b[1]}" width="${Math.max(b[2] - b[0], 0)}" height="${Math.max(b[3] - b[1], 0)}" fill="${color}" fill-opacity=".12" stroke="${color}" stroke-width="1.5" vector-effect="non-scaling-stroke" ${dash ? 'stroke-dasharray="4 3"' : ""}><title>${esc(title)}</title></rect>` : "";
    let parts = "";
    if (level === "word") {
      if (state.layers.ref) for (const m of d.matches || []) parts += rect(m.reference_bbox, m.hit ? "var(--ok)" : "var(--bad)", `reference: ${m.text}${m.hit ? " (located)" : " (not located)"}`, true);
      if (state.layers.pred) for (const b of d.scored_boxes || []) parts += rect(b.bbox, "var(--accent)", `predicted: ${b.text}`);
      if (state.layers.unmatched) for (const b of d.unmatched_predictions || []) parts += rect(b.bbox, "var(--part)", `unmatched prediction: ${b.text}`);
    } else if (level === "line") {
      if (state.layers.ref) for (const x of d.line_matches || []) parts += rect(x.bbox, x.hit ? "var(--ok)" : "var(--bad)", `reference line: ${x.text}${x.hit ? " (located)" : " (not located)"}`, true);
      if (state.layers.pred) for (const x of d.predicted_lines || []) parts += rect(x.bbox, "var(--accent)", `engine line: ${clip(x.text)}`);
    } else {
      if (state.layers.ref) for (const x of d.block_matches || []) parts += rect(x.bbox, x.covered ? "var(--ok)" : "var(--bad)", `reference: ${x.text}${x.covered ? ` (inside block ${x.block + 1})` : " (not inside a block with its text)"}`, true);
      if (state.layers.pred) (d.blocks || []).forEach((x, i) => { parts += rect(x.bbox, "var(--accent)", `block ${i + 1}: ${clip(x.text)}`); });
    }
    svg = `<svg viewBox="0 0 1 1" preserveAspectRatio="none">${parts}</svg>`;
    const labels = {
      word: {ref: "reference boxes (green located / red missed)", pred: "predicted boxes (as scored)", unmatched: "unmatched predictions"},
      line: {ref: "reference lines (green located / red missed)", pred: "engine lines (its boxes merged into visual lines)"},
      block: {ref: "reference units (green inside a block with their text / red not)", pred: "engine blocks"}}[level];
    controls = `<div class="geo-levels">${geoLevels(row).map(([k, label, v]) => `<button data-geo="${k}" class="${k === level ? "on" : ""}">${label} ${fmt(v)}</button>`).join("")}</div>
      <div class="layers">${Object.entries(labels).map(([k, text]) => `<label><input type="checkbox" data-layer="${k}" ${state.layers[k] ? "checked" : ""}> ${text}</label>`).join("")}</div>`;
  }
  return `<div class="panel sticky"><h3>${esc(doc.path)}</h3>${controls}<div class="imgwrap"><img src="${doc.image}" alt="${esc(state.doc)}" onerror="this.outerHTML='<p class=muted>Page image not found. Download the images once with <code>python scripts/download_data.py</code> in the repository directory, then reload.</p>'">${svg}</div></div>`;
}

// Geometry: Word is the score; Line and Block cov are shown beside it where the scorer recorded them.
const geoExtra = row => row && !row.error ? [["Line", row.metrics?.geometry_line_score], ["Block cov", row.metrics?.geometry_block_score]]
  .filter(([, v]) => v != null).map(([k, v]) => `${k} <b>${fmt(v)}</b>`).join(" · ") : "";

function scorePanel(row) {
  const leg = state.leg, geo = leg === "geometry";
  const all = engines.map(e => ({e, row: legRow(e, state.doc, leg)})).filter(x => x.row)
    .sort((a, b) => (b.row.score ?? -1) - (a.row.score ?? -1));
  const bars = all.length ? `<div class="bars">${all.map(x => `<div class="bar ${geo ? "geo" : ""} ${x.e === state.engine ? "me" : ""}" data-e="${x.e}">
      <span>${esc(x.e)}</span><span class="track"><span class="fill" style="display:block;width:${(x.row.score ?? 0) * 100}%"></span></span>
      <span>${(st => st === "s" ? fmt(x.row.score) : st === "err" ? '<span class="err">error</span>' : STATUS_TEXT[st])(legStatus(x.e, state.doc, leg))}</span>${geo ? `<span class="geo-extra">${geoExtra(x.row)}</span>` : ""}</div>`).join("")}</div>`
    : '<span class="muted">none</span>';
  const metrics = Object.entries(row?.metrics || {}).filter(([, v]) => v == null || typeof v !== "object")
    .map(([k, v]) => `<span>${esc(k)}: <b>${fmt(v)}</b></span>`).join("");
  return `<div class="panel"><h3>${LEG_TITLES[leg]} score on this page${geo ? " · Word" : ""}</h3>
    ${row ? `<div class="score-big" style="color:${scoreColor(row.score)}">${row.error ? "error" : row.scoring_status === "unscored" ? "Unscored" : row.scoring_status === "pending_review" ? "Pending review" : fmt(row.score)}</div>
      ${geo && geoExtra(row) ? `<div class="geo-sub">${geoExtra(row)}</div>` : ""}
      ${row.error ? `<pre class="warn">${esc(row.error)}</pre>` : ""}
      ${row.reason ? `<div class="muted">${esc(row.reason)}</div>` : ""}
      ${metrics ? `<details><summary class="muted">all metrics</summary><div class="metrics">${metrics}</div></details>` : ""}`
      : `<div class="muted">No ${LEG_TITLES[leg]} result is recorded for ${esc(state.engine)} on this page.</div>`}
    <h3 style="margin-top:12px">All engines, this page${geo ? ": Word, with Line and Block cov where recorded" : ""} (click to switch)</h3>${bars}</div>`;
}

function callPointer(row, info, call) {
  if (!info) return "";
  if (info.missing || !call) {
    const where = esc(info.missing || rel(info.url));
    return `<div class="panel prompt"><h3>How we asked</h3><p class="warn">${row.error ? `This call failed and no call file was saved (${where}). Error: ${esc(row.error)}` : `Call file not found: ${where}`}</p></div>`;
  }
  return callBlock(info, call, state.leg);
}

// The "Why" table for the geometry level shown on the image.
function geoWhy(row, d) {
  const level = geoLevel(row);
  if (level === "line") return `<div class="panel"><h3>Why: reference lines</h3>
    <p class="muted">Reference and engine boxes are each merged into visual lines. Each reference line is paired with the unused engine line of most similar text (at least 0.7) and is located when each box contains the other's centre.</p>
    ${verdictTable((d.line_matches || []).map(x => ({...x, verdict: x.hit ? "correct" : x.predicted_text != null ? "text matched, box missed" : "no line with similar text"})), [
      ["reference line", r => r.text], ["engine line", r => r.predicted_text], ["text similarity", r => r.similarity], ["verdict", r => r.verdict]])}</div>`;
  if (level === "block") return `<div class="panel"><h3>Why: reference units inside blocks</h3>
    <p class="muted">A reference unit is covered when an engine block contains its text and its centre. This shows where text sits, not word-level boxes.</p>
    ${verdictTable((d.block_matches || []).map(x => ({...x, verdict: x.covered ? "correct" : "not covered", inBlock: x.block == null ? null : `block ${x.block + 1}: ${clip(d.blocks?.[x.block]?.text, 120)}`})), [
      ["reference", r => r.text], ["covering block", r => r.inBlock], ["verdict", r => r.verdict]])}</div>`;
  return `<div class="panel"><h3>Why: reference units</h3>${verdictTable((d.matches || []).map(x => ({...x, verdict: x.hit ? "correct" : x.matched ? "text matched, box missed" : "missing"})), [
    ["reference", r => r.text], ["engine text", r => r.predicted_text], ["IoU", r => r.iou], ["verdict", r => r.verdict]])}</div>`;
}

function verdictTable(rows, cols) {
  const bad = r => !["correct", "blank_ok", "present_correct"].includes(r.verdict);
  // Table verdicts record `numeric` on every cell and `invented` on numeric extras
  // (evaluation/table_cells.numeric_diagnostics); the filters exist only when the rows carry them.
  const hasNumeric = rows.some(r => "numeric" in r), hasInvented = rows.some(r => r.invented);
  const filters = [["all", "all"], ["bad", "only not correct"]];
  if (hasNumeric) filters.push(["numeric", "numbers only"], ["numeric-bad", "numbers not correct"]);
  if (hasInvented) filters.push(["invented", "invented numbers"]);
  const active = filters.some(f => f[0] === state.filter) ? state.filter : "all";
  const shown = active === "all" ? rows : active === "bad" ? rows.filter(bad)
    : active === "numeric" ? rows.filter(r => r.numeric) : active === "numeric-bad" ? rows.filter(r => r.numeric && bad(r))
    : rows.filter(r => r.invented);
  const counts = {};
  for (const r of rows) counts[r.verdict] = (counts[r.verdict] || 0) + 1;
  if (hasInvented) counts.invented = rows.filter(r => r.invented).length;
  const chips = Object.entries(counts).map(([k, v]) => `<span class="chip ${k === "correct" || k === "blank_ok" ? "ok" : k === "partial" ? "part" : "bad"}">${esc(k)} ${v}</span>`).join("");
  return `<div>${chips}</div><div class="filter">${filters.map(([f, label]) => `<button data-f="${f}" class="${active === f ? "on" : ""}">${label}</button>`).join("")}</div>
    <div class="vwrap"><table class="v"><tr>${cols.map(c => `<th>${esc(c[0])}</th>`).join("")}</tr>
    ${shown.map(r => `<tr>${cols.map(c => {
      const v = c[1](r);
      return c[0] === "verdict" ? `<td><span class="chip ${v === "correct" || v === "blank_ok" ? "ok" : v === "partial" ? "part" : "bad"}">${esc(v)}</span></td>` : `<td>${esc(v)}</td>`;
    }).join("")}</tr>`).join("")}</table></div>`;
}
const cell = c => c ? `${c.text ?? ""}` : "";
const pos = c => c ? `r${c.row} c${c.col}` : "";
const sub = text => `<div class="sub">${esc(text)}</div>`;
// Reference changes the scorer recorded with this row, if any.
const repairs = v => isEmpty(v) ? "" : `<details><summary>Reference changes recorded with this score</summary>${tree(v)}</details>`;

// ---------- leg bodies ----------
function legBody(row, call, ref) {
  const d = row.detail || {}, m = row.metrics || {};
  call = call || {};
  switch (state.leg) {
    case "text":
      return `${row.scoring_status === "unscored" ? '<div class="panel muted">Word coverage has not been scored for this call.</div>' : `<div class="panel"><h3>Missed reference words (${(d.missed_words || []).length})</h3>
        ${(d.missed_words || []).map(w => `<span class="chip bad">${esc(w)}</span>`).join("") || '<span class="muted">none</span>'}</div>`}
        <div class="two"><div class="panel"><h3>Engine text</h3>${sub("raw_text, what the scorer read")}<pre>${esc(call.raw_text)}</pre></div>
        <div class="panel"><h3>Reference text</h3>${ref ? sub(ref.file) : ""}${ref ? `<pre>${esc(ref.text)}</pre>` : '<p class="muted">No text sidecar found for this page.</p>'}
          ${repairs(m.text_reference_repairs)}</div></div>`;
    case "markdown": {
      const mdText = (call.structured_output || {}).markdown ?? call.raw_text ?? "";
      const mdField = (call.structured_output || {}).markdown != null ? "structured_output.markdown" : "raw_text";
      return `<div class="two"><div class="panel"><h3>Engine Markdown</h3>${sub(`${mdField}, as recorded; one outer \`\`\` fence hidden, as the scorer removes it`)}
          <pre>${esc(unfence(mdText))}</pre>
          <details><summary>rendered</summary><div class="rendered">${md(unfence(mdText))}</div></details></div>
        <div class="panel"><h3>Reference Markdown</h3>${ref ? sub(ref.file) : ""}
          ${ref ? `<div class="rendered">${md(ref.text)}</div>` : '<p class="muted">No _markdown_gt.md sidecar for this page, and the row records no reference file; the items the headline was scored against are in the verdicts below.</p>'}
          <p class="muted">Supporting metrics only (CER/WER, block recall, reading order); their recorded text reference: ${esc(m.markdown_text_reference_source ?? "not recorded")}. The headline score checks the catalogue items below.</p></div></div>
        <div class="panel"><h3>Why: catalogue item verdicts (${(d.verdicts || []).length} items)</h3>${verdictTable(d.verdicts || [], [
          ["#", r => r.n], ["type", r => r.type], ["reference content", r => r.content], ["verdict", r => r.verdict], ["how", r => r.how],
          ["matched engine text", r => (r.evidence || []).map(e => `[${e.kind}] ${e.text}`).join(" | ")]])}</div>`;
    }
    case "tables": {
      // The tables the scorer read are saved with the score.
      const tables = d.output_tables || [];
      return `<div class="two"><div class="panel"><h3>Engine tables (${tables.length})</h3>${sub("as recorded by the scorer")}
          ${tables.map((t, i) => `<div class="muted">table ${i}</div><div class="tablebox">${clean(t)}</div>`).join("") || `<span class="muted">${d.output_tables ? "the scorer recorded no tables" : "not recorded for this row"}</span>`}</div>
        <div class="panel"><h3>Reference table</h3>${sub(ref ? ref.file : `selected_reference: ${d.selected_reference ?? "not recorded"}`)}
          ${ref ? `<div class="tablebox">${clean(ref.html || "")}</div>` : '<p class="muted">The reference file is not recorded for this row; each reference cell is in the verdicts below.</p>'}
          <div class="metrics">reference cells <b>${fmt(d.reference_cells)}</b> · output cells <b>${fmt(d.output_cells)}</b></div></div></div>
        <div class="panel"><h3>Why: cell verdicts</h3>${verdictTable(d.verdicts || [], [
          ["ref cell", r => pos(r.reference)], ["reference", r => cell(r.reference)], ["engine", r => (r.output_components || []).map(cell).join(" + ") || cell(r.output)],
          ["credit", r => r.credit], ["verdict", r => r.verdict === "extra" && r.invented ? "extra (invented number)" : r.verdict], ["how", r => r.how]])}</div>`;
    }
    case "key_value":
    case "schema": {
      // Recorded only: the engine's reply as saved, the native pairs saved in the call, and the scorer's verdicts.
      const extra = state.leg === "schema" && d.compliance ? `<div class="metrics">valid JSON as returned <b>${d.compliance.valid_json}</b> · schema valid <b>${d.compliance.schema_valid}</b> · extra fields <b>${(d.extra_fields || []).length}</b></div>
          ${(d.compliance.errors || []).length ? `<p class="muted">Recorded validity errors: ${esc(d.compliance.errors.join("; "))}</p>` : ""}` : "";
      const textract = state.engine === "textract_structured";
      const forms = !textract && Array.isArray((call.structured_output || {}).forms) && call.structured_output.forms.length ? call.structured_output.forms : null;
      return `<div class="two"><div class="panel"><h3>Engine reply</h3>${sub("raw_text, as recorded")}${extra}
          ${textract ? `<p class="muted">The scorer does not read this text for Textract. It rebuilds the FORMS pairs and tables from the saved Blocks (run_benchmark._read_stored_call); they are not saved separately. The Blocks are under "Full provider reply"; the key and value the scorer matched for each field are in the verdicts below.</p>` : ""}
          <pre>${esc(call.raw_text ?? "")}</pre></div>
        <div class="panel"><h3>Reference</h3>${ref ? sub(`true_json in ${ref.file}`) : ""}
          ${ref ? `<pre>${esc(JSON.stringify(ref.json, null, 2))}</pre>` : '<p class="muted">This score row does not record its reference file; the expected value of every field is in the verdicts below.</p>'}
          ${state.leg === "schema" ? repairs(m.schema_reference_repairs) : ""}</div></div>
        ${forms ? `<div class="panel"><h3>Native key/value pairs</h3>${sub("structured_output.forms, as saved in the call")}
          <div class="vwrap"><table class="v"><tr><th>key</th><th>value</th></tr>${forms.map(f => `<tr><td>${esc(f.key)}</td><td>${esc(f.value)}</td></tr>`).join("")}</table></div></div>` : ""}
        <div class="panel"><h3>Why: field verdicts (as recorded by the scorer)</h3>${verdictTable(d.verdicts || [], [
          ["reference field", r => r.path], ["expected", r => r.expected], ["engine key", r => r.key], ["engine value", r => r.value],
          ["verdict", r => r.verdict], ["how", r => r.how]])}
          ${(d.unresolved_keys || []).length ? `<details><summary>unmatched engine pairs (${d.unresolved_keys.length})</summary><table class="v"><tr><th>key</th><th>value</th><th>decision</th></tr>${d.unresolved_keys.map(u => `<tr><td>${esc(u.key)}</td><td>${esc(u.value)}</td><td>${esc(u.decision)}</td></tr>`).join("")}</table></details>` : ""}</div>`;
    }
    case "geometry": {
      return `<div class="panel"><div class="metrics"><span>engine box unit: <b>${esc(m.geometry_prediction_granularity || "not recorded")}</b></span>
          <span>scored against: <b>${esc(m.geometry_scoring_unit || "not recorded")}</b></span>
          <span>recall (headline): <b>${fmt(m.geometry_recall ?? row.score)}</b></span><span>precision: <b>${fmt(m.geometry_precision)}</b></span>
          ${m.geometry_block_score != null ? `<span>block score (diagnostic): <b>${fmt(m.geometry_block_score)}</b></span><span>words per box: <b>${fmt(m.geometry_words_per_box)}</b></span>` : ""}
          ${m.geometry_line_score != null ? `<span>line score (diagnostic): <b>${fmt(m.geometry_line_score)}</b> of <b>${fmt(m.geometry_reference_lines)}</b> lines</span>` : ""}
          <span>predicted boxes: <b>${(d.scored_boxes || []).length}</b></span>
          <span>matched: <b>${fmt(m.geometry_matched_words)}</b> / located: <b>${fmt(m.geometry_located_words)}</b> of <b>${fmt(m.geometry_reference_words)}</b></span>
          <span>rescaled from 1000: <b>${fmt(m.geometry_rescaled_from_thousand)}</b></span><span>axes swapped: <b>${fmt(m.geometry_axes_swapped)}</b></span></div>
          ${m.geometry_prediction_granularity === "layout_block" && m.geometry_scoring_unit !== "layout_block" ? `<p class="warn">The engine returned layout blocks; the headline counts ${esc(m.geometry_scoring_unit)}-level reference units one-to-one, so a block covering several ${esc(m.geometry_scoring_unit)}s can locate at most one of them. The block score above shows box placement separately.</p>` : ""}
          <p class="muted">Boxes are drawn on the page image to the left, in the scorer's normalised coordinates (${esc(d.scored_box_coordinates || "not recorded")}); hover a box for its text.</p>
          ${repairs(m.geometry_reference_repairs)}</div>
        ${geoWhy(row, d)}`;
    }
  }
  return "";
}

async function renderContent() {
  const seq = ++contentSeq;
  const row = legRow(state.engine, state.doc, state.leg);
  const entry = D.docs[state.doc]?.engines[state.engine];
  const info = row && entry?.calls[row.call];
  const [call, ref] = row ? await Promise.all([loadCall(info), loadReference(D.docs[state.doc], state.leg, row)]) : [null, null];
  if (seq !== contentSeq || state.view !== "page") return;
  let right = scorePanel(row);
  if (row) right += callPointer(row, info, call) + (row.error ? "" : legBody(row, call, ref));
  el("content").innerHTML = `<div class="grid">${imagePanel(state.leg === "geometry" && row && !row.error)}<div style="min-width:0">${right}</div></div>`;
  el("content").querySelectorAll(".bar").forEach(n => n.onclick = () => go({engine: n.dataset.e}));
  el("content").querySelectorAll(".filter button").forEach(b => b.onclick = () => go({filter: b.dataset.f}));
  el("content").querySelectorAll("[data-layer]").forEach(c => c.onchange = () => { state.layers[c.dataset.layer] = c.checked; renderContent(); });
  el("content").querySelectorAll("[data-geo]").forEach(b => b.onclick = () => { state.geoLevel = b.dataset.geo; renderContent(); });
}

// ---------- routing ----------
function go(patch) {
  // Picking an engine, page or leg opens the page view; the Results entry (or a filter change) stays in results.
  if ((patch.engine || patch.doc || patch.leg) && !patch.view) patch.view = "page";
  Object.assign(state, patch);
  // A newly picked engine keeps the page when it has it, else jumps to its first.
  const avail = availableDocs(state.engine);
  if (avail.length && !avail.includes(state.doc)) state.doc = avail[0];
  location.hash = state.view === "results" ? Results.toHash() : [state.engine, state.doc, state.leg, state.howOpen ? "how" : "view"].join("/");
  return render();
}
function render() {
  renderSidebar();
  if (state.view === "results") { Results.render(); return Promise.resolve(); }
  renderTabs();
  return Promise.all([renderHead(), renderContent()]);
}
function fromHash() {
  const parts = decodeURIComponent(location.hash.slice(1)).split("/");
  if (parts[0] === "results") { state.view = "results"; Results.fromHash(parts.slice(1)); return; }
  const [e, d, l, how] = parts;
  state.howOpen = how === "how";
  if (engines.includes(e)) state.engine = e;
  if (docs.includes(d)) state.doc = d;
  if (D.legs.includes(l)) state.leg = l;
}
document.addEventListener("keydown", ev => {
  if (!docs.length || ["INPUT", "TEXTAREA", "SELECT"].includes(ev.target.tagName)) return;
  if (state.view === "results" && /^[0-6]$/.test(ev.key)) { Results.update({leg: ev.key === "0" ? "all" : D.legs[+ev.key - 1] || "all"}); ev.preventDefault(); return; }
  const i = engines.indexOf(state.engine);
  const avail = availableDocs(state.engine), j = avail.indexOf(state.doc);
  if (ev.key === "ArrowDown" || ev.key === "j") { go({engine: engines[Math.min(i + 1, engines.length - 1)]}); ev.preventDefault(); }
  else if (ev.key === "ArrowUp" || ev.key === "k") { go({engine: engines[Math.max(i - 1, 0)]}); ev.preventDefault(); }
  else if (ev.key === "ArrowRight") go({doc: avail[(j + 1) % avail.length]});
  else if (ev.key === "ArrowLeft") go({doc: avail[(j - 1 + avail.length) % avail.length]});
  else if (/^[1-6]$/.test(ev.key) && D.legs[+ev.key - 1]) go({leg: D.legs[+ev.key - 1]});
});

const ready = load().then(() => {
  if (!docs.length) { el("content").innerHTML = '<div class="empty">No results found.</div>'; return; }
  state.engine = engines[0]; state.doc = docs[0];
  fromHash();
  // Open on the Results overview unless the address names a view.
  if (location.hash.length <= 1) state.view = "results";
  // The first engine may have rows on a few pages only; start on one it has.
  const avail = availableDocs(state.engine);
  if (avail.length && !avail.includes(state.doc)) state.doc = avail[0];
  return render();
}).catch(err => { el("content").innerHTML = `<div class="empty warn" style="white-space:pre-line">${esc(err.message)}</div>`; throw err; });
// checks/sweep.js renders every engine × page × leg; it is loaded only with ?sweep in the address.
if (new URLSearchParams(location.search).has("sweep")) {
  const s = document.createElement("script");
  s.src = new URL("checks/sweep.js", HERE);
  document.body.append(s);
}
