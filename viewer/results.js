// Results view: comparison tables and charts over every recorded row of the published run, read from the same
// files the page view reads. Nothing is scored here. The means reproduce evaluation/reporting.py (family-weighted;
// errors and N/A excluded), and cost/time reproduce reporting._operations (per leg) and run_benchmark.engine_operational
// (per engine) over the recorded ledger. checks/verify_results.py recomputes every cell in Python and compares.
const Results = (() => {
  const TIERS = ["easy", "medium", "hard"];
  const LEG_SHORT = {text: "Text", markdown: "Markdown", tables: "Tables", key_value: "KV", schema: "Schema", geometry: "Geometry"};
  // evaluation/reporting.py METRICS followed by EXTRA_METRICS: label, saved metric (or derived rate), higher is better.
  const METRICS = {
    text: [["CER", "cer", false], ["WER", "wer", false], ["Word F1", "word_f1", true], ["Word P", "word_precision", true], ["Word R", "word_recall", true]],
    markdown: [["Text CER", "markdown_cer", false], ["Text WER", "markdown_wer", false], ["Block F1", "element_f1", true], ["Order", "order_score", true], ["Block P", "element_precision", true], ["Block R", "element_recall", true]],
    tables: [["Numbers", "numeric_accuracy", true], ["Invented", "invented_number_rate", false], ["Strict F1", "cell_f1", true], ["Extra rate", "extra_cell_rate", false], ["Structure", "structure_similarity", true], ["Numbers anywhere", "numeric_found_rate", true], ["Strict P", "cell_precision", true], ["Strict R", "cell_recall", true]],
    geometry: [["Line", "geometry_line_score", true], ["Block cov", "geometry_block_score", true], ["Words/box", "geometry_words_per_box", false], ["Precision", "geometry_precision", true], ["F1", "geometry_f1", true], ["Box CER", "geometry_cer", false], ["Box WER", "geometry_wer", false]],
    key_value: [["Missing", "missing_rate", false], ["Wrong", "wrong_rate", false], ["Unmapped", "kv_unresolved_keys", false], ["Populated fields", "kv_populated_accuracy", true], ["Blank fields", "kv_blank_accuracy", true]],
    schema: [["Missing", "missing_rate", false], ["Wrong", "wrong_rate", false], ["Extra/page", "schema_extra_fields", false], ["Valid JSON", "schema_valid_json", true]],
  };
  // evaluation/reporting.py METHOD, verbatim.
  const METHOD = {
    text: "Distinct-token coverage; CER/WER measure sequence errors separately.",
    markdown: "Correct structure + half partial credit per reference element; text fidelity is separate.",
    tables: "Reference value credit, including returned blanks, checkbox states and partial text matches; gold blanks with no returned slot and unprinted header rows are not counted. Labels and headers align supported split, merged or reordered fields; identical physical grids are not required. Numbers = gold numeric cells correct in place; Invented = returned numeric cells printed nowhere on the page. Extras, strict F1 and structure are separate diagnostics.",
    key_value: "Correct values / counted printed-label fields; unreturned reference blanks are excluded. Unmapped is a count of returned keys per page.",
    schema: "Correct values / (requested fields + unexpected fields). Same frozen field sample.",
    geometry: "Full = Word: located reference-unit recall (words; line segments on receipts). Line: both sides merged into visual lines, fuzzy text, mutual centre containment. Block cov: reference words inside a returned block that contains their text; block engines only, read with Words/box. Boxes are never invented at a finer unit than returned.",
  };
  // Columns whose sort starts ascending: every lower-is-better metric, cost and time.
  const LOWER = new Set(["cost", "rec", "cost_all", "rec_all", ...Object.values(METRICS).flat().filter(m => !m[2]).map(m => "m:" + m[1])]);
  const LOW_THRESHOLD = 0.25, MIN_ENGINES = 3;
  // The two recorded operating measures, as the report and the README's cost and latency tables define them.
  const XM = {
    // Stored per page; shown per 1,000 pages, which reads more easily than fractions of a cent. `mult` scales axis labels.
    cost: {label: "$/1k pages", title: "recorded inference cost per 1,000 pages (USD): steady_cost_per_page, else cost_per_page, of the call serving the scored page, × 1,000", fmt: v => v == null ? "—" : f(v * 1000, v * 1000 < 1 ? 3 : 2), mult: 1000},
    rec: {label: "s/page", title: "recorded seconds per page: steady_latency_seconds, else latency_seconds. For a Modal batch this is the warm batch's seconds divided by its items (throughput basis, startup excluded), not request latency. † = the primary pass shared its batch with other passes, so this share includes their generation time", fmt: v => f(v, 2)},
  };
  const YMIN = [["0", "0 (full axis)"], ["auto", "lowest point"], ["0.5", "0.5"], ["0.8", "0.8"], ["0.9", "0.9"]];
  const DEFAULTS = {leg: "all", tier: "all", type: "all", sort: "score", dir: "desc",
    x: "cost", xlog: "1", labels: "auto", strip: "score", color: "type", ymin: "0", vol: "1000000", near: "0.1", top: "10"};
  const VOLUMES = [["1000", "1,000"], ["10000", "10,000"], ["100000", "100,000"], ["1000000", "1 million"]];
  // At-scale panel: how far below a leg's best score an engine may sit, and how many of those engines to show.
  const NEAR = [["0.05", "0.05"], ["0.1", "0.10"], ["0.15", "0.15"]];
  const TOP = [["1", "top 1"], ["2", "top 2"], ["3", "top 3"], ["5", "top 5"], ["10", "top 10"]];
  const rf = {...DEFAULTS};

  // ---------- numbers, exactly as the report computes and prints them ----------
  const isNum = v => typeof v === "number" && Number.isFinite(v);
  const numOf = v => typeof v === "boolean" ? +v : v;          // Python counts a bool metric as 0/1
  // Python 3.12+ sum() of floats is Neumaier-compensated; reproduce it so family means match to the bit.
  const fsum = xs => { let s = 0, c = 0; for (const x of xs) { const t = s + x; c += Math.abs(s) >= Math.abs(x) ? (s - t) + x : (x - t) + s; s = t; } return s + c; };
  const mean = xs => xs.length ? fsum(xs) / xs.length : null;
  // pyFixed (viewer.js) prints a number as Python does, ties to even; round4 is Python's round(x, 4).
  const round4 = x => Number(pyFixed(x, 4));
  const f = (v, n) => v == null ? "—" : pyFixed(v, n);
  const f4 = v => f(v, 4);
  const fi = v => v == null ? "—" : String(v);
  const cap = s => s.charAt(0).toUpperCase() + s.slice(1);
  const push = (obj, k, v) => (obj[k] ||= []).push(v);

  const valid = r => !!(r.applicable && !r.error && isNum(r.score));
  function metricValue(r, key) {
    if (key === "score") return r.score;
    const m = r.metrics || {};
    if (key === "missing_rate" || key === "wrong_rate") {
      const kv = r.leg === "key_value";
      const total = m[kv ? "kv_pairs_total" : "schema_fields_total"];
      const count = m[(kv ? "kv_pairs_" : "schema_fields_") + (key === "missing_rate" ? "missing" : "wrong_value")];
      return total && count != null ? count / total : null;
    }
    return numOf(m[key]);
  }
  // reporting.mean_metric: mean of the (tier, doc_type) family means; the headline rounds each family to 4 d.p.
  function meanMetric(rows, key = "score") {
    const groups = {};
    for (const r of rows) { const v = metricValue(r, key); if (valid(r) && isNum(v)) push(groups, r.tier + "|" + r.doc_type, v); }
    const means = Object.values(groups).map(v => key === "score" ? round4(mean(v)) : mean(v));
    return means.length ? mean(means) : null;
  }
  // reporting.low_cases: page/leg cases where every scored engine (>= 3) is <= 0.25 and no participant failed or is pending.
  function lowCases(rows) {
    const groups = {};
    for (const r of rows) if (r.leg !== "judge") push(groups, r.stem + "|" + r.leg, r);
    const cases = new Map();
    for (const [key, records] of Object.entries(groups)) {
      const scored = records.filter(valid), engines = new Set(scored.map(r => r.engine_name));
      const unresolved = records.some(r => r.error || (r.applicable && !valid(r)) || (r.metrics || {}).pending || /pending/i.test(String(r.reason ?? "")));
      if (!unresolved && engines.size >= MIN_ENGINES && Math.max(...scored.map(r => r.score)) <= LOW_THRESHOLD)
        cases.set(key, {page: scored[0].stem, leg: scored[0].leg, doc_type: scored[0].doc_type, engines: engines.size, max: Math.max(...scored.map(r => r.score))});
    }
    return cases;
  }

  // ---------- the recorded ledger: which call served a scored page, and what it cost ----------
  const mixedMark = r => r.mixed && r.mixed.length ? `<sup title="share of a batch that also ran: ${esc(r.mixed.join(", "))}">†</sup>` : "";
  const recorded = c => { let v = c.steady_latency_seconds; if (v == null) v = c.latency_seconds; return isNum(v) ? v : null; };
  const costOf = c => { let v = c.steady_cost_per_page; if (v == null) v = c.cost_per_page; return isNum(v) ? v : null; };
  // run_benchmark.unique_inference_calls: leg-selection rows that refer to one persisted call are merged.
  function uniqueCalls(calls) {
    const seen = new Map();
    for (const r of calls) {
      const key = [r.engine_name, r.stem, r.pass_name || "default", r.source_timestamp || r.timestamp,
        r.source_pass_name || r.pass_name || "default", !!r.error].join("|");
      if (!seen.has(key)) seen.set(key, {...r, legs: [...(r.legs || [])]});
      else { const u = seen.get(key); u.legs = [...new Set([...u.legs, ...(r.legs || [])])]; }
    }
    return [...seen.values()];
  }
  const callFor = r => D.results.index.get([r.engine_name, r.stem, r.pass_name || "default"].join("|"));
  // reporting._operations: the recorded pass serving each scored page of this leg, document-weighted.
  function operations(rows) {
    const acc = {};
    for (const r of rows) {
      if (!valid(r)) continue;
      const call = callFor(r);
      if (!call) continue;
      const a = acc[r.engine_name] ||= {cost: [], rec: [], tin: [], tout: []};
      const c = costOf(call), s = recorded(call);
      if (c != null) a.cost.push(c);
      if (s != null) a.rec.push(s);
      if (isNum(call.input_tokens)) a.tin.push(call.input_tokens);
      if (isNum(call.output_tokens)) a.tout.push(call.output_tokens);
    }
    const out = {};
    for (const [e, a] of Object.entries(acc)) out[e] = {cost: mean(a.cost), rec: mean(a.rec), cost_n: a.cost.length, rec_n: a.rec.length, tin: mean(a.tin), tout: mean(a.tout)};
    return out;
  }
  // run_benchmark.engine_operational: the "primary" pass (one call per page) and all passes, per engine.
  function engineOperational(calls, allCalls = calls) {
    const docCost = {}, docLat = {}, passCost = {}, passLat = {}, passes = {}, batchPasses = {};
    for (const r of uniqueCalls(calls)) {
      if (r.error || r.measurement_scope === "replayed_saved_call") continue;
      const e = r.engine_name, d = r.stem, cost = costOf(r), lat = recorded(r);
      if (cost != null) { docCost[e] ||= {}; docCost[e][d] = (docCost[e][d] || 0) + cost; }
      if (lat != null) { docLat[e] ||= {}; docLat[e][d] = (docLat[e][d] || 0) + lat; }
      const p = r.pass_name || "default";
      (passes[e] ||= new Set()).add(p);
      if (cost != null) push(passCost[e] ||= {}, p, cost);
      if (lat != null) push(passLat[e] ||= {}, p, lat);
    }
    for (const r of allCalls) {
      if (r.error) continue;
      const p = r.pass_name || "default";
      ((batchPasses[r.engine_name] ||= {})[r.batch_id || `solo|${r.stem || r.document_path}|${p}`] ||= new Set()).add(p);
    }
    const primary = e => {
      const ps = passes[e] || new Set();
      if (ps.has("default")) return "default";
      const text = [...ps].filter(p => p.includes("text"));
      if (text.length) return text.reduce((a, b) => b.length < a.length ? b : a);
      let best = "default", n = -1;
      for (const p of ps) { const k = (passCost[e]?.[p] || []).length; if (k > n) { best = p; n = k; } }
      return best;
    };
    const stats = {};
    for (const e of new Set([...Object.keys(docCost), ...Object.keys(docLat)])) {
      const costs = Object.values(docCost[e] || {}), lats = Object.values(docLat[e] || {}), p = primary(e);
      // Batches in which the primary pass ran beside other passes: their share of the batch is not that pass alone.
      const mixed = [...new Set(Object.values(batchPasses[e] || {}).filter(s => s.has(p) && s.size > 1).flatMap(s => [...s]))].filter(x => x !== p).sort();
      stats[e] = {cost_per_page: mean(passCost[e]?.[p] || []), latency_s: mean(passLat[e]?.[p] || []),
        cost_all_passes: mean(costs), latency_all_passes: mean(lats), passes: passes[e].size,
        documents: Object.keys(docCost[e] || {}).length || Object.keys(docLat[e] || {}).length, primary: p, mixed};
    }
    return stats;
  }

  // ---------- selection ----------
  let memo = {sig: null, data: {}};
  function cached(key, make) {
    const sig = JSON.stringify(rf);
    if (memo.sig !== sig) memo = {sig, data: {}};
    return key in memo.data ? memo.data[key] : (memo.data[key] = make());
  }
  const selection = () => D.results.rows.filter(r => (rf.tier === "all" || r.tier === rf.tier) && (rf.type === "all" || r.doc_type === rf.type));
  const selectedCalls = () => { const stems = new Set(selection().map(r => r.stem)); return D.results.ledger.filter(c => stems.has(c.stem)); };
  function docTypes() {
    const seen = new Map();
    for (const r of D.results.rows) seen.set(r.doc_type, r.tier);
    return [...seen].map(([type, tier]) => ({type, tier})).sort((a, b) => TIERS.indexOf(a.tier) - TIERS.indexOf(b.tier) || a.type.localeCompare(b.type));
  }

  // Marks are fixed for the session whatever the filter shows. Document types: hue by tier, one of four validated
  // shades within the tier, a circle for the first four types of the tier (alphabetical) and a square after.
  // Engines: the eight categorical hues in their fixed order × four shapes, in sidebar order, so up to 32 engines
  // keep distinct marks; labels, legend and tooltip carry identity as well.
  const SHAPES = ["circle", "square", "triangle", "diamond"];
  let TYPE_MARKS = {}, ENGINE_MARKS = {};
  function init() {
    const R = D.results;
    // reporting._operations indexes the ledger by (engine, page, pass); a later row for the same key replaces an earlier one.
    R.index = new Map();
    for (const c of R.ledger) if (!c.error) R.index.set([c.engine_name, c.stem, c.pass_name || "default"].join("|"), c);
    // Exclusion cases come from the run's own rows, as its report chooses them; never per filter.
    R.cases = lowCases(R.rows);
    TYPE_MARKS = {};
    for (const tier of TIERS) docTypes().filter(t => t.tier === tier).forEach((t, i) =>
      TYPE_MARKS[t.type] = {tier, cls: `t-${tier}-${i % 4 + 1}`, shape: i >= 4 ? "square" : "circle"});
    ENGINE_MARKS = {};
    engines.forEach((e, i) => ENGINE_MARKS[e] = {cls: "e-" + (i % 8), shape: SHAPES[Math.floor(i / 8) % 4]});
  }

  // ---------- sidebar dots: each engine's standing under the current filter ----------
  function dots(engine) {
    const byKey = cached("dots", () => { const m = {}; for (const r of selection()) push(m, r.engine_name + "|" + r.leg, r); return m; });
    return D.legs.map(leg => {
      const rows = byKey[engine + "|" + leg] || [];
      return {leg, score: meanMetric(rows), n: rows.filter(valid).length, errors: rows.filter(r => r.error).length, dim: rf.leg !== "all" && rf.leg !== leg};
    });
  }

  // ---------- tables ----------
  const tint = v => v == null ? "" : v >= .8 ? "ok" : v >= .5 ? "part" : "bad";
  // cols: {key, label, fmt, higher (marks the best and sets the sort direction), tint, countKey, title}
  function table(name, cols, rows) {
    const sortable = cols.filter(c => c.fmt);
    const sc = sortable.find(c => c.key === rf.sort) || sortable[0];
    const dir = sc && sortable.some(c => c.key === rf.sort) ? rf.dir : (sc?.higher === false ? "asc" : "desc");
    const sign = dir === "asc" ? 1 : -1;
    const ordered = rows.slice().sort((a, b) => {
      const x = a[sc?.key], y = b[sc?.key];
      if ((x == null) !== (y == null)) return x == null ? 1 : -1;
      if (x != null && x !== y) return sign * (x < y ? -1 : 1);
      return a.engine.localeCompare(b.engine);
    });
    const better = (x, y) => dir === "asc" ? x < y : x > y;
    const ranks = Object.fromEntries(ordered.map(r => [r.engine, r[sc?.key] == null ? "—" : 1 + ordered.filter(o => o[sc.key] != null && better(o[sc.key], r[sc.key])).length]));
    const best = {};
    for (const c of cols) if (c.fmt && c.higher != null) { const vs = rows.map(r => r[c.key]).filter(isNum); if (vs.length) best[c.key] = c.higher ? Math.max(...vs) : Math.min(...vs); }
    const head = cols.map(c => `<th data-key="${c.key}" class="${c.fmt ? "sortable" : ""} ${sc?.key === c.key ? "sorted" : ""}" title="${esc(c.title || "")}">${esc(c.label)}${c.higher === true ? " ↑" : c.higher === false ? " ↓" : ""}${sc?.key === c.key ? (dir === "asc" ? " ▲" : " ▼") : ""}</th>`).join("");
    const body = ordered.map(r => `<tr data-e="${esc(r.engine)}">` + cols.map(c => {
      if (c.key === "rank") return `<td data-v="${ranks[r.engine]}" class="muted">${ranks[r.engine]}</td>`;
      if (c.key === "engine") return `<td data-v="${esc(r.engine)}" class="eng-cell">${esc(r.engine)}</td>`;
      const v = r[c.key], text = (c.fmt ? c.fmt(v) : esc(v ?? "—")) + (c.note ? c.note(r) : "");
      const n = c.countKey ? r[c.countKey] : null, pages = r.pages;
      const suffix = c.countKey && n != null && pages != null && n > 0 && n < pages ? ` <span class="muted">(${n})</span>` : "";
      const isBest = isNum(v) && best[c.key] === v;
      return `<td data-v="${v == null ? "" : v}" class="${c.tint ? tint(v) : ""} ${isBest ? "best" : ""}">${isBest ? `<b>${text}</b>` : text}${suffix}</td>`;
    }).join("") + "</tr>").join("");
    return `<div class="twrap"><table class="v r" data-table="${esc(name)}"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  }

  // One row per participating engine for a leg, with every column the report prints.
  function legData(leg, rows) {
    const R = D.results, legRows = rows.filter(r => r.leg === leg);
    const engines = [...new Set(legRows.map(r => r.engine_name))].sort();
    const buckets = Object.fromEntries(engines.map(e => [e, legRows.filter(r => r.engine_name === e)]));
    const participants = engines.filter(e => buckets[e].some(r => valid(r) || r.error));
    const absent = engines.filter(e => !participants.includes(e));
    const present = TIERS.filter(t => legRows.some(r => valid(r) && r.tier === t));
    const tiers = present.length > 1 ? present : [];   // a single tier would only repeat the score
    const kept = e => buckets[e].filter(r => !R.cases.has(r.stem + "|" + r.leg));
    const ops = operations(legRows);
    const data = participants.map(e => {
      const rs = buckets[e], k = kept(e), row = {engine: e,
        score: meanMetric(rs), noout: meanMetric(k), pages: rs.filter(valid).length, kept: k.filter(valid).length,
        errors: rs.filter(r => r.error).length, unscored: rs.filter(r => r.unscored).length, pending: rs.filter(r => r.pending && !r.error).length};
      for (const t of tiers) row["tier:" + t] = meanMetric(rs.filter(r => r.tier === t));
      for (const [, key] of METRICS[leg]) { row["m:" + key] = meanMetric(rs, key); row["mn:" + key] = rs.filter(r => valid(r) && isNum(metricValue(r, key))).length; }
      Object.assign(row, {cost: null, rec: null, cost_n: 0, rec_n: 0, tin: null, tout: null}, ops[e] || {});
      return row;
    });
    return {leg, legRows, data, tiers, absent, buckets, pages: new Set(legRows.filter(valid).map(r => r.stem)).size};
  }
  // Geometry leads with Word (the score), then Line and Block cov beside it where the scorer records them.
  const GEO_LEAD = ["geometry_line_score", "geometry_block_score"];
  function legColumns(L) {
    const showNoOut = L.data.some(r => f4(r.noout) !== f4(r.score));
    const metricCol = ([label, key, up]) => ({key: "m:" + key, label, fmt: f4, higher: up, countKey: "mn:" + key, title: `${key}: family-weighted mean over scored pages that record it`});
    const geo = L.leg === "geometry";
    return [
      {key: "rank", label: "#"}, {key: "engine", label: "Engine"},
      {key: "score", label: geo ? "Word" : "Score", fmt: f4, higher: true, tint: true, title: "Full: mean of the document-family means of page scores; errors and N/A excluded"},
      ...(geo ? METRICS.geometry.filter(m => GEO_LEAD.includes(m[1])).map(m => ({...metricCol(m), tint: true})) : []),
      showNoOut && {key: "noout", label: "Without low", fmt: f4, higher: true, tint: true, title: "the same without the low-scoring page cases listed under Notes"},
      ...L.tiers.map(t => ({key: "tier:" + t, label: cap(t), fmt: f4, higher: true, tint: true, title: `family-weighted mean over the ${t} tier`})),
      {key: "pages", label: "Pages", fmt: fi, title: "pages with a score"},
      {key: "errors", label: "Err", fmt: fi, title: "rows recorded with an error"},
      L.data.some(r => r.unscored) && {key: "unscored", label: "Unscored", fmt: fi, title: "collected replies this run never scored"},
      L.data.some(r => r.pending) && {key: "pending", label: "Pending", fmt: fi, title: "rows awaiting a key review"},
      ...METRICS[L.leg].filter(m => !(geo && GEO_LEAD.includes(m[1]))).map(metricCol),
      {key: "cost", label: XM.cost.label, fmt: XM.cost.fmt, higher: false, countKey: "cost_n", title: XM.cost.title},
      {key: "rec", label: XM.rec.label, fmt: XM.rec.fmt, higher: false, countKey: "rec_n", title: XM.rec.title},
    ].filter(Boolean);
  }
  function familiesTable(L) {
    const kinds = docTypes().filter(t => L.legRows.some(r => valid(r) && r.doc_type === t.type));
    if (kinds.length < 2) return "";
    const rows = L.data.map(d => { const r = {engine: d.engine, pages: d.pages};
      for (const k of kinds) { const rs = L.buckets[d.engine].filter(x => x.doc_type === k.type); r["t:" + k.type] = meanMetric(rs); r["tn:" + k.type] = rs.filter(valid).length; }
      return r; });
    const cols = [{key: "engine", label: "Engine"}, ...kinds.map(k => ({key: "t:" + k.type, label: `${k.type} · ${k.tier}`, fmt: f4, higher: true, tint: true, countKey: "tn:" + k.type, title: "family mean (pages)"}))];
    return `<details class="panel" data-k="families"><summary>By document type · ${LEG_TITLES[L.leg]} full score per family (pages in brackets where fewer than the engine's total)</summary>${table("families:" + L.leg, cols, rows)}</details>`;
  }
  function judgePanel() {
    const rows = D.results.judge.filter(r => isNum(r.score) && (rf.tier === "all" || r.tier === rf.tier) && (rf.type === "all" || r.doc_type === rf.type));
    if (!rows.length) return "";
    const kinds = [...new Set(rows.map(r => r.doc_type))].sort();
    const engines = [...new Set(rows.map(r => r.engine_name))];
    const data = engines.map(e => { const rs = rows.filter(r => r.engine_name === e), d = {engine: e, acc: mean(rs.map(r => r.score)), docs: rs.length};
      for (const k of kinds) { const ks = rs.filter(r => r.doc_type === k); d["j:" + k] = ks.length ? mean(ks.map(r => r.score)) : null; d["jn:" + k] = ks.length; }
      d.pages = d.docs; return d; });
    const cols = [{key: "rank", label: "#"}, {key: "engine", label: "Engine"}, {key: "acc", label: "Item accuracy", fmt: v => f(v, 3), higher: true, tint: true, title: "checklist items right / items on the page, mean over judged pages"},
      {key: "docs", label: "Pages", fmt: fi}, ...kinds.map(k => ({key: "j:" + k, label: k, fmt: v => f(v, 3), higher: true, tint: true, countKey: "jn:" + k}))];
    return `<details class="panel" data-k="judge"><summary>Document judge estimate · the ${new Set(rows.map(r => r.stem)).size} silver-GT pages, scored against the image (not a leg)</summary>
      <p class="muted">Item accuracy = checklist items the engine got right / items on the page. The checklist is model-generated and not validated against human judgement; not comparable to the GT-scored legs. Engines without saved judge verdicts are omitted.</p>
      ${table("judge", cols, data)}</details>`;
  }

  // ---------- charts (inline SVG; the tables above and the per-page table are their table views) ----------
  const tickText = v => v >= 1 ? (Number.isInteger(v) ? String(v) : String(+v.toFixed(2))) : String(+v.toPrecision(2));
  function linearScale(lo, hi, n = 5) {
    const span = hi - lo || 1, raw = span / n, p = 10 ** Math.floor(Math.log10(raw)), q = raw / p, step = (q >= 5 ? 5 : q >= 2 ? 2 : 1) * p;
    const a = Math.floor(lo / step) * step, b = Math.ceil(hi / step) * step, ticks = [];
    for (let v = a; v <= b + step / 2; v += step) ticks.push(+v.toFixed(12));
    return {lo: a, hi: b === a ? a + step : b, ticks, minor: []};
  }
  function logScale(lo, hi) {
    const a = Math.floor(Math.log10(lo)), b = Math.max(Math.ceil(Math.log10(hi)), a + 1), ticks = [], minor = [];
    for (let e = a; e <= b; e++) { ticks.push(10 ** e); for (const m of [2, 5]) if (e < b) minor.push(m * 10 ** e); }
    return {lo: 10 ** a, hi: 10 ** b, ticks, minor, log: true};
  }
  // The score axis from 0, from a chosen value, or from just below the lowest point shown.
  function scoreScale(values) {
    let lo = 0;
    if (rf.ymin === "auto") { const m = Math.min(...values.filter(isNum)); lo = isNum(m) ? Math.max(0, Math.min(0.95, Math.floor((m - 0.01) * 20) / 20)) : 0; }
    else lo = Math.max(0, Math.min(0.95, Number(rf.ymin) || 0));
    const span = 1 - lo, step = span <= 0.25 ? 0.05 : span <= 0.5 ? 0.1 : 0.2, ticks = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= 1 + 1e-9; v += step) ticks.push(+v.toFixed(2));
    if (ticks[0] !== +lo.toFixed(2)) ticks.unshift(+lo.toFixed(2));
    return {lo, hi: 1, ticks, minor: [], score: true};
  }
  const scaler = (s, from, to) => v => from + (s.log ? (Math.log10(v) - Math.log10(s.lo)) / (Math.log10(s.hi) - Math.log10(s.lo)) : (v - s.lo) / (s.hi - s.lo)) * (to - from);
  const overlap = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
  // The mark for a point under a colour mode: a class carrying the colour, and the shape.
  function markOf(p, mode) {
    if (mode === "engine") { const m = ENGINE_MARKS[p.engine]; if (m) return m; }
    if (mode === "type") { const m = TYPE_MARKS[p.doc_type]; if (m) return m; }
    if (mode === "tier" && p.tier) return {cls: "tier-" + p.tier, shape: "circle"};
    return {cls: "pub", shape: "circle"};
  }
  function markSVG(shape, cx, cy, r) {
    switch (shape) {
      case "square": return `<rect class="dot" x="${cx - r}" y="${cy - r}" width="${2 * r}" height="${2 * r}"/>`;
      case "triangle": return `<path class="dot" d="M${cx},${cy - r * 1.25} L${cx + r * 1.2},${cy + r * 0.9} L${cx - r * 1.2},${cy + r * 0.9} Z"/>`;
      case "diamond": return `<path class="dot" d="M${cx},${cy - r * 1.35} L${cx + r * 1.35},${cy} L${cx},${cy + r * 1.35} L${cx - r * 1.35},${cy} Z"/>`;
      default: return `<circle class="dot" cx="${cx}" cy="${cy}" r="${r}"/>`;
    }
  }

  // Score (y) against one recorded measure (x), one point per engine. Points under the axis start sit on
  // the baseline as hollow triangles, so nothing is silently dropped.
  function scatterSVG(pts, {xm, log, labels, w = 680, h = 380, small = false}) {
    const L = small ? 34 : 48, R = 14, T = 10, B = small ? 28 : 38, pw = w - L - R, ph = h - T - B;
    const xv = pts.filter(p => !p.zero).map(p => p.x);
    const xs = !xv.length ? linearScale(0, 1) : log ? logScale(Math.min(...xv), Math.max(...xv)) : linearScale(Math.min(0, ...xv), Math.max(...xv));
    const ys = scoreScale(pts.map(p => p.y));
    const X = scaler(xs, L, L + pw), Y = v => T + (1 - (v - ys.lo) / (1 - ys.lo)) * ph;
    let g = "";
    for (const v of ys.ticks) g += `<line class="grid" x1="${L}" x2="${L + pw}" y1="${Y(v)}" y2="${Y(v)}"/><text class="tick" x="${L - 6}" y="${Y(v) + 3.5}" text-anchor="end">${tickText(v)}</text>`;
    for (const v of xs.ticks) g += `<line class="grid" y1="${T}" y2="${T + ph}" x1="${X(v)}" x2="${X(v)}"/><text class="tick" x="${X(v)}" y="${T + ph + 14}" text-anchor="middle">${tickText(v * (xm.mult || 1))}</text>`;
    for (const v of xs.minor) g += `<line class="grid minor" y1="${T}" y2="${T + ph}" x1="${X(v)}" x2="${X(v)}"/>`;
    g += `<line class="axis" x1="${L}" x2="${L + pw}" y1="${T + ph}" y2="${T + ph}"/><line class="axis" x1="${L}" x2="${L}" y1="${T}" y2="${T + ph}"/>`;
    g += `<text class="tick" x="${L + pw / 2}" y="${h - 4}" text-anchor="middle">${esc(xm.label)}${log ? " (log)" : ""}</text>`;
    if (!small) g += `<text class="tick" transform="translate(12 ${T + ph / 2}) rotate(-90)" text-anchor="middle">score${ys.lo > 0 ? ` from ${ys.lo}` : ""}</text>`;
    const show = labels === "on" || (labels === "auto" && pts.length <= 40 && !small);
    const order = pts.map((p, i) => i).sort((a, b) => pts[b].y - pts[a].y);
    const pos = pts.map(p => { const below = p.y < ys.lo; return {cx: p.zero ? L : X(p.x), cy: below ? T + ph : Y(p.y), below}; });
    const taken = pos.map(q => ({x: q.cx - 7, y: q.cy - 7, w: 14, h: 14}));   // every mark keeps a little clear space
    // Label positions tried in turn: beside the point first, then nudged with a leader line; a label that fits
    // nowhere stays in the tooltip and legend.
    const CANDS = [[8, 0, "start"], [-8, 0, "end"], [7, -10, "start"], [7, 11, "start"], [-7, -10, "end"], [-7, 11, "end"],
      [0, -12, "middle"], [0, 14, "middle"], [16, -20, "start"], [16, 22, "start"], [-16, -20, "end"], [-16, 22, "end"], [24, -30, "start"], [24, 32, "start"], [-24, -30, "end"], [-24, 32, "end"]];
    let dotsHTML = "", textHTML = "", leaders = "";
    for (const i of order) {
      const p = pts[i], {cx, cy, below} = pos[i], m = markOf(p, "engine");
      dotsHTML += `<g class="pt ${m.cls} ${below ? "below" : ""} ${p.zero ? "zero" : ""}" tabindex="0" data-i="${i}" data-engine="${esc(p.engine)}"><circle class="hit" cx="${cx}" cy="${cy}" r="12"/>${below ? `<path class="dot" d="M${cx - 5},${cy - 10} L${cx + 5},${cy - 10} L${cx},${cy - 1} Z"/>` : p.zero ? `<path class="dot" d="M${cx + 1},${cy - 6} L${cx + 11},${cy} L${cx + 1},${cy + 6} Z"/>` : markSVG(m.shape, cx, cy, 5)}</g>`;
      if (!show || below) continue;
      const bw = p.engine.length * 6.3 + 4;
      for (const [dx, dy, anchor] of CANDS) {
        const ax = cx + dx, box = {x: anchor === "start" ? ax : anchor === "end" ? ax - bw : ax - bw / 2, y: cy + dy - 6, w: bw, h: 12};
        if (box.x < 2 || box.x + box.w > w - 2 || box.y < 0 || box.y + box.h > h - B + 4) continue;
        if (taken.some((q, j) => j !== i && overlap(q, box))) continue;
        taken.push(box);
        textHTML += `<text class="lbl" data-engine="${esc(p.engine)}" x="${ax}" y="${cy + dy + 3.5}" text-anchor="${anchor}">${esc(p.engine)}</text>`;
        if (dy !== 0 || Math.abs(dx) > 8) {
          const ex = anchor === "start" ? ax - 2 : anchor === "end" ? ax + 2 : ax, ey = anchor === "middle" ? cy + dy + (dy < 0 ? 7 : -7) : cy + dy;
          leaders += `<line class="leader" x1="${cx}" y1="${cy}" x2="${ex}" y2="${ey}"/>`;
        }
        break;
      }
    }
    return {svg: `<svg class="chart" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img">${g}${leaders}${dotsHTML}${textHTML}</svg>`, lo: ys.lo};
  }
  // One row per engine, one dot per page.
  function stripSVG(rowsE, {xm, log}) {
    const w = 880, L = 176, R = 16, T = 8, rh = 20, B = 30, pw = w - L - R, h = T + rowsE.length * rh + B;
    const vals = rowsE.flatMap(r => r.points.map(p => p.v));
    const xs = xm.key === "score" ? scoreScale(vals) : log ? logScale(Math.min(...vals), Math.max(...vals)) : linearScale(Math.min(0, ...vals), Math.max(...vals));
    const X = scaler(xs, L, L + pw);
    let g = "";
    for (const v of xs.ticks) g += `<line class="grid" y1="${T}" y2="${T + rowsE.length * rh}" x1="${X(v)}" x2="${X(v)}"/><text class="tick" x="${X(v)}" y="${T + rowsE.length * rh + 14}" text-anchor="middle">${tickText(v * (xm.mult || 1))}</text>`;
    for (const v of xs.minor) g += `<line class="grid minor" y1="${T}" y2="${T + rowsE.length * rh}" x1="${X(v)}" x2="${X(v)}"/>`;
    g += `<text class="tick" x="${L + pw / 2}" y="${h - 3}" text-anchor="middle">${esc(xm.label)}${xs.score && xs.lo > 0 ? ` from ${xs.lo}` : ""}${log && xm.key !== "score" ? " (log)" : ""}</text>`;
    let dotsHTML = "";
    rowsE.forEach((r, i) => {
      const y = T + i * rh + rh / 2;
      g += `<line class="grid" x1="${L}" x2="${L + pw}" y1="${T + (i + 1) * rh}" y2="${T + (i + 1) * rh}"/>`;
      g += `<text class="lbl" x="${L - 10}" y="${y + 3.5}" text-anchor="end">${esc(r.engine)}</text>`;
      r.errors.forEach((e, j) => { dotsHTML += `<g class="pt err" tabindex="0" data-r="${i}" data-e="${j}" data-stem="${esc(e.stem)}" data-type="${esc(e.doc_type)}" data-engine="${esc(r.engine)}"><circle class="hit" cx="${L + 6 + j * 8}" cy="${y}" r="8"/><text class="x" x="${L + 6 + j * 8}" y="${y + 3.5}" text-anchor="middle">✕</text></g>`; });
      r.points.forEach((p, j) => {
        const dy = ((j * 37) % 11 - 5) * .7, below = xs.score && p.v < xs.lo, cx = below ? L + 4 : X(p.v);
        const m = markOf({engine: r.engine, tier: p.tier, doc_type: p.doc_type}, rf.color);
        dotsHTML += `<g class="pt ${m.cls} ${below ? "below" : ""}" tabindex="0" data-r="${i}" data-p="${j}" data-stem="${esc(p.stem)}" data-type="${esc(p.doc_type)}" data-engine="${esc(r.engine)}"><circle class="hit" cx="${cx}" cy="${y + dy}" r="9"/>${below ? `<path class="dot" d="M${cx + 4},${y + dy - 4} L${cx - 4},${y + dy} L${cx + 4},${y + dy + 4} Z"/>` : markSVG(m.shape, cx, y + dy, 4)}</g>`;
      });
    });
    g += `<line class="axis" x1="${L}" x2="${L + pw}" y1="${T + rowsE.length * rh}" y2="${T + rowsE.length * rh}"/>`;
    return {svg: `<svg class="chart" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img">${g}${dotsHTML}</svg>`, lo: xs.score ? xs.lo : null};
  }
  const legendHTML = items => `<div class="legend">${items.map(([cls, label, shape]) => `<span><i class="sw ${cls} ${shape || ""}"></i>${esc(label)}</span>`).join("")}</div>`;
  const entry = (key, value, m, label) => `<span class="lg" data-h="${esc(value)}" tabindex="0"><i class="sw ${m.cls} ${m.shape}"></i>${esc(label)}</span>`;
  // Legends whose entries are highlighters (hover lights the matching marks up, click pins): engines, or document
  // types grouped by tier.
  const engineLegend = (list, id, hint) => `<div class="legend types" id="${id}" data-key="engine">${list.map(e => entry("engine", e, ENGINE_MARKS[e] || {cls: "pub", shape: "circle"}, e)).join("")}<span class="muted">${esc(hint)}</span></div>`;
  function typeLegend(typesShown, id) {
    const groups = TIERS.map(tier => { const types = docTypes().filter(t => t.tier === tier && typesShown.has(t.type));
      return types.length ? `<span class="tiergrp"><b>${tier}</b> ${types.map(t => entry("type", t.type, TYPE_MARKS[t.type], t.type)).join("")}</span>` : ""; }).join("");
    return `<div class="legend types" id="${id}" data-key="type">${groups}<span class="muted">hover a type to light it up across every row, click to pin; hover a dot to follow that page down the rows</span></div>`;
  }
  function stripLegend(typesShown, enginesShown) {
    if (rf.color === "engine") return engineLegend(enginesShown, "strip-legend", "hover an engine to light its row up, click to pin; hover a dot to follow that page down the rows");
    if (rf.color === "tier") return legendHTML(TIERS.map(t => ["tier-" + t, t + " tier"]));
    return typeLegend(typesShown, "strip-legend");
  }
  const scatterLegend = (enginesShown, id) => engineLegend(enginesShown, id, "hover an engine here or in the table to light its point up, click to pin");

  let tipEl = null;
  function tip(lines, x, y) {
    tipEl ||= Object.assign(document.body.appendChild(document.createElement("div")), {id: "tip"});
    tipEl.replaceChildren(...lines.map((t, i) => { const d = document.createElement("div"); d.textContent = t; if (!i) d.className = "t0"; return d; }));
    tipEl.style.display = "block";
    const vw = window.innerWidth, vh = window.innerHeight, bw = tipEl.offsetWidth, bh = tipEl.offsetHeight;
    tipEl.style.left = `${x + 14 + bw > vw ? x - 14 - bw : x + 14}px`;
    tipEl.style.top = `${y + 14 + bh > vh ? y - 14 - bh : y + 14}px`;
  }
  const hideTip = () => { if (tipEl) tipEl.style.display = "none"; };
  function wire(svg, lookup, onClick) {
    svg.querySelectorAll(".pt").forEach(node => {
      const show = ev => { const lines = lookup(node.dataset); if (!lines) return; const r = node.getBoundingClientRect(); tip(lines, ev.clientX ?? r.right, ev.clientY ?? r.top); };
      node.addEventListener("pointerenter", show); node.addEventListener("pointermove", show); node.addEventListener("focus", show);
      node.addEventListener("pointerleave", hideTip); node.addEventListener("blur", hideTip);
      node.addEventListener("click", () => onClick(node.dataset));
      node.addEventListener("keydown", ev => { if (ev.key === "Enter") onClick(node.dataset); });
    });
  }
  // Highlighting: legend entries (hover; click to pin) and, for the scatter, table rows light up every mark and
  // label whose data-<key> matches; `follow` lets a hovered mark light up its siblings by another attribute.
  function wireHighlight(svgs, legend, key, rows = []) {
    let pinned = null;
    const nodes = () => svgs.flatMap(s => [...s.querySelectorAll(".pt, .lbl")]);
    const emph = sel => { for (const s of svgs) s.classList.toggle("emph", !!sel); for (const n of nodes()) n.classList.toggle("on", !!sel && sel(n)); };
    const by = v => n => n.dataset[key] === v;
    const back = () => emph(pinned && by(pinned));
    for (const l of [...(legend?.querySelectorAll(".lg[data-h]") || []), ...rows]) {
      const v = l.dataset.h ?? l.dataset.e, show = () => emph(by(v));
      l.addEventListener("pointerenter", show); l.addEventListener("focus", show);
      l.addEventListener("pointerleave", back); l.addEventListener("blur", back);
      if (l.dataset.h != null) {
        l.addEventListener("click", () => { pinned = pinned === v ? null : v;
          legend.querySelectorAll(".lg").forEach(x => x.classList.toggle("pinned", x.dataset.h === pinned)); back(); });
        l.addEventListener("keydown", ev => { if (ev.key === "Enter") l.click(); });
      }
    }
    return {follow: attr => { for (const n of nodes()) if (n.dataset[attr] != null) {
      n.addEventListener("pointerenter", () => emph(x => x.dataset[attr] === n.dataset[attr]));
      n.addEventListener("pointerleave", back);
    } }};
  }

  function scatterPanel(L, {small = false, id} = {}) {
    const xm = {...XM[rf.x] || XM.cost, key: rf.x in XM ? rf.x : "cost"}, log = rf.xlog === "1";
    const pts = L.data.filter(r => r.score != null).map(r => ({engine: r.engine, y: r.score, x: r[xm.key], n: r.pages, xn: r[xm.key + "_n"]}));
    for (const p of pts) p.zero = log && p.x === 0;   // unpriced local engine: drawn at the axis start
    const plotted = pts.filter(p => isNum(p.x) && (!log || p.x >= 0)), left = pts.filter(p => !plotted.includes(p));
    const drawn = plotted.length ? scatterSVG(plotted, {xm, log, labels: rf.labels, small, w: small ? 300 : 680, h: small ? 210 : 380}) : null;
    const below = drawn ? plotted.filter(p => p.y < drawn.lo) : [];
    const notes = [];
    if (left.length) notes.push(`Not plotted: ${left.map(p => `${esc(p.engine)} (${p.x == null ? "not recorded" : xm.fmt(p.x)})`).join(", ")}`);
    const zero = plotted.filter(p => p.zero);
    if (zero.length) notes.push(`At the axis start, shown as ▷ ($0 recorded: local engine, machine cost unpriced): ${zero.map(p => esc(p.engine)).join(", ")}`);
    if (below.length) notes.push(`Below the axis start (${drawn.lo}), shown as ▼ on the baseline: ${below.map(p => `${esc(p.engine)} ${f4(p.y)}`).join(", ")}`);
    return {html: `<div class="chartbox" id="${id}">${drawn ? drawn.svg : `<p class="muted">No engine in this selection records ${esc(xm.label)}.</p>`}${notes.map(n => `<div class="muted" style="font-size:11px">${n}</div>`).join("")}</div>`, pts: plotted, leg: L.leg, engines: plotted.map(p => p.engine)};
  }
  function stripPanel(L) {
    const key = rf.strip in XM ? rf.strip : "score", xm = key === "score" ? {label: "score", key, fmt: f4} : {...XM[key], key};
    const log = key !== "score" && rf.xlog === "1";
    const order = L.data.slice().sort((a, b) => (b.score ?? -1) - (a.score ?? -1) || a.engine.localeCompare(b.engine));
    const rowsE = order.map(d => {
      const rs = L.buckets[d.engine], points = [], errors = [], other = [];
      for (const r of rs) {
        if (r.error) { errors.push(r); continue; }
        if (!valid(r)) { other.push(r); continue; }
        const call = callFor(r), v = key === "score" ? r.score : call ? (key === "cost" ? costOf(call) : recorded(call)) : null;
        if (isNum(v) && (!log || v > 0)) points.push({...r, v}); else other.push(r);
      }
      return {engine: d.engine, points, errors, other};
    }).filter(r => r.points.length || r.errors.length);
    if (!rowsE.length) return {html: `<p class="muted">Nothing to plot.</p>`, rowsE: [], types: new Set(), engines: []};
    const drawn = stripSVG(rowsE, {xm, log});
    const below = drawn.lo != null ? rowsE.reduce((n, r) => n + r.points.filter(p => p.v < drawn.lo).length, 0) : 0;
    return {html: drawn.svg + (below ? `<div class="muted" style="font-size:11px">${below} page${below === 1 ? "" : "s"} below the axis start (${drawn.lo}) sit at the left edge as ◀.</div>` : ""),
      rowsE, xm, types: new Set(rowsE.flatMap(r => [...r.points, ...r.errors].map(p => p.doc_type))), engines: rowsE.map(r => r.engine)};
  }
  function pagesTable(L) {
    const rows = [];
    for (const d of L.data) for (const r of L.buckets[d.engine].slice().sort((a, b) => a.stem.localeCompare(b.stem))) {
      const call = callFor(r);
      rows.push({engine: r.engine_name, page: r.stem, tier: r.tier, type: r.doc_type, score: valid(r) ? r.score : null,
        cost: call ? costOf(call) : null, rec: call ? recorded(call) : null,
        status: r.error ? "error: " + String(r.error).slice(0, 80) : r.unscored ? "unscored" : r.pending ? "pending review" : r.applicable === false ? "n/a: " + (r.reason || "") : valid(r) ? "" : r.reason || "no score"});
    }
    const cols = [{key: "engine", label: "Engine"}, {key: "page", label: "Page"}, {key: "tier", label: "Tier"}, {key: "type", label: "Type"},
      {key: "score", label: "Score", fmt: f4, tint: true}, {key: "cost", label: XM.cost.label, fmt: XM.cost.fmt}, {key: "rec", label: XM.rec.label, fmt: XM.rec.fmt}, {key: "status", label: "Status"}];
    const body = rows.map(r => `<tr data-e="${esc(r.engine)}" data-d="${esc(r.page)}" class="link">` + cols.map(c => `<td data-v="${c.fmt ? (r[c.key] ?? "") : esc(r[c.key] ?? "")}" class="${c.tint ? tint(r[c.key]) : ""}">${c.key === "type" && TYPE_MARKS[r.type] ? `<i class="sw ${TYPE_MARKS[r.type].cls} ${TYPE_MARKS[r.type].shape}"></i>` : ""}${c.fmt ? c.fmt(r[c.key]) : esc(r[c.key] ?? "")}</td>`).join("") + "</tr>").join("");
    return `<details class="panel" data-k="pages"><summary>Per-page values · table view of the chart (${rows.length} rows; click a row to open the page)</summary>
      <div class="twrap tall"><table class="v r" data-table="pages:${L.leg}"><thead><tr>${cols.map(c => `<th data-key="${c.key}">${esc(c.label)}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table></div></details>`;
  }

  // ---------- views ----------
  const selectHTML = (opt, options, label) => `<label>${esc(label)} <select data-o="${opt}">${options.map(([v, t]) => `<option value="${v}" ${rf[opt] === v ? "selected" : ""}>${esc(t)}</option>`).join("")}</select></label>`;
  const xControls = () => selectHTML("x", Object.entries(XM).map(([k, m]) => [k, m.label]), "x")
    + `<label><input type="checkbox" data-o="xlog" ${rf.xlog === "1" ? "checked" : ""}> log axis</label>`
    + selectHTML("ymin", YMIN, "score axis from");
  function filterBar() {
    const rows = selection(), stems = new Set(rows.map(r => r.stem)), eng = new Set(rows.map(r => r.engine_name));
    el("leg-tabs").innerHTML = ["all", ...D.legs].map(l => `<span class="tab ${rf.leg === l ? "active" : ""}" data-f="leg" data-v="${l}">${l === "all" ? "0 All legs" : `${D.legs.indexOf(l) + 1} ${LEG_TITLES[l]}`}</span>`).join("");
    const sel = (name, options, title) => `<label title="${esc(title)}">${esc(name)} <select data-f="${name.toLowerCase()}">${options.map(([v, t]) => `<option value="${v}" ${rf[name.toLowerCase()] === v ? "selected" : ""}>${esc(t)}</option>`).join("")}</select></label>`;
    el("doc-tabs").innerHTML = sel("Tier", [["all", "all tiers"], ...TIERS.map(t => [t, t])], "difficulty tier")
      + sel("Type", [["all", "all document types"], ...docTypes().map(t => [t.type, `${t.type} · ${t.tier}`])], "document family")
      + `<span class="sc muted">${rows.length} rows · ${stems.size} page${stems.size === 1 ? "" : "s"} · ${eng.size} engine${eng.size === 1 ? "" : "s"}</span>`;
    el("leg-tabs").querySelectorAll(".tab").forEach(n => n.onclick = () => update({leg: n.dataset.v}));
    el("doc-tabs").querySelectorAll("select").forEach(s => s.onchange = () => update({[s.dataset.f]: s.value}));
  }
  const REPO = "https://github.com/ilsilfverskiold/ocr-engine-evaluation";
  const SCORING = REPO + "/blob/main/public_docs/scoring-design.md";
  const link = (href, text) => `<a href="${href}" target="_blank" rel="noopener">${esc(text)}</a>`;
  // Written for a reader who has not seen the code: what was tested, how a score is averaged, what cost means.
  function headHTML() {
    const R = D.results, nEngines = new Set(R.rows.map(r => r.engine_name)).size, nPages = new Set(R.rows.map(r => r.stem)).size;
    const types = docTypes();
    return `<h1>Results</h1>
      <div class="meta"><span>Published run ${esc(D.baseline)}: ${nEngines} engines · ${nPages} pages · ${R.rows.length} result rows</span></div>
      <details class="panel" data-k="how"><summary>How these numbers are computed</summary>
        <ul class="notes">
          <li><b>What was tested.</b> ${nEngines} OCR engines and engine configurations, from local tools (Tesseract, Docling) and self-hosted open models to cloud APIs, each ran on the same ${nPages} single-page documents: ${types.length} document types (${types.map(t => esc(t.type.replace(/_/g, " "))).join(", ")}) in three difficulty tiers, easy, medium and hard.</li>
          <li><b>Six tasks, scored separately.</b> Each task is called a leg: Text (did it read the words), Markdown (is the page structure right), Tables (are the cells right), Key-value (did it pull the printed fields), Schema (can it fill a given set of fields) and Geometry (are the text boxes in the right place). Every leg has its own reference answers and a score from 0 to 1. There is no combined score.</li>
          <li><b>How an engine's score is averaged.</b> Pages are averaged within each document type first, then those averages are averaged, so every type counts equally whatever its number of pages. Failed calls and pages without a reference are left out, not counted as zero; the Err and Pages columns show how many.</li>
          <li><b>Cost and time.</b> ${esc(XM.cost.label)} is the recorded cost of 1,000 pages: the provider's price for cloud APIs, and measured GPU processing time at Modal's rates for self-hosted models (container startup and idle time not included). Tesseract and Docling run on your own CPU and are unpriced. ${esc(XM.rec.label)} is the recorded time per page. A call that serves several legs is paid once, so do not add costs across legs. † on a time: the call ran in a batch with other calls, so its share includes their time.</li>
          <li><b>Without low</b> appears where it changes a figure. It leaves out page/leg cases where at least ${MIN_ENGINES} engines all scored ${LOW_THRESHOLD} or less, the same cases for every engine. It is a sensitivity check, not a ranking.</li>
          <li><b>Charts</b> are drawn from the tables beside them. Hover a point for its values; click it to open that engine and page with the prompt sent, the full reply and the reference.</li>
          <li><b>Where it comes from.</b> Every value is read from the saved results of this one run; nothing is re-scored in the browser. The code, the references, every prompt sent and every reply received are public: ${link(REPO, REPO.replace("https://", ""))}. The full rules for each leg are in ${link(SCORING, "scoring-design.md")}.</li>
        </ul></details>`;
  }
  function notesPanel(legs) {
    const cases = [...D.results.cases.values()].filter(c => legs.includes(c.leg));
    return `<details class="panel" data-k="notes"><summary>Notes · exclusions and definitions</summary>
      ${cases.length ? `<p>Low-page cases excluded in "Without low": ${cases.map(c => `${esc(c.page)} (${esc(c.doc_type)}, ${LEG_TITLES[c.leg]}, ${c.engines} engines, max ${f(c.max, 3)})`).join("; ")}.</p>` : "<p class=\"muted\">No low-page exclusion applies to this leg; Full and Without low are identical.</p>"}
      <p class="muted">Key-value and schema: Missing and Wrong are field fractions; Unmapped and Extra are mean counts per page. A bracketed count after a metric means fewer pages record it than are scored. Unequal page coverage can change rankings; read Pages alongside every column.</p></details>`;
  }

  const charts = [];
  // Collapsible panels the reader has opened stay open across re-renders (sorting, a filter or chart control).
  const openPanels = new Set();
  const panels = () => document.querySelectorAll("#engine-head details[data-k], #content details[data-k]");
  const rememberPanels = () => { for (const d of panels()) d.open ? openPanels.add(d.dataset.k) : openPanels.delete(d.dataset.k); };
  const restorePanels = () => { for (const d of panels()) { d.open = openPanels.has(d.dataset.k); d.ontoggle = () => d.open ? openPanels.add(d.dataset.k) : openPanels.delete(d.dataset.k); } };
  function legView(rows) {
    const L = cached("leg:" + rf.leg, () => legData(rf.leg, rows));
    const scatter = scatterPanel(L, {id: "scatter"}), strip = stripPanel(L);
    charts.push({sel: "#scatter svg", pts: scatter.pts, leg: L.leg, kind: "scatter"}, {sel: "#strip svg", rowsE: strip.rowsE, xm: strip.xm, leg: L.leg, kind: "strip"});
    const stripSel = selectHTML("strip", [["score", "score"], ...Object.entries(XM).map(([k, m]) => [k, m.label])], "value")
      + selectHTML("color", [["type", "document type"], ["tier", "tier"], ["engine", "engine"]], "colour")
      + (rf.strip === "score" ? selectHTML("ymin", YMIN, "score axis from") : "");
    return `<div class="panel"><h3>${LEG_TITLES[L.leg]} — ${L.pages} scored page${L.pages === 1 ? "" : "s"} in this selection</h3>
        <p class="muted">${esc(METHOD[L.leg])} Click a column to sort; bold marks the best value per column.</p>
        ${L.data.length ? table("leg:" + L.leg, legColumns(L), L.data) : '<p class="muted">No engine has a scored or failed row here.</p>'}
        ${L.absent.length ? `<p class="muted">N/A only (not ranked): ${L.absent.map(esc).join(", ")}</p>` : ""}</div>
      <div class="panel"><h3>Score against ${esc((XM[rf.x] || XM.cost).label)} · one point per engine</h3>
        <div class="chart-controls">${xControls()}${selectHTML("labels", [["auto", "auto"], ["on", "on"], ["off", "off"]], "labels")}</div>${scatterLegend(scatter.engines, "scatter-legend")}${scatter.html}</div>
      <div class="panel"><h3>Every page · one mark per scored page, ✕ = error</h3>
        <div class="chart-controls">${stripSel}</div>${stripLegend(strip.types, strip.engines)}<div id="strip">${strip.html}</div>
        ${pagesTable(L)}</div>
      ${familiesTable(L)}
      ${L.leg === "text" ? judgePanel() : ""}
      ${notesPanel([L.leg])}`;
  }
  // ---------- best per leg at scale: the leg table's own $/page and s/page, multiplied by a page volume ----------
  const money = v => v == null ? "—" : "$" + v.toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2});
  function duration(sec) {
    if (sec == null) return "—";
    if (sec < 60) return `${Math.round(sec)} sec`;
    const mins = Math.round(sec / 60);
    if (sec < 86400) return mins < 60 ? `${mins} min` : `${Math.floor(mins / 60)} hr${mins % 60 ? ` ${mins % 60} min` : ""}`;
    const hrs = Math.round(sec / 3600), d = Math.floor(hrs / 24);
    return `${d} day${d === 1 ? "" : "s"}${hrs % 24 ? ` ${hrs % 24} hr` : ""}`;
  }
  // reporting.parallel_rate: [pages per second, binding limit] at the engine's documented default limits.
  function parallelRate(lim, secPerPage, tin, tout) {
    lim = lim || {};
    if (lim.requests_per_second) return [lim.requests_per_second, "requests_per_second"];
    if (lim.requests_per_minute) {
      const caps = [[lim.requests_per_minute / 60, "requests_per_minute"]];
      if (lim.input_tokens_per_minute && tin) caps.push([lim.input_tokens_per_minute / 60 / tin, "input_tokens_per_minute"]);
      if (lim.output_tokens_per_minute && tout) caps.push([lim.output_tokens_per_minute / 60 / tout, "output_tokens_per_minute"]);
      return caps.reduce((a, b) => b[0] < a[0] ? b : a);
    }
    for (const key of ["concurrent_jobs", "concurrent_gpus"]) if (lim[key] && secPerPage) return [lim[key] / secPerPage, key];
    return [null, null];
  }
  // reporting.parallel_seconds: a daily request cap binds only beyond one day's allowance.
  function parallelSeconds(lim, pages, secPerPage, tin, tout) {
    const [rate, by] = parallelRate(lim, secPerPage, tin, tout);
    if (!rate) return [null, null, null];
    const seconds = pages / rate, perDay = (lim || {}).requests_per_day;
    if (perDay && pages > perDay && pages / perDay * 86400 > seconds) return [pages / perDay * 86400, "requests_per_day", rate];
    return [seconds, by, rate];
  }
  const big = n => n >= 1e6 ? `${+(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${+(n / 1e3).toFixed(0)}k` : String(n);
  function limitText(lim, by, rate, rec) {
    if (!by) return lim?.basis?.startsWith("local") ? "local: scales with CPU cores" : "no default limit declared";
    const what = {requests_per_second: `${lim[by]} request${lim[by] === 1 ? "" : "s"}/s`, requests_per_minute: `${big(lim[by])} requests/min`,
      input_tokens_per_minute: `${big(lim[by])} input tokens/min`, output_tokens_per_minute: `${big(lim[by])} output tokens/min`,
      requests_per_day: `${big(lim[by])} requests/day`, concurrent_jobs: `${lim[by]} jobs at once`, concurrent_gpus: `${lim[by]} GPUs at once`}[by];
    const text = by.startsWith("concurrent") || by === "requests_per_day" ? what : `${what} · about ${Math.round(rate * rec)} calls at once`;
    return lim.basis?.startsWith("estimate") ? `${text} (estimate)` : text;
  }
  function scalePanel(Ls) {
    const vol = +rf.vol || 1e6, near = Math.max(0, +rf.near || 0), top = +rf.top || 1, volText = (VOLUMES.find(v => +v[0] === vol) || [, String(vol)])[1];
    const rows = [], unpriced = [];
    // Lower $/page, then lower s/page; unrecorded values rank last.
    const low = (x, y) => (x == null) - (y == null) || (x ?? 0) - (y ?? 0);
    const cheaper = (a, b) => low(a.cost, b.cost) || low(a.rec, b.rec) || round4(b.score) - round4(a.score) || a.engine.localeCompare(b.engine);
    for (const l of D.legs) {
      // Score against cost: every engine within `near` of the leg's best score (4 d.p.), cheapest first, the first `top` shown.
      const scored = Ls[l].data.filter(r => isNum(r.score));
      if (!scored.length) continue;
      const bestScore = Math.max(...scored.map(r => round4(r.score)));
      const best = scored.filter(r => round4(r.score) === bestScore).sort(cheaper)[0];
      const within = scored.filter(r => round4(r.score) >= bestScore - near - 1e-9).sort(cheaper);
      for (const r of within) if (!isNum(r.cost)) unpriced.push(`${r.engine} (${LEG_TITLES[l]})`);
      within.slice(0, top).forEach((r, i) => {
        const lim = D.limits?.[r.engine], [parAt, by, rate] = parallelSeconds(lim, vol, r.rec, r.tin, r.tout);
        rows.push({leg: l, rank: i + 1, engine: r.engine, score: r.score, cost: r.cost, rec: r.rec, best, count: within.length,
          costAt: isNum(r.cost) ? r.cost * vol : null, timeAt: isNum(r.rec) ? r.rec * vol : null,
          parAt, limit: limitText(lim, by, rate, r.rec), basis: lim ? `${lim.basis}${lim.source ? ` · ${lim.source}` : ""}` : "no default_limits in engines.py"});
      });
    }
    const empty = D.legs.filter(l => !Ls[l].data.some(r => isNum(r.score)));
    const where = `${rf.tier === "all" ? "all tiers" : rf.tier + " tier"} · ${rf.type === "all" ? "all document types" : rf.type}`;
    const maxCost = Math.max(0, ...rows.map(r => r.costAt || 0)), maxTime = Math.max(0, ...rows.map(r => r.timeAt || 0)),
      maxPar = Math.max(0, ...rows.map(r => r.parAt || 0));
    const bar = (v, max, text, cls) => `<div class="scale-cell"><span class="track"><span class="fill ${cls}" style="width:${max > 0 && v ? Math.max(v / max * 100, 0.6) : 0}%"></span></span><span class="val">${text}</span></div>`;
    // Each row against the leg's best scorer: the score given up and the money saved (or spent) at this volume.
    const vsBest = r => {
      const drop = round4(r.best.score) - round4(r.score), diff = isNum(r.best.cost) && isNum(r.cost) ? (r.best.cost - r.cost) * vol : null;
      return {score: r.engine === r.best.engine ? "best score" : `−${f4(drop)} vs best`,
        cost: r.engine === r.best.engine || diff == null ? "" : diff >= 0 ? `saves ${money(diff)} vs best` : `${money(-diff)} more than best`};
    };
    const body = rows.map(r => { const v = vsBest(r); return `<tr data-e="${esc(r.engine)}" class="${r.rank === 1 ? "lead-row" : ""}">
        <td data-v="${r.leg}">${r.rank === 1 ? `${esc(LEG_TITLES[r.leg])}<div class="muted limit">best ${f4(r.best.score)} · ${esc(r.best.engine)}<br>${r.count} engine${r.count === 1 ? "" : "s"} within ${esc(rf.near)}</div>` : ""}</td><td data-v="${esc(r.engine)}" class="eng-cell">${esc(r.engine)}</td>
        <td data-v="${r.score}" class="${tint(r.score)}">${f4(r.score)}<div class="muted limit">${v.score}</div></td>
        <td data-v="${r.cost ?? ""}">${XM.cost.fmt(r.cost)}</td><td data-v="${r.rec ?? ""}">${XM.rec.fmt(r.rec)}</td>
        <td data-v="${r.costAt ?? ""}">${bar(r.costAt, maxCost, r.cost === 0 ? "unpriced (local)" : money(r.costAt), "cost")}${v.cost ? `<div class="muted limit">${v.cost}</div>` : ""}</td>
        <td data-v="${r.timeAt ?? ""}">${bar(r.timeAt, maxTime, duration(r.timeAt), "time")}</td>
        <td data-v="${r.parAt ?? ""}" title="${esc(r.basis)}">${bar(r.parAt, maxPar, duration(r.parAt), "time")}<div class="muted limit">${esc(r.limit)}</div></td></tr>`; }).join("");
    const heads = [["leg", "Leg"], ["engine", "Engine"], ["score", "Score"], ["cost", XM.cost.label], ["rec", XM.rec.label],
      ["costAt", `Cost for ${volText} pages`], ["timeAt", "Time, one at a time"], ["parAt", "Time in parallel at default limits"]];
    return `<div class="panel"><h3>Score against cost per leg at scale — engines within ${esc(rf.near)} of the best score, cheapest first · ${esc(volText)} pages · ${esc(where)}</h3>
        <p class="muted">Per leg: the best score in this selection (named under the leg), then every engine scoring within the chosen margin of it (4 d.p.), cheapest first by ${esc(XM.cost.label)}, then ${esc(XM.rec.label)}. "Within" sets the margin; "show" how many of those engines to list. Each row shows the score it gives up and the money it saves against the best scorer. Cost = ${esc(XM.cost.label)} × pages; running in parallel leaves it the same. One at a time = ${esc(XM.rec.label)} × pages: one call after another, or one warm GPU container at the measured throughput. In parallel = pages ÷ the fastest rate the provider's default limits allow a standard paid account, with no increase requested: AWS Textract quotas in US East, Anthropic's Start tier (whose $500 monthly spend cap also limits volume), LlamaCloud's Starter plan and Modal's Starter plan of 10 GPUs. Hover a time for the limit's source. Gemini and Mistral publish no default limits, so theirs are estimates and marked as such: Gemini at third-party Tier 1 values, Mistral at an assumed 1 request per second. Local engines scale with your CPU cores. ${esc(XM.cost.label)} and ${esc(XM.rec.label)} are each leg's own means over its scored pages, as in the leg tables. Legs that share a call are paid once, so do not add rows. Modal figures cover GPU processing only, without container startup or idle time; local engines are unpriced and count as $0.</p>
        <div class="chart-controls">${selectHTML("tier", [["all", "all tiers"], ...TIERS.map(t => [t, t])], "tier")}${selectHTML("type", [["all", "all document types"], ...docTypes().map(t => [t.type, `${t.type} · ${t.tier}`])], "type")}${selectHTML("near", NEAR, "within")}${selectHTML("top", TOP, "show")}${selectHTML("vol", VOLUMES, "pages")}</div>
        <div class="twrap"><table class="v r scale" data-table="scale"><thead><tr>${heads.map(([k, t]) => `<th data-key="${k}">${esc(t)}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table></div>
        ${unpriced.length ? `<p class="muted">No ${esc(XM.cost.label)} recorded, ranked last: ${unpriced.map(esc).join(", ")}.</p>` : ""}
        ${empty.length ? `<p class="muted">No scored pages in this selection for ${empty.map(l => esc(LEG_TITLES[l])).join(", ")}.</p>` : ""}</div>`;
  }
  function overviewView(rows) {
    const Ls = cached("legs", () => Object.fromEntries(D.legs.map(l => [l, legData(l, rows)])));
    const engines = [...new Set(rows.filter(r => valid(r) || r.error).map(r => r.engine_name))].sort();
    const ops = engineOperational(selectedCalls(), D.results.ledger);
    const data = engines.map(e => { const d = {engine: e, pages: new Set(rows.filter(r => r.engine_name === e && valid(r)).map(r => r.stem)).size};
      for (const l of D.legs) d["leg:" + l] = Ls[l].data.find(x => x.engine === e)?.score ?? null;
      Object.assign(d, {cost: ops[e]?.cost_per_page ?? null, rec: ops[e]?.latency_s ?? null, mixed: ops[e]?.mixed ?? []});
      return d; });
    const cols = [{key: "rank", label: "#"}, {key: "engine", label: "Engine"},
      ...D.legs.map(l => ({key: "leg:" + l, label: LEG_SHORT[l], fmt: f4, higher: true, tint: true, title: `${LEG_TITLES[l]} full score`})),
      {key: "pages", label: "Pages", fmt: fi, title: "pages with any scored leg"},
      {key: "cost", label: XM.cost.label, fmt: XM.cost.fmt, higher: false, title: "primary pass, " + XM.cost.title}, {key: "rec", label: XM.rec.label, fmt: XM.rec.fmt, higher: false, title: "primary pass, " + XM.rec.title, note: mixedMark}];
    const opsData = engines.map(e => ({engine: e, cost: ops[e]?.cost_per_page ?? null, rec: ops[e]?.latency_s ?? null,
      cost_all: ops[e]?.cost_all_passes ?? null, rec_all: ops[e]?.latency_all_passes ?? null, passes: ops[e]?.passes ?? null, documents: ops[e]?.documents ?? null, primary: ops[e]?.primary ?? null, mixed: ops[e]?.mixed ?? []})).filter(d => d.cost != null || d.rec != null);
    const opsCols = [{key: "rank", label: "#"}, {key: "engine", label: "Engine"}, {key: "primary", label: "Primary pass"},
      {key: "cost", label: XM.cost.label, fmt: XM.cost.fmt, higher: false, title: "primary pass"}, {key: "rec", label: XM.rec.label, fmt: XM.rec.fmt, higher: false, title: "primary pass", note: mixedMark},
      {key: "cost_all", label: `all passes ${XM.cost.label}`, fmt: XM.cost.fmt, higher: false, title: "every pass the benchmark asked, summed per page"}, {key: "rec_all", label: "all passes s/page", fmt: XM.rec.fmt, higher: false},
      {key: "passes", label: "Passes", fmt: fi}, {key: "documents", label: "Pages", fmt: fi, title: "pages with a successful call"}];
    const shownEngines = new Set();
    const multi = D.legs.map(l => { const sp = scatterPanel(Ls[l], {small: true, id: "scatter-" + l}); charts.push({sel: `#scatter-${l} svg`, pts: sp.pts, leg: l, kind: "scatter"});
      sp.engines.forEach(e => shownEngines.add(e));
      return `<div><h4>${LEG_TITLES[l]} · ${Ls[l].pages} pages</h4>${sp.html}</div>`; }).join("");
    return `<div class="panel"><h3>At a glance — full scores per leg; bold marks the column's best</h3>
        <p class="muted">Each leg is scored on its own pages; no cross-leg blended score. Cost and time are the engine's primary pass; † marks a time taken from a batch that also ran other passes (hover for which). Click a column to sort.</p>
        ${table("glance", cols, data)}</div>
      ${scalePanel(Ls)}
      <div class="panel"><h3>Cost and time per engine — recorded calls in this selection</h3>
        <p class="muted">Primary pass = one call per page ("default", else the text pass). All-passes figures add every pass the benchmark asked of the engine, so they depend on how many legs it was prompted for.</p>
        ${table("ops", opsCols, opsData)}</div>
      <div class="panel"><h3>Score against ${esc((XM[rf.x] || XM.cost).label)} per leg · one point per engine</h3>
        <div class="chart-controls">${xControls()}</div>${scatterLegend(engines.filter(e => shownEngines.has(e)), "scatter-legend")}<div class="multi">${multi}</div></div>
      ${notesPanel(D.legs)}`;
  }

  function render() {
    hideTip();
    charts.length = 0;
    rememberPanels();
    filterBar();
    el("engine-head").innerHTML = headHTML();
    const rows = selection();
    el("content").innerHTML = rf.leg === "all" ? overviewView(rows) : legView(rows);
    restorePanels();
    const c = el("content");
    c.querySelectorAll("th.sortable").forEach(th => th.onclick = () => {
      const key = th.dataset.key;
      update(rf.sort === key ? {dir: rf.dir === "asc" ? "desc" : "asc"} : {sort: key, dir: LOWER.has(key) ? "asc" : "desc"});
    });
    c.querySelectorAll("[data-o]").forEach(n => n.onchange = () => update({[n.dataset.o]: n.type === "checkbox" ? (n.checked ? "1" : "0") : n.value}));
    c.querySelectorAll("table[data-table^='leg:'] tr[data-e], table[data-table='glance'] tr[data-e], table[data-table='ops'] tr[data-e], table[data-table='scale'] tr[data-e], table[data-table^='families:'] tr[data-e], table[data-table='judge'] tr[data-e]").forEach(tr => {
      tr.classList.add("link"); tr.ondblclick = () => go({view: "page", engine: tr.dataset.e, leg: rf.leg === "all" ? state.leg : rf.leg}); tr.title = "double-click to open this engine in the page view";
    });
    c.querySelectorAll("table[data-table^='pages:'] tr[data-e]").forEach(tr => tr.onclick = () => go({view: "page", engine: tr.dataset.e, doc: tr.dataset.d, leg: rf.leg}));
    const scatterSvgs = [];
    for (const ch of charts) {
      const svg = c.querySelector(ch.sel);
      if (!svg) continue;
      if (ch.kind === "scatter") {
        scatterSvgs.push(svg);
        wire(svg, ds => { const p = ch.pts[+ds.i]; if (!p) return null; const xm = XM[rf.x] || XM.cost;
          return [p.engine, `${LEG_TITLES[ch.leg]} score ${f4(p.y)} · ${p.n} page${p.n === 1 ? "" : "s"}`, `${xm.label} ${xm.fmt(p.x)}${p.xn != null && p.xn !== p.n ? ` (${p.xn} calls)` : ""}${p.zero ? " · unpriced local engine, drawn at the axis start" : ""}`, "click: open in the page view"]; },
        ds => { const p = ch.pts[+ds.i]; if (p) go({view: "page", engine: p.engine, leg: ch.leg}); });
      } else {
        wire(svg, ds => { const r = ch.rowsE[+ds.r]; if (!r) return null;
          if (ds.e != null) { const e = r.errors[+ds.e]; return [`${r.engine} · ${e.stem}`, `${e.doc_type} · ${e.tier}`, "error: " + String(e.error).slice(0, 160), "click: open in the page view"]; }
          const p = r.points[+ds.p]; return p ? [`${r.engine} · ${p.stem}`, `${p.doc_type} · ${p.tier}`, `${ch.xm.label} ${ch.xm.fmt(p.v)}${ch.xm.key !== "score" ? ` · score ${f4(p.score)}` : ""}`, "click: open in the page view"] : null; },
        ds => { const r = ch.rowsE[+ds.r]; const x = ds.e != null ? r?.errors[+ds.e] : r?.points[+ds.p]; if (x) go({view: "page", engine: r.engine, doc: x.stem, leg: ch.leg}); });
        const legend = c.querySelector("#strip-legend");
        wireHighlight([svg], legend, legend?.dataset.key || "type").follow("stem");
      }
    }
    if (scatterSvgs.length) {
      // Engines light up from the legend and from their rows in the tables above the chart.
      const rows = [...c.querySelectorAll("table[data-table^='leg:'] tr[data-e], table[data-table='glance'] tr[data-e], table[data-table='ops'] tr[data-e]")];
      wireHighlight(scatterSvgs, c.querySelector("#scatter-legend"), "engine", rows).follow("engine");
    }
  }

  // ---------- state ----------
  function set(patch) { for (const k of Object.keys(DEFAULTS)) if (k in patch && patch[k] != null) rf[k] = String(patch[k]); }
  function update(patch) { set(patch); return go({view: "results"}); }
  const toHash = () => ["results", ...Object.keys(DEFAULTS).filter(k => rf[k] !== DEFAULTS[k]).map(k => `${k}=${rf[k]}`)].join("/");
  function fromHash(parts) {
    Object.assign(rf, DEFAULTS);
    for (const part of parts) { const [k, ...v] = part.split("="); if (k in DEFAULTS && v.length) rf[k] = v.join("="); }
    if (rf.leg !== "all" && !D.legs.includes(rf.leg)) rf.leg = "all";
    if (rf.tier !== "all" && !TIERS.includes(rf.tier)) rf.tier = "all";
  }
  // Machine-readable copy of what is on screen, for checks/verify_results.py.
  function dump() {
    const tables = {};
    for (const t of document.querySelectorAll("table[data-table]")) {
      const heads = [...t.querySelectorAll("thead th")].map(th => th.dataset.key || th.textContent.trim());
      tables[t.dataset.table] = [...t.querySelectorAll("tbody tr")].map(tr => Object.fromEntries([...tr.children].map((td, i) => [heads[i], td.dataset.v ?? td.textContent.trim()])));
    }
    const dotsOut = [...el("engine-list").querySelectorAll(".eng[data-e]")].map(n => [n.dataset.e, [...n.querySelectorAll(".dot")].map(d => d.dataset.v ?? "")]);
    return {filter: {...rf}, tables, dots: dotsOut, charts: charts.map(ch => ({sel: ch.sel, points: ch.kind === "scatter" ? ch.pts.length : ch.rowsE.reduce((n, r) => n + r.points.length + r.errors.length, 0)}))};
  }

  return {DEFAULTS, rf, init, render, dots, set, update, toHash, fromHash, docTypes, dump, f4};
})();
