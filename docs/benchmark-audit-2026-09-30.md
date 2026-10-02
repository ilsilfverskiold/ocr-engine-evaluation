# Audit of the reference run 20260810T130609Z

> Historical planning/audit record. Current status and final scores are in
> [handoff-2026-10-01.md](handoff-2026-10-01.md), run `20261002T095730Z`.
> Prototype paths and pending-work counts below describe the earlier stage;
> scratch artifacts and superseded rescoring runs have since been removed.

Date: 2026-09-30. Offline and read-only against the saved run: the code,
the saved `.call.json` outputs, the judge caches and the result JSONL were
read; nothing was re-run against a model, and no saved score, cache or
result was changed. Every number below was reproduced in this session by
re-running the analysis scripts (kept in the session scratchpad; they import
the repository's pure scoring functions and read saved files). Where a figure
rests on a heuristic classification rather than a direct check, it says so.

Read [scoring-design.md](scoring-design.md) first. Section 0 summarises the
part of it that the findings depend on.

## What we said, and whether the run bears it out

Every factual claim in the README and in scoring-design.md, checked against
`results/runs/20260810T130609Z_results.jsonl`, the summary CSV the report
wrote, and the ground-truth loader. "Rows" means applicable, scored result
rows for that engine and leg.

### Claims that are true

| Claim (where) | Checked |
| --- | --- |
| 14 engines × 93 documents, all outputs and judge caches shipped, `--report` re-renders for free (README) | 2,332 inference records, 2,242 passes present; every `.call.json` on disk |
| Per-tier ground-truth table: easy 12/12/6/11/12/6, medium 35/35/35/23/35/0, hard 46/35/16/8/18/11 (README) | Recomputed from the loader: identical |
| text: 82 docs, 14 engines · markdown: 57, 11 · tables: 42, 12 · schema: 65, 8 · geometry: 17 docs (README) | 82 text rows for all 14; 57 markdown rows for exactly 11; 12 engines with table rows (42 after the chart exclusion); 65 schema rows for exactly 8; 17 geometry docs |
| 11 silver-text documents; 8 model-transcribed; 17 schemas extended by hand; 5 fabricated table references removed (README) | gt_limitations.json and gt_provenance.json: 11, 8, 17, 5; no schema-derived table reference remains |
| Judge models: markdown 2.5-pro checklist / 3.6-flash verdicts, tables 3.6-flash, KV/schema 2.5-pro, document judge 3.6-flash (README) | Module defaults match; the markdown verdict files record 2.5-pro / 3.6-flash |
| Markdown/table/document-judge checklists are built from the image once, cached, and shared by every engine; KV/schema checklists are built from the GT JSON with no model (README, design) | Code and cache files as described; KV/schema checklists equal today's GT, 0 differences |
| Scores are never pooled across document types (README) | `engine_leg_means` = mean of per-doc-type means |
| Text and geometry are deterministic (README) | Recomputed offline from saved outputs: text 1,148/1,148 rows identical, geometry 165/165 |

### Claims that are false or no longer true

| Claim (where) | What the run contains | Effect |
| --- | --- | --- |
| key_value: **6 engines** (README six-legs table); `glm_ocr` and `deepseek_ocr` answer key_value (README engines table) | **4** engines have KV rows: textract_structured, gemini, claude, qwen. glm and deepseek have none (the registry dropped them after the README was written) | No score affected. **Re-enabled in the registry 2026-09-30** (interface rule); the README becomes true once they are run — 65 KV passes each |
| geometry: **6 engines** (README); layout parsers and deepseek "do not compete" (design, Leg 6); mineru/paddle/marker/deepseek listed without geometry (README engines table) | **10** engines have geometry rows, and the published `engines.csv` prints geometry for mineru 0.105, marker 0.146, **paddle 0.000, deepseek 0.000** | The report publishes four geometry numbers the README says don't exist; two of them are adapter artefacts (§5.1) |
| The 11 silver-text documents "get the judged estimate … labelled as judged in the report and kept separate" (README); document judge "Engines: all 14" (design) | Judge rows exist for 10 engines (deepseek, glm, qwen, marker have none), and the report **never printed** the judged section for anyone — the rows were dropped before the printer ran | The estimate was computed for 10/14 and shown for 0/14. Both fixed in the harness 2026-09-30; the four engines need re-running on the 11 pages |
| The local substring check "resolves the ~85–90% of field items that match exactly" (README); "~85–90%" (design) | 71.7% of KV fields, 73.0% of schema fields; it is a substring test after stripping punctuation, not an exact match | Overstated; and the mechanism is looser than described |
| "In practice the false-positive rate is negligible because GT values are specific" (design, Known issues) | Four confirmed wrong accepts in §1 of the verdict below; 16/6/8/3 strictly suspicious accepts per KV engine, 27–121 per schema engine; enough to reorder the schema top three | The stated assumption does not hold for this corpus (zeros, booleans, repeated amounts) |
| "The model used is stored in every JSONL record (`judge_model` field)" (README) | markdown rows: yes (627/627); tables: 525/530; **key_value, schema and document-judge rows: no model field at all** | Traceability claim false for three of the four judged paths |
| tables GT "45 documents" (design); key_value "~60 of 65" (design) | 42 (charts excluded) and 56 | Stale numbers in the design doc; README has the right ones |

### Things that are wrong that we never claimed otherwise

- Verdict caches can be reused for a changed output, and were, for 155
  KV/schema pairs (§4). The README says re-scoring "costs nothing" because
  of the cache; it does not claim the cache is keyed on the output.
- KV passes were run on 9 chart/financial-table documents the scorer then
  refused (30 calls, $0.14) — the README footnote does say those documents
  are "reported not applicable at scoring time".

## Verdict in short

**Most of it was done right.** The design is what the design document says
it is, and the run is complete: 2,242 passes planned, all present except 5
Marker geometry failures; text and geometry scores reproduce exactly from the
saved outputs offline; the KV/schema checklists match the ground truth
field for field; the image-built checklists are shared across engines so no
engine is graded against a different reference; every call's prompt, raw
reply and cost are on disk. The judged legs' method (Gemini catalogues the
page once, then grades each engine against that catalogue) is sound and is
implemented as documented.

**Four things are wrong.** Each one below says what the mechanism is, shows
a real example from the saved run, says why it matters, and gives the check.

### 1. The KV and schema legs score value *presence*, not extraction

*What happens.* For key_value and schema, the list of expected values comes
from the ground-truth JSON. A field is scored correct if its value is
**present in the engine's output** — not if it sits under the right label,
not if it is in JSON, not if it is attributed to anything. Two mechanisms do
this. First, before any of them is sent to Gemini, our own code runs a quick
check on each one: lowercase both sides, delete everything that is not a
letter or digit, and if the expected value is found *anywhere* inside the
engine's output, record `correct` and never ask Gemini
(`evaluation/schema_judge.py`, `_verify_locally`). Second, the values that
fail that check go to Gemini with a prompt that says "match by VALUE, not by
field name … what matters is whether the correct value is present and
attributable to the right concept" — attribution is asked for, but the
example in this section shows the local check never gets that far.

*The decisive test: score a plain text dump.* Tesseract and Docling do not
compete in these legs — they return the page's text, nothing extracted. Fed
through the same local check against the same checklists:

| Plain OCR text, no extraction | KV fields accepted | KV leg-style score | schema fields accepted | schema leg-style score |
| --- | ---: | ---: | ---: | ---: |
| tesseract default output | 502 / 877 (57%) | **0.683** | 1,834 / 2,687 (68%) | **0.686** |
| docling default output | 629 / 877 (72%) | **0.760** | 2,154 / 2,687 (80%) | **0.789** |
| published: engines that actually extracted | | textract 0.959, claude 0.876, qwen 0.852, gemini 0.811 | | gemini 0.969, claude 0.967, textract 0.962, mistral 0.954, llama 0.893, qwen 0.866 |

(Leg-style score = mean of document-type means, blank fields counted as
`blank_ok` because an empty dump never invents a value.) A page dump with
no extraction lands within 0.05–0.10 of engines that returned filled JSON,
before Gemini even looks at the remainder. What separates the top engines
from a text dump in this leg is therefore mostly text coverage, not
extraction — and the leg's stated questions ("did it pull out label → value
pairs?", "given your JSON schema, did it fill it correctly?") are not what
the number measures.

This is a design flaw, not only the shortcut. The design document chose
"match by value, not by field name" deliberately, to stop punishing an
engine that says `Company Name` instead of `company`. That fairness goal is
right; the implementation dropped the other half — that the value has to be
*the engine's answer for that field*. The shortcut then made it worse by
never even asking Gemini for the fields it could find as substrings.

*Example.* Lease `lease_001`, field `rent.securityDeposit`, expected **1180**.
Claude's key_value answer, line 21 of
`results/ocr_outputs/20260810T130609Z/medium__lease__lease_001/claude_sonnet_4_6/key_value.txt`:

```
"A security deposit of dollars ($1180)": "180",
```

Claude read the deposit as **180** — wrong. But the label it printed contains
the digits `1180`, so after stripping punctuation the quick check finds
`1180` inside the output and records `correct`. Gemini never saw this field.
The cached verdict is in
`results/schema_judge_cache/verdicts/lease_001__claude_sonnet_4_6__kv.json`:

```
{"path": "rent.securityDeposit", "expected": "1180", "verdict": "correct"}
```

Three more, same mechanism, all confirmed against the saved files:

| Document, field | Expected | What the engine wrote | Why it passed |
| --- | --- | --- | --- |
| payslip_002 (glm), `consultants[4].sales` | 1140 | `"sales": 21.0` | another field is `"sfmCode": "114025"`, which contains `1140` |
| tax_000 (qwen), `ageBlindness.spouseBlind` | True | `"spouseBlind": false` | some other field on the form is `true` |
| sroie_002 (mistral), `gst_summary[0].amount` | 6.89 | `"amount": "5.89"` | `"total_sales_excluding_gst": "6.89"` elsewhere |

*Why it matters.* The quick check decided about 72% of all KV and schema
fields in the run (§2 has the per-engine counts). Most of those are fine —
long identifiers, names and amounts rarely occur by accident. The risk is
concentrated in short numbers (`0`, `3`, `7`), booleans, and amounts that
appear in more than one field, which this corpus has a lot of (tax forms
full of zeros, medical forms full of true/false). Counting only the
clearly-wrong kinds of accept, the KV ranking does not change, but the
schema leg's top three — gemini 0.969, claude 0.967, textract 0.962 — sit
closer together than the error and can come out textract > claude > gemini
(§2). Nothing has been rescored; these are bounds.

*Check it.* Open the two files quoted above.

### 2. 155 sets of judge verdicts belong to older engine outputs

*What happens.* When Gemini judges an engine on a document, the verdicts are
saved to a file named `<document>__<engine>.json` so a re-run does not pay
again. The file name is the only key: nothing in it records *which output*
was judged. So if an engine was re-run and produced a different output, the
scorer finds the old file, uses it, and reports cost $0.

*Example.* Invoice `invoice_005`, deepseek, schema leg, field
`items[1].item_desc`, expected "Sony PlayStation 4 Slim 1TB (Upgraded to
SSD) Black Game Console". The cached verdict file
`results/schema_judge_cache/verdicts/invoice_005__deepseek_ocr.json` says
**`missing`**. The saved output
`results/ocr_outputs/20260810T130609Z/easy__invoice__invoice_005/deepseek_ocr/schema.txt`
contains it verbatim:

```
{"item_desc": "Sony PlayStation 4 Slim 1TB (Upgraded to SSD) Black Game Console", "item_qty": "2.00"
```

The verdict file was written about 90 minutes before that output. It is a
verdict on an earlier deepseek output that is not in the saved run.

*How many.* Comparing every verdict file's timestamp with the output it
scores: schema — deepseek 40 of 65, glm 55 of 65, qwen 4 of 65; key_value —
qwen 56 of 56. Total 155. In 10 of those, 52 fields are saved as
missing/wrong although the current output plainly contains the value (in
the non-stale files: 0 such cases). Markdown and table verdict caches: 0
stale. Lower bound on what it cost the engines: deepseek schema 0.338 → at
least 0.351. What Gemini would say about the rest of those fields is
unknown without re-judging.

*Check it.* `ls -l --time-style=full-iso` (or `stat`) on the two invoice_005
files above; the verdict is older.

### 3. DeepSeek and Paddle scored 0.000 on geometry because of our adapters

*What happens.* The geometry leg matches predicted boxes to reference boxes
**by their text first** (a box is "located" when its text matches a
reference word and the boxes agree on position). So a box with the wrong
text can never match.

*DeepSeek.* Its grounding output looks like
`<|ref|>title<|/ref|><|det|>[[78,190,155,206]]<|/det|>Invoice no: 97159829`.
The word between the `ref` tags is the region *type*; the recognised text
comes after the `det` tag. The v1 app stored the type as the box's text.
Open
`results/ocr_outputs/20260810T130609Z/easy__invoice__invoice_001/deepseek_ocr/markdown+tables+geometry.call.json`
→ `structured_output.words`: ten boxes, texts `text` ×5, `title` ×3,
`table` ×2. No reference word is called "title", so 0 of 17 pages match.
The real text is still in `structured_output.raw_model_output`, so this can
be re-derived without re-running the engine.

*Paddle.* The app looked for line boxes under `overall_ocr_res`, which the
PaddleX version used does not return. 0 of the 93 saved Paddle outputs have
`text_regions`. The scorer then tried to parse boxes out of Paddle's
markdown, failed, and recorded — for all 17 pages —
`reason: "engine output was not parseable word boxes"`, score 0.0
(`results/runs/20260810T130609Z_results.jsonl`, `engine_name`
`paddleocr_vl`, `leg` `geometry`).

*Was geometry actually computed for them?* Different answer for each, and
the saved metrics show which (`geometry_parse_ok`, `geometry_predicted_words`,
`geometry_matched_words` on every geometry row):

| Engine, sroie_000 | parse_ok | reference words | predicted boxes | text-matched | located | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| tesseract (for comparison) | true | 46 | 26 | 33 | 30 | 0.652 |
| deepseek_ocr | true | 46 | 22 | 0 | 0 | 0.0 |
| paddleocr_vl | **false** | 46 | **0** | 0 | 0 | 0.0 |

- **Paddle: no, nothing was computed.** On all 17 pages the scorer received
  no boxes (`predicted_words` = 0, `parse_ok` = false), and *assigned* 0.0
  with the reason "engine output was not parseable word boxes". It is a
  placeholder zero, and it should have been recorded as not-applicable or
  as an adapter error rather than as a score — that is finding §5.1's
  "parse failures are scored applicable 0.0".
- **DeepSeek: yes, the matcher ran, on the wrong input.** It received
  between 1 and 31 boxes per page (22 on sroie_000), compared their texts
  (`title`, `text`, `table`) against the reference words, and matched 0 on
  16 pages and 1 on one page. The zero is a genuine computation over boxes
  whose text field held the region label instead of the words.

*Why it matters.* Both 0.000s are published as if the engines had tried and
failed. Paddle's pipeline does produce boxes (per layout block); DeepSeek's
grounding output does carry text. Neither number says anything about the
engine.

*Can they be scored properly from what was saved?* Yes, without re-running
either engine — DeepSeek's text is in `raw_model_output` after each tag,
Paddle's `layout_blocks` carry content and a 0–1 box — and it was done
offline to see what the leg would say. Both engines return one box per
layout **block** (a whole line, paragraph or table), and the leg's references
are **words** (FUNSD, SynFinTabs) or **lines** (SROIE). The scorer matches a
predicted unit to a reference unit by text, so a 20-word block can never
equal a single reference word:

| Engine (block boxes, correct text) | 6 line-level pages (SROIE) | 11 word-level pages (FUNSD, fintab) |
| --- | ---: | ---: |
| deepseek_ocr (re-derived) | 0.214 | 0.008 |
| paddleocr_vl (re-derived) | 0.125 | 0.014 |
| mineru (as saved) | 0.286 | 0.016 |
| marker (as saved) | 0.421 | 0.011 |

*Are the boxes themselves right?* Yes for three of the four. Checked
independently of granularity: for every reference word whose text appears
inside some returned block, is the word's centre inside that block's box?

| Engine | Boxes | Outside [0,1] | Reference words whose centre lies inside the block that contains their text |
| --- | ---: | ---: | ---: |
| deepseek_ocr (re-derived) | 212 | 0 | **97%** (per page 0.83–1.00) |
| paddleocr_vl (layout_blocks) | 186 | 0 | **98%** (0.88–1.00) |
| mineru | 274 | 0 | **99%** (0.93–1.00) |
| marker | 134 | 0 | 53%, and text found on only 9 of 17 pages |

DeepSeek, Paddle and MinerU returned correctly placed boxes; their low leg
scores are entirely the block-vs-word mismatch below. Marker's geometry pass
is genuinely broken (its OCRConverter tree walker returns near-empty output
— already noted in `engines.py`), so its 0.146 is a defect of the adapter,
not a measurement either.

So the corrected numbers are not 0.000, but they are still near zero on 11
of 17 pages for a structural reason: block-level output is not what this leg
asks for ("for every word, its box"). The four block-level engines were
declared into the leg on 2026-08-10, the day before the commit, on the
grounds that their outputs "carry a bbox"; the README and design were
written earlier and still say six engines. The honest options are to report
these four as not applicable ("block granularity, not word boxes") — which
is what the README already implies — or to add a separate block-level
measurement. Publishing them as low scores in a word-box leg says the
engines are bad at something they were never asked for.

### 4. The 11 hard pages were judged for 10 engines, not 14

*What happens.* The newspaper and report pages have no usable text ground
truth, so instead of a text score they get the document judge: Gemini
catalogues the page, then grades each engine's default output against the
catalogue. The design says all 14 engines get this.

*What the run has.* `grep -c '"leg": "judge"' results/runs/20260810T130609Z_results.jsonl`
→ 110 = 10 engines × 11 pages. deepseek_ocr, glm_ocr, qwen3_vl and marker
have no judge rows and no output at all on those pages.

*Why.* Those four serve the text leg through a *prompted* pass (deepseek and
glm send `Free OCR.`/`Text Recognition:` on a pass named `default`; qwen's
pass is named `text`, marker's `text+markdown`). Two gates in
`run_benchmark.py` combined: a prompted pass was skipped on any document
with no ground truth for its legs (these 11 have none for text), and the
judge only fired on a pass named exactly `default`. So for those four
engines the 11 pages were never run.

*Why it matters.* The judged column for those pages compares 10 engines,
not 14; nothing is wrong in the numbers that exist. Fixed in the harness on
2026-09-30 (the judge follows whichever pass serves text; collect-only runs
do not skip that pass), but the four engines need re-running on those 11
pages to fill the column.

Everything else in §5 is display, bookkeeping or a documented limit, listed
for completeness, not as a claim that the results are wrong.

Classification: **(a)** confirmed defect that affects saved results, with
measured magnitude; **(b)** confirmed defect that does not affect saved
results; **(c)** documented design limit; **(d)** checked and not a problem.

Provenance caveat: the repository's git history has one commit for the
code and the run together (2026-08-11), so code changes cannot be dated
against the run from git. Where "changed after the run" is claimed below, the
evidence is the saved scores themselves.

## 0. How the judged legs work (from the code, not from memory)

Four legs and the document judge use Gemini, in two patterns.

**Pattern A — Gemini reads the page itself.** The **markdown** leg (57
documents), the **tables** leg (42) and the **document judge** for the 11
newspaper/report pages whose text ground truth is unusable.

1. *Checklist, once per document, cached, shared by every engine.* Gemini is
   shown the page **image** and catalogues it: structural elements for
   markdown (`markdown_judge.py`, 40-item cap), table cells with row/col for
   tables (`table_judge.py`, 60-item cap), checkable values for the document
   judge (`judge.py`, 40-item cap). Cached as
   `results/{markdown_judge_cache,table_judge_cache,judge_checklists}/<stem>.json`.
2. *Verdicts, once per document × engine, cached.* Gemini sees the checklist
   and the engine's output and returns one verdict per item. Cached as
   `.../verdicts/<stem>__<engine>.json`. The markdown and table judges send
   **every** item to Gemini. The document judge first settles items whose
   value is a normalised substring of the output (`judge.py:_verify_locally`)
   and sends the rest. (The table judge's docstring says it does the same;
   its code does not — a docstring error, not a scoring defect.)
3. Score = correct ÷ items (markdown: `wrong_type`/`partial` count ½).

**The reference in pattern A is Gemini's reading of the page.** The human
markdown/table sidecars are used only for the `--skip-judge` fallback
(block F1, TEDS) and for eligibility. This is deliberate: the design document
records that the hand-written markdown references are over-segmented and
that comparing HTML dialects punished correct tables.

**Pattern B — the checklist comes from the ground-truth JSON.** The
**key_value** leg (56 scoreable documents) and the **schema** leg (65).

0. *Where that JSON came from (corpus building, before any run).* For 48 of
   the 65 documents it is the source dataset's own annotation. For **17
   documents** (the 6 SROIE receipts, 5 shipping documents, chart_003,
   scanned_000/005, funsd_000, lease_001/004) the dataset's schema was too
   thin to separate engines, so **an LLM (Claude Opus 5, the
   `schema-annotator` agent) read the page image and wrote the extra
   fields**, each value corroborated against the 14 engines' outputs, with
   two corrected on human review. Each file's `field_provenance` records
   which fields are `dataset_human` and which `model_verified`. Share of the
   reference that is model-extracted: key_value 167 of 922 scalar fields
   (18%); schema 1,301 of 3,831 leaf fields (34%, most of them shipping
   line-item cells). On sroie_000, 18 of its 22 KV fields are model-written.
   Two consequences: those expected values are the model's reading (with
   consensus as the check, so values no engine read were dropped or
   human-confirmed — e.g. sroie_002's PAID stamp), and they lean toward what
   the 14 engines under test could read.
1. *Checklist, once per document, cached.* Built by **code**: walk the
   document's `_structured_gt.json` (+ its `json_schema`) and make one item
   per leaf field — path, expected value, blank or not (80-item cap; KV
   keeps scalar fields only). Cached as
   `results/schema_judge_cache/<stem>.json` / `<stem>_kv.json`.
2. *Verdicts, once per document × engine, cached.* First **our code** checks
   each item: lowercase, strip everything but `[a-z0-9]`, and if the expected
   value occurs anywhere in the engine's output as a substring, record
   `correct` without asking Gemini (`schema_judge.py:_verify_locally`). The
   items that fail go to Gemini in one call together with the engine's
   whole output (60,000-character cap), for
   `correct / wrong / missing / blank_ok / invented`. Cached as
   `results/schema_judge_cache/verdicts/<stem>__<engine>.json` (`__kv`
   suffix for key_value). The verdict file records path, expected value and
   verdict — not whether the verdict came from the local check or Gemini.
3. Score = (`correct` + `blank_ok`) ÷ items.

The design document states two things about pattern B that this audit
tests: the local check "resolves ~85–90%" of items, and "in practice the
false-positive rate is negligible because GT values are specific (amounts,
identifiers, names)".

Which findings apply where: §2, §3 and §5.6 are pattern B. §4 is both
patterns in mechanism; staleness was measured for both and found only in
pattern B. §5.2, §5.3 and §5.4 are pattern A. §5.14 is expected pattern-A
behaviour, recorded because it was observed.

## 1. The six prior findings (from engine-leg-and-cost-matrix.md)

| # | Prior finding | Verdict |
| --- | --- | --- |
| 1 | Local field verifier accepts false positives | **Confirmed and measured** (§2). Also accepts `True` against any `true`, `0` against `2020`, `Müller` against `Mller`. |
| 2 | Schema score measures value presence, not schema conformance | **Confirmed, (c)** — and wider than stated: the substring test also matches inside JSON *keys*, and 266 of the 2,687 non-blank schema values are accepted when checked against the prompt's own schema text alone (enum values such as `Widowed`, `HI`, `True`, and `0`), so an engine that echoes the schema back gets credit for them. |
| 3 | Verdict-cache identity is unsafe for reruns | **Confirmed — and it happened in the baseline** (§4). |
| 4 | DeepSeek KV exclusion conflicts with the interface rule | **Confirmed.** The registry note says it "never once returned JSON"; on the schema leg 59 of its 65 saved outputs parse as JSON. |
| 5 | Historical Mistral identity unresolved | **Confirmed.** 158/158 inference records say `mistral-ocr-latest`. New: its schema (65) and table (45) verdict files are stored under the old engine name `mistral_ocr`; a rescore under `mistral_ocr_4` misses them and re-judges. |
| 6 | Accuracy averages drop failed calls | **Confirmed.** 89 inference error rows, 89 result error rows; exactly 5 have no later success, all Marker geometry (sroie_000, sroie_002, sroie_005, fintab_001, fintab_004). New: parse failures are scored *applicable 0.0*, not as errors, so the two failure kinds are treated differently (§5.1). |

## 2. Pattern B: how much the local check decided, and what it got wrong

Origin of each verdict was reconstructed by re-running only
`_verify_locally` (a pure function) over the saved outputs and the cached
checklists. Consistency checks: the cached checklists equal today's ground
truth (0 differences); the cached verdicts equal the per-field detail stored
in `results.jsonl` (0 differences); in the 589 (document, engine) sets that
are not stale (§4), every locally accepted field carries the saved verdict
`correct` — so the reconstruction is exact there. In the 155 stale sets it
describes the current output, not what was judged.

Fields settled by the local check (KV: 56 documents, 914 fields per engine;
schema: 65 documents, 2,954 fields per engine):

| Leg | Engine | Local | Sent to Gemini | Whole call avoided |
| --- | --- | ---: | ---: | ---: |
| KV | claude_sonnet_4_6 | 728 (79.6%) | 186 | 12 of 56 docs |
| KV | textract_structured | 680 (74.4%) | 234 | 19 |
| KV | qwen3_vl | 607 (66.4%) | 307 | 11 |
| KV | gemini_flash | 606 (66.3%) | 308 | 12 |
| schema | claude_sonnet_4_6 | 2562 (86.7%) | 392 | 14 of 65 |
| schema | gemini_flash | 2556 (86.5%) | 398 | 15 |
| schema | mistral_ocr_4 | 2542 (86.1%) | 412 | 10 |
| schema | textract_structured | 2394 (81.0%) | 560 | 9 |
| schema | llamaparse | 2376 (80.4%) | 578 | 12 |
| schema | qwen3_vl | 2288 (77.5%) | 666 | 8 |
| schema | glm_ocr | 1728 (58.5%) | 1226 | 1 |
| schema | deepseek_ocr | 810 (27.4%) | 2144 | 0 |

Overall 71.7% of KV fields and 73.0% of schema fields were settled locally
(74.7% / 80.3% of non-blank fields). Local accepts are 80.9% (KV) and 89.3%
(schema) of every field finally scored correct. "85–90%" holds only for the
top schema engines.

### Wrong accepts: confirmed examples

Each of these was checked directly: the cached verdict file says `correct`,
and the saved output shows the value at the wrong place.

| Leg, engine, document, field | Expected | What the output actually has | Why it was accepted |
| --- | --- | --- | --- |
| schema, glm_ocr, payslip_002, `consultants[4].sales` | 1140 | `"sales": 21.0` | `"sfmCode": "114025"` contains `1140` |
| KV, claude_sonnet_4_6, lease_001, `rent.securityDeposit` | 1180 | `"…deposit of dollars ($1180)": "180"` — the engine's value is 180 | 1180 appears inside the label |
| schema, qwen3_vl, tax_000, `ageBlindness.spouseBlind` | True | `"spouseBlind": false` | some other field is `true` |
| schema, mistral_ocr_4, sroie_002, `gst_summary[0].amount` | 6.89 | `"amount": "5.89"` | `"total_sales_excluding_gst": "6.89"` (verdict file under `mistral_ocr`) |

Further examples from the classification pass (same pattern, checked by the
audit script's context dump, not re-opened by hand here): deepseek lease_003
`lessor.name` "Doyle Grant" present only under `lessee`; deepseek payslip_002
`idNumber` 9689414383 present only as `sfmCode`; qwen tax_004 `income.line10`
= 0 matched inside 139490; deepseek sroie_001 `items[3].qty` = 3 matched
inside a hallucinated 16-digit number; textract chart_004 0.24 matched
"2024" in "Survey 2024"; llamaparse lease_004 `expirationDay` = 7 matched
inside "2025-07-07".

### How many accepts look like that, and the bound on score change

Every local accept was classified by a script (heuristic, not hand-checked
item by item). **Strict** = a number found only inside a longer number; a
separator/decimal mismatch; the value present in the output JSON at a
different path while the correct path holds another value or null.
**Strong** adds matches only inside a JSON key or across a key/value
boundary, and text found only inside a longer token. **Weak** adds values of
≤3 characters, values shared with another GT field, booleans and
array-index misalignment.

| Leg / engine | Local accepts | Strict | Strong | Any flag |
| --- | ---: | ---: | ---: | ---: |
| KV claude / gemini / qwen / textract | 728 / 606 / 607 / 680 | 16 / 6 / 8 / 3 | 58 / 60 / 141 / 3 | 348 / 264 / 307 / 179 |
| schema claude | 2562 | 47 | 48 | 1272 |
| schema gemini | 2556 | 60 | 60 | 1233 |
| schema mistral | 2542 | 43 | 44 | 1234 |
| schema textract | 2394 | 27 | 27 | 946 |
| schema llamaparse | 2376 | 27 | 31 | 1109 |
| schema qwen | 2288 | 53 | 67 | 1223 |
| schema glm | 1728 | 121 | 165 | 795 |
| schema deepseek | 810 | 101 | 130 | 577 |

Booleans: 1,255 local accepts, 87 of which contradict the value at the
correct path. A random sample of 20 "strong" items was read by hand in the
classification pass: about 10 clearly wrong, 9 where the value exists only
as a KV key with `null` as the value (whether that counts is a policy
question), 1 false flag.

**Upper bound if every strict accept were wrong** (score = mean of
per-doc-type means, recomputed from the saved per-field verdicts):

| Leg / engine | Saved | Strict bound | Δ |
| --- | ---: | ---: | ---: |
| KV textract_structured | 0.9586 | 0.9574 | −0.0012 |
| KV claude_sonnet_4_6 | 0.8762 | 0.8696 | −0.0065 |
| KV qwen3_vl | 0.8517 | 0.8483 | −0.0034 |
| KV gemini_flash | 0.8111 | 0.8062 | −0.0049 |
| schema gemini_flash | 0.9688 | 0.9503 | −0.0185 |
| schema claude_sonnet_4_6 | 0.9670 | 0.9520 | −0.0149 |
| schema textract_structured | 0.9621 | 0.9540 | −0.0081 |
| schema mistral_ocr_4 | 0.9542 | 0.9406 | −0.0136 |
| schema llamaparse | 0.8929 | 0.8812 | −0.0117 |
| schema qwen3_vl | 0.8659 | 0.8477 | −0.0182 |
| schema glm_ocr | 0.6709 | 0.6289 | −0.0421 |
| schema deepseek_ocr | 0.3377 | 0.2961 | −0.0416 |

KV order is unchanged. **The schema top three reorder**: gemini > claude >
textract (spread 0.007) becomes textract > claude > gemini. Places 4–8 do
not move. Under "strong", KV drops by 0.13–0.21 for the three VLMs (textract
−0.001) with order unchanged; under "any flag" scores drop 0.17–0.43, which is
not realistic (it counts every boolean and shared value). Detection is
uneven: textract's schema output is not JSON, so path-based checks cannot
fire for it and its bound is understated relative to the JSON engines.

These are bounds, not corrections. Nothing has been rescored.

## 3. What the local check saved

Whole Gemini calls avoided (every field settled locally): 54 of 224 KV pairs,
69 of 520 schema pairs. The other 621 pairs still made a call with a shorter
list; of those, 477 were paid and 144 were cache hits recorded at $0 (§4).
Recorded KV+schema judge spend: $1.461. Estimated saving from the avoided
calls, at each engine's median call cost: about $0.26. Shortening the list
saves little because each call carries the engine's whole output. The KV/
schema cost function omits Gemini's thinking tokens (`schema_judge.py:_cost`),
so the recorded spend is a floor.

## 4. Cache identity: stale verdicts in the baseline

The verdict cache key is `{stem}__{engine}` in all four judges, read before
the output is examined; no hash of output, prompt, model, ground truth or
scorer version is stored. Checked for all four caches by comparing each
verdict file's mtime with the `.call.json` it scores:

| Cache | Verdict files | Older than the output they score |
| --- | ---: | ---: |
| schema | 455 (+65 under `mistral_ocr`) | **99**: deepseek_ocr 40/65, glm_ocr 55/65, qwen3_vl 4/65 |
| key_value | 224 | **56**: qwen3_vl 56/56 |
| markdown | 627 | 0 |
| tables | 485 (+45 under `mistral_ocr`) | 0 |

So 155 pattern-B verdict sets were reused from an earlier output. Two
independent confirmations: the result rows for exactly those pairs record
judge cost $0 although Gemini verdicts were needed (cache hits), and in 10
of those pairs 52 locally checkable fields are saved as missing/wrong while
the saved output contains the value (0 such contradictions in non-stale
pairs). Lower bound on the effect: deepseek schema 0.3377 → ≥0.3507; glm
+0.0011; qwen KV +0.0004. The effect on the Gemini-judged fields in those
pairs is unknown without re-judging. The earlier outputs the verdicts belong
to are not in the saved JSONL.

The same defect is demonstrated by `tests/test_reference_self_consistency.py`:
scoring a reference against itself under the engine name `gemini_flash`
returns gemini_flash's cached verdicts for its own output. Those two
parametrisations are `xfail(strict)` until the key includes the judged
output.

## 5. Other findings, most severe first

1. **Geometry: two engines' 0.000 scores are adapter defects — (a).**
   Reproduced from the saved outputs: `deepseek_ocr` reads its geometry from
   `structured_output["words"]`, and every one of those entries has the
   layout *label* as its text (`title`, `text`, `table`) — the app stored
   the `<|ref|>` group, which is the region category, not the recognised
   content that follows the tag. Text matching cannot succeed: **0.000 on all
   17 pages.** `paddleocr_vl` has `text_regions` in 0 of 93 outputs (PaddleX
   3.7 returns no `overall_ocr_res`), so the scorer fell back to parsing
   boxes out of raw markdown and failed: **0.0, applicable, on all 17.**
   Also: `qwen3_vl` has 3 unparseable outputs scored 0 (it reads `words` on
   14 pages, raw text on 3) and its boxes are lines labelled as words;
   `mineru` stores layout blocks as `words`; `marker` has 3 empty outputs
   scored 0 (without them 0.114 → ~0.2). The saved deepseek and paddle
   geometry numbers are not measurements of those engines.
2. **mineru's markdown score was computed from a different input than the
   current code would use — (a).** `run_benchmark.py:score_leg` now prefers
   `structured_output["markdown"]` when present; mineru's outputs have it in
   92 of 93 cases and it differs from `raw_text` in 90. The saved
   deterministic `markdown_score` equals the raw-text-based value on 57 of
   57 rows and the structured-markdown-based value on 2, so the saved rows —
   the judged headline included, since both are fed the same text — scored
   the flattened text. A rescore today would feed the markdown to the judge
   but hit the existing cache (§4) unless it is invalidated.
3. **The document judge covered 10 of 14 engines — (a) for that
   estimate.** Judge rows exist for tesseract, docling, textract_text,
   textract_structured, mistral_ocr_4, llamaparse, gemini_flash,
   claude_sonnet_4_6, mineru, paddleocr_vl (11 rows each) and for none of
   deepseek_ocr, glm_ocr, qwen3_vl, marker. Two gates in `run_benchmark.py`
   caused it: a prompted pass was skipped on any document with no reference
   for its legs (the 11 pages have no text reference, and those four engines
   serve text through a prompted pass), and the judge only fired on a pass
   literally named `default` (qwen's is `text`, marker's `text+markdown`).
   The design document says all 14. **Both gates fixed 2026-09-30** for
   future runs (the judge follows whichever pass serves the text leg; in a
   collect-only run the text-serving pass is not gated on text GT).
4. **The report never printed the judged-documents section — (b).** Judge
   rows were dropped before `_print_judged_documents` ran (0 of 110 rows
   reached it). **Fixed 2026-09-30.**
5. **Judge errors are hidden — (b), 0 occurrences in the saved run.** KV/
   schema judge errors become applicable 0.0 with no `error` field; a
   markdown or table judge failure falls back to F1/TEDS inside the same leg
   average with no marker in the row.
6. **KV eligibility mismatch — (b).** `has_gt_for("key_value")` ignores the
   scorer's 3-field minimum: 30 KV inference calls ($0.14) on chart and
   financial-table pages produced rows the scorer then marked not
   applicable, and the report's "of 65" denominators overstate coverage.
7. **Cost display — (b).** The per-leg table shows the primary pass's $/page
   for every leg: claude shows $0.0152 beside its geometry score, while its
   geometry pass cost $0.0546/page (schema $0.0192, key_value $0.0168).
   `_print_per_leg_costs` maps legs to passes from today's planner, not the
   recorded `legs` field. In the Modal audit every item in a batch is costed
   at `batch_seconds / batch_size` regardless of pass (8 of 57 batches mix
   passes); GPU time of 6 failed batch slots is attributed to no record. The
   audit itself passes with 0 errors.
8. **T4 fallback — (b).** `calculate_modal_cost` fell back to the T4 rate for
   an unknown GPU type and the non-batch path defaulted a missing `gpu_type`
   to T4. The saved run is unaffected (all Modal rows are batch rows at the
   registry's rate). **Both fixed 2026-09-30.**
9. **Resume/rerun bookkeeping — (b).** `write_run_manifest` overwrote the
   manifest on every resume (**fixed: merge**). `--rerun-errors --skip-judge`
   would drop KV/schema from the manifest and the report would hide them.
   `zip(chunk, results)` would silently drop items if a batch came back
   short (none did: 2,242 passes planned, all present except the 5 Marker
   failures).
10. **Report GT re-check — (d).** Only demotes rows; deterministic rescoring
    offline reproduces text 1,148/1,148 and geometry 165/165; TEDS differs
    only on chart rows, now excluded. Minor: the per-doc CSV is written from
    the unfiltered rows.
11. **Table scoring — (b)/(c); TEDS is diagnostic only.** Candidate choice
    uses the ground truth as an oracle against the first reference table
    only; `cell_match_rate` uses the first reference table; with several
    reference tables one hypothesis may match several; the TSV path keeps
    only the first table (no engine declares TSV). Chart exclusion is
    correct.
12. **Geometry heuristics — (b)/(d); no misfire in saved data.** The
    (1.5, 1000] rescale is all-or-nothing on the single largest coordinate
    (would wrongly divide pixel coordinates ≤1000, would not rescale >1000);
    only gemini triggered it (16/17, max coordinate 96–988). Axis swap is a
    per-document majority vote (gemini 16/17). Line merging groups by
    vertical overlap only, so multi-line regions never match line
    references — correct, undisclosed. Word matching is greedy.
13. **Text leg — (c).** Whitespace tokens keep punctuation attached
    ("total:" ≠ "total"); coverage ignores order and repetition (documented).
14. **Judged and deterministic markdown are different metrics — (d), by
    design.** Observed during the accidental judge run disclosed in the
    implementation document §8: the human markdown references, fed through
    the markdown judge, score below 1.0 on 42 of 57 pages (chart_003 0.79,
    fintabs ~0.93), while deterministic block F1 scores them 1.0 on all 57.
    Expected under pattern A: the judge's reference is Gemini's catalogue of
    the page and the human sidecar was never its reference. The only
    practical point: a judged markdown number and a `--skip-judge` F1 number
    are not on the same scale and must not share a column.

## 6. Minimal fixes, and whether the baseline needs rescoring

| Issue | Class | Minimal fix | Rescore? Judge calls? |
| --- | --- | --- | --- |
| Local check false positives (pattern B) | a | Match on token boundaries; compare numbers by value; match values, never keys; check the schema path first; send short values, booleans, shared values and zeros to Gemini; store per-field origin in the verdict file | Yes, KV and schema. Gemini only for fields no longer accepted locally (most pairs already make a call; roughly under $2 at recorded rates) |
| Stale verdicts / cache identity (both patterns) | a | Key the verdict cache on a hash of (judged output, checklist, prompt, judge model, scorer version) | At minimum re-judge the 155 stale pairs plus Mistral's 65 schema + 45 table under its current name. A clean rebuild of KV/schema is all 744 pairs, about $1.5–2 |
| Geometry adapters | a | Score parse/adapter failures as errors or not-applicable, never 0.0; pair deepseek boxes with the text after each tag (the saved `raw_model_output` still has it); put block/line boxes under `text_regions` with a stated granularity | Deterministic, no judge. Paddle v1 needs re-inference to get line boxes |
| mineru markdown input | a | Choose one canonical input and invalidate its markdown verdict cache | 57 markdown judge calls |
| Document judge on 10/14 engines | a (estimate) | Done in the harness; needs the four engines re-run on the 11 pages | Inference + 44 judge calls |
| Judge error → 0.0 / silent fallback | b | Record as an error row | No |
| Cost display, per-doc CSV, zip truncation | b | Report/bookkeeping | No |
| KV `has_gt_for` vs ≥3-field cutoff | b | One eligibility rule in both places | No; saves 30 calls |
| Schema conformance and field association | c | Separate metrics beside value recovery | New metrics only |

Nothing in this table has been applied to scores. The harness changes of
2026-09-30 (`--collect-only`, `--score`, manifest merge, cost fallbacks,
judged section, judge gates, the test-suite credential guard) alter no saved
number; they make a later paired rescore possible without re-paying for
inference.

## 7. Method notes

- Scripts: `audit_local_verify.py` (origin reconstruction, consistency
  checks), `suspicious_accepts.py` + `bounds_strict.py` (classification and
  bounds), `judge_cost.py`, `schema_leak.py`, `geometry_audit.py`,
  `det_rescore.py`, `completeness.py`, `stale_impact.py`, plus ad-hoc checks
  for stale caches across all four judges, judge-row coverage, the mineru
  input comparison, per-pass costs and the four example verdicts. All read
  saved files and call only pure functions; `GOOGLE_API_KEY` was unset.
- Read-only `git log` was used to date commits (one commit covers code and
  run together, so it dates nothing).
- The unit-test suite was run to verify harness edits; the paid-call incident
  that caused on the first run is disclosed in the implementation document
  §8. No other judge or engine was called.
