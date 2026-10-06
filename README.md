# OCR Benchmark

Runs 14 OCR engines over 93 real documents and reports, per engine, **how good
the output is, what it cost, and how long it took** — printed to your terminal.

We built this because we were curious how the engines actually compare across
the different jobs people use OCR for — reading text, preserving structure,
recovering tables, extracting fields, filling schemas, locating words — and
couldn't find numbers we trusted. So we measured it ourselves and published
everything: documents, ground truth, raw outputs, and the scoring code.

```bash
python run_benchmark.py --report 20261002T162630Z    # corrected saved results
```

That saved run predates the current evaluation changes. The new scorers and
accepted references are implemented, but schema reruns, key matching and a
replacement report remain in progress. See the current
[execution checklist](docs/follow-up-2026-10-03.md) for what has actually run.

With the document corpus installed, build the browser report (no API calls):

```bash
python scripts/build_evidence_report.py 20261002T162630Z
```

Open `results/reports/ocr-benchmark-final.html`. It includes source pages, actual
returned boxes, extracted field pairs, cell grids, prompts and saved verdicts.
The generated HTML is local and gitignored; the builder and saved results are versioned.

Historical routing calculator (uses the original scores, not the saved rescore): [artifact](https://claude.ai/code/artifact/d92e781d-11e6-4dd1-ae0f-d98eb5e2e4b0
)

---

## Getting started

### 1. View the reference benchmark — no downloads, no keys

The full reference run (14 engines × 93 documents × all legs) ships **in this
repository** under `results/`: the raw JSONL records, every engine's raw output
per document, and saved score/verdict details. Superseded per-engine judge
cache files were removed; shared image checklists and document-judge caches
are retained. Cloning is all it takes:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_benchmark.py --report 20260810T130609Z
```

That re-renders the complete report in your terminal — free, instant, and any
number in it can be traced to the raw engine output under
`results/ocr_outputs/`.

### 2. Get the documents (for inference or the browser evidence report)

The 93 document images and their ground truth live on HuggingFace at
[`ilsilfverskiold/ocr-benchmark`](https://huggingface.co/datasets/ilsilfverskiold/ocr-benchmark)
— they are third-party dataset excerpts, kept out of the code repo. One
command rebuilds the corpus locally:

```bash
python scripts/download_data.py
```

Accepted references ship with this repository: 42 human-reviewed table files,
65 shared KV/schema files and five original shipping-table alternatives.
The downloader restores image bytes from Hugging Face, checks their manifest
hashes and fills missing source sidecars. It refuses to overwrite differing
local files. Review drafts and save history remain local. Both reference
loaders verify that gold belongs to the downloaded source image.

### 3. Run locally for free

Tesseract and Docling run on your CPU with no API keys:

```bash
python run_benchmark.py --engines tesseract,docling --tiers easy --skip-judge
```

That covers 12 easy documents on the legs these two engines can answer —
text, markdown, tables (docling) and text, geometry (tesseract) — all scored
by code using existing references. `--skip-judge` disables the separate
silver-document judge; it does not disable key-value or schema scoring.

### 4. Set up cloud engines

Copy `.env.example` to `.env` and fill in only the keys for the engines you
want. Engines whose keys are missing are reported by `--check`; you simply
leave them out of `--engines`.

The self-hosted models run as serverless GPU apps on [Modal](https://modal.com)
(new accounts get free credits):

```bash
modal setup
python scripts/setup_modal.py     # stores your HuggingFace token as a Modal secret
python scripts/deploy_modal.py    # deploys the six GPU apps
```

### 5. Run a benchmark

```bash
python run_benchmark.py --check                              # keys and deployments OK?
python run_benchmark.py --estimate                           # what will it cost?

# start small: one cheap engine, one tier, one leg
python run_benchmark.py --engines mistral_ocr --tiers easy --legs text --skip-judge

# then whatever combination you actually care about (LLM judge skipped)
python run_benchmark.py --engines gemini_flash,mistral_ocr_4,glm_ocr \
    --tiers easy,medium,hard --legs text,markdown,tables

# or the whole thing
python run_benchmark.py

# re-print the finished run, free
python run_benchmark.py --report    
```

The rest of this README explains how to pick engines and legs, what
`--skip-judge` changes for each leg, and what different flag combinations
actually run.

---

## The dataset

All 93 documents and their ground truth are published as a HuggingFace
dataset:
**[huggingface.co/datasets/ilsilfverskiold/ocr-benchmark](https://huggingface.co/datasets/ilsilfverskiold/ocr-benchmark)**

One `train` split, one row per document: the page image plus every
ground-truth sidecar as columns — `gt_text`, `gt_json`, `structured_gt`
(the JSON schema + expected values for the schema/key_value legs),
`markdown_gt`, and `geometry_gt`. The JSON columns are stored as strings
because their structure varies per document type; parse them with
`json.loads()`.

`scripts/download_data.py` rebuilds the exact on-disk layout the harness
expects from this dataset. You can also use it directly, without the
harness:

```python
from datasets import load_dataset
ds = load_dataset("ilsilfverskiold/ocr-benchmark", split="train")
```

Every image is an excerpt from a public dataset — sources, licences and
attribution are in
[docs/documents-and-licensing.md](docs/documents-and-licensing.md).

## The six legs

Each leg is a different job, asked for differently and scored differently. The
`--legs` column is the exact token to pass on the command line; "docs" is how
many distinct pages were scored in the saved-output rescore. Engine
counts describe that run, not every integration available in the registry.

| `--legs` value | The question | Docs | Engines |
|---|---|---|---|
| `text` | Did it read the words? | 82 | 14 |
| `markdown` | Did it preserve structure and reading order? | 57 | 11 |
| `tables` | Did it recover the table, not just the cell text? | 42 | 12 |
| `key_value` | Did it pull out label → value pairs? | 65 | 4 |
| `schema` | Given *your* JSON schema, did it fill it correctly? | 65 | 7 |
| `geometry` | Does it know *where* on the page each word is? | 17 | 9 |

KV and schema draw from the same 65 accepted structured references. KV uses all
eligible fields; schema uses the frozen common sample in
`documents/schema_sample.json` (3,316/4,251 fields), identical across engines.
KV resolves unmatched wording during evaluation: complete-value candidates go
to a judge only when needed; confirmed variations are cached for later engines.
Geometry includes two later geometry-only engines; Marker has fewer successful pages.

Re-evaluate stored KV answers without rerunning extraction:

```bash
python run_benchmark.py --rescore 20260810T130609Z --legs key_value --estimate
OCR_KV_JUDGE_BUDGET_USD=2 python run_benchmark.py --rescore 20260810T130609Z --legs key_value
# Cache-only replay (no paid calls): add --skip-judge.
```

The budget caps recorded spend plus reservations for in-flight requests. The
estimate assumes no incremental alias reuse and reserves maximum token output;
actual calls/cost are saved in `results/kv_synonyms/value_assisted/calls/`.
Missing or invalid responses stay pending. Existing confirmations are shared
across engines; each response value is still checked separately.

Default is all six. Underscore matters: it is `--legs key_value`, not
`key-value` — an unknown token fails immediately and lists the valid ones.

### Which tiers carry which ground truth

Ground truth is uneven by design — the source datasets annotate what they were
built for. This is what each tier can actually score (docs with GT per leg):

| Tier | Docs | text | markdown | tables | key_value/schema | geometry |
|---|---|---|---|---|---|---|
| easy | 12 | 12 | 6 | 11 | 12 | 6 |
| medium | 35 | 35 | 35 | 23 | 35 | 0 |
| hard | 46 | 35² | 16 | 8 | 18 | 11 |

² the other 11 hard documents (newspapers, reports) are the silver-GT set:
no usable text reference, judged estimate instead — see below.

A run only calls the documents that have GT for at least one selected leg, so
`--tiers medium --legs geometry` plans zero calls and tells you why. The
easy-tier markdown GT is the 6 invoices (the receipts have none); geometry GT
exists only for the SROIE receipts (6), FUNSD forms (6) and SynFinTabs
financial tables (5).

Not every engine can answer every leg, and the line is drawn at the interface,
not the score. An engine with **no prompt parameter at all** cannot be asked
for a schema, so it is reported **not applicable**. An engine that *can* be
asked competes — and if it answers badly, that low score is its result. We
never withdraw an engine from a leg because it does poorly; we only make sure
the ask was the best one available to it.

## How each leg is scored

The saved-output rescore is `20261002T095730Z`, **provisional and not cleared for publication**. The causal audit found scorer defects and schema/GT conflicts; see the [audit handoff](docs/handoff-2026-10-01.md#causal-audit-after-the-report-was-challenged-2026-10-02). The saved run uses older scorers. Current scoring differs from that saved run; see [scoring design](docs/scoring-design.md).
The completed KV-only rescore is **`20261006T195826Z`**: 260 scored outputs,
65 accepted pages, four engines, zero pending decisions/errors. Its 342 judge
calls cost **$1.4756**, including all responses; final replay made no new calls.
See [the follow-up](docs/follow-up-2026-10-03.md#latest-kv-decision--inline-value-assisted-evaluation)
and `results/runs/20261006T195826Z_kv_judge_audit.json`. The table below describes current scoring. Historical full-run scores still
need replacement after the remaining approved reference/inference work.
Rebuild the KV viewer with:

```bash
python scripts/build_evidence_report.py 20261006T195826Z --output results/reports/ocr-benchmark-kv.html --status 'Completed KV-only rescore; other legs are separate.'
```

KV can call a judge during evaluation for unmatched pairs, caching confirmations.
The other deterministic legs do not make per-answer judge calls.

| Leg | Headline | Reference / supporting metrics |
|---|---|---|
| `text` | Unique GT-word coverage | 82 pages; CER, WER, word F1 support it |
| `markdown` | `(correct + 0.5 × partial) / elements` | Actual syntax, disjoint structural matches; image catalogue shared across engines; CER/WER separate |
| `tables` | Cell F1 | Human gold; one-to-one cells including blanks/extras; precision, recall and span-aware structure diagnostic; shipping takes the better whole-page gold/original comparison |
| `key_value` | `(correct + blank_ok) / selected fields` | Shared accepted references; known keys first, complete-value candidates judged when needed, confirmed variations cached |
| `schema` | `(correct + blank_ok) / (selected fields + extra fields)` | Shared sampled subset of accepted KV references; exact supplied keys; JSON/schema compliance separate; stale contracts require new inference |
| `geometry` | Located / reference units | One-to-one word/line matches; precision/F1, invalid boxes and box-text CER/WER; annotation scope and overlays recorded |

The saved run covers 82 text, 57 Markdown, 42 table, 55 key-value, 65 schema
and 17 geometry pages. Coverage also varies by engine: Marker has only nine
scored geometry pages and eight error rows. Errors and N/A are excluded from
means. Headlines are means of document-type means, not pooled page averages.

`--skip-judge` prevents new silver-document and KV judge calls. KV reuses saved
confirmations and leaves genuinely unresolved candidates pending. It does **not**
disable a selected Gemini OCR engine. Markdown needs a cached image checklist;
KV learns needed variations during evaluation when judging is enabled.

Missing Markdown references are prepared separately from scoring:

```bash
python scripts/build_markdown_catalogues.py prepare --output /tmp/markdown-catalogue-plan.json
```

This command is free and records exact image hashes, prompt, configuration and
cost reserve. Generation is a separate paid command requiring approval. Review
proposals against source images before `accept`; scoring never activates them
automatically. A sampled catalogue does not become a full-page text reference.

The document judge is an estimate, not a seventh leg. The saved rescore
copies 110 previously saved judge rows unchanged and makes no new judge call.
`--judge-model` affects that optional document judge; inline KV uses Gemini 2.5 Flash.
Its supported model identifiers and internal cost assumptions are in
`evaluation/judge.py`; they are not current vendor price quotations.

Claude/Gemini Markdown and table scores in this rescore use their historical
default transcription outputs. Those runner prompts already request structure,
Markdown tables and reading order. Dedicated leg prompts are implemented but
have not been tested on these pages; their effect cannot be inferred from the
saved scores. Treat these as saved-output measurements, not a completed
prompt-sensitivity comparison.

Examples:

```bash
# Deterministic field scoring; the selected OCR engine still costs money.
python run_benchmark.py --engines gemini_flash --tiers easy --legs schema --skip-judge
# Inspect the saved results without inference or judge calls.
python run_benchmark.py --report 20261002T095730Z
# Recompute from the three original inference sources, writing a new run.
python run_benchmark.py --rescore 20260810T130609Z,20261001T102236Z,20261001T102808Z --skip-judge
```

Current rules, limitations and handoff:
[scoring design](docs/scoring-design.md),
[audit handoff](docs/handoff-2026-10-01.md).

## The engines

The `Engine` column is the exact token for `--engines` (comma-separated, any
subset). The `Answers` column is which legs the engine competes in — asking it
for a leg it cannot answer is fine: it is reported "not applicable", never
penalised.

| Engine | Runs on | $/page | Answers |
|---|---|---|---|
| `tesseract` | your CPU | free | text, geometry |
| `docling` | your CPU | free | text, markdown, tables |
| `textract_text` | AWS | $0.0015 | text, geometry |
| `textract_structured` | AWS | $0.0650 | text, tables, key_value¹, geometry |
| `mistral_ocr_4` | Mistral API | $0.0040 | text, markdown, tables, schema² |
| `llamaparse` | LlamaCloud | $0.00375 | text, markdown, tables, schema² |
| `gemini_flash` | Google API | ~$0.0007 | all six legs |
| `claude_sonnet_4_6` | Anthropic API | ~$0.0120 | all six legs³ |
| `mineru` | Modal GPU | ~$0.0009 | text, markdown, tables |
| `glm_ocr` | Modal GPU | ~$0.0006 | text, markdown, tables, key_value, schema |
| `paddleocr_vl` | Modal GPU | ~$0.0014 | text, markdown, tables |
| `marker` | Modal GPU | ~$0.0014 | text, markdown, tables |
| `deepseek_ocr` | Modal GPU | ~$0.0010 | text, markdown, tables, key_value, schema |
| `qwen3_vl` | Modal GPU | ~$0.0041 | all six legs |

Sixteen more engines were added on 2026-09-30 and have **not been run**: they
are `default=False`, carry no measured cost, and are not part of the reference
run. The list, where each one runs, which legs it enters and how to test it are
in [docs/engine-expansion-implementation.md](docs/engine-expansion-implementation.md).

¹ `textract_structured` has no prompt interface. Its key_value answer is its
FORMS output (label → value pairs), judged like everyone else's. It is not in
the schema leg: AnalyzeDocument takes no schema, so nothing was ever asked of
it there (Textract Queries would be a separate integration).
² via the vendor's own extraction feature (Mistral Annotations, LlamaExtract) —
their documented structured path, not a prompt we invented.
³ one asymmetry to know about: for schema/key_value, Gemini runs with the
provider's JSON mode enabled, but Claude does not — the Anthropic API has no
JSON mode and the current model rejects assistant prefill, so Claude answers
from the prompt's "Output only JSON" instruction alone, at temperature 0. This
costs it little in *our* scores, because the judge matches expected values
against the raw output and never parses the JSON — but it means the benchmark
does not test Claude's guaranteed-structure path (forced tool use), which is
what you would use in production.

## What it costs to run

`--estimate` prints the exact cost breakdown for your engine/leg/tier
selection before anything is spent — use it, the numbers below are
orientation.

```bash
python run_benchmark.py --estimate
python run_benchmark.py --check      # verify keys and Modal deployments
python run_benchmark.py --dry-run    # plan the passes without any inference
python run_benchmark.py --dry-run --legs schema --plan-output /tmp/schema-plan.json
```

`--plan-output` saves eligible page/pass requests, exact task prompts and schemas,
image hashes and runner/configuration hashes without loading API credentials.
It does not include expected answer values or authorize execution. Review model
context/output limits and pricing before using the plan for a paid run.

**Engine inference** is per page, per the table above: the two local engines
are free; most Modal-hosted models land between $0.0006 and $0.004 per page;
the API engines range from $0.0007 (gemini_flash) to $0.065
(textract_structured).

**Scoring stored outputs** with `--rescore ... --skip-judge` makes no model
calls. The optional silver-document judge is separate and can incur checklist
and verdict charges. Existing image checklists and printed-label aliases are
read from disk; the six leg scorers do not create them automatically. New
reference creation needs its own cost estimate and authorization.

The engine figures above are repository assumptions, not current vendor quotes.
Use measured saved costs where available and review the exact pass/page manifest
before a new run. Dedicated Markdown/table prompts for Claude and Gemini have
not been measured yet, so an old all-engine total is not a budget for them.

**Cost is per pass, and some legs share one.** Native converters can serve
text, Markdown and tables in one conversion call. Promptable engines may need
separate calls for those legs; the registry plans this per engine. Schema,
key-value and geometry can also need separate calls. Do not sum per-leg costs
without checking shared passes.

## Running it

```bash
# Available values:
#   --engines  tesseract, docling, textract_text, textract_structured,
#              mistral_ocr_4, llamaparse, gemini_flash, claude_sonnet_4_6,
#              mineru, glm_ocr, paddleocr_vl, marker, deepseek_ocr, qwen3_vl
#   --legs     text, markdown, tables, key_value, schema, geometry
#   --tiers    easy, medium, hard

# pick your engines and legs
python run_benchmark.py --engines gemini_flash,mistral_ocr_4 --legs text,tables

# structured extraction only, on the engines that can answer it
python run_benchmark.py --engines gemini_flash,claude_sonnet_4_6,textract_structured \
    --legs schema,key_value

# the geometry leg: 17 documents, 6 engines
python run_benchmark.py --engines tesseract,textract_text,gemini_flash --legs geometry

# re-print any stored run — no re-run, no cost
python run_benchmark.py --report                     # the latest
python run_benchmark.py --report 20260810T130609Z    # the reference run

# continue an interrupted run, or retry only what errored
python run_benchmark.py --resume
python run_benchmark.py --rerun-errors
```

To add engines to an existing run without rescoring anything, collect first
and score later:

```bash
# collect every output (all legs, all 93 pages), no scoring, no judge
python run_benchmark.py --engines rapidocr,ovisocr2 --collect-only --extends 20260810T130609Z
# score the six legs from stored outputs without a new document judge
python run_benchmark.py --score <TIMESTAMP> --skip-judge
# one report over the extension run and the baseline it extends
python run_benchmark.py --report <TIMESTAMP>,20260810T130609Z
```

Useful flags: `--tiers easy,medium,hard`, `--max-docs N` for a quick check,
`--docs stem,stem` for a chosen cross-section,
`--skip-judge` to prevent new document/KV judge calls while using existing
confirmations, and `--workers N` to cap inference parallelism. A selected Gemini OCR
engine still makes paid inference calls.

**How a run executes.** Local CPU engines go first, one at a time, one document
at a time — your machine never runs two engines at once (first runs also
download models here, e.g. docling's). Cloud and Modal engines then run in
parallel, one worker per engine, but each engine still processes its documents
sequentially — so no provider ever sees more than one concurrent request from
a run. `--workers N` lowers that parallelism if you want it gentler.

Every run is stored under `results/` — raw records as JSONL, CSV summaries,
and each engine's raw output per document. `--report <timestamp>` re-renders
any of them for free, so you never pay twice to look at the same numbers.

## Reading the results

The report prints, in order:

1. **Ground truth provenance** — how many documents rest on human annotation,
   on model-transcribed text, and on unreliable machine-generated text. Before
   any score, because it frames them.
2. **Known limitations** — what is imperfect about this corpus, ranked by
   severity. See below.
3. **One section per leg** — a table per difficulty tier, document types
   across, engines down, with `(n)` for how many documents back each cell, and
   the scoring method (and judge model, where one was used) in the header.
4. **A summary line per engine**: score, `$/page`, latency, documents scored,
   failures.
5. **Failing results** — every engine/document/leg that scored badly, so any
   number can be traced back to a page.

**Scores are never pooled across document types.** A 43-field tax form and a
4-field receipt are not the same difficulty. An engine's leg score is the mean
of its per-document-type means.

## What is imperfect, and why we say so

Benchmarks are only as good as their reference data, and pretending otherwise
produces numbers that describe the annotation rather than the engines. Every
known problem with this corpus is recorded in
[documents/gt_limitations.json](documents/gt_limitations.json) and **printed
with every run**.

The current code-based headlines still depend on reference quality. Markdown
uses a cached model-authored image checklist. Table selection also uses human
reference HTML, and field scoring uses annotated GT values. Normalization,
sampling, blank-field credit and parser behavior affect what the metrics mean.
Known dataset limitations include:

- **11 documents cannot be scored on text at all.** Their ground truth was
  machine-generated when the corpus was built and is too unreliable (42–66% of
  the page missing). They get the judged estimate instead. Re-transcribing
  them was attempted and declined — multi-column newspapers, a rotated numeric
  table, a handwritten ledger — because a model transcription would have added
  more error than it removed.
- **19 documents have model-transcribed, reviewed or partially repaired text
  ground truth.** Eight have historical engine-consensus checks; six invoice
  references and targeted corrections to five receipts were reviewed against
  the source images. Receipt repairs do not certify the remaining transcription.
  These are disclosed separately from independent human annotations.
- **17 documents had their schemas extended by hand**, because the source
  datasets scope their annotations to the task *they* were built for. SROIE
  ships four fields where its receipts carry about twenty-five extractable
  values — which put 12 of 14 engines at a perfect score and measured nothing.
  The schema/key_value matcher checks values from this annotation; matching
  improvements do not repair a wrong reference value.
- **6 handwriting pages are composites** of individual line images, so the
  handwriting is genuine but the page layout is synthetic.

And one that now only limits coverage: **5 fabricated table references were
removed** (they described schema field names rather than the columns printed
on the page), so those 5 documents simply do not compete in the tables leg.

## Documentation

| | |
|---|---|
| [docs/scoring-design.md](docs/scoring-design.md) | What each leg asks for, how it is scored, the LLM judge pipeline, and known methodology issues |
| [docs/documents-and-licensing.md](docs/documents-and-licensing.md) | Where the 93 documents come from, and their licences |
| [docs/benchmark-audit-2026-09-30.md](docs/benchmark-audit-2026-09-30.md) | What an offline audit of the reference run found: local-verifier false positives, stale judge caches, geometry adapter defects, and what each would take to fix |
| [docs/engine-expansion-implementation.md](docs/engine-expansion-implementation.md) | The 16 unrun expansion engines: where each runs, which legs it enters, how to test it, and what is unverified |
| [docs/engine-expansion-research.md](docs/engine-expansion-research.md), [docs/engine-leg-and-cost-matrix.md](docs/engine-leg-and-cost-matrix.md) | The research those engines were built from |

## Licence

Code is MIT. The document images are excerpts from third-party datasets and
keep their original licences — see
[docs/documents-and-licensing.md](docs/documents-and-licensing.md), which is
explicit about the ones restricted to research use.


Schema requests use the committed common sample, with the same selected fields
for every engine. Rebuild it offline with `scripts/build_schema_sample.py
--tokenizer-dir PATH` only when references or engine budgets change. The builder
uses pinned DeepSeek/Qwen/GLM tokenizer JSON files, not model inference; the
manifest records their hashes and token budgets. Runtime evaluation does not
need these tokenizer files. Sampling is schema-only and leaves full gold intact.

To continue an interrupted run without submitting an already attempted call
again (including errors), use `--resume-unattempted TIMESTAMP`. Ordinary
`--resume` retries error rows. `--stop-on-error` halts further requests for an
engine when a provider call fails; poor or truncated answers stay recorded and
scored. No engine tuning or automatic regeneration is performed.
