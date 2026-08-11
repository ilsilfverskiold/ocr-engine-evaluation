# OCR Benchmark

Runs 14 OCR engines over 93 real documents and reports, per engine, **how good
the output is, what it cost, and how long it took** — printed to your terminal.

We built this because we were curious how the engines actually compare across
the different jobs people use OCR for — reading text, preserving structure,
recovering tables, extracting fields, filling schemas, locating words — and
couldn't find numbers we trusted. So we measured it ourselves and published
everything: documents, ground truth, raw outputs, and the scoring code.

```bash
python run_benchmark.py --report 20260810T130609Z    # view the reference run
```

---

## Getting started

### 1. View the reference benchmark — no downloads, no keys

The full reference run (14 engines × 93 documents × all legs) ships **in this
repository** under `results/`: the raw JSONL records, every engine's raw output
per document, and all cached LLM judge verdicts. Cloning is all it takes:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_benchmark.py --report 20260810T130609Z
```

That re-renders the complete report in your terminal — free, instant, and any
number in it can be traced to the raw engine output under
`results/ocr_outputs/`.

### 2. Get the documents (only needed to rerun engines)

The 93 document images and their ground truth live on HuggingFace at
[`ilsilfverskiold/ocr-benchmark`](https://huggingface.co/datasets/ilsilfverskiold/ocr-benchmark)
— they are third-party dataset excerpts, kept out of the code repo. One
command rebuilds the corpus locally:

```bash
python scripts/download_data.py
```

### 3. Run locally for free

Tesseract and Docling run on your CPU with no API keys:

```bash
python run_benchmark.py --engines tesseract,docling --tiers easy --skip-judge
```

### 4. Run with cloud engines

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

---

## The six legs

Each leg is a different job, asked for differently and scored differently. The
`--legs` column is the exact token to pass on the command line; "docs" is how
many of the 93 documents carry ground truth for that leg — a run only calls
the documents that do.

| `--legs` value | The question | Docs | Engines |
|---|---|---|---|
| `text` | Did it read the words? | 82 | 14 |
| `markdown` | Did it preserve structure and reading order? | 57 | 11 |
| `tables` | Did it recover the table, not just the cell text? | 42 | 12 |
| `key_value` | Did it pull out label → value pairs? | 65¹ | 6 |
| `schema` | Given *your* JSON schema, did it fill it correctly? | 65 | 8 |
| `geometry` | Does it know *where* on the page each word is? | 17 | 6 |

¹ key_value uses the same structured GT as schema, but only its scalar fields —
documents with fewer than 3 scalar fields (charts, financial tables, whose
schemas are almost entirely arrays) are reported not applicable at scoring
time, so the report shows fewer.

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

Two legs are scored purely deterministically, against gold ground truth. Four
are scored by an LLM judge — each for a specific reason, stated below.

| Leg | Default scoring | With `--skip-judge` | Why a judge at all |
|---|---|---|---|
| `text` | **Deterministic** — GT coverage: share of unique reference words present | same | — |
| `geometry` | **Deterministic** — words located: text match + box centers agree | same | — |
| `markdown` | **LLM judge** — element accuracy against a checklist built from the page image | deterministic structural F1 vs. the reference markdown | our hand-written markdown references encode *one* valid rendering and are over-segmented — engines that correctly merge text into flowing prose scored low against them |
| `tables` | **LLM judge** — cell accuracy against a checklist built from the page image | TEDS (tree edit distance) vs. the reference table | same problem: a byte-perfect table scored 0.846 against our reference for its HTML dialect; TEDS stays in the metrics as a diagnostic |
| `key_value` | **LLM judge** — field accuracy, matching by *value* | unchanged — still judged (no deterministic scorer exists) | engines legitimately name fields differently — `"company"` vs `"Company Name"`. Deterministic key-matching punished naming, not extraction |
| `schema` | **LLM judge** — field accuracy, matching by *value* | unchanged — still judged (no deterministic scorer exists) | same — and Textract returns FORMS pairs, not JSON, so there is no key to match deterministically |

So `--skip-judge` changes exactly three things: markdown falls back to the
deterministic structural F1, tables falls back to TEDS, and the silver-text
estimate (below) is skipped. It does **not** affect schema or key_value —
those have no non-judge way to be scored, so selecting them always needs
`GOOGLE_API_KEY`, flag or no flag.

One honest caveat about the fallbacks: the deterministic markdown and tables
scorers are exactly what the judge was built to replace. They compare against
references with known problems (over-segmentation, dialect punishment), so
their numbers are cheaper but less reliable — fine for smoke tests and
relative comparisons on a budget, not what we would publish.

**And one estimate that is not a leg:** 11 of the 93 documents (newspapers,
reports) have machine-generated reference text too unreliable to score against
— 42–66% of the page is missing from it. For those, the text leg reports no
deterministic score; instead a **document judge** builds a checklist of the
page's content from the image and estimates how much of it each engine
captured. Those numbers are labelled as judged in the report and kept separate.
They are directional, not definitive. So the text story is: **gold GT for 82
documents, a judged estimate for the other 11** — and `--skip-judge` means
those 11 simply carry no score.

### Worked examples — what actually runs

`--tiers easy --legs schema --skip-judge`
: 12 easy documents have schema GT, so all 12 are called. Schema is
judge-scored — the flag does not change that — so this still makes Gemini
verdict calls and still needs `GOOGLE_API_KEY`. The flag saves nothing here;
it only matters for markdown, tables, and the silver-text estimate.

`--engines docling --legs markdown` (no flag)
: docling emits markdown, so it competes on all 57 markdown-GT documents. With
`GOOGLE_API_KEY` set, scoring is the markdown judge — one cached checklist per
document (shared with every other engine, so if the reference run already
built it, it is free) plus one verdict call per document. Without the key,
scoring falls back to the deterministic F1 automatically — same run, judge
quality traded for zero judge cost.

`--engines tesseract,docling --tiers easy --skip-judge`
: entirely free: local engines, deterministic text/tables/markdown scoring,
no API keys at all.

The pattern for spending less: the judge is the main non-engine cost, so run
cheap first (`--skip-judge`, or just the deterministic legs) to check your
setup and get rough rankings, and pay for judged runs when the numbers need
to hold up. `--estimate` prints both parts — engine cost and judge cost —
before anything is spent.

Full methodology, prompts, and normalisation rules per leg:
[docs/scoring-design.md](docs/scoring-design.md).

## The LLM judge

### How it works

Every judge in the benchmark is the same three-stage shape:

**Stage 1 — Build a checklist** (once per document, cached, shared by all
engines). For `schema`/`key_value` the checklist is built deterministically
from the ground-truth JSON — no model involved, one item per expected field.
For `markdown`, `tables`, and the document judge, a model reads the page
*image* and catalogues what is on it: headings, paragraphs, table cells, list
items. Every engine is then measured against the identical checklist, so the
judge cannot favour the engine whose output it happened to see first.

**Stage 2 — Verdict per engine** (one call per document × engine, cached).
The model sees the checklist and the engine's output, and returns a verdict
per item — `correct` / `partial` / `missing`, or for fields:
`correct` / `wrong` / `missing` / `blank_ok` / `invented`. Before any model
call, a local substring check resolves the ~85–90% of field items that match
exactly; only undecided items reach the model.

**Stage 3 — Arithmetic.** Field accuracy is
`(correct + blank_ok) / total`; element accuracy is
`(correct + 0.5 × partial) / total`. No model involved.

All verdicts are cached under `results/*_judge_cache/`, and the reference
run's caches ship with this repo — re-scoring the reference run costs nothing.

### Which model judges what

All judges are Gemini models, called with your `GOOGLE_API_KEY`. Defaults, and
the file whose constant to edit to change them:

| Judge | Checklist model | Verdict model | Change it in |
|---|---|---|---|
| markdown | `gemini-2.5-pro` | `gemini-3.6-flash` | `evaluation/markdown_judge.py` |
| tables | `gemini-3.6-flash` | `gemini-3.6-flash` | `evaluation/table_judge.py` |
| schema + key_value | — (deterministic) | `gemini-2.5-pro` | `evaluation/schema_judge.py` |
| document judge (silver text) | `gemini-3.6-flash` | `gemini-3.6-flash` | `evaluation/judge.py` |

Supported models and per-million-token pricing:

| Model | Input | Output | Notes |
|---|---|---|---|
| `gemini-2.5-flash` | $0.30 | $2.50 | cheapest |
| `gemini-2.5-pro` | $1.25 | $10.00 | strongest; used where verdicts are hardest |
| `gemini-3.5-flash` | $1.50 | $9.00 | |
| `gemini-3.5-flash-lite` | $0.30 | $2.50 | |
| `gemini-3.6-flash` | $1.50 | $7.50 | default for image checklists and most verdicts |

The model used is stored in every JSONL record (`judge_model` field), and the
report prints it in each leg's header — a number is never separated from the
model that produced it.

### What the judge is not

The judge is from the Gemini family, and `gemini_flash` is a contestant. Stage
1 checklists are either deterministic or built before any engine output is
seen, which removes one path for bias; stage 2 makes binary observations ("is
this value present"), not holistic ratings, which limits — but does not
eliminate — same-family advantage. We disclose it rather than claim it away.
Checklists are capped (80 items for schema, 60 for tables, 40 for the document
judge) with even sampling beyond the cap, because structured output degrades
on very long checklists.

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
| `textract_structured` | AWS | $0.0650 | text, tables, key_value, schema¹, geometry |
| `mistral_ocr_4` | Mistral API | $0.0040 | text, markdown, tables, schema² |
| `llamaparse` | LlamaCloud | $0.00375 | text, markdown, tables, schema² |
| `gemini_flash` | Google API | ~$0.0007 | all six legs |
| `claude_sonnet_4_6` | Anthropic API | ~$0.0120 | all six legs |
| `mineru` | Modal GPU | ~$0.0009 | text, markdown, tables |
| `glm_ocr` | Modal GPU | ~$0.0006 | text, markdown, tables, key_value, schema |
| `paddleocr_vl` | Modal GPU | ~$0.0014 | text, markdown, tables |
| `marker` | Modal GPU | ~$0.0014 | text, markdown, tables |
| `deepseek_ocr` | Modal GPU | ~$0.0010 | text, markdown, tables, key_value, schema |
| `qwen3_vl` | Modal GPU | ~$0.0041 | all six legs |

¹ `textract_structured` has no prompt interface. For the schema and key_value
legs, its FORMS/TABLES output goes to the same LLM judge as everyone else —
the judge checks whether each expected field value appears in the output
regardless of format.
² via the vendor's own extraction feature (Mistral Annotations, LlamaExtract) —
their documented structured path, not a prompt we invented.

## What it costs to run

`--estimate` prints the exact cost breakdown for your engine/leg/tier
selection before anything is spent — use it, the numbers below are
orientation.

```bash
python run_benchmark.py --estimate
python run_benchmark.py --check      # verify keys and Modal deployments
python run_benchmark.py --dry-run    # plan the passes without any inference
```

**Engine inference** is per page, per the table above: the two local engines
are free; most Modal-hosted models land between $0.0006 and $0.004 per page;
the API engines range from $0.0007 (gemini_flash) to $0.065
(textract_structured).

**The judge** adds Gemini calls on top: one checklist per document that needs
one (cached and shared across engines), plus one verdict call per
document × engine pair (cached). As orientation from the reference run's
estimates: a document-judge checklist is ~$0.006/doc and each verdict
~$0.005/doc/engine; table-judge checklists ~$0.008/doc (image input) and
verdicts ~$0.004/doc/engine. Judge costs are tracked separately in the results
(`judge_cost_usd`, `kv_judge_cost_usd`), never folded into engine $/page.

**A full run** — all 14 engines, all 93 documents, all legs, with the judge —
comes to roughly **$15–25** depending on judge model choice. Caches make
re-scoring free: the judge is only ever paid once per (document, engine,
model).

**Cost is per pass, and some legs share one.** Text, markdown and tables all
come from a single conversion call, so all three report that call's cost.
Schema, key_value and geometry each need their own call. Do not add per-leg
costs together — you would count the shared call more than once.

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

Useful flags: `--tiers easy,medium,hard`, `--max-docs N` for a quick check,
`--skip-judge` to score markdown/tables deterministically and skip the
silver-text estimate (schema/key_value stay judged — see the scoring table
above), `--workers N` to cap parallelism.

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
with every run**. In brief:

- **11 documents cannot be scored on text at all.** Their ground truth was
  machine-generated when the corpus was built and is too unreliable (42–66% of
  the page missing). They get the judged estimate instead. Re-transcribing
  them was attempted and declined — multi-column newspapers, a rotated numeric
  table, a handwritten ledger — because a model transcription would have added
  more error than it removed.
- **8 documents have model-transcribed ground truth**, checked against the
  consensus of 14 engines, replacing text that was unusable.
- **17 documents had their schemas extended by hand**, because the source
  datasets scope their annotations to the task *they* were built for. SROIE
  ships four fields where its receipts carry about twenty-five extractable
  values — which put 12 of 14 engines at a perfect score and measured nothing.
- **5 fabricated table references were removed** — they described schema field
  names rather than the columns printed on the page.
- **6 handwriting pages are composites** of individual line images, so the
  handwriting is genuine but the page layout is synthetic.

## Documentation

| | |
|---|---|
| [docs/scoring-design.md](docs/scoring-design.md) | What each leg asks for, how it is scored, the LLM judge pipeline, and known methodology issues |
| [docs/documents-and-licensing.md](docs/documents-and-licensing.md) | Where the 93 documents come from, and their licences |

## Licence

Code is MIT. The document images are excerpts from third-party datasets and
keep their original licences — see
[docs/documents-and-licensing.md](docs/documents-and-licensing.md), which is
explicit about the ones restricted to research use.
