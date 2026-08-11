# OCR Benchmark

Runs 14 OCR engines over 93 real documents and reports, per engine, **how good
the output is, what it cost, and how long it took** — printed to your terminal.

Most OCR comparisons measure one thing: did the engine read the words. That is
only the first of seven questions people actually have. This measures all seven.

```bash
python run_benchmark.py --engines tesseract,docling --tiers easy --skip-judge
```

---

## The seven legs

Each leg is a different job, asked for differently and scored differently. The
`--legs` column is the exact token to pass on the command line; "docs" is how
many of the 93 documents carry ground truth for that leg — a run only calls
the documents that do.

| `--legs` value | The question | Scored on | Docs | Engines |
|---|---|---|---|---|
| `text` | Did it read the words? | GT coverage — share of unique reference words present | 82 | 14 |
| `markdown` | Did it preserve structure and reading order? | structural element F1 + block order | 57 | 11 |
| `tables` | Did it recover the table, not just the cell text? | TEDS (tree edit distance on structure) | 40 | 12 |
| `key_value` | Did it pull out label → value pairs? | LLM-judged field accuracy (scalar fields) | ~60 | 6 |
| `schema` | Given *your* JSON schema, did it fill it correctly? | LLM-judged field accuracy (all fields) | 65 | 8 |
| `geometry` | Does it know *where* on the page each word is? | words located — text-matched, box centers agree | 17 | 6 |
| `judge` | How well did it capture the page overall? | LLM-judged element checklist (headings, paragraphs, tables, lists) | 11 | 14 |

Default is all seven. Underscore matters: it is `--legs key_value`, not
`key-value` — an unknown token fails immediately and lists the valid ones.

The **text** leg covers 82 of the 93 documents: the other 11 have
machine-generated reference text too unreliable to score against. Those 11 are
evaluated by the **judge** leg instead — an LLM builds a checklist of structural
elements from the page image and checks each engine's output against it.

Not every engine can answer every leg, and the line is drawn at the interface,
not the score. An engine with **no prompt parameter at all** cannot be asked
for a schema, so it is reported **not applicable**. An engine that *can* be
asked competes — and if it answers badly, that low score is its result. We
never withdraw an engine from a leg because it does poorly; we only make sure
the ask was the best one available to it.

Full detail: [docs/scoring-design.md](docs/scoring-design.md).

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
| `gemini_flash` | Google API | ~$0.0007 | all seven legs |
| `claude_sonnet_4_6` | Anthropic API | ~$0.0120 | all seven legs |
| `mineru` | Modal GPU | ~$0.0009 | text, markdown, tables |
| `glm_ocr` | Modal GPU | ~$0.0006 | text, markdown, tables, key_value, schema |
| `paddleocr_vl` | Modal GPU | ~$0.0014 | text, markdown, tables |
| `marker` | Modal GPU | ~$0.0014 | text, markdown, tables |
| `deepseek_ocr` | Modal GPU | ~$0.0010 | text, markdown, tables, key_value, schema |
| `qwen3_vl` | Modal GPU | ~$0.0041 | all seven legs |

¹ `textract_structured` has no prompt interface. For the schema and key_value
legs, its FORMS/TABLES output is judged by the same LLM judge used for all
other engines — the judge checks whether each expected field value appears in
the engine's output regardless of format.
² via the vendor's own extraction feature (Mistral Annotations, LlamaExtract) —
their documented structured path, not a prompt we invented.

All seven legs are evaluated for every engine regardless of selection. The
**judge** leg runs automatically on the 11 silver-GT documents unless you pass
`--skip-judge`.

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

That runs free, locally, on 12 documents, and needs no API keys.

### 4. Run with cloud engines

For anything else, copy `.env.example` to `.env` and fill in only the keys for
the engines you want. Engines whose keys are missing are reported by `--check`;
you simply leave them out of `--engines`.

The self-hosted models run as serverless GPU apps on [Modal](https://modal.com)
(new accounts get free credits):

```bash
modal setup
python scripts/setup_modal.py     # stores your HuggingFace token as a Modal secret
python scripts/deploy_modal.py    # deploys the six GPU apps
```

## Estimating cost before running

```bash
# see what it will cost before spending anything
python run_benchmark.py --estimate

# check your keys and deployments
python run_benchmark.py --check

# plan the run without doing any inference
python run_benchmark.py --dry-run
```

### Cost per engine (approximate)

These are the per-page costs for the OCR inference itself (not the judge):

| Tier | Engines | $/page |
|---|---|---|
| Free | tesseract, docling | $0 |
| Cheap cloud | gemini_flash | ~$0.0007 |
| Cheap Modal | glm_ocr, mineru, deepseek_ocr | $0.0006–0.0010 |
| Mid Modal | paddleocr_vl, marker | ~$0.0014 |
| Mid cloud | textract_text, llamaparse, mistral_ocr_4 | $0.0015–0.004 |
| Mid Modal | qwen3_vl | ~$0.004 |
| Expensive cloud | claude_sonnet_4_6 | ~$0.012 |
| Expensive cloud | textract_structured | $0.065 |

A full run of all 14 engines × 93 documents × all legs costs roughly **$15–25**
depending on judge model choice. The `--estimate` flag shows the exact breakdown
before you spend anything.

### Cost of the LLM judge

The LLM judge adds cost on top of per-engine pricing. Judge costs are not
included in per-engine $/page — they are tracked separately.

| Judge stage | Applies to | Cost per call | Notes |
|---|---|---|---|
| Schema/KV checklist build | schema, key_value legs | ~$0.006/doc | Once per document, shared across engines |
| Schema/KV verdict | schema, key_value legs | ~$0.005/doc/engine | One call per (document, engine) pair |
| Table checklist build | tables leg | ~$0.008/doc | Image input, more expensive |
| Table verdict | tables leg | ~$0.004/doc/engine | One call per (document, engine) pair |
| Document judge (silver GT) | judge leg | ~$0.006/doc checklist + $0.005/doc/engine | 11 documents only |

Skip the judge entirely with `--skip-judge`. This disables the judge leg and
falls back to deterministic scoring for tables (TEDS) and schema/key_value
(local substring matching only).

## Running it

```bash
# Available values:
#   --engines  tesseract, docling, textract_text, textract_structured,
#              mistral_ocr_4, llamaparse, gemini_flash, claude_sonnet_4_6,
#              mineru, glm_ocr, paddleocr_vl, marker, deepseek_ocr, qwen3_vl
#   --legs     text, markdown, tables, key_value, schema, geometry, judge
#   --tiers    easy, medium, hard

# pick your engines and legs
python run_benchmark.py --engines gemini_flash,mistral_ocr_4 --legs text,tables

# structured extraction only, on the engines that can answer it
python run_benchmark.py --engines gemini_flash,claude_sonnet_4_6,textract_structured \
    --legs schema,key_value

# the geometry leg: 17 documents, 6 engines
python run_benchmark.py --engines tesseract,textract_text,gemini_flash --legs geometry

# re-print the last run's report — no re-run, no cost
python run_benchmark.py --report

# ...or the reference run
python run_benchmark.py --report 20260810T130609Z

# continue an interrupted run, or retry only what errored
python run_benchmark.py --resume
python run_benchmark.py --rerun-errors
```

Useful flags: `--tiers easy,medium,hard`, `--max-docs N` for a quick check,
`--skip-judge` to skip the LLM judge, `--workers N` to cap parallelism.

## The LLM judge

Four legs use an LLM judge: **tables** (cell accuracy), **key_value** and
**schema** (field accuracy), and **judge** (document-level element checklist for
silver-GT documents). All judges use Gemini models.

### How the judge works

The judge uses a three-stage pipeline:

**Stage 1 — Build a checklist** (one call per document, cached and shared
across all engines). For schema/key_value, the checklist is built
deterministically from the ground truth JSON — no LLM call needed. For tables
and the document judge, a capable model reads the page image and catalogues
every structural element (headings, table rows, paragraphs, list items).

**Stage 2 — Judge each engine** (one LLM call per document × engine pair,
cached). The model sees the checklist and the engine's output and returns a
verdict per item. Before calling the LLM, a local substring verifier resolves
~85–90% of items without an API call — only undecided items go to the model.

**Stage 3 — Compute the score** (arithmetic, no LLM). For schema/key_value:
`field_accuracy = (correct + blank_ok) / total_fields`. For tables/judge:
`element_accuracy = (correct + 0.5*partial) / total`.

Verdicts are cached per (document, engine) pair, so re-running with the same
data is free.

### Judge models

The judge uses two Gemini models:

| Role | Default | Used for |
|---|---|---|
| Checklist builder | `gemini-2.5-pro` | Stage 1: building element checklists from page images (tables, markdown, document judge) |
| Verdict model | `gemini-3.6-flash` | Stage 2: judging each engine's output against the checklist |

To change the judge model, modify the `DEFAULT_CHECKLIST_MODEL` and
`DEFAULT_VERDICT_MODEL` constants in the relevant evaluation files:

| File | Controls |
|---|---|
| `evaluation/judge.py` | Document judge (silver-GT pages) |
| `evaluation/table_judge.py` | Table leg judge |
| `evaluation/markdown_judge.py` | Markdown leg judge |

The schema/key_value judge models are set the same way in
`evaluation/judge.py`.

Supported models and their per-million-token pricing:

| Model | Input | Output | Thinking | Notes |
|---|---|---|---|---|
| `gemini-2.5-flash` | $0.30 | $2.50 | budget=0 | Cheapest option |
| `gemini-2.5-pro` | $1.25 | $10.00 | budget=128 | Default checklist builder |
| `gemini-3.5-flash` | $1.50 | $9.00 | budget=0 | |
| `gemini-3.5-flash-lite` | $0.30 | $2.50 | minimal | |
| `gemini-3.6-flash` | $1.50 | $7.50 | minimal | **Default verdict model** |

The model used is stored in each JSONL record (`judge_model` field) so reports
always show which model produced the verdicts.

### Judge considerations

The schema/key_value judge is from the Gemini family. When it scores
`gemini_flash` output, judge and contestant are related. Stage 1 (checklist) is
deterministic and engine-blind. Stage 2 (verdicts) makes binary observations
("is this value present"), not holistic ratings — this limits but does not
eliminate same-family advantage.

Large schemas are capped at 80 checklist items (schema leg) or 40 (document
judge), with even sampling beyond that cap. The cap exists because structured
output degrades above ~100 items.

## How a run executes

Local CPU engines go first, one at a time, one document at a time — your machine
never runs two engines at once (first runs also download models, e.g. docling's).
Cloud and Modal engines then run in parallel, one worker per engine, but each
engine still processes its documents sequentially — so no provider ever sees more
than one concurrent request from a run. `--workers N` lowers that parallelism.

Every run is stored under `results/` — raw records as JSONL, CSV summaries, the
full log, and each engine's raw output per document. `--report` re-renders any
of them for free, so you never pay twice to look at the same numbers.

## Reading the results

The report prints, in order:

1. **Ground truth provenance** — how many documents rest on human annotation,
   on model-transcribed text, and on unreliable machine-generated text.
2. **Known limitations** — what is imperfect about this corpus, ranked by
   severity.
3. **One section per leg** — a table per difficulty tier, document types
   across, engines down, with `(n)` for how many documents back each cell.
4. **A summary line per engine**: score, `$/page`, latency, documents scored,
   failures.
5. **Failing results** — every engine/document/leg that scored badly, so any
   number can be traced back to a page.

Two things to know when reading them:

**Scores are never pooled across document types.** A 43-field tax form and a
4-field receipt are not the same difficulty. An engine's leg score is the mean
of its per-document-type means.

**Cost is per pass, and some legs share one.** Text, markdown and tables all
come from a single conversion call, so all three report that call's cost.
Schema needs its own call and reports its own. Do not add per-leg costs
together to get a run total — you would count the shared call more than once.

## What is imperfect, and why we say so

Benchmarks are only as good as their reference data, and pretending otherwise
produces numbers that describe the annotation rather than the engines. Every
known problem with this corpus is recorded in
[documents/gt_limitations.json](documents/gt_limitations.json) and **printed
with every run**. In brief:

- **11 documents cannot be scored on text at all.** Their ground truth was
  machine-generated when the corpus was built and is too unreliable (42–66% of
  the page missing). They are evaluated by the judge leg instead. Re-transcribing
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

Code is MIT. The documents under `documents/` are excerpts from third-party
datasets and keep their original licences — see
[docs/documents-and-licensing.md](docs/documents-and-licensing.md), which is
explicit about the ones restricted to research use.
