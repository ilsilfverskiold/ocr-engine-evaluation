# Scoring design: how each leg is prompted and scored

Six legs, each measuring a different job people actually do with OCR. This
document fixes, per leg: **what we ask the engine for**, **which engines can
answer**, **how the answer is scored**, and **what we normalise so we measure
content rather than formatting**. A seventh section covers the document judge
— not a leg, but the estimate used where text ground truth is unusable.

The governing principle: **a benchmark must hold one variable still.** Engines
legitimately return different shapes — in production you would adapt your
integration to whichever you chose. But to compare them we fix the target and
ask every engine for the same thing. Where an engine cannot be asked (a fixed
pipeline with no prompt), it competes through deterministic conversion code we
publish, or it does not compete in that leg. Both are honest results.

> **Saved scoring: 2026-10-02, run `20261002T095730Z` — under audit.**
> The v4 scores are provisional. See the handoff for confirmed scorer and
> schema/reference contract defects; current field code is v5, whose offline replay
> changes 32 field scores. A replacement final run awaits task/reference
> repairs and the necessary controlled inference checks. All six legs
> use code for per-engine verdicts. `code-v3` handles Markdown and tables;
> `kv-schema-code-v4` handles key-value and schema. Image checklists and
> printed-label aliases were created earlier by models and are reused.
> Current audit code includes table code-v5, field v7, geometry-v3 (global
> word assignment and explicit axis policy), and native Textract merged-cell reconstruction. The handoff
> records isolated replay impacts; these diagnostics have not replaced the
> archived run. Do not add their change counts: some affect the same rows.
> `--skip-judge` keeps all six legs and disables new silver-document judge
> calls. The rescore copies 110 old document-judge rows unchanged.
>
> Current coverage is 82 text, 57 Markdown, 42 table, 55 key-value, 65 schema
> and 17 geometry pages; Marker geometry scores nine pages and records eight
> errors. Errors/N/A do not enter means. Headlines average document-type
> means. Detailed rules, policy switches and final numbers are in the
> [handoff](handoff-2026-10-01.md) and scorer module docstrings.
>
> For blank scalar fields, an explicitly named semantic field takes precedence
> over ambiguous printed labels. This preserves repeated-label lease blanks
> without excusing an explicitly invented spouse surname. Values are compared
> using documented field/row matching and normalization, not a global
> punctuation-stripped substring search.
>
> **Historical sections below:** the LLM methods described for legs 2–5
> record how the original August run was scored, not current execution.
> Claude/Gemini Markdown/table scores in the final rescore still describe
> stored default transcription outputs. Their default prompts request
> structure, Markdown tables and reading order; dedicated leg-prompt
> sensitivity has not yet been tested. The code does not silently relabel
> those historical passes as newly prompted runs.

---

## Prompting is part of the measurement

**A wrong prompt does not measure a weak engine. It measures a weak prompt.**
If an engine scores badly on a leg, the first question is always whether we
asked it correctly — and that question has to be settled *before* the number is
published, not raised afterwards as an explanation.

This is not theoretical. `paddleocr_vl` was originally sent a 25-word English
instruction when its model card specifies short task prefixes (`"OCR:"`,
`"Table Recognition:"`). Switching to the documented prefix changed its score
on its own. A generic prompt applied to a specialised model is a benchmark
defect, not an engine result.

Five rules follow.

**1. Prompts are per engine AND per leg, not one prompt for everyone.**
General instruction-following VLMs (Claude, Gemini, Qwen3-VL) take a natural
instruction and a JSON schema. OCR-specialised fine-tunes (GLM-OCR,
DeepSeek-OCR, PaddleOCR-VL, MinerU) have prescribed prompts or task prefixes on
their model cards and degrade when given anything else. Fixed pipelines
(Marker, Docling, Tesseract, Textract) take no prompt at all. Each engine's
prompt for each leg lives in the registry beside that engine, sourced from the
model card where one exists, and cited.

**2. The prompt used is recorded with every result.** The exact prompt string
goes into each record, so any number can be traced to what was actually asked.
A result whose prompt cannot be reproduced is not a result.

**3. Before publishing a leg, prompt sensitivity is checked.** For any engine
scoring poorly, re-run a handful of documents with the model card's own prompt
and with a reasonable alternative. If the score moves materially, the original
number was measuring the prompt — fix it and re-run. If it does not move, the
engine genuinely cannot do the task, and *that* is publishable. Record which
prompt won.

**4. "Cannot be prompted" is a real finding, reported as such — and the bar is
the interface, not the score.** An engine with *no prompt parameter* does not
compete in the legs needing one: that is a capability limit worth knowing,
reported as "not applicable", never as a zero, because a zero would say it
tried and failed. But an engine that *can* be asked always competes, and a
zero it earns is published. Withdrawing an engine from a leg because it does
badly is editing the result, not measuring it. Rules 1–3 exist so that when a
zero appears we already know it belongs to the model.

**5. Output dialect is not capability.** An engine is never penalised for the
syntax it chose. Tables are read as pipes, HTML or tab-separated, whichever
the engine emits (`table_format` in the registry); the markdown leg normalises
HTML and tab-separated tables to pipe rows before parsing; an answer wrapped
entirely in a ```` ```markdown ```` fence is unwrapped first; and a schema is
rendered as either a JSON Schema document or an empty example instance
depending on which form the engine's card demonstrates. Every one of these was
added after a correct answer scored zero for its packaging.

The prompts given per leg below are the **starting point** for promptable
engines, not the final word. Where a model card prescribes something, the model
card wins.

---

## Leg 1 — Text accuracy

**Ask:** the engine's default conversion. No special prompting; this is what
each product does out of the box.

**Engines:** all 14.

**Score:** **GT coverage** (primary) — the fraction of unique ground-truth words
present anywhere in the output. Set-based, so it is blind to word order and to
extra words the engine adds. That is deliberate: engines format output
differently (labels, headers, markdown syntax) and precision-style metrics
punish correct reading that happens to be formatted unlike the annotation.
CER, WER and word P/R/F1 are recorded as supporting detail, not headline.

**Normalise:** Unicode NFKC, whitespace collapse, lowercase, markdown/HTML
rendered to plain text — applied to both sides.

**Known limits:** coverage is repetition-blind (an engine looping a phrase can
still score 1.0 — check output length against GT length) and order-blind (so
the FUNSD reading-order defect does not affect it).

---

## Leg 2 — Markdown / reading order

**Ask:** *"Convert this document to Markdown. Preserve reading order, headings,
lists and tables. Output only the Markdown."*

**Engines:** the eleven that emit markdown (mistral_ocr_4, llamaparse, marker,
docling, mineru, deepseek_ocr, glm_ocr, gemini_flash, claude_sonnet_4_6,
qwen3_vl, and paddleocr_vl — whose pipeline output is its own markdown
rendering with tables embedded as HTML, which the scorer normalises).
tesseract, textract_text and textract_structured cannot and do not compete.

**Score:** the headline comes from the **markdown LLM judge** — a checklist of
the page's structural elements built from the image (checklist model reads the
page once, cached, shared by all engines), against which each engine's
markdown is judged. This replaced the deterministic scorer as headline because
the hand-written references are over-segmented — lease_000 has 52 paragraph
blocks where a reader sees 12 logical elements, and engines that correctly
merge text into flowing prose could not match the fragments. The deterministic
scorer remains as the `--skip-judge` fallback and works as described below:
structural, not literal. Compare **which structural elements are present and
what they contain** — headings, list items, table blocks — plus reading order
via block sequence. Never string-equality against our markdown.

**Normalise:** heading depth, emphasis markers, list bullet style, table
alignment rows, and trailing punctuation are all ignored. `## Totals` and
`**Totals**` are the same heading.

**Why so tolerant:** the reference encodes *one* valid rendering. This is the
same mistake that cost us real numbers in the table leg — comparing HTML
dialects instead of structure scored a byte-perfect table 0.846 instead of
1.000. The same principle covers packaging: tables embedded as HTML
(paddleocr_vl) or tab-separated (mineru) are normalised to pipe rows, and an
answer wrapped entirely in a ```markdown fence (qwen3_vl) is unwrapped, before
any block is counted. A reference block also matches a consecutive run of
output blocks, because splitting one annotated paragraph across several lines
is segmentation, not structure.

**GT:** 57 documents — 40 human-authored markdown, 11 derived from other
annotations, and 6 (the invoices) transcribed from the page images after the
derived versions were found to use the source dataset's schema field names
(`invoice_date`, `item_net_price`) where the page prints "Date of issue" and
"Net price". No faithful conversion could match that vocabulary; see
documents/gt_limitations.json.

---

## Leg 3 — Table extraction

**Ask:** nothing extra from the layout parsers — tables come out of the default
conversion. The schema-capable VLMs are asked: *"Extract every table as a
Markdown table. Preserve rows, columns and merged cells. Output only the
tables."*

**Engines:** the eleven whose output is traversable (mistral_ocr_4, llamaparse,
marker, docling, mineru, deepseek_ocr, glm_ocr, paddleocr_vl, gemini_flash,
claude_sonnet_4_6, qwen3_vl — the general VLMs' default conversion preserves
tables as markdown) plus textract_structured through its cell grid. tesseract
and textract_text produce no table structure.

**Score:** the headline comes from the **table LLM judge** — cell accuracy
against a checklist built from the page image, for the same reason as the
markdown leg: comparing against our reference HTML punished dialect and
inherited the reference's own defects. **TEDS** — tree edit distance between
the extracted table tree and the reference, via `apted` — remains as the
`--skip-judge` fallback and as a diagnostic printed beside the judge score.

**Normalise:** both sides reduced to a presentation-free grid — `thead`/`tbody`
unwrapped, `th` treated as `td`, all attributes dropped. Without this an engine
is penalised for its HTML dialect rather than its structure.

**GT:** 45 documents. Provenance varies and is recorded per file in
`table_html_source`; the 5 references derived from schema field names rather
than printed page columns are the weakest and should be rebuilt or excluded.

---

## Leg 4 — Key-value pairs (LLM-judged)

**Ask:** *"Extract every printed label and its value from this document as a
flat JSON object of label → value. Use the label exactly as printed. Use null
where a printed field is blank. Output only JSON."*

**Engines:** textract_structured natively (its FORMS output); the
schema-capable VLMs by prompt. Fixed pipelines cannot compete.

**Score:** **field accuracy** — `(correct + blank_ok) / total_fields`, where
each field is judged by an LLM (Gemini 2.5 Pro) against the engine's output.

**How it works.** The checklist is the scalar (non-array) fields from the
document's `true_json` — arrays are excluded because they belong to the schema
leg, and including them would score the same values twice. Documents with
fewer than 3 scalar fields are marked not applicable (this excludes charts and
financial tables whose schemas are almost entirely arrays).

The judge is the same three-stage pipeline described under Leg 5 below, invoked
with `scalar_only=True`. Per-field verdicts are cached separately from the
schema leg (engine suffix `__kv`).

**Why LLM-judged, not deterministic.** The original scorer matched labels
deterministically: it normalised case and punctuation, stripped trailing colons,
and looked up the engine's output by label name. This penalised engines whose
label naming differed from the ground truth's schema identifiers — a receipt
with `"company"` in the GT scored zero when the engine returned `"Company
Name"` with the correct value. The LLM judge matches by *value*, not by field
name: it checks whether the correct value appears in the engine's output and is
attributable to the right concept, regardless of what the engine called it.

**Blank fields are the point.** A printed-but-empty field must come back
`null`; an engine that invents a value there is wrong in the way that matters
most in production. The judge distinguishes `blank_ok` (both GT and engine
agree the field is empty) from `invented` (GT is blank but the engine
hallucinated a value).

---

## Leg 5 — Schema JSON (LLM-judged)

**Ask:** the document's own `json_schema` injected into the prompt —
*"Extract the values from this document as JSON conforming exactly to this
schema. Use null for fields not present on the page. Output only JSON."* —
with the provider's JSON mode enabled where available.

**This is what settles the shape problem.** We supply the schema, so every
engine is asked for the same structure. An engine returning a different shape
has not expressed a preference — it failed to follow the instruction, which is
a real and reportable result.

**Engines (8):** gemini_flash with the provider's JSON mode enabled
(`response_mime_type="application/json"`); claude_sonnet_4_6 by prompt at
temperature 0 — the Anthropic API has no JSON mode, the current model rejects
assistant prefill, and we have not implemented forced tool use, so the prompt's
own "Output only JSON" instruction is the only enforcement (the judge matches
by value against the raw output, so fenced or imperfect JSON is not penalised
in scoring); qwen3_vl, glm_ocr and deepseek_ocr by prompt — glm_ocr through its
card's own Chinese extraction prompt, with the schema rendered as an *empty
example object* rather than a JSON Schema document, because that is the form
the card demonstrates and handing it the schema document makes it echo the
document back instead of filling it; mistral_ocr_4 via the OCR API's Annotations
feature and llamaparse via LlamaExtract — the vendors' own structured-extraction
paths; textract_structured through its FORMS/TABLES output, judged by the
same LLM judge as everyone else. marker, mineru, paddleocr_vl, docling,
tesseract and textract_text have no prompt interface and are not applicable.

**deepseek_ocr scores near zero here, and it competes anyway.** Its card
documents exactly two prompts (`Free OCR.` and the grounding markdown
conversion), and a prompt-sensitivity check over four phrasings returned an
empty string twice, `"No line breaks detected."`, and a markdown conversion of
the page. That check is what makes the zero publishable: it is the model
failing a task it has no head for, not us asking badly. Withdrawing an engine
because it scores poorly would be editing the result — the only engines
excluded from a leg are the ones with no way to be asked at all.

**Score:** **field accuracy** — `(correct + blank_ok) / total_fields`, judged
by an LLM (Gemini 2.5 Pro) per (document, engine) pair.

### The three-stage judge pipeline

This replaced the deterministic key-matching scorer that penalised engines
whose output format differed from the ground truth's schema identifiers. The
old approach required the engine to return values under the exact schema key
names — a fundamental unfairness, because engines legitimately use different
field names, different nesting, or no field names at all (Textract returns
FORMS key-value pairs, not JSON). The LLM judge matches by *value*, not by
field name.

**Stage 1 — Build the checklist (deterministic, no LLM call).**
Walk `true_json` + `json_schema` and create one `SchemaChecklistItem` per leaf
field: its dotted path, expected value, and whether the GT value is blank.
Cached once per document at `results/schema_judge_cache/`. Capped at 80 items;
larger schemas are evenly sampled. No LLM involved — the checklist is the same
for every engine on the same document.

**Stage 2 — Judge each field (one LLM call per document×engine).**
For each checklist item, first try local verification: strip both strings to
lowercase alphanumeric and check if the expected value is a substring of the
engine's output. This resolves the majority of fields (typically ~85–90%)
without an LLM call. The remaining undecided items are batched into one
Gemini 2.5 Pro call with structured JSON output (temperature 0). The model
sees the numbered checklist and the engine's full output and returns a verdict
per item.

For engines that provide structured output (Textract FORMS/TABLES), that
structured data is prepended to the raw text so the judge can see both. Engine
text is truncated to 60,000 characters.

**Stage 3 — Compute the score (arithmetic, no LLM).**
`field_accuracy = (correct + blank_ok) / total`. Verdicts and costs are cached
per (document, engine) at `results/schema_judge_cache/verdicts/`.

### Verdict categories

| Verdict | Meaning |
|---|---|
| `correct` | Value appears in the output and matches (same meaning, same digits; minor formatting like whitespace, casing, currency symbols or date format is OK) |
| `wrong` | A value for this field is present but incorrect (wrong digits, wrong name, wrong amount) |
| `missing` | No value for this field appears in the output at all |
| `blank_ok` | Expected value is blank/null AND the engine also left it blank or omitted it (both agree the field is empty) |
| `invented` | Expected value is blank/null BUT the engine filled in a value (hallucinated content for an empty field) |

### What the judge does NOT normalise

The judge prompt instructs: *"match by VALUE, not by field name. The engine may
use different field names, different nesting, or no field names at all."* Minor
formatting differences (whitespace, casing, currency symbols, date format) are
explicitly allowed. But one wrong digit in an identifier or amount = wrong. No
partial credit; this is where OCR actually fails in production.

**Aggregate per document type, never pooled.** A 43-field 1040 and a 4-field
receipt are not the same difficulty.

**GT:** 65 documents. 16 were extended from dataset-scoped schemas to the full
page; see `documents/gt_limitations.json`.

---

## Leg 6 — Geometry

**Ask:** *"For every word, return its text and bounding box as
`[{"text":…, "bbox":[x0,y0,x1,y1]}]`, normalised to 0–1. Output only JSON."*

**Engines (6):** textract_text and textract_structured natively (every block
carries a box and a confidence); tesseract via `image_to_data` (boxes
normalised 0–1); gemini_flash, claude_sonnet_4_6 and qwen3_vl by prompt. The
layout parsers (marker, docling, mineru, paddleocr_vl) and the OCR APIs
without native box output (mistral_ocr_4, llamaparse, glm_ocr, deepseek_ocr)
do not compete.

**Score:** **words located** — `located / reference_words`. A predicted word
counts as **located** when (1) it text-matches a reference word (exact
normalised match) and (2) each box contains the other's center point (mutual
center containment). Not raw IoU, because IoU punishes box-drawing convention:
tesseract boxes the full text line where SynFinTabs boxes the glyphs, and a
word placed exactly right scored ~0.45 for being twice as tall. Center
agreement scores position; raw IoU stays in the metrics as a diagnostic.

**Granularity handling.** Where the reference is line-level (SROIE), predicted
words are merged into lines by vertical overlap and lines pair by fuzzy text
similarity (SequenceMatcher, threshold 0.7) — otherwise multi-word lines could
never match a word-level engine.

**Auto-corrections.** Predicted boxes arriving in the 0–1000 grounding
convention (qwen, gemini emit it despite the 0–1 ask) are rescaled to 0–1, and
the rescale is flagged in the metrics (`rescaled_from_thousand`).
Portrait-vs-landscape axis swaps are detected and corrected
(`axes_swapped`).

**GT:** 17 documents (`<stem>_geometry_gt.json`), built from the source
datasets' own boxes and normalised 0–1 against the shipped images: SROIE 6
(line-level boxes), FUNSD 6 (word-level; source coords are LayoutLM 0–1000,
divided by 1000 — dividing by image size corrupts x), SynFinTabs 5
(word-level, from cell annotations). Granularity is recorded per sidecar;
compare engines within a document, not across sources.

---

## The document judge (silver-GT pages) — an estimate, not a leg

The judge estimates text accuracy where there is no gold ground truth. It does
not appear in `--legs`, and its scores are labelled as judged in the report and
kept separate from the deterministic text scores.

**Ask:** nothing extra from the engine — it evaluates the engine's default
conversion output, same as leg 1.

**Engines:** all 14.

**Applies to:** the 11 documents whose machine-generated ground-truth text is
too unreliable to score against (6 newspaper, 5 report pages — 42–66% of the
page missing from GT). These get the judged estimate instead of a text-leg
score.

**Score:** **element accuracy** —
`(correct + 0.5 × partial) / total_checklist_items`, judged by an LLM.

### The three-stage document judge pipeline

**Stage 1 — Build the checklist (one LLM call per document, cached).**
A capable model (default: `gemini-2.5-pro`) reads the page *image* and
catalogues every structural element a reader would recognise: headings,
paragraphs, list items, table rows, key-value pairs. The checklist is capped
at 40 items (evenly sampled beyond that). Cached once per document; every
engine is measured against the identical checklist.

**Stage 2 — Judge each engine (one LLM call per document × engine, cached).**
The verdict model (default: `gemini-3.6-flash`) sees the checklist and the
engine's raw text output. For each item it returns `correct`, `partial`, or
`missing`. Cached per (document, engine) pair.

**Stage 3 — Compute the score (arithmetic, no LLM).**
`element_accuracy = (correct + 0.5 × partial) / total`.

**Known limits:** the checklist itself is model-generated from the image, not
from human annotation. Scores on these documents are labelled as judged and
separated in the report — they are directional, not definitive.

**GT:** 11 documents. No text GT; the page image is the reference.

---

## Known issues and methodology considerations

Every benchmark has blind spots. These are ours, stated so the numbers can be
read with the right caveats.

### The LLM judge is not neutral

The schema and key-value judge is Gemini 2.5 Pro. When it scores
`gemini_flash` output, judge and contestant are from the same model family.
Stage 1 (checklist building) is deterministic and engine-blind, which removes
one path for bias. Stage 2 (verdict calls) is not engine-blind — the judge
sees the engine's output — but it makes binary observations ("is this value
present"), not holistic ratings, which limits but does not eliminate
same-family advantage. This is disclosed in the results; we have not measured
the effect size.

### Local substring verification is binary

The local verification step (`_verify_locally`) resolves ~85–90% of checklist
items without an LLM call by checking whether the expected value (stripped to
lowercase alphanumeric) is a substring of the engine output. This saves cost
but cannot distinguish "correct with different formatting" from "coincidental
substring match". In practice the false-positive rate is negligible because GT
values are specific (amounts, identifiers, names), but it means some fields
never reach the LLM even when a human reviewer might disagree.

### Large schemas are sampled

The checklist is capped at 80 items (schema leg) or 40 items (document judge).
Documents with more fields are evenly sampled. This means a field that was
sampled out is untested for all engines equally — it shifts coverage, not
ranking. The cap exists because Gemini structured output degrades above ~100
items.

### Silver GT documents are judge-scored, not text-scored

11 documents (6 newspaper, 5 report pages) have machine-generated ground-truth
text too unreliable to score against. These are evaluated by a separate
document judge (`evaluation/judge.py`) that builds a checklist from the page
*image* — model-generated and therefore imperfect. Scores on these documents
are labelled as judged and separated in the report. They are directional, not
definitive.

### Ground truth coverage is uneven across legs

| Leg | Documents with GT | What lacks GT |
|---|---|---|
| text | 82 of 93 | 11 silver-GT docs (judged instead) |
| markdown | 57 of 93 | receipt, most medium-tier docs |
| tables | 45 of 93 | docs without tabular content |
| key_value | ~60 of 65 with structured GT | chart/financial_table excluded (< 3 scalar fields) |
| schema | 65 of 93 | docs without `_structured_gt.json` |
| geometry | 17 of 93 | most documents (only SROIE, FUNSD, SynFinTabs) |
| (document judge) | 11 of 93 | the silver-GT documents — an estimate, not a leg |

Engines are never penalised for missing GT — the doc is reported "not
applicable". But sparse GT means some legs rest on fewer data points than
others, and geometry in particular is narrow.

### Receipt table GT is hand-transcribed

The 5 receipt table references (`sroie_000` through `sroie_005`, excluding
`sroie_004` which has no line items) were transcribed by reading the receipt
images. Interpretation decisions were made: which columns to include, whether
tax codes belong in the amount column, how to combine multi-line item
descriptions. These are recorded as `table_html_source: "hand-transcribed"` and
should be reviewed against the images before treating the scores as definitive.

### Not every engine gets the same question format

Engines differ in how they receive the schema: general VLMs get the JSON Schema
document in the prompt; `glm_ocr` gets an empty example instance (its model
card's documented format); Textract gets no prompt at all and competes through
its FORMS output. `mistral_ocr_4` and `llamaparse` use their vendor's own
structured-extraction features. This is by design — each engine is asked in the
way its documentation prescribes — but it means the "schema" leg is not
purely measuring extraction accuracy. It also measures how well each engine
handles its own prescribed interface. That is a real-world factor, not an
artefact, but it should be understood.

### Cost is per pass, not per leg

Text, markdown and tables all come from one default conversion call — all three
report that call's cost. Schema and key_value each need their own prompted
call. Geometry needs its own. Do not sum per-leg costs to get a run total; you
would count the shared call multiple times. The judge's LLM calls are an
additional cost not reflected in per-engine pricing (tracked separately as
`judge_cost_usd` and `kv_judge_cost_usd`).

---

## Run structure

Legs 1–3 all score from one default conversion call per (engine, document).
Legs 4–5 each need a schema-prompted call. Leg 6 needs its own geometry call.
The document judge scores the same default conversion output as legs 1–3, so
it costs no extra engine call — only its own LLM calls.
The judge adds one Gemini call per (document, engine) pair for schema,
key_value, tables, markdown, and the document judge, minus items settled
locally.
