# Rescoring plan and open items

> Historical planning/audit record. Current status and final scores are in
> [handoff-2026-10-01.md](handoff-2026-10-01.md), run `20261002T095730Z`.
> Prototype paths and pending-work counts below describe the earlier stage;
> scratch artifacts and superseded rescoring runs have since been removed.

Working document for the rescore of run `20260810T130609Z`. Everything here was
decided or found in the 2026-09-30 session; each item says what was decided,
what evidence it rests on, and whether it is done. Dated 2026-09-30.

Ground rules that apply to all of it: engine outputs from the run are reused,
nothing under `results/` is edited in place, old scores are superseded (moved
aside) rather than deleted, no paid call is made without an explicit go, and
every number in the documents must be reproducible from the saved files.

## 1. Decisions per leg

| leg | old method (the run) | decided method | status |
|---|---|---|---|
| text | unique-GT-word coverage after `normalize_text` | unchanged | reproduces exactly on every document checked |
| markdown | image checklist (Gemini 2.5 Pro, cached) judged per engine by gemini-3.6-flash | same checklist, verdicts by **code**: typed blocks, most-similar block, rubric applied literally (correct / wrong_type / partial / absent); key-value = label and value both present | prototype done (`docs/deterministic/`), harness code not written |
| tables | image cell checklist (cached) judged per engine; empty cells counted correct | same checklist, verdicts by **code**: grid lookup anchored on row-mates and column-mates, transposed tables accepted, empty cells excluded | prototype done, harness code not written |
| key_value | GT value searched as a substring of the whole output, leftovers to a judge | **stage 1** once per document: gemini-3.1-pro-preview reads the image + our field list and returns each field's printed label(s) and whether it is printed as a pair (no values asked for). **Stage 2** code: parse pairs (JSON leaves; Textract FORMS + 2-column tables), split empty-valued "Label: value" keys at the colon, key = field name or printed label (one-typo tolerant), value **exactly equal** (numbers as numbers, dates as dates, strings normalised — no containment). Denominator = fields printed as pairs | stage 1 run for sroie_000 only; prototype scorer done; 64 documents pending (~$2) |
| schema | substring test, then judge | same stage 2 as key_value on all fields, keys by path; truncated JSON salvaged as far as it parses; textract_structured removed (no schema interface) | prototype done |
| geometry | mutual centre containment on text-matched words, line merging for SROIE | unchanged for word-box engines. Layout parsers (mineru, paddle, deepseek) = **N/A** on the word leg + a block diagnostic (reference inside a block that contains its text, reported with words-per-box). marker rescored as a line-box engine once its adapter is fixed | prototype done (`docs/geometry-block-level.html`), harness code not written |
| document judge (11 silver pages) | image value checklist, substring then gemini-3.6-flash, 10 of 14 engines | same, all 14 engines, section printed | needs ~150 calls (~$0.5) |

Evidence pages (all built from the saved run, no model calls):
`docs/how-we-measure-{sroie_000,invoice_000,funsd_000}.html` (every engine, every
leg, prompt → raw output → parsed → reference → verdicts → arithmetic),
`docs/geometry-how-scored-{sroie_000,funsd_000}.html` (box by box),
`docs/deterministic/index.html` (code vs judge, all engines, tables and markdown),
`docs/geometry-block-level.html`, `docs/geometry-in-pictures.html`,
`docs/kv-schema-v3-sroie_000.html`, `docs/scoring-defects-to-fix.html`.

## 2. What the code-vs-judge comparison showed (tables and markdown)

Per-document means over the run's documents; agreement = same verdict on the same item.

| leg | engine | judge | code | agreement |
|---|---|---|---|---|
| tables | claude 0.971→0.871 · textract 0.867→0.815 · qwen 0.837→0.788 · llamaparse 0.822→0.765 · paddle 0.803→0.747 · mistral 0.868→0.737 · deepseek 0.828→0.736 · gemini 0.932→0.723 · marker 0.764→0.684 · glm 0.737→0.673 · mineru 0.816→0.660 · docling 0.835→0.586 | | | 69–89 % |
| markdown | claude 0.860→0.800 · qwen 0.856→0.798 · mistral 0.804→0.740 · llamaparse 0.773→0.739 · docling 0.739→0.721 · gemini 0.745→0.680 · deepseek 0.707→0.680 · paddle 0.746→0.668 · marker 0.739→0.637 · glm 0.542→0.517 · mineru 0.361→0.308 | | | 72–87 % |

Dominant cause of disagreement, both legs: the judge credited structure that is not
there — cells on pages where the engine produced no table (docling 14 of 45 pages,
gemini 10, mineru 9; 344/409 such cells credited for docling), rows flattened into
one cell, headings written as bold table rows, checkbox lists as one line, receipts
left in code blocks. Only 3 (engine, page) cases across all engines were tables our
extractor failed to read.

## 3. Policy decisions (applied uniformly; change here if you disagree)

- Rubric as written: a heading rendered as a bold table row, a list rendered as
  `☐ a ☐ b` on one line, a receipt left in a ``` block, a form the checklist calls a
  table rendered as lines → wrong_type / misplaced.
- Tables: split sub-rows (an engine puts a multi-line cell on extra rows) → still to
  add a two-row window; today scored partial/misplaced.
- Tables eligibility: the leg applies wherever the image checklist has non-empty
  cells (funsd_000 has 13 and the run stored "n/a: no table reference"). Decide.
- key_value denominator = fields the page prints as label+value pairs; a masked
  card number must match as printed; dates and amounts must match exactly
  (`15/01/2019 11:05:16 AM` satisfies `datetime`, not `date`; `193.00 SR` is not `193.00`).
- Geometry: word leg for word-box engines only; block diagnostic never ranked
  against it; granularity (median words per box) printed beside every block number.

## 4. Defects list status (`docs/scoring-defects-to-fix.html`)

| # | defect | resolution | status |
|---|---|---|---|
| 1 | KV/schema score value presence, not extraction | alias pass + exact match | agreed, running soon |
| 2 | verdict caches reused after outputs changed | no verdict caches survive; code scoring | resolved by design |
| 3 | mineru markdown scored on flattened text | code scoring reads `structured_output["markdown"]` | in prototype: not yet (compares on the run's input); harness: to do |
| 4 | empty table cells counted correct | excluded | done in prototype |
| 5 | table judge credits pages with no table | code | done in prototype |
| 6 | deepseek geometry: label stored as text | re-pair boxes with the text after each `<\|det\|>` from `raw_model_output` | done in prototype |
| 7 | paddle geometry: no boxes reached the scorer | use `layout_blocks` | done in prototype |
| 8 | block engines in a word-level leg | N/A + block diagnostic | done in prototype |
| 9 | document judge on 10 of 14 engines, section never printed | run the 4, print | pending (paid) |
| 10 | failures scored 0.0 | error rows (already so for Claude's 17 credit-balance rows) | harness: to do |
| 11 | bookkeeping | see §7 | to do |

New since the list was written:

- **marker geometry adapter**: HTML tags left in `text_regions` text broke text matching
  (`<h2>tan chay yee</h2>`); with tags stripped marker scores 0.54/0.43 on the two
  receipts where it emits regions. Its OCR pass (`OCRConverter`) crashed on 5 of 17
  documents (`TypeError: 'NoneType' object is not iterable`) and captured 1–31 lines
  on the others, so the line polygons it can produce were never stored.
- **SROIE geometry GT**: 15 of 297 reference lines have a box far wider than their
  text (`NO 2 & 4` boxed as the whole line `NO 2 & 4, JALAN BAYU 4,`; `LOT 13`,
  `NO. 5-7`, `22A-1`, single digits). Every engine loses those lines. Fix the 15 from
  the image (shrink box to the words or extend the text).
- **Qwen geometry**: the Modal app divides `bbox_2d` by 1000 on the assumption of a
  0–1000 convention; Qwen's numbers follow no consistent frame (y up to 2036 on
  one page, 704 for the last line of another). No conversion rescues it (≤ 0.006
  under four frames); the assumption should be stated as unverified.
- **Claude geometry**: re-prompted with pixel coordinates and the image size stated
  (its computer-use convention), 2 pages, $0.12: 0.000 on both, boxes drifting the
  same way (y scale 1.33 on the receipt, +0.12–0.14 offsets on the form). The leg's
  0.03 is fair. Replies in the scratchpad, not `results/`.
- **Geometry matcher, repeated values (fixed 2026-10-01 in `evaluation/geometry_leg.py`,
  regression test added)**: `_match_lines` ranked candidate windows by text
  similarity first, so an exact `193.00` anywhere on a receipt beat a slightly
  misread `1193.00` in the right place; with six identical amounts and each word
  usable once, every later reference shifted one instance down. Now, among
  windows with text ratio ≥ 0.7, one that overlaps the reference box wins. Effect
  on stored outputs (17 docs, means): PP-OCRv5 0.800→0.815 (sroie_000 0.783→0.913,
  sroie_001 0.838→0.950), Tesseract 0.746→0.749, marker 0.136→0.139, MinerU
  pipeline 0.113→0.114; Textract, Gemini, Claude, Qwen unchanged. No engine drops.
  The reference run's stored geometry rows predate this; they are rescored with
  everything else.
- **glm_ocr / deepseek_ocr key_value**: the leg was added to them after the run; no
  key_value pass exists in the run. Needs a Modal run to score them.
- **GLM schema truncation**: its own whitespace loop hits the 4096-token cap; parsed
  as far as it goes. A proper fix is a re-run.

## 5. Geometry: which engines can give finer boxes (corrected)

Checked against the deployed pipelines and the run's own diagnostic dumps:

| engine | what the deployed pipeline returns | finer boxes possible? |
|---|---|---|
| paddleocr_vl | PP-DocLayoutV3 blocks recognised by the 0.9B VLM; the v1 result has `parsing_res_list` + `layout_det_res` only — **no OCR sub-result** (`_debug_result_shape` in every stored call) | not from this pipeline; PP-OCRv5 text detection would be a different engine |
| mineru | `two_step_extract`: layout detection → per-block recognition → ContentBlocks | not from this pipeline; MinerU's pipeline backend would be a different engine |
| deepseek_ocr | `<\|grounding\|>` markdown: one box per layout element; model card documents no line/word mode | no |
| marker | `OCRConverter` produces line polygons (`Line` nodes with text and bbox) — the adapter walked them wrong and crashed on 5 pages | **yes** — fix the walker, re-run the OCR pass on the 17 geometry documents on Modal |

So the "re-run four engines for word boxes" reduces to marker. The other three
stay block-level by design.

## 5a. Two line-box engines added and run (2026-10-01)

Both are the vendors' own text-line pipelines, run as they ship them, added
beside (never instead of) the VLM engines. Geometry leg only so far; the text
output is stored and can be scored later without a re-run.

| engine | what it is | pinned | GPU | measured |
|---|---|---|---|---|
| `paddleocr_ppocrv5` | PaddleOCR 3.7.0 general OCR pipeline: PP-OCRv5_server_det (one polygon per text line) + PP-OCRv5_server_rec; doc orientation / unwarping / text-line orientation off; `return_word_box` on → word boxes from `text_word` + `text_word_boxes`; det limits read back (64 / min, max_side 4000) | det `ca867c89`, rec `b26c3587`, paddlepaddle-gpu 3.2.1 cu126 | A10 | 0.5 GPU-s/page, 21 s cold start; run `20261001T102808Z` |
| `mineru_pipeline` | MinerU 2.5 **pipeline** backend (`-b pipeline -m ocr`, lang en): DocLayout-YOLO + text-line OCR + formula + table models; line boxes from middle.json lines, spans kept too; markdown and HTML tables from the same pass | `mineru[pipeline]==2.5.4`, weights `PDF-Extract-Kit-1.0@ed6b654c`, torch 2.6.0 cu124 | T4 | 1.7 GPU-s/page, 33 s cold start; run `20261001T102236Z` |

Files: `modal_apps/paddleocr_ppocrv5_app.py`, `modal_apps/mineru_pipeline_app.py`,
two `engines.py` entries (`default=False`); registry derives the Modal runner
entries. Tests: `tests/test_registry_and_legs.py` 16 passed. Commands:
`modal deploy modal_apps/<app>.py`; `python run_benchmark.py --engines <e>
--legs geometry --skip-judge --extends 20260810T130609Z --docs <17 stems>`.

Results on the 17 geometry documents (`docs/geometry-line-engines.html`,
per-document tables; box-by-box on `docs/geometry-how-scored-*.html`):

| engine | current rule (word/segment) | line-level rule |
|---|---|---|
| paddleocr_ppocrv5 | **0.800** (2nd, after Textract 0.930) | **0.976** (1st; Textract 0.969) |
| mineru_pipeline | 0.113 | 0.496 |

MinerU pipeline's zeros are pages its layout stage classified as one table
(sroie_003, fintab_000/003/004 → one HTML block, no line geometry) and the
receipt totals its OCR detector never found; where it emits lines they sit
well (0.93–1.00 on three pages). Line-level rule = reference merged into lines
with the scorer's own merge, fuzzy text ≥ 0.7, mutual centre.

Next experiments (one variable each, not done): T4 for Paddle (sm_75 with
the cu126 wheel); Paddle's `text_rec_batch_size` and list-input `predict`
for throughput; MinerU with `table_enable=False` to see whether the one-table
pages then yield lines (changes what the engine is — report separately if
tried). Claude pixel-prompt variant: tested, no gain, closed.

## 6. Runs still needed (all need an explicit go)

| run | scope | cost | status |
|---|---|---|---|
| key_value stage 1 (alias labels) | 64 remaining documents, 1 Gemini 3.1 Pro call each | ≈ $2 | pending |
| document judge | 11 pages × 14 engines (re-judge all on one scorer version) | ≈ $0.5 | pending |
| marker OCR pass with fixed adapter | 17 geometry documents, Modal | minutes of GPU | pending |
| glm_ocr, deepseek_ocr key_value pass | 65 documents, Modal | small | pending |
| GLM schema without truncation | 65 documents, Modal, needs an output-cap change | small | undecided |
| Claude/Gemini/Qwen geometry in native conventions | not needed: Claude tested, no gain; Gemini/Qwen already get their 0–1000 | — | closed |

## 7. Harness work (code, no cost)

1. `evaluation/`: markdown and table scorers from `det_all.py`; key_value/schema scorer from `kv_schema_v3.py` with exact matching; alias cache under `results/kv_alias_cache/`; geometry block diagnostic + N/A; adapters for deepseek (text after `<\|det\|>`), paddle (`layout_blocks`), marker (strip HTML; fixed OCR walker).
2. `run_benchmark.py --score`: new timestamp with manifest `rescore_of`, scorer version, alias model; old key_value/schema/markdown/tables/geometry rows moved to `<ts>_results.superseded.jsonl`; verdict caches (744 + 627 + 530 files) moved to `superseded/`; stage-1 checklists kept and stamped with the prompt version that built them; `report(old, new)` combines.
3. Error rows for every failed call (never 0.0); `n_fail`/`n_not_applicable` semantics documented.
4. Tests: pin sroie_000 (all engines, all legs) and the Claude tables/markdown comparison numbers.
5. Documentation: caps (40 markdown / 60 tables / 40 doc judge / 80 schema), the doc judge's "discrete values only" rule, which documents were sampled, judge cost split (stage 1 ≈ $2.7 once; stage 2 ≈ $10.5 per full pass at the code's prices), the Gemini price table (code says 3.6-flash $1.50/$7.50; list price found 2026-09-30 $0.75/$3.75), README numbers (85–90 %, "6 geometry engines"), textract out of schema, KV engine count.

## 8. Still to audit

- The 161 stage-1 checklists have never been spot-checked against the page by a
  person, and they do not record the prompt version that built them.
- Markdown checklists: paragraph fingerprints are the first ~50 words; whether
  Gemini's element typing (heading vs bold row, list vs checkbox) is consistent
  across documents has not been checked.
- Text leg: punctuation stays attached to words (§5.13 of the audit); not changed.
- The new-engine integrations (16 engines added 2026-09-30) have not been run.
- The SROIE GT box/text mismatches (15 lines) — fix before the geometry rescore.
- KV/schema GT for 17 documents was extended by Claude Opus 5 (recorded in
  `field_provenance`); the alias pass will surface disagreements between the page
  and those values where an engine reads differently — review them when they appear.
