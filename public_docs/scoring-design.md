# Scoring design

How each leg is scored, as implemented in `evaluation/` and used for published run
`20261020T120222Z`. [Results and commands](../README.md) ·
[Engine versions and pricing](engines-and-pricing.md).

The published rescore retains the original Textract KV adapter. The row-label
change from `131394ea` was reverted after it mislabelled repeated date/amount
columns. Only unreturned blank reference fields change the scored KV denominator.

## Overview

Six legs, each a separate task with its own reference and score. There is no
combined score across legs.

| Leg | Question | Headline | Scorer version |
| --- | --- | --- | --- |
| Text | Did the engine read the words? | Unique reference-token coverage | `evaluation/text_leg.py` |
| Markdown | Is the page structure right? | `(correct + 0.5 × partial) / elements` | `markdown-structure-v3-spans-reflow` |
| Tables | Are table cells right? | Value credit / counted reference cells | `gold-table-values-v9-numeric-diagnostics` |
| Key/value | Did it pull printed fields? | `(correct + blank_ok) / counted fields` | `shared-fields-v9-unreturned-blanks-not-counted` |
| Schema | Can it fill a given schema? | `(correct + blank_ok) / (fields + extra fields)` | `shared-fields-v1` |
| Geometry | Are the boxes in the right place? | Located reference units / reference units | `geometry-v8` |

**Which engines take part.** `engines.py` declares, per engine and leg, whether
the engine answers natively (its normal output), when prompted, or not at all
(N/A). `evaluation/legs.py` (`plan_passes`) turns that into calls: all native
legs share one default call; each prompted leg is its own call, except that legs
with an identical prompt and response format share one (DeepSeek-OCR's Markdown,
tables and geometry share one call). An N/A leg is not a zero. A low score never
makes an engine ineligible.

Engines with scored pages in the published run:

| Leg | Engines |
| --- | --- |
| Text | 15: every engine in the run except PP-OCRv5 (MinerU pipeline on 17 pages) |
| Markdown | 12 (MinerU pipeline on 17 pages): Sonnet, DeepSeek-OCR, Docling, Gemini, GLM-OCR, LlamaParse, Marker, MinerU, MinerU pipeline, Mistral OCR 4.1, PaddleOCR-VL, Qwen3-VL |
| Tables | 12: Sonnet, DeepSeek-OCR, Docling, Gemini, GLM-OCR, LlamaParse, Marker, MinerU, Mistral OCR 4.1, PaddleOCR-VL, Qwen3-VL, Textract structured |
| Key/value | 5: Textract structured, Sonnet, Gemini, Mistral OCR 4.1, Qwen3-VL |
| Schema | 7: Sonnet, DeepSeek-OCR, Gemini, GLM-OCR, LlamaParse, Mistral OCR 4.1, Qwen3-VL |
| Geometry | 15: Sonnet, DeepSeek-OCR, Docling, Gemini, LlamaParse, LlamaParse granular, Marker, MinerU, MinerU pipeline, PP-OCRv5, PaddleOCR-VL, Qwen3-VL, Tesseract, Textract text, Textract structured |

**Retries.** A call whose reply is empty, a generation loop, or cannot be
parsed into the leg's format (recorded as `reply_condition`) gets one retry on
the hosted engines, whose output can differ between calls. The self-hosted
Modal apps decode at temperature 0 and Docling is a fixed local library, so
their replies are kept as returned; LlamaParse is not retried because its
service version cannot be pinned. A retry is collected as its own run and
merged with `--rescore <current>,<retry> --best-attempt`: the retry and the
original answer are both scored, the attempt with the higher mean over the
retried legs is published (ties keep the original), only the published call is
kept, and the manifest's `retries` entry and the row's `attempt` field record
the choice.

**References.** Source images are pinned by `documents/manifest.json` and
downloaded separately. Text and geometry references ship as sidecars beside each
image; accepted table and structured references live in
`documents/table_references/gold/` and `documents/structured_references/gold/`.
Markdown catalogues and judge rulings live under `results/`. Scoring does not
regenerate catalogues or replace accepted references with proposals.

**What was sent.** Each call's saved `.call.json` holds the prompt sent, response
format, provider reply, cost and latency, with these gaps:
- Generation settings (temperature, output cap) are recorded on 414 of 2,636
  calls (Sonnet and Gemini).
- The provider reply is empty for Mistral OCR, Docling, Tesseract and LlamaParse;
  their output is kept in `structured_output`.
- `prompt_sent` is empty on Sonnet's and Gemini's default (transcription) calls.
- For Mistral OCR 4.1 and LlamaParse, `prompt_sent` on the schema call shows the
  generic schema prompt, but their runners send only the JSON schema through the
  vendor's extraction feature (Mistral `document_annotation_format`, LlamaExtract
  `data_schema`); no prompt text is sent.

---

## Text

**What it measures.** Whether the engine read the words on the page.

**Pages and engines.** 82 pages with usable text references; 15 engines (MinerU
pipeline on 17 pages).
The 11 pages without a reliable text reference are covered by the separate
judged-text estimate at the end of this document.

**How each engine's answer is read.** The plain text of the call that serves the
text leg (`raw_text`): the engine's default conversion, or its text prompt where
one is declared (GLM-OCR, DeepSeek-OCR, Marker, Qwen3-VL).

**How it is scored.**

```
engine text ─┐
             ├─► render to plain text (Markdown/HTML formatting removed, entities decoded),
reference ───┘   Unicode NFKC, lowercase, whitespace collapsed
                     │
                     ▼
     coverage = unique reference tokens found in the output / unique reference tokens
```

**Score and limits.**
- Coverage ignores repetition, extra words and reading order.
- CER: character insertions, deletions and substitutions / reference characters.
  WER: the same on words. Lower is better; zero is an exact normalized match;
  added text can push either above 1.
- Word precision/recall/F1, normalized edit similarity, output/reference length
  ratio and a repetition flag are supporting metrics; they do not change coverage.
- Reference omissions or reading-order differences can raise CER/WER even when
  the words were read correctly.

---

## Markdown

**What it measures.** Whether the page's structure (headings, lists, tables,
key/value lines, paragraphs) is rendered correctly.

**Pages and engines.** 92 pages with shared structural catalogues; 12 engines (11
on all 92 pages, MinerU pipeline on 17). Docling, LlamaParse, MinerU, MinerU
pipeline, Mistral OCR 4.1 and PaddleOCR-VL answer natively; Sonnet, Gemini,
GLM-OCR, DeepSeek-OCR and Qwen3-VL through a Markdown request; Marker through its
converter setting.

**How each engine's answer is read.** A Markdown rendering the engine stored
(`structured_output.markdown`) if present; otherwise the call's `raw_text` when
the engine answers Markdown natively or by prompt (`markdown_for_scoring`,
`evaluation/markdown_leg.py`).

**How it is scored.** Each page has a cached catalogue of up to 40 expected
elements. Older catalogues drop elements marked uncertain; no element in the
current catalogues is marked uncertain. Accepted image-reviewed catalogues
check source hashes; a missing catalogue makes the page N/A. A catalogue can
establish eligibility without a full Markdown reference.

```
catalogue element ──► 1 find its content in the output ──► 2 check the block type ──► 3 credit once
                        (exact or boundary-aligned span)     it sits in                   (1, 0.5 or 0)
```

1. Find exact occurrences and boundary-aligned partial spans, comparing the
   whole expected content. Normalize Unicode, case and typography whitespace;
   keep meaningful numeric and identifier differences.
2. Check the actual Markdown/HTML block type. Compatible content similarity
   ≥0.85 earns 1; ≥0.50 earns 0.5. Full credit also requires matching numeric and
   identifier signatures. Lower-similarity character matches need token overlap.
3. Assign credited spans once. Separate references may use disjoint spans of
   one paragraph; adjacent paragraphs may satisfy one reference. Wrong-type
   evidence earns zero but remains visible. Content already used by another
   element leaves the verdict absent; its explanation says it was already allocated.

Structural rules:
- ATX/Setext headings and HTML heading tags count. Bold emphasis, table-cell text
  and isolated glyphs do not establish headings or list markers.
- Tables require valid Markdown separators or HTML tables. One outer Markdown
  fence may be removed as transport packaging; JSON/Python fences remain code.
- A key/value pair may appear in a paragraph, heading, list, up to 8 adjacent
  paragraphs, or one table row. A single-record table may pair headers with
  values below. Values are never joined across table rows.
- `table_header` requires column-header structure; a row-label `<th>` is not
  enough. Generic `table_row` references accept header or data rows.
- Full table-row credit requires ordered cell matches. Empty padding is ignored.
  Split item rows are not merged; a fragment row earns partial credit only when
  its similarity reaches 0.5.
- Repeated side-by-side column groups may reflow into a narrow table only with
  matching headers and unique first-cell anchors in the same table. Values stay
  with their items; duplicate anchors or changed amounts cannot earn full credit.
- Content inside an oversized heading/list entry or a wider table row can receive
  partial structure credit instead of being called absent.

**Score and limits.**
- Headline: `(correct + 0.5 × partial) / catalogue elements`.
- Verdicts show matched types, token ranges, normalized excerpts and similarities.
- Markdown-text CER/WER compare the answer's rendered text with a trusted
  full-page transcript. Pages with a full Markdown reference fall back to it;
  catalogue-only pages without a transcript have no CER/WER. With a full
  Markdown reference, block recall counts complete compatible matches; block
  precision requires at least 85% of an output block's tokens to be covered by
  complete matches; reading order uses those matches' positions. Sampled
  catalogues alone cannot establish full-page block precision or reading order.
  None of these change the headline.

---

## Tables

**What it measures.** Whether table cell values are returned under the right
rows and columns.

**Pages and engines.** 42 pages with human-reviewed gold tables; 12 engines.
Sonnet, Gemini, GLM-OCR and Marker answer a dedicated table request; DeepSeek-OCR
and Qwen3-VL answer it in a call shared with other legs; the others use their
default conversion.

**How each engine's answer is read** (`evaluation/table_leg.py`,
`evaluation/table_extractor.py`):
- Engines that store native tables (`structured_output.tables[].html`, e.g.
  Textract) are scored on those HTML tables.
- Otherwise every table is extracted from the answer text using the engine's
  declared `table_format` in `engines.py` (Markdown or HTML), with tab-separated
  text recognised when present. An engine without a declared format (Textract
  structured) is scored on its native tables only. Native HTML is preferred over
  a Markdown export of the same answer.
- A reply that hides a table where a consuming pipeline would not find it,
  such as a Markdown table inside a JSON string, scores as that pipeline would
  score it: no table. No reading rule is added after inspecting an engine's
  reply.
- Every result row of every leg records `reply_condition`, never scored: `ok`;
  `empty`; `degenerate` (an unterminated HTML table, a line with content
  repeated 10+ times in a row, an empty table row repeated 100+ times, or a
  3-40 character unit with a letter or digit repeated 20+ times, so dot leaders,
  separator rows, `&nbsp;`, tags and JSON literals do not count, and a run counts only when it exceeds twice
  the number of times the page text reference contains that line or unit);
  `no_table` (tables: readable text without a table); `no_pairs` (key/value
  and schema: a readable reply with no field pairs); `unparsed` (geometry: no
  usable boxes or blocks); `truncated` (key/value and schema: pairs salvaged
  from a cut-off reply). Rows are scored as returned; the report lists the unusable replies
  under each leg.

**How it is scored.**

```
engine tables ──► logical cells (merged cells once) ──► match to gold rows/columns ──► cell credit
                                                        (each returned span used once)
```

| Cell comparison | Credit |
| --- | --- |
| Exact normalized text or numerically equivalent amount | 1 |
| Nonnumeric text similarity ≥0.5 | the similarity value |
| Printed checkbox (`□`, `☑`) against any recognised rendering of the same state (`[x]`, `✓`, `[ ]`; an empty returned cell is unchecked) | 1 |
| Lower similarity, wrong number, wrong checkbox state or missing cell | 0 |

Currency formatting may differ. Signs, percentage units and leading-zero
identifiers stay significant. A checkbox is compared as a state, using the
same true/false sets as the key/value leg, only when the gold cell is a
checkbox glyph. A blank earns credit from a returned blank, or from an empty
component established by headers and a combined column layout or by a returned
merged cell spanning several gold columns. A gold blank with no returned slot
at all (`blank_not_returned`) is not counted, as in key/value: the blank may be
a layout artefact, and a missing row costs its printed values only. A value
returned in a blank slot is wrong. Gold header cells marked
`data-unprinted="true"` (the `Line / Description / Amount` row on the five
1040 pages, which the form does not print) keep column identity for alignment
and are never scored, returned or not. Empty output against a nonempty gold
table scores zero.

Matching keeps logical row/column associations and uses each returned span
once; it does not require identical physical grids:
- Unique descriptions can anchor adjacent continuation rows. Another item, a
  conflicting label, a label that is not part of this description, a blank row
  or completion of the record ends the group. A
  unique item code can link the preceding amounts row to its description;
  unrelated codes in separate cells remain extras.
- Unique headers, including multi-tier and spanning headers, identify added,
  missing, reordered or split columns. Merged values split into ordered,
  disjoint spans. Repeated ambiguous headers cannot donate a distant column.
- Parenthetical header qualifiers may move around the field name: `(RM) RSP`
  and `RSP (RM)` identify the same column and receive full header credit.
  The complete field name, qualifier content and their respective word order
  remain significant. This normalization applies only to headers.
- An explicit reference colspan can collect returned cells from the mapped
  columns it covers, in source order. For example, a three-column `TOT QTY: 45`
  cell can match `TOT QTY:` and `45` in separate covered slots. It cannot take
  a value from the amount column outside that span.
- Descriptions, record IDs and ISO dates identify records. A label/value record
  may share a printed row with another record, or repeat its ID before the
  amount. Expected amounts never select the record (choosing which returned
  table goes with which gold table does use all cell values). Multiple independent value
  fields still need column evidence; unresolved layouts keep positional comparison.
- Repeated header groups can reflow. Repeated header/unit bands define sections;
  unique row labels can locate section boundaries when returned headers are
  absent. Units keep column order and can establish a Notes column combined into
  a label. Indented slots under one spanning unit header can map to a combined
  field; its empty components keep blank credit without inventing a record.
- A blank leading label column may fold into its neighbour when every named
  header agrees and their body slots are mutually exclusive. A row label may
  move into an adjacent empty slot only when the exact label and an actual empty
  counterpart are returned. Populated fields cannot swap this way.
- Matching sections can cross table boundaries. Identical-header chunks may join;
  repeated data rows remain extras. Transposition requires header evidence.

Shipping pages: the whole page is scored against both the gold and the frozen
original reference; the higher score is used, gold on ties, and both comparisons
are kept. This applies to every engine and never mixes cells from the two.

**Score and limits.**
- Headline: summed cell credit / counted reference cells, where counted cells
  are the logical cells minus unreturned blanks and unprinted headers. A merged
  cell counts once; returned blanks count, parser padding does not.
- Numbers are also reported on their own, as diagnostics beside the headline.
  A gold cell is numeric when its text parses as a number (amounts, quantities,
  percentages, numeric identifiers such as a spaced SSN; dates and codes such
  as `1a` are text). `numeric_accuracy` = numeric gold cells with full credit
  in place / counted numeric gold cells. `numeric_found_rate` = numeric gold
  values present in any returned cell, ignoring position. `invented_number_rate`
  = returned numeric cells whose digit string and value occur nowhere in the
  page text reference or the gold tables / returned numeric cells; every
  verdict carries `numeric`, and such extras carry `invented`. A merged cell
  such as `7 4927` passes when each digit group is printed. A number
  reformatted beyond separators and currency (`1,232` returned as `1232.00`)
  can be flagged, and so can a number the page shows but the text reference
  omits (the handwritten `1809` beside two items on sroie_001); a number
  printed elsewhere on the page never is.
- Extra cells, rows and tables do not lower the headline. Extra-cell counts,
  unmatched tables, table-count differences and strict cell precision/recall/F1
  are separate diagnostics. The metric named `teds` is a custom row/cell
  edit-distance diagnostic, not the official PubTabNet implementation;
  `structure_similarity` omits text costs.
- Every scored verdict keeps original coordinates, component cells and span
  offsets; the browser viewer shows the returned tables beside the reference
  with every cell verdict.

---

## Key/value

**What it measures.** Whether the engine pulled the printed fields off the page
as label/value pairs.

**Pages and engines.** 65 pages with accepted structured references. Five
engines in the published run: Textract structured (native) and Sonnet, Gemini,
Mistral OCR 4.1, Qwen3-VL (prompted). GLM-OCR and DeepSeek-OCR declare a key/value prompt in
`engines.py` but the published run has no key/value call for them.

**Which fields are scored.** The answer key is the accepted structured reference
(`documents/structured_references/gold/<page>.json`, `true_json`), minus its
`excluded_fields`, minus fields that `documents/structured_references/kv_label_review.json`
does not mark `keep`. That review is an image-based agent review, not human
approval: `remove` covers keys that name an inferred role for unlabeled text,
`uncertain` covers absent or ambiguous printed labels. Printed section, row and
column labels count as labels, so table cells with printed row and column labels
are scored here. Result: 3,902 fields.

**How each engine's answer is read.** `kv_schema_code.engine_pairs` turns every
answer into an ordered list of `(label, value)` pairs, plus, for Textract table
cells, each cell's position.

| Engine | What it was asked | How the answer becomes pairs |
| --- | --- | --- |
| Sonnet, Gemini, Qwen3-VL | The key/value prompt in `evaluation/legs.py`: *"Extract every printed label and its value … as a flat JSON object of label → value. Use the label exactly as printed. Use null where a printed field is blank. Output only JSON."* Sonnet: prompt only, temperature 0. Gemini: JSON mode (`response_mime_type`), no schema. Qwen3-VL: `json_object` response format. | One outer ```` ``` ```` fence is removed and the text parsed as JSON, keeping duplicate keys in order. If that fails: several fenced JSON blocks are all read in order; then embedded JSON; then a truncated answer is salvaged. Nested objects join keys with `.` (`Flame.31.1.04 = 781`), arrays add `[i]`, `null` becomes an empty value. |
| Textract structured | Nothing; FORMS and TABLES have no prompt. | FORMS: each key/value pair as returned. Tables whose rows all have two cells: `left = right`. Other tables: see below; their cells keep their position (table, row, column, row label, column header, checkbox state) as native context. |

How the other Textract tables are read:

```
row 0:  |                       | Current Period | YTD Period   |
row 1:  | Account Transfers Out | $0.00          | -$73,579.95  |
pairs:  ""             = Account Transfers Out
        Current Period = $0.00
        YTD Period     = -$73,579.95
```

The original published adapter is preserved: each cell below the first row
uses its column header as its key. Physical row/column context is retained
separately. It does not prepend the first cell to every key, which would
mislabel side-by-side `Date / Amount` groups. The only score-rule change from
the published evaluator is excluding unreturned blank reference fields in KV.

**How it is scored.** `evaluation/keyvalue_leg.py` → `kv_value_groups.resolve` →
`shared_fields.score_shared`:

```
 answer key fields                     engine pairs (label = value)
        └──────────────┬──────────────────────┘
                       ▼
 ① native position   Textract cell whose row label + column header spell the
                     field name ("<row> — <column>"), unique on the page
 ② exact key         label == field name
 ③ rule              normalized label, a printed alias from the gold file
                     (kv_aliases), or the same form line number
 ④ stored ruling     the document's rulings file: a synonym (label → field) or
                     an exact label + value pair → field
                       │  each field takes at most one pair and each pair fills
                       │  at most one field; ① before ② before ③ before ④
                       ▼
 ⑤ output ruling     a stored judge decision about this exact engine output
 ⑥ value candidates  field still open and an unused pair's value overlaps the
                     expected value → ask the judge, store the ruling;
                     no overlap → missing, no judge call
                       ▼
 value check         correct / wrong / blank_ok / invented / missing
```

- Value overlap (⑥) means whole-value equality or complete-token containment in
  either direction; numbers must match as whole numbers (`100` never matches
  inside `1000`) and identifiers keep every digit group in order. Overlap only
  nominates candidates; it never earns credit.
- The value check (`kv_schema_code.value_equal`) compares numbers as numbers
  (currency glyphs and thousands separators do not matter; a malformed amount
  fails), dates as dates, identifiers with every digit kept, checkboxes as
  true/false, and everything else as lowercase letters and digits with spacing
  ignored. Textract's native checkbox state can establish true or false. An empty
  value matched to a checkbox field by its key counts as false.

The judge (`evaluation/kv_value_groups.py`, `PROMPT`):

```
 up to 24 fields per request            gemini-2.5-flash, temperature 0,
 + every returned pair, with its   ──►  thinking budget 1,024, max output 4,096,
   native table position                JSON response schema
 + all reference fields and values
                       │
                       ▼
 per field: {"output_ids": [..], "synonym": true/false, "reason": ".."}
                       │
                       ▼
 code checks: chosen outputs were candidates · no output used twice ·
 parts add up exactly to the expected value (one selected output may contain it
 inside a longer value) · identifier parts in order
   pass → accepted        fail → rejected (reason kept)
                       │
                       ▼
 stored in the document's rulings file; never asked again
```

A synonym is kept only when the judge flags it, exactly one output was selected,
its value equals the reference exactly, and the last part of the field name is
not shared with another field (item vs summary `Net worth`). Spending stops at
`OCR_KV_JUDGE_BUDGET_USD` (default $2). Requests are journaled locally before
sending (`results/kv_synonyms/value_assisted/calls/`, not in Git) and are not
retried automatically.

The rulings file, one per document (`results/kv_synonyms/value_assisted/<page>.json`),
keyed by reference field, holds the judge state scoring reads. A reply already in
the local `calls/` journal is also reused instead of asking again.

```json
"Account Overview — Beginning of Period Value — Current Period": {
  "expected": "$637,990.00",
  "pairs": [{"key": "Beginning of Period Value", "value": "$637,990.00", "request_id": "9d30f8e1…"}],
  "output_rulings": [{
    "engine": "claude_sonnet_4_6", "output": "e3b7fc20…", "verdict": "rejected",
    "looked_at": [{"output_id": 2, "key": "Account Transfers Out - Current Period", "value": "$637,990.00"}],
    "reason": "Value matches, but key 'Account Transfers Out' is incorrect …",
    "request_id": "794bb1c7…"
  }]
}
```

| Entry | Means | Reused for |
| --- | --- | --- |
| `synonyms` | this label always means this field | every engine on this document |
| `pairs` | this label with this value is this field | every engine with that exact label and value |
| `output_rulings` | accepted or rejected decision about one engine output, with the pairs looked at | that identical output only |

A synonym claimed by two fields is ignored by label; its exact pairs still apply.
A changed reference invalidates the file. Rulings never cross documents.

**Score and limits.**
- Headline: `(correct + blank_ok) / counted fields`. A reference blank whose label
  the engine did not return (`blank_not_returned`) is not counted: the blank may
  be a reference problem. A value returned for a reference blank (`invented`)
  still counts against the engine, and a missing key for a filled field is
  still missing. Populated-only accuracy, blank accuracy and unmatched extra
  pairs are separate diagnostics; extras do not lower the headline.
- Wrong numbers and Missing numbers (`kv_numbers_*`): the share of counted
  fields that `value_equal` compares as numbers (`kv_schema_code.is_number_field`:
  amounts, totals, quantities, rates; not dates, checkboxes or identifiers such
  as invoice or phone numbers) whose value was wrong or not returned. Pages
  without such fields have no value. Diagnostics; the headline is unchanged.
- A page with a field that still needs the judge publishes no headline; its
  provisional score stays visible and is excluded from means. `--skip-judge`
  disables new judge calls, not inference.
- Textract table reading is not repaired layout by layout. Known misreadings:
  side-by-side repeated column groups (`Date | Amount | Date | Amount` labels
  every amount with the row's first date), headers spread over two rows (tax
  forms), repeated identical headers (`£ | £` for two years) and footer fragments
  without a row label.

---

## Schema

**What it measures.** Whether the engine can fill a JSON schema it is given.

**Pages and engines.** 65 pages; seven engines (Sonnet, DeepSeek-OCR, Gemini,
GLM-OCR, LlamaParse, Mistral OCR 4.1, Qwen3-VL). Textract FORMS has no
supplied-schema interface and does not take part.

**Which fields are scored.** The same accepted structured references as
key/value, but a different selection: `documents/schema_sample.json` requests
3,316 of 4,251 fields, identical for all seven engines; 45 pages are complete
and 20 sampled. The key/value label review does not apply to this leg. Reference
and image hashes protect the sample; a changed reference requires an updated
sample.

**How each engine's answer is read.** Every engine receives the literal keys to
fill, without value defaults or filled examples. Keys may themselves include
printed row identities, such as consultant names. This is guided extraction;
the model does not have to discover those identities independently.

| Engine | How the schema is sent |
| --- | --- |
| Mistral OCR 4.1 | Vendor extraction: `document_annotation_format` with the schema; no prompt text |
| LlamaParse | LlamaExtract `data_schema`; no prompt text |
| Gemini | Schema in the prompt; JSON mode (`response_mime_type`), no schema constraint |
| Sonnet | Schema in the prompt; temperature 0; no constraint (see below) |
| Qwen3-VL, GLM-OCR, DeepSeek-OCR | Schema in the prompt (GLM: its model-card prompt with an empty instance template) plus a `json_schema` response-format constraint |

Sonnet is not given Anthropic's structured-output constraint. A test call on
2026-10-17 (invoice_000) was rejected by the API: at most 16 fields may be
nullable, and every field here is nullable (24 on that page, 87 on the largest). Giving Sonnet a different, non-nullable schema would change
its contract relative to the other engines, so it stays prompt-only.

The answer is parsed like a key/value answer (fence removed, duplicate keys kept,
nested keys joined) into `(key, value)` pairs.

**How it is scored.**

```
saved request schema ── same as the current contract? ── no ──► N/A, rerun required
                                 │ yes
                                 ▼
returned keys ──► exact literal key match (no synonyms, no judge) ──► value check
```

1. Score only against the contract the engine received; a changed or unrecorded
   contract requires new output.
2. Keys match by exact literal name only.
3. A blank field is correct only when its matched key returns blank/null; a
   missing key is missing. Values use the same value check as key/value.
4. Extras: any returned key that is not a requested field or allowed by the
   contract, and any repeated key. A returned key that is excluded from the
   reference also counts as an extra.

**Score and limits.**
- Headline: `(correct + blank_ok) / (requested fields + extras)`.
- Unpenalized accuracy and JSON/schema compliance are separate diagnostics.
- Wrong numbers and Missing numbers (`schema_numbers_*`) are counted as in
  key/value, over the requested fields compared as numbers.
  Recovering values from fenced or malformed JSON does not make the JSON valid:
  Sonnet's answers are fenced, so its `schema_valid_json` is false although its
  values are scored.

---

## Geometry

**What it measures.** Whether the engine's text boxes are in the right place.

**Pages and engines.** 17 pages with box references: SROIE lines, FUNSD and
SynFinTabs words; 15 engine configurations (Docling and LlamaParse element/item
boxes and the `llamaparse_granular` word/line configuration added 2026-10-20).

**How each engine's answer is read.** Each adapter delivers boxes from the
engine's stored output. Layout-only engines (MinerU, PaddleOCR-VL, DeepSeek-OCR)
keep their actual `layout_block` boxes; no word boxes are invented. Mixed
line/table interfaces keep each region's type. Box text comes from region
content, not region-type labels. Docling's element boxes are `prov[0].bbox` of
every text item and table in the saved export (bottom-left origin flipped,
divided by the recorded page size). LlamaParse's item boxes (`items[].bBox`,
x/y/w/h) are in the source image's pixel frame, not the page's reported 72-dpi
point size; both are `layout_block` and are recovered from the stored replies.
`llamaparse_granular` reads the parse endpoint's `grounded_items` sidecar: word
and line boxes in page points with the page size declared per row, word text
sliced from the item's Markdown by UTF-8 byte span; its words are scored.

Coordinates: boxes are expected in a 0–1 frame. Layout-block engines are never
rescaled (DeepSeek-OCR's 0–999 grid is divided by 999 in its adapter). For the
others, unless the output declares its coordinate frame (or the engine is MinerU
pipeline), boxes whose largest coordinate is between 1.5 and 1000 are divided by
1000. For Gemini only, x and y
are swapped when more than 70% of its boxes are portrait-shaped while fewer than
30% of reference boxes are. Sonnet's pixel recipe requests coordinates in a
pre-resized image of known size; x/y are divided by the sent width/height and
the frame is declared, so no rescaling or swap applies. It uses temperature 0, a
32,768-token cap and streaming.

**How it is scored.**

```
reference unit ◄── matching text ──► predicted box
        │                                        │
        └──── each box contains the other's centre? ──── yes → located
```

- Units are text-matched one-to-one: word pages by identical normalized text,
  line pages by text similarity of at least 0.7.
- Line pages use disjoint matched word fragments and remaining line segments,
  labelled `line_segment`.
- Nonfinite, reversed, zero-area and out-of-frame boxes earn no location credit,
  count as predictions and remain visible.
- Precision/F1 and box-text errors use the rectangular envelope of the reference
  annotations, excluding reviewed unreadable regions. Predictions intersecting it
  take part unless their centre is in an excluded region.

**Score and limits.**
- Headline (Word): recall = located reference units / reference units.
  Precision and F1 expose extra, duplicated and misplaced predictions.
- Line: the reference merged into visual lines, the prediction merged the same
  way, paired by text similarity of at least 0.7, one-to-one, same centre rule.
  The fair number for engines that return one box per line (PaddleOCR-VL 1.6
  Spotting, Gemini in practice); for word engines it tracks the headline.
- Block coverage (block engines only): a reference word is covered when a
  returned block contains its centre and its text. A page-sized box scores 1.0
  on it, so it is shown with the median words per box and never replaces the
  headline.
- Boxes do two jobs. Acting on text (redaction, searchable-PDF layers, layout
  models) needs words: the Word column. Navigating to text (citations, review,
  RAG provenance) tolerates lines or blocks: the Line and Block coverage
  columns. No number is inflated to look like another.
- Geometry CER/WER use the text from the returned boxes, including unmatched,
  duplicate and invalid ones, in visual line order then left to right, with
  NFKC, case folding and whitespace collapse. The default transcription is never
  substituted. Empty extra boxes affect precision, not text errors (Marker and
  MinerU pipeline drop empty regions before scoring).
- This measures performance inside the annotated region, not certified
  full-page precision.

---

## Judged text (estimate, not a leg)

For 11 pages without reliable text references, a shared image-derived checklist
is evaluated with model verdicts: `correct items / checked items`. These
estimates are separate from text coverage and CER/WER. New verdicts use
`silver-document-v2-no-substring-2026-10-06`. The published report uses cached
verdicts that may include substring-matching errors; report generation does not
refresh them. Judged-text rows exist for 10 engines.

---

## Averaging, costs and published files

- Scores are averaged per document family first, then across family means.
  Failed calls, missing references and pending key/value pages are excluded from
  means, not counted as zero. Difficulty breakdowns use easy, medium and hard.
- "Without low" excludes a page/leg only when at least three engines scored it,
  all scores are ≤0.25 and no participant failed or is pending. One exclusion set
  is computed before display filtering; family means are recomputed and remaining
  coverage is reported. It is a sensitivity check, not a corrected leaderboard.
  Cost and latency keep their original cohorts.
- Costs belong to unique calls: a conversion serving several legs is paid once.
  Modal per-page cost and latency use processing-only batch time; startup and
  shutdown are excluded. See [pricing](engines-and-pricing.md#pricing-audit).
- `current` is the newest complete, scored run in `results/runs/`; its
  `_results.jsonl` is committed gzipped (`.jsonl.gz`) and every reader accepts
  either form. Every published call lives under
  `results/ocr_outputs/reconciled_outputs/`; each inference row's engine,
  document and `pass_name` (`source_pass_name` when set) select the file. One
  physical call scored as several views has one file per view. Report commands read these files without
  changing scores. `--rescore` uses cached judgments by default and
  preserves document verdicts exactly once. Only `--allow-kv-judge` permits new
  KV rulings; it cannot refresh document judgments. Missing source calls fail
  the rescore rather than producing an apparently complete partial run.

## Published result limitations

- Some saved calls omit exact prompts or immutable model versions; recovered
  configuration is labelled separately from per-call evidence.
- Schema scores cover a sampled field set on 20 of 65 pages.
- Cached judged-text scores carry the limitation above.
