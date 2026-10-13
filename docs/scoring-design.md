# Scoring design

Six legs measure text, Markdown, tables, key-value extraction, supplied-schema
extraction and geometry. The silver-document judge is a separate estimate.

Checked against the current scorer code and published run `20261013T081846Z`
on October 13, 2026. Active scorer versions are:

| Leg | Scorer |
| --- | --- |
| Markdown | `markdown-structure-v3-spans-reflow` |
| Tables | `gold-table-values-v4-sections` |
| KV | `shared-fields-v8-identifiers-native-checkboxes` |
| Schema | `shared-fields-v1` |
| Geometry | `geometry-v8` |

The published bundle records source calls and coverage. Offline reporting never changes scores.
See the [README](../README.md) for commands, current limitations and coverage.

## Prompts and provenance

`engines.py` declares each engine's native and prompted legs. `evaluation/legs.py`
plans shared or separate passes. Use the author's task recipe for specialized
models and the supported vendor extraction interface where available. An engine
with a prompt interface competes even if it performs poorly; a fixed interface
that cannot supply the requested output is N/A.

Current calls retain the request, model, schema, raw answer, native structured
payload, costs and transformations. Some historical calls omit prompts or exact
model versions; recovered configuration is labeled as such. Dedicated Markdown/table controls must be
identified separately from the older default transcription. Investigate input,
prompt, adapter and reference defects before attributing low scores to a model.
Do not use a second model to manufacture missing structure and credit it to OCR.

## Text

Default transcription is rendered to plain text; HTML entities, Markdown syntax
and layout whitespace are normalized consistently on both sides. Unicode NFKC,
lowercasing and whitespace normalization precede comparison.

- **Coverage:** fraction of unique reference tokens found in the output. This
  does not penalize repetitions, extra words or wrong reading order.
- **CER:** character insertions, deletions and substitutions divided by reference
  characters. **WER** uses words instead. Lower is better; zero is an exact
  normalized match. Insertions can make either rate exceed 100%.
- Word precision/recall/F1 remain supporting metrics. No combined headline with
  CER/WER has been agreed.

82 pages have usable text references. The 11 silver pages are excluded from
reference-based text scores. Reference omissions and reading-order differences
can increase error rates without being recognition errors; disclose them.

## Markdown

The headline is `(correct + 0.5 × partial) / checklist elements`, with deterministic
verdicts against a shared, cached image-authored structural catalogue. It is not
a per-engine model judgment. Existing catalogues are sampled to 40 elements.
Missing catalogues remain N/A; receipt coverage is now included in the 92 cached catalogues.
Accepted catalogues ship in `results/current.zip`; running the benchmark does
not regenerate them. Uncertain elements are excluded and source hashes checked.
A reviewed catalogue can establish structure eligibility without a Markdown
sidecar. It is not substituted for a full transcript: full-page block F1/order
remain N/A without one, and CER/WER need an independent trusted text reference.

Actual Markdown/HTML structure determines type. ATX/Setext headings and HTML
heading tags count; `**bold**` is emphasis, not a heading. A cell containing a
heading's words is not a heading. A glyph alone is not a Markdown list marker.
Tables require a valid Markdown separator or HTML table; plain TSV is not
Markdown table formatting. One outer Markdown code fence may be unwrapped as
transport packaging; JSON/Python code fences remain code.

The active scorer is `markdown-structure-v3-spans-reflow`. Locate the expected content
first, then inspect its actual rendered block type. Search exact occurrences
and boundary-aligned partial spans; compare the **whole expected content** with
the candidate span. A partial-search similarity is never itself the score.
NFKC, case and typography whitespace normalize; punctuation and identifier
digits are retained. Catalogue spacing/OCR differences can therefore match
without pretending every reference character is authoritative.

A compatible type and content similarity at least 0.85 earns correct; at least
0.50 earns partial. Full credit also requires matching numeric/identifier
signatures, including signs and decimal punctuation. Character resemblance
alone below 0.85 needs token overlap as corroboration. Wrong type earns zero
and reports where the text was found. Text embedded in a larger heading/list
entry or wider table row is present, but may receive partial structure credit
rather than being called absent. If a matching occurrence has already
been allocated, the reason explicitly says so instead of claiming no content.

Distinct reference items may occupy disjoint spans of one paragraph. Several
adjacent paragraphs may supply one reference item. Credited output tokens can
be used only once. Wrong-type evidence earns no credit and may be reused as a
diagnostic. There are no engine/document-specific matching rules.

`key_value` is a semantic category: a complete pair may be in a paragraph,
heading, list item, or cells of the same table row. A single-record table may
also pair each header with the value directly below in its column. Merged or
multi-record grids are not silently converted into these field pairs. A value
elsewhere on the page cannot be borrowed. Actual `table_header` blocks retain header semantics;
a data row with just a row-label `<th>` is not a column header. Older generic
`table_row` references allow either header or data rows. Full row credit requires
ordered cell matches, not just finding the same words somewhere in a table.
Empty padding cells are ignored for this Markdown comparison; the separate
gold-table leg scores blanks and grid shape. Split item rows are not silently
merged and cannot receive full row credit. Partial verdicts retain content and
ordered-cell similarities so a description alone is not evidence for prices.

Repeated side-by-side column groups may reflow into one narrow table. This
requires an explicit repeating reference header, matching output headers, and a
unique first-cell anchor for each group in that same output table. Each date or
item stays paired with its own values. Duplicate anchors, different tables and
changed amounts cannot earn full credit. Fully credited rows consume and show
all their tokens, including content around blank padding cells.

New catalogue text is literal. Legacy catalogue inline emphasis is decoded
without interpreting numbered cell text as a Markdown list. Saved verdicts
include matched block types, token ranges and normalized text excerpts.

With a full Markdown reference, supporting block recall uses complete compatible
matches from this same matcher. Output-block precision requires at least 85% of
that block's tokens to be covered by complete matches; unrelated extra prose
is not credited merely because it contains a matching phrase. Reading order
uses the positions of these matches, including their positions within a block.
A sampled catalogue alone still cannot establish full-page block precision.

CER/WER are computed separately on the actual Markdown answer's rendered text,
using trusted full-page text where available, otherwise the Markdown reference.
Block precision/recall/F1 and reading order are supporting diagnostics. Text
fidelity does not make incorrect structural syntax correct.

## Tables

Human gold at `documents/table_references/gold/` takes precedence over original
references; a proposal is never selected automatically. All 42 reviewed pages
have gold. Schema/KV gold is a separate reference collection.

**Headline: reference value score (`gold-table-values-v4-sections`).** Sum per-cell
value credit and divide by reference logical cells. Exact normalized text and
numerically equivalent amounts receive 1; nonnumeric text with character
similarity at least 0.5 receives that fractional credit; lower similarity and
incorrect numeric amounts receive 0. Currency formatting may differ; percentage
units and leading-zero identifiers remain significant. Actual blank cells count;
parser padding does not. A merged cell counts once.

Extra cells, rows and tables do **not** reduce this headline. Report extra-cell
count/rate, unmatched returned-table count, physical table-count difference and the old strict cell precision/recall/F1 as
separate diagnostics. Empty output against nonempty gold scores zero.

Keep logical item/column associations and use each returned content span at
most once. A unique description may anchor several adjacent physical rows.
Empty label cells or fragments of the same description may continue an item;
a different label/subtotal, another item anchor, or completion of the record
ends the group. Ambiguous/repeated labels do not justify an inferred association.
Preserve value order within columns. This is deterministic grouping, not a model
inventing missing structure.

Adjacent columns may combine when complete headers establish their identity;
HTML colspan also establishes which columns a cell covers. Values in one merged
cell partition into ordered, disjoint word spans. Never reorder those values
or borrow from another item. Blank cells need a returned blank or an empty component established by a combined field’s header/column span; missing whole rows never receive blank credit.
Every verdict records the combined content, original component cells and their
normalized-text offsets; the report keeps the original returned table visible.
A physically different row layout alone does not lose content credit.
Unique headers permit added, missing or reordered columns. Unique descriptive
row labels or ISO dates identify rows; otherwise positions govern. Repeated
side-by-side header groups may reflow into a narrow table. Repeated header/unit bands define logical sections on both sides; unique row labels locate section boundaries when returned headers are omitted. Match sections independently of physical table boundaries. Repeated units preserve column order and can establish a Notes-like column combined into the label. Actual units rows cannot be consumed by year headers or section-label blank cells. Identical-header
chunks may combine when they belong to the same reference section; repeated data
rows remain extras. A transpose is accepted only when header matching supports
it, not because it maximizes matching values. Preserve original coordinates and
component cells in every verdict. Arbitrarily collapsed grids, unrelated values
elsewhere, and wrong row/value associations are not repaired into perfect scores.
Native HTML is preferred over a second Markdown export of the same answer.

Supporting structure similarity preserves `rowspan`/`colspan`, ignores decorative
HTML wrappers/attributes and penalizes unmatched tables. The current `teds` metric
uses a custom ordered row/cell edit-distance diagnostic with cell-text costs;
`structure_similarity` removes text costs. Do not describe it as the official
PubTabNet APTED implementation. Table pairing and normalization are recorded.

**Shipping exception:** evaluate the whole page against gold and the frozen
original separately. Select the higher reference value score, preferring gold on ties, and
retain both comparisons. Never choose individual cells from different references.
This policy applies to every engine.

The current planner supplies an explicit table-only request when the engine's
registry configuration selects a prompted table pass, allowing Markdown for
simple grids and HTML for spans. Native pipelines use their conversion output.
The accepted human references retain review notes for table scope and ambiguous
glyphs. Historical default answers are still scored from their saved table output;
dedicated table-only controls have not been run.

## Shared KV/schema references

The accepted fresh, flat key/value extractions at
`documents/structured_references/gold/` supply the same accepted keys and expected values for both legs. Storage shape
does not disqualify a KV field. KV additionally applies the image-reviewed
`documents/structured_references/kv_label_review.json`: 3,902 fields remain
eligible, 246 invented-label fields are removed, and 103 uncertain fields are
withheld, on top of 93 prior exclusions. Printed section/row/column labels may
supply context; an unlabeled company name does not establish a “Company” key.
This is a separate agent review, not a new human approval of gold. Gold values
and schema selection stay unchanged. The overlay records per-field reasons and
source/gold hashes and refuses stale references. For the current
schema run, the user requested one shared field sample sized to fit every engine:
`documents/schema_sample.json` selects 3,316 of 4,251 fields, identically for all
seven engines. Forty-five pages keep every field; twenty are sampled. Schema
requests and scoring both use that frozen selection; unexpected keys still
incur the extra-key penalty. Full gold is unchanged; KV selection is recorded separately. Reference
edits invalidate the sample. There is no three-field minimum. The corpus has 65
structured pages; uncertain readings remain explicitly excluded from both legs.
The old nested references remain available as provenance, not the active target.

**KV resolves wording and split values during evaluation.** Match literal keys,
unambiguous printed aliases and cached positive confirmations first. For other
fields, compare whole values and check complete-token containment in either
direction. A returned fragment inside a reference value (or a reference inside
a longer output) nominates a candidate; it never earns credit without judgment.
Leading/trailing currency glyphs are ignored in numeric comparison, with signs,
decimals and identifier digits preserved. `100` does not match inside `1000`.

Native record tables retain printed row identities, column headers and cell
coordinates alongside the returned pairs. An unambiguous row identity + header
can establish the reference key without a judge; the ordinary value checker
then checks the value, including blanks. Duplicate row identities, conflicting
renderings and headerless grids whose first row is a known record do not use
this shortcut. Values never select the row. This is shared by live evaluation,
rescores and saved-request planning. Native matches supersede cached reviews
for those fields; unrelated paid decisions remain reusable.

The judge receives reference-centered items plus all returned keys/values for
context, including native source context where available. A matching empty cell
must be identified by its output ID; an empty value is not an empty selection.
 It may select multiple output components for one complete value, even
when a parent field is empty. Code validates complete coverage and one-use output
allocation after judgment. There is no special Seller/Client rule, component-name
whitelist or automatic key-prefix credit. Repeated amounts are judged with their
surrounding fields, not arbitrarily allocated by occurrence.

Existing valid positive cache entries are reused. Old negative decisions do not
suppress new partial-value/context requests. Group decisions bind to the entire
returned-pair context and reference identity. Only an explicitly judge-approved
single-key synonym with a complete matching value can become reusable across
contexts. The stored reason and request ID remain visible. Reference/context
edits invalidate affected decisions; incorrect numeric substring matches cannot
become credit through a cached synonym.

The shared path is `evaluation/keyvalue_leg.py` → `kv_value_groups.py`, using
`kv_value_judge.py` for legacy-cache compatibility, budget control and journaling.
Live runs, saved scoring and rescoring use the same path. `--skip-judge` makes no
new judge calls; required uncached reviews stay pending. A page with pending
review has no headline score and is excluded from the mean; its provisional
calculation remains available as a diagnostic. This flag does not disable OCR inference.
Paid replies are saved and never automatically retried. Existing but ineligible
selected components, incomplete coverage and output reuse reject the proposed
field match while retaining other valid batch decisions. Invented IDs or malformed
reply shapes remain errors. Validated journal decisions can be recovered even
when later aliases change batch composition, avoiding duplicate payment.

KV headline remains `(correct + blank_ok) / selected fields`. Extra pairs are
visible but do not directly penalize incomplete-reference pages. Native Textract
uses FORMS and native table fields. Fields, including array leaves, are eligible
after shared gold exclusions and the additional KV printed-label review above.
Legacy printed-pair flags are not used. Schema uses its frozen sample and shared
gold exclusions; the KV-only label review does not change its requested fields.

**Schema:** supply the current answer-free schema and require its literal keys.
Changed or unrecorded saved request contracts require a new extraction; they are
not scored against a contract the engine never received. Headline is
`(correct + blank_ok) / (selected fields + unexpected or duplicate fields)`.
Allowed optional keys and explicitly excluded reference fields are not extras.
Unpenalized field accuracy and raw JSON/schema compliance are separate diagnostics.
Recovering values from fenced/malformed JSON does not make that JSON valid.
Textract FORMS has no supplied-schema interface and does not enter this leg.

For either leg, an expected blank earns credit only when its matched key returns
blank/null. A missing key is missing, not a correct blank. Per-field verdicts,
exclusions, extras, unresolved keys and parsing diagnostics remain inspectable.

## Geometry

17 pages have box references: FUNSD and SynFinTabs words; SROIE lines.
Text-matched units are located when each box contains the other's centre.
Matching is one-to-one. Recall remains the headline; **precision and F1** expose
extra, duplicated and misplaced predictions. Invalid predictions are counted.
Line pages use disjoint matched word fragments plus remaining line segments as
the prediction units; this conversion is explicitly labeled `line_segment`.
DeepSeek-OCR, MinerU and PaddleOCR-VL now enter this headline with their actual
returned layout boxes, explicitly marked `layout_block`. Do not invent word
boxes by splitting their regions. Coarse-block diagnostics remain separate;
poor location scores are displayed. Box text uses the region content, with known
Markdown/HTML rendered to plain text, not the model's region type label.

**Geometry CER/WER use text from the returned boxes**, including unmatched,
duplicate and invalid-coordinate items. Both sides use deterministic visual
line order, then left-to-right ordering, NFKC, case folding and whitespace
collapse. Invalid-coordinate text follows the valid boxes in returned order.
The default transcription is never substituted. Correct text in misplaced
boxes may have low CER/WER and poor location scores; empty extra boxes affect
precision but not text errors.

Sonnet's pixel recipe requests absolute coordinates in a pre-resized image with
its exact width/height included in the prompt. The adapter saves the original
pixel boxes and divides x by sent width, y by sent height. No inferred rescaling
or axis swapping is applied to these known-frame results. Geometry requests use
temperature 0, a 32,768-token cap and streaming; stop reasons remain visible.
The exact saved call determines which recipe a report describes.

Adapters' declared granularity and coordinate frame are authoritative. Block outputs retain their
actual boxes; mixed MinerU line/table regions retain their individual types and
render native table HTML to text. Normalized coordinates are not clipped or
scaled a second time. Finite out-of-frame, reversed and zero-area rectangles
receive no location credit and count as invalid predictions; their text and
boxes remain in the evidence. An unresolved provider coordinate frame is an
adapter error, not a zero inferred from its prose output.

Coordinate-frame conversions must be explicit and recorded. Keep source image,
returned text/box associations, matched/misplaced/missing/extra overlays and
invalid-box counts. Corpus precision/F1 and box-text errors use the rectangular
envelope of retained dataset annotations, with reviewed unreadable regions
excluded. A prediction intersecting that envelope participates unless its centre
falls in an excluded region. Outside-region boxes remain visible in the overlay
and exclusion list; malformed and out-of-frame boxes still incur penalties.
This fixed rule is independent of engine output. It measures annotated-region
performance, not certified full-page precision: omissions inside the envelope
can still affect comparisons. Explicit custom regions use the same mechanism.
Sonnet prompt/crop controls require separate approved inference.

## Silver-document estimates

The document judge evaluates 11 pages with unreliable text ground truth. A
cached image-derived checklist is shared across engines. Every sampled item
now receives a contextual model verdict; the unsafe punctuation-stripped
substring shortcut was removed. Accuracy is correct items / checked items.

New verdicts record `silver-document-v2-no-substring-2026-10-06`. Historical
verdicts copied by an offline rescore are marked historical and are not evidence
of the fix. Refreshing estimates costs money and requires approval. Model-created
references/verdicts can be wrong and sharing a faulty reference can affect engine
rankings differently. These remain estimates, separate from deterministic text.

## Aggregation, costs and reproduction

Average pages within each document type, then average document-type means.
Errors and missing references stay visible and are excluded from means; report
coverage beside quality. Do not compare aggregates on different page sets without
showing that difference. Schema rerun requirements are not engine failures.

Costs attach to inference passes. A shared conversion serving several legs is
paid once; do not sum its copied per-leg costs. Modal batches report one measured
GPU interval, GPU type, startup and amortized document costs. Per-page cost and
latency exclude startup/shutdown; volume projections multiply processing-only
means, not session overhead. No model upgrades
or GPU changes are implicit in a scorer update.

`python run_benchmark.py --rescore <source-run> --skip-judge` replays saved calls
without new inference or model judging. New schema calls use the current
`gt.inference_schema`; record them before scoring. `--collect-only`, `--score`,
`--docs`, `--engines` and `--legs` support separately approved batches. The repo
includes accepted table/structured gold and the five original shipping tables;
original sidecars ship in Git and the downloader restores manifest-verified source images.

## Consistently low cases (reporting only)

Full corpus scores remain primary. The secondary comparison excludes a page/leg
case when at least three distinct engines have scored it, all their scores are
≤0.25, and no participant failed or remains pending. Nonapplicable engines do not
count as zeros. The complete saved cohort determines one common exclusion set
before display filtering. Recompute document-type means and report remaining
pages and lost types; all-excluded results are N/A. Costs retain all saved calls.
This sensitivity check cannot distinguish hard documents from bad references,
prompt problems or scorer defects. It is not a corrected leaderboard.

## Published evidence

`results/current.zip` contains one score set, its inference records, exact source
calls and shared catalogues/KV decisions. Readers use it directly; local working
files override bundled files during new runs. `scripts/publish_results.py` verifies
SHA-256 hashes and atomically replaces the bundle. Images download separately.
The HTML is generated locally and is not committed. Source identifiers inside
the archive preserve provenance, not additional published result sets.


## Reference and reporting limitations
KV retains its existing blank-inclusive headline pending a policy choice and
now also reports populated-only and blank-field accuracy. Identifier fragment
validation preserves digit groups, order and leading zeros. Native Textract
cell selection metadata can establish an explicit false checkbox value; a
plain empty cell cannot. Saved model approvals rejected by the old guards can
be revalidated without another call, with original requests/replies retained.
New unreviewed candidates remain pending, never silently awarded credit.

Saved silver-document estimates appear by default in a separate judged-text
comparison in HTML and terminal reports. Restored pre-fix verdicts are labeled
historical; they are not blended with deterministic text scores or CER/WER.

## Engine versions and provenance

Versions below come from saved replies, with source pins and the dated Mistral
alias resolution identified separately. Exact prompts, task options and replies
are in `results/current.zip` and the evidence report. An engine ID is a stable
result identifier, not proof of its model version.

| Engine ID | Recorded model/version | Configuration |
| --- | --- | --- |
| `claude_sonnet_4_6` | `claude-sonnet-4-6` | Anthropic API; dedicated schema/KV/Markdown and pixel-frame geometry passes |
| `deepseek_ocr` | `deepseek-ai/DeepSeek-OCR` (v1); source pin `9f30c71f441d` | vLLM, Modal A100-80GB |
| `docling` | `2.118.1` (reconstructed environment; calls saved `docling-latest`) | Standard pipeline: Heron layout + TableFormer Accurate + RapidOCR/PP-OCRv6; see provenance below |
| `gemini_flash` | `gemini-3.1-flash-lite` | Google API |
| `glm_ocr` | `zai-org/GLM-OCR`; source pin `ca5d8b3e287e` | vLLM, Modal A10 |
| `llamaparse` | v1 Balanced / Parse with LLM, August 10 run (dated release unconfirmed); schema `openai-gpt-4-1` | Hosted Parse/Extract APIs; recovered client packages `llama-cloud-services==0.6.94`, `llama-cloud==0.1.46` |
| `marker` | `marker-pdf==1.8.2` | Modal A100-80GB; 11 conversion workers; separate OCR geometry pass |
| `mineru` | `MinerU2.5-2509-1.2B`; source pin `1aa090b41282` | vLLM native batch, Modal A100-80GB |
| `mineru_pipeline` | `mineru[pipeline]==2.5.4`; Torch `2.6.0+cu124`; weights `ed6b654c018d` | DocLayout-YOLO + PP-OCR + UniMERNet/table models, Modal T4; geometry only |
| `mistral_ocr_4` | **4.1 conversion** (dated alias resolution); **4.0 schema** (saved reply) | Mistral OCR/Annotations API; mixed versions, see below |
| `paddleocr_ppocrv5` | PP-OCRv5 server det `ca867c89` / rec `b26c3587`; PaddleOCR `3.7.0`; Paddle GPU `3.2.1` | Modal A10; geometry only |
| `paddleocr_vl` | `PaddleOCR-VL@7fa00a8c55b7`, pipeline v1 | Layout + vLLM recognition, Modal A100-80GB |
| `qwen3_vl` | `Qwen/Qwen3-VL-8B-Instruct`; source pin `0c351dd01ed8` | vLLM, Modal A100-80GB |
| `tesseract` | `5.5.2` | Local CPU |
| `textract_structured` | `AnalyzeDocumentModelVersion: 1.0` | AWS FORMS/TABLES; provider-reported version; no supplied-schema interface |
| `textract_text` | `DetectDocumentTextModelVersion: 1.0` | AWS text/word detection; provider-reported version |

**Mistral:** all 93 conversion replies from `20260810T130609Z` saved
`mistral-ocr-latest`. Mistral's [changelog](https://docs.mistral.ai/resources/changelogs)
states that this alias switched to `mistral-ocr-4-1` on July 16, 2026. Thus the
conversion results (text, judged text, Markdown and tables) correspond to 4.1
according to the published alias history, not an immutable ID in those replies.
All 65 schema replies from `20261006T205850Z` explicitly report
`mistral-ocr-4-0`. The runner pins conversion to 4.1 and schema to 4.0 separately;
it does not silently substitute a model. Mistral now lists 4.0 as retired, so
reproducing that schema run may no longer be possible through the hosted API.

**Source pins:** the four revisions marked above already appear in the initial
benchmark commit `e649e5a`; full hashes are in the corresponding
[Modal apps](../modal_apps). Their old replies did not record the deployed hash,
so these establish the configured weights rather than independent per-call
verification. New replies include the revision.

### Recovered Docling and LlamaParse versions

**Docling:** [2.118.1 was released August 7, 2026](https://github.com/docling-project/docling/releases/tag/v2.118.1).
The retained environment's package/source timestamps are August 9, before the
August 10 run, and the October 2 environment inspection recorded the same
versions: Docling `2.118.1`, docling-core `2.91.0`, docling-ibm-models `3.13.3`,
docling-parse `7.11.0`, RapidOCR `3.9.2`. This reconstructs the likely run
environment; the original per-call records do not independently verify it.

The runner used `DocumentConverter()` defaults. In that environment these select
Heron (RT-DETR/ResNet50) for layout and TableFormer **Accurate** for tables.
The model-cache refs were written during the run, August 10 at 13:13 UTC:
`docling-project/docling-layout-heron@8f39ad3c0b4c58e9c2d2c84a38465abf757272d8`
and `docling-project/docling-models@fc0f2d45e2218ea24bce5045f58a389aed16dc23`
(TableFormer release `v2.3.0`). Auto-OCR resolves to RapidOCR with Torch in the
retained environment: ocrmac, ONNX Runtime and EasyOCR are absent. Its cached
August 9 checkpoints are `PP-OCRv6_det_small.pth`, `PP-OCRv6_rec_small.pth` and
`ch_ptocr_mobile_v2.0_cls_mobile.pth`. These identify a multi-model document
pipeline, not a single Docling language model. New calls record the installed
Docling release.

**LlamaParse:** the retained client packages have August 9 installation
timestamps, but SDK versions do not identify the hosted parser model. The
original runner selected Markdown output without a tier, parser version or
vendor model. The [legacy mode documentation](https://developers.llamaindex.ai/llamaparse/parse/v1/presets_and_modes/advance_parsing_modes/)
identifies this default as **Parse with LLM / Balanced** (`parse_page_with_llm`).
The [published parser version list](https://developers.llamaindex.ai/reference/resources/parsing/methods/list_versions/)
places Cost Effective `2026-08-08` immediately before our August 10 run, with
the next release on August 11. Thus **`2026-08-08` is a date-based candidate
if legacy Balanced followed v2 Cost Effective's latest release**. The inspected
changelogs and migration documentation do not establish that mapping, so this
is not a verified version pin. Other tiers had different release dates:
Agentic `2026-07-24`, Agentic Plus `2026-07-08`, Fast `2026-06-15`.
An October 13
read-only lookup of saved job `9b19a059-c95c-49b0-a983-3120f5bb9a97` returned
only its ID and `SUCCESS`; the v2 lookup did not expose that legacy job.
The underlying parsing model remains unresolved and needs provider confirmation
against the saved job IDs. Separately, all 65 saved **LlamaExtract schema**
replies name `openai-gpt-4-1`; that does not identify the parsing model.



**Comparison limits:** historical default table/Markdown prompts vary; the saved
Markdown coverage was completed with dedicated calls where needed. Sonnet's 17
geometry calls use the updated pixel prompt. Output-cap stops and provider errors
remain recorded. Reference omissions and different output granularity can affect
scores; inspect same-page evidence before attributing a loss to recognition.

## Pricing audit

Checked October 13, 2026.

| Tested service / pipeline | Rate used and audit finding |
| --- | --- |
| Claude Sonnet 4.6, direct Anthropic API | $3 / million input tokens + $15 / million output tokens. Saved usage reproduces the recorded costs; image tokens are included. [Pricing](https://platform.claude.com/docs/en/about-claude/pricing). |
| Gemini 3.1 Flash-Lite, Gemini API | $0.25 / million input tokens + $1.50 / million output tokens, including thinking. Saved usage reproduces the recorded costs. [Pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.1-flash-lite). |
| Mistral OCR 4.1 conversion / 4.0 annotations | $4 / 1,000 OCR pages; $5 / 1,000 annotated pages. Published results already correct the old conversion logger's $0.001/page to $0.004. Schema is a separate annotated call, not the conversion price. [OCR pricing](https://docs.mistral.ai/models/ocr-4-1). |
| LlamaParse v1 Parse page with LLM / LlamaExtract v1 Multimodal | Parse: 3 credits/page × $1.25 / 1,000 credits = $0.00375/page. Published results already correct the old logger's $0.003. Schema: 20 credits = $0.025/page for an uncached file; cached extraction can be 14 credits = $0.0175. These are list-price estimates, not captured credit charges. [Mode-specific pricing](https://developers.llamaindex.ai/llamaparse/general/pricing/). |
| Textract DetectDocumentText / AnalyzeDocument TABLES + FORMS | $0.0015/page for text; $0.015 tables + $0.05 forms = $0.065/page, with OCR included. These match the first-million-page US rates, including the template's `us-east-1`; the old calls did not persist their region. The runner's fixed rate must be checked if using another region. [Pricing](https://aws.amazon.com/textract/pricing/), [US East price list](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonTextract/current/us-east-1/index.json). |
| Modal model pipelines | GPU rate × measured batch seconds / attempted pages. A100-80GB: $0.000694/s for DeepSeek, MinerU VLM, PaddleOCR-VL, Marker and Qwen; A10: $0.000306/s for GLM and PP-OCRv5; T4: $0.000164/s for MinerU pipeline. Saved calculations match these rates. **CPU/RAM and container overhead are missing from these costs.** [Resource pricing](https://modal.com/pricing). |
| Local Docling / Tesseract | Saved `$0` means no provider charge was assigned. It excludes the machine and electricity; the tables now show **Unpriced**. |

### What changes at scale

**GPU hosting:** the volume table uses
`pages × processing-only GPU cost/page`. If estimating a complete deployment bill,
add startup and shutdown/idle GPU charges **once per actual container session**,
plus CPU/RAM charges. Do not multiply a small test run's startup-inclusive
per-page average by the production page count. CPU/RAM usage was not recorded,
so a complete historical bill cannot be reconstructed.
As an illustration, a 190.8-second DeepSeek startup plus the configured 30-second
shutdown delay adds about **$0.153 in GPU charges per container session**: $0.00153
per page over 100 pages, or $0.0000153 over 10,000. Multiple containers repeat that
cost. Sustained batches spread it out; isolated requests do not. Modal also bills
CPU/RAM at the higher of requested or used resources. [Billing rules](https://modal.com/pricing).

**Token-priced engines:** longer outputs and extra task prompts change cost.
Async batch endpoints offer 50% lower token prices for
[Sonnet](https://platform.claude.com/docs/en/about-claude/pricing#batch-processing)
and [Gemini](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.1-flash-lite),
and [Mistral's OCR Batch API](https://docs.mistral.ai/resources/cookbooks/mistral-ocr-batch_ocr)
offers 50% off. The benchmark uses standard API calls, so these savings are
**not applied** above; running calls concurrently does not earn a batch discount.
For the same token mix, 1 million pages would be approximately $7,564.55 for
Sonnet or $625.64 for Gemini via their batch APIs; Mistral conversion would be $2,000.
These are price projections, not tested batch runs.

**Volume tiers:** Textract's next pages after the first million per month cost
$0.0006 for text or $0.05 for Forms + Tables at the US rates used here. Thus
2 million pages would cost $2,100 or $115,000 respectively, rather than doubling
the first-million price. [AWS volume examples](https://aws.amazon.com/textract/pricing/).

**Multiple legs:** count unique inference passes, not score rows. One conversion
can supply text, Markdown and tables at no additional inference charge. Dedicated
Markdown, schema, KV or geometry calls cost extra. For example, Mistral conversion
plus a separate annotated extraction is $9 / 1,000 pages; LlamaParse plus uncached
v1 Multimodal extraction is $28.75 / 1,000. Cached vendor work can reduce charges.
Evaluation judge calls are a separate benchmark expense, not a production OCR price.

### Latency projections

README volume durations multiply unrounded mean seconds/page by page count,
using the same 82 successful text-scored pages per engine. GPU timings are
processing time amortized across batches on one warm container; CPU/API timings
assume sequential page calls. Geometry-only estimates use their 17 scored pages.
Startup/shutdown is excluded. These are workload projections, not measured
production completion times. Parallelism, rate limits, document mix and batch
utilization change elapsed time. Durations round to minutes below one day and
to hours at or above one day; per-page timings retain two decimals.
