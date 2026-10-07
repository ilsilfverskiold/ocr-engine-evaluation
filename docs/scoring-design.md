# Scoring design

Six legs measure text, Markdown, tables, key-value extraction, supplied-schema
extraction and geometry. The silver-document judge is a separate estimate.

The published bundle records exact scorer versions, source calls and coverage.
Current table scoring is `gold-table-values-v4-sections`; Markdown uses
`markdown-structure-v3-spans-reflow`. Offline reporting never changes scores.
See the [README](../README.md) for commands, current limitations and coverage.

## Prompts and provenance

`engines.py` declares each engine's native and prompted legs. `evaluation/legs.py`
plans shared or separate passes. Use the author's task recipe for specialized
models and the supported vendor extraction interface where available. An engine
with a prompt interface competes even if it performs poorly; a fixed interface
that cannot supply the requested output is N/A.

Saved calls retain the exact request, model, schema, raw answer, native structured
payload, costs and transformations. Dedicated Markdown/table controls must be
identified separately from the older default transcription. Investigate input,
prompt, adapter and reference defects before attributing low scores to a model.
Do not use a second model to manufacture missing structure and credit it to OCR.

## Text

Default transcription is rendered to plain text; HTML entities, Markdown syntax
and layout whitespace are normalized consistently on both sides. Unicode NFKC,
case folding and whitespace normalization precede comparison.

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
Image-only catalogue generation is a separate, cost-approved action through
`scripts/build_markdown_catalogues.py`. Proposals stay inactive until source-image
review is recorded; uncertain elements are excluded and source hashes checked.
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

Promptable engines receive an explicit table-only request allowing Markdown for
simple grids and HTML for spans. Native pipelines retain their documented table
interface. The accepted human references retain review notes for table scope and ambiguous
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
new calls; required uncached reviews stay pending and scores remain provisional.
Paid replies are saved and never automatically retried. Existing but ineligible
selected components, incomplete coverage and output reuse reject the proposed
field match while retaining other valid batch decisions. Invented IDs or malformed
reply shapes remain errors. Validated journal decisions can be recovered even
when later aliases change batch composition, avoiding duplicate payment.

KV headline remains `(correct + blank_ok) / selected fields`. Extra pairs are
visible but do not directly penalize incomplete-reference pages. Native Textract
uses FORMS and native table fields. All structured reference fields participate (including array leaves);
old printed-pair flags no longer remove fields. The same explicit human field
exclusions apply to KV and schema.

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
GPU interval, GPU type, startup and amortized document costs. No model upgrades
or GPU changes are implicit in a scorer update.

`python run_benchmark.py --rescore <source-run> --skip-judge` replays saved calls
without new inference or model judging. New schema calls use the current
`gt.inference_schema`; record them before scoring. `--collect-only`, `--score`,
`--docs`, `--engines` and `--legs` support separately approved batches. The repo
includes accepted table/structured gold and the five original shipping tables;
the downloader restores manifest-verified source images and original sidecars.

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
