# Scoring design

Rules implemented by the scorers and used in published run `20261013T112225Z`,
checked October 13, 2026. [Results and commands](../README.md) ·
[Engine versions and pricing](engines-and-pricing.md).

## Tasks and references

Each leg measures a separate task. Native outputs or supported task prompts
establish eligibility; unsupported tasks are N/A. A low score does not make an
engine ineligible. The benchmark does not add another model to infer missing
structure and credit it to the OCR engine.

| Leg | Headline | Active scorer |
| --- | --- | --- |
| Text | Unique reference-token coverage | `evaluation/text_leg.py` |
| Markdown | `(correct + 0.5 × partial) / elements` | `markdown-structure-v3-spans-reflow` |
| Tables | Value credit / reference logical cells | `gold-table-values-v4-sections` |
| Key/value | `(correct + blank_ok) / selected fields` | `shared-fields-v8-identifiers-native-checkboxes` |
| Schema | `(correct + blank_ok) / (selected fields + extra fields)` | `shared-fields-v1` |
| Geometry | Located reference units / reference units | `geometry-v8` |

Source images are pinned by `documents/manifest.json`. Text and geometry
references ship as sidecars; accepted table and structured references live in
`documents/table_references/gold/` and `documents/structured_references/gold/`.
Shared Markdown catalogues and cached judge decisions ship in `results/current.zip`.
Scoring does not regenerate Markdown catalogues or replace human gold with proposals.

`engines.py` and `evaluation/legs.py` define native and prompted passes. Shared
passes are called once. Prompted passes use the engine's supported task recipe;
schema requests contain the field contract, never the expected values. Saved
requests and responses establish what was actually tested. Sonnet and Gemini
use dedicated Markdown and table requests on the same 92- and 42-page cohorts
as the other participating engines.

## Text

**82 pages** have usable text references. Both output and reference are rendered
to plain text, with Markdown/HTML formatting and entities decoded, Unicode NFKC,
lowercasing and whitespace normalization.

- **Coverage:** fraction of unique reference tokens found in the output. It
  ignores repetition, extra words and reading order.
- **CER:** character insertions, deletions and substitutions / reference characters.
- **WER:** the same edit calculation using words. CER/WER are better when lower;
  zero is an exact normalized match, and added text can push either above 1.
- Word precision/recall/F1, normalized edit similarity, output/reference length
  ratio and a repetition flag are supporting metrics. They do not change coverage.

Reference omissions or reading-order differences can increase CER/WER even when
the words were recognized correctly. The 11 pages without reliable text references
use the separate judged-text estimate below.

## Markdown

**92 pages** have shared cached structural catalogues, capped at 40 elements.
Uncertain elements are excluded. Accepted image-reviewed catalogues check source
hashes; a missing catalogue makes the checklist score N/A. A catalogue can
establish eligibility without a full Markdown sidecar.

The scorer locates expected content before checking its rendered structure:

1. Find exact occurrences and boundary-aligned partial spans, comparing the
   whole expected content. Normalize Unicode, case and typography whitespace;
   retain meaningful numeric and identifier differences.
2. Check the actual Markdown/HTML block type. Compatible content similarity
   ≥0.85 earns 1; ≥0.50 earns 0.5. Full credit also requires matching numeric and
   identifier signatures. Lower-similarity character matches need token overlap.
3. Assign credited spans once. Separate references may use disjoint spans of
   one paragraph; adjacent paragraphs may satisfy one reference. Wrong-type
   evidence earns zero but remains visible. Already-used content is identified
   as allocated, not reported as absent.

### Structural rules

- ATX/Setext headings and HTML heading tags count. Bold emphasis, table-cell text
  and isolated glyphs do not establish headings or list markers.
- Tables require valid Markdown separators or HTML tables. One outer Markdown
  fence may be removed as transport packaging; JSON/Python fences remain code.
- A semantic key/value pair may appear in a paragraph, heading, list or one table
  row. A single-record table may pair headers with values below. Values elsewhere
  on the page cannot be borrowed.
- `table_header` requires column-header structure; a row-label `<th>` is insufficient.
  Generic `table_row` references accept either header or data rows.
- Full table-row credit requires ordered cell matches. Empty padding is ignored
  in this leg. Split item rows receive partial credit rather than an inferred merge.
- Repeated side-by-side column groups may reflow into a narrow table only with
  matching headers and unique first-cell anchors in the same table. Values stay
  with their items; duplicate anchors or changed amounts cannot earn full credit.
- Content inside an oversized heading/list entry or wider table row can receive
  partial structure credit instead of being called absent.

Verdicts expose matched types, token ranges, normalized excerpts and similarities.
Catalogue text is literal; inline markup in older catalogue records is decoded.

### Supporting metrics

Markdown-text CER/WER use the returned answer's rendered text against a trusted
full-page transcript, falling back to the Markdown reference. They do not replace
structure scoring. With a full Markdown reference, block recall counts complete
compatible matches; block precision requires at least 85% of an output block's
tokens to be covered by complete matches. Reading order uses those matches'
positions, including positions inside a block. Sampled catalogues alone cannot
establish full-page block precision or reading order.

## Tables

**42 pages** have human-reviewed gold tables. The headline sums value credit over
reference logical cells. A merged cell counts once; actual blanks count and
parser padding does not.

| Cell comparison | Credit |
| --- | --- |
| Exact normalized text or numerically equivalent amount | 1 |
| Nonnumeric text similarity ≥0.5 | Similarity value |
| Lower similarity, wrong numeric value or missing cell | 0 |

Currency formatting may differ. Signs, percentage units and leading-zero
identifiers remain significant. A blank needs a returned blank or an empty
component established by a combined header/column span. A missing row cannot earn
blank credit. Empty output against nonempty gold scores zero.

### Layout flexibility

Matching preserves logical row/column associations and uses each returned span
once. It does not require identical physical grids:

- Unique descriptions can anchor adjacent continuation rows. Another item,
  subtotal, conflicting label or completion of the record ends the group.
- Unique headers allow added, missing or reordered columns. Complete combined
  headers or `colspan` can establish combined columns; merged values partition
  into ordered, disjoint spans. Values cannot be borrowed from another item.
- Unique descriptive labels or ISO dates identify rows; otherwise positions govern.
- Repeated header groups can reflow. Repeated header/unit bands define sections;
  unique row labels can locate section boundaries when returned headers are absent.
  Units preserve column order and can establish a Notes column combined into a label.
- Matching sections can cross table boundaries. Identical-header chunks may join;
  repeated data rows remain extras. Transposition requires header evidence.

Native HTML is preferred over another Markdown export of the same answer. Every
verdict retains original coordinates, component cells and normalized span offsets.
The report displays the original returned tables alongside the normalized matches.

### Extras and alternative references

Extra cells, rows and tables **do not reduce the headline**. Extra-cell counts/rates,
unmatched tables, table-count differences and strict cell precision/recall/F1 are
separate diagnostics. Structure diagnostics preserve spans and penalize unmatched
tables. The metric named `teds` is a custom row/cell edit-distance diagnostic,
not the official PubTabNet implementation; `structure_similarity` omits text costs.

For shipping, score the whole page against gold and the frozen original reference,
select the higher score, and prefer gold on ties. Keep both comparisons. This
exception applies to every engine; it never mixes cells from different references.

## Shared structured references

KV and schema use the same accepted flat keys and values across **65 pages**,
but their field selections differ:

| Leg | Selection |
| --- | --- |
| KV | Gold exclusions plus `kv_label_review.json`: 3,902 eligible fields; invented or uncertain printed labels excluded. |
| Schema | `schema_sample.json`: 3,316 of 4,251 fields, identical for all seven engines; 45 pages complete, 20 sampled. |

The KV overlay is an image-based agent review, not additional human approval.
Printed section/row/column labels may provide context; an unlabeled company name
does not establish a “Company” key. Array storage does not itself exclude a field.
The overlay does not change schema requests or gold values. Reference/image hashes
protect review and sample validity; changed references require updated selections.

## Key/value extraction

Match known keys first; use value evidence to nominate unresolved candidates:

1. Try exact keys, unambiguous normalized printed aliases and valid cached mappings.
   An explicit native row identity plus column header can also establish a key;
   values never select the row. Ambiguous or conflicting native context cannot.
2. For unresolved fields, compare whole values and complete-token containment in
   either direction. Partial values nominate candidates but do not earn credit.
   Currency glyphs may normalize; `100` must not match inside `1000`.
3. When needed, the judge sees reference-centered candidates and all returned
   pairs, including native context. It may select multiple components for one
   value, including when the returned parent field is empty.
4. Code validates complete value coverage, numeric/identifier integrity and
   one-use component allocation. An empty returned value must identify its output
   component; an empty selection is not a match. There is no Seller/Client-specific rule.

Group approvals bind to the full returned context and reference identity. Only
explicitly approved single-key synonyms with complete matching values can be
reused across contexts. Negative cache decisions do not suppress new partial-value
candidates. Changed context/references invalidate affected decisions. Saved judge
replies are journaled and not automatically retried.

The headline includes correct nonblank and blank fields. Missing keys do not
count as correct blanks. Populated-only accuracy, blank accuracy and extra pairs
are separate diagnostics; extras do not penalize the KV headline. Native Textract
FORMS and table context can establish checkbox values, including explicit false;
an ordinary empty cell cannot establish false.

**Pending reviews:** no headline score is published for a page needing an uncached
review. Its provisional calculation remains visible but is excluded from means.
`--skip-judge` disables new judge calls, not OCR inference. Live evaluation and
rescoring use the same matching, validation and cache path.

## Schema extraction

Supply an answer-free schema with literal requested keys. Score only against the
contract the engine received; changed or unrecorded contracts require new outputs.

The headline is `(correct + blank_ok) / (selected fields + unexpected or duplicate
fields)`. Allowed optional keys and explicitly excluded fields are not extras.
A blank is correct only when its matched key returns blank/null; a missing key
is missing. Values are checked by code without a judge or key-synonym substitution.

Unpenalized accuracy and JSON/schema compliance are separate diagnostics. Recovering
values from fenced or malformed JSON does not make the JSON valid. Textract FORMS
has no supplied-schema interface and does not enter this leg.

## Geometry

**17 pages** have box references: SROIE lines, FUNSD and SynFinTabs words.
A text-matched unit is located when each box contains the other's centre.
Matching is one-to-one. Recall is the headline; precision/F1 expose extra,
duplicated and misplaced predictions. Invalid boxes count as predictions.

Line pages use disjoint matched word fragments and remaining line segments,
labeled `line_segment`. Layout-only engines retain their actual `layout_block`
boxes; no word boxes are invented. Mixed line/table interfaces retain each
region's type. Box text comes from region content, not region-type labels.

Geometry CER/WER use **text from the returned boxes**, including unmatched,
duplicate and invalid-coordinate entries. Both sides use deterministic visual
line order, then left-to-right order, NFKC, case folding and whitespace collapse.
Invalid-coordinate text follows valid boxes in returned order. Default transcription
is never substituted. Empty extra boxes affect precision but not text errors.

Adapters declare coordinate frames. Normalized coordinates are not clipped or
rescaled. Nonfinite, reversed, zero-area and out-of-frame boxes earn no location
credit and remain visible. An unresolved frame is an adapter error.
Sonnet's saved pixel recipe requests coordinates in a pre-resized image with its
exact dimensions; x/y are divided by sent width/height. It uses temperature 0,
a 32,768-token cap and streaming. No inferred rescaling or axis swap is applied.

Precision/F1 and box-text errors use the rectangular envelope of retained reference
annotations, excluding reviewed unreadable regions. Intersecting predictions
participate unless their centres fall in an excluded region; invalid predictions
still incur penalties. Outside-region boxes remain inspectable. This measures
annotated-region performance, not certified full-page precision.

## Judged text

For **11 pages** without reliable text references, a shared image-derived checklist
is evaluated with contextual model verdicts: `correct items / checked items`.
These estimates are separate from deterministic text accuracy and CER/WER.
New verdicts use `silver-document-v2-no-substring-2026-10-06`.

The published report uses cached verdicts that may include substring-matching
errors. They are not fresh evaluations by the current judge, and report generation
does not refresh them. Model-created checklists and verdicts can be wrong.

## Aggregation and reporting

Average pages within each document family, then average family means. Report
coverage with quality. Failed calls, missing references and pending KV reviews
are excluded from score means; they are not zero scores. Difficulty breakdowns
use easy, medium and hard. There is no combined headline across legs.

“Without low” excludes a page/leg only when at least three engines scored it,
all scores are ≤0.25, and no participant failed or is pending. One common exclusion
set is computed before display filtering. Recompute family means; report remaining
coverage. This is a sensitivity analysis, not a corrected leaderboard. Cost and
latency retain their original scored-page cohorts.

Costs belong to unique inference passes: a conversion feeding several legs is
paid once. Modal per-page cost and latency use processing-only batch time;
startup/shutdown is shared session overhead and excluded from volume multiplication.
See [pricing and latency assumptions](engines-and-pricing.md#pricing-audit).

`results/current.zip` contains the published scores, source calls and shared
catalogues/decisions. Local working files override bundled files. Report commands
read saved scores without changing them. An explicit offline rescore recomputes
scores from retained outputs; `--skip-judge` prevents new judging. Source images
are downloaded separately; references ship in the repository.

## Published result limitations

- Some saved calls omit exact prompts or immutable model versions. Recovered
  configuration is labeled separately from per-call evidence.
- Reference omissions, granularity differences and sampled schema fields limit
  what can be concluded from aggregate scores.
- Cached judged-text scores carry the limitation described above. Low scores
  require inspecting source pages and returned evidence.
