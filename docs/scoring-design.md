# Scoring design

Six legs measure text, Markdown, tables, key-value extraction, supplied-schema
extraction and geometry. The silver-document judge is a separate estimate.

**Implementation status, 2026-10-06:** the reviewed-reference scorers below are
implemented and tested offline. The last published scores, run
`20261002T162630Z`, predate these changes. The sampled schema run
`20261006T214240Z` has 453 scored results and two server errors. Markdown
catalogues now cover 92 pages; newspaper_000 remains missing. The remaining
prompt controls and combined all-leg report are still pending. The inline KV rescore
`20261006T195826Z` is complete across all 65 accepted structured-reference pages. The execution checklist is
[follow-up-2026-10-03.md](follow-up-2026-10-03.md).

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
Missing catalogues remain N/A; receipt coverage must be completed explicitly.
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

Content similarity uses normalized full-string edit similarity, not global
word presence. Same type and similarity at least 0.85 earns correct; 0.50–0.85
earns partial; wrong type earns zero. Output blocks cannot be credited twice.
Consecutive paragraph fragments may satisfy one paragraph. Heading depth,
bullet style and trailing punctuation are normalized. Record every expected
block, match and verdict so bad catalogue labels can be reviewed.

CER/WER are computed separately on the actual Markdown answer's rendered text,
using trusted full-page text where available, otherwise the Markdown reference.
Block precision/recall/F1 and reading order are supporting diagnostics. Text
fidelity does not make incorrect structural syntax correct.

## Tables

Human gold at `documents/table_references/gold/` takes precedence over original
references; a proposal is never selected automatically. All 42 reviewed pages
have gold. Schema/KV gold is a separate reference collection.

**Headline: cell F1.** A logical reference/output cell can match only once.
Cells must retain their row and column association; finding the value elsewhere
is insufficient. Explicit blank cells participate. Missing cells lose recall;
wrong or extra cells lose precision; duplicate rows/tables are extra predictions.
Parser padding is never a real blank cell. A merged cell counts once.

`precision = correct cells / output cells`, `recall = correct cells / reference
cells`, and F1 is their harmonic mean. Empty output against nonempty gold scores
zero. Cell text uses NFKC, case folding and collapsed whitespace while preserving
punctuation and digits. This is currently stricter than field amount normalization.

Tables are assigned one-to-one. Unique first-row headers can identify column
permutations. Unique descriptive row labels anchor row alignment; otherwise
positions govern. Every cell verdict retains the association used. Native HTML
is preferred over a second Markdown export of the same answer; duplicate tables
within the native result still count as duplicates.

Supporting structure similarity preserves `rowspan`/`colspan`, ignores decorative
HTML wrappers/attributes and penalizes unmatched tables. The current `teds` metric
uses a custom ordered row/cell edit-distance diagnostic with cell-text costs;
`structure_similarity` removes text costs. Do not describe it as the official
PubTabNet APTED implementation. Table pairing and normalization are recorded.

**Shipping exception:** evaluate the whole page against gold and the frozen
original separately. Select the higher cell F1, preferring gold on ties, and
retain both comparisons. Never choose individual cells from different references.
This policy applies to every engine.

Promptable engines receive an explicit table-only request allowing Markdown for
simple grids and HTML for spans. Native pipelines retain their documented table
interface. Table scope, ambiguous glyphs and equivalent shipment representations
still require the reference metadata checks listed in the follow-up.

## Shared KV/schema references

The accepted fresh, flat key/value extractions at
`documents/structured_references/gold/` supply the same accepted keys and expected values for both legs. Storage shape
does not disqualify a KV field. KV uses the full eligible set. For the current
schema run, the user requested one shared field sample sized to fit every engine:
`documents/schema_sample.json` selects 3,316 of 4,251 fields, identically for all
seven engines. Forty-five pages keep every field; twenty are sampled. Schema
requests and scoring both use that frozen selection; unexpected keys still
incur the extra-key penalty. Full gold and KV coverage are unchanged. Reference
edits invalidate the sample. There is no three-field minimum. The corpus has 65
structured pages; uncertain readings remain explicitly excluded from both legs.
The old nested references remain available as provenance, not the active target.

**KV now resolves wording during evaluation.** Match literal keys, unambiguous
printed aliases and cached confirmations first. For an unmatched key, compare
its complete value against the reference values. No value candidate means
incorrect/unmatched without a judge call. A plausible candidate is not credit:
Gemini 2.5 Flash sees the returned pair and the page's accepted reference fields
and must confirm the field's identity. Equal totals/subtotals, repeated dates,
zeros and blanks are not enough to disambiguate a field.

Confirmed key variations are cached immediately and reused by subsequent engines.
If the judge needs the value to identify the field, cache the exact pair, not an
unconditional key alias. Repeated reference labels also stay pair-specific even
if the judge proposes a wording alias. Replan later batches after each saved
confirmation. Code still checks every returned value after matching;
a confirmed synonym cannot make a future wrong value correct. Reference changes
invalidate the page cache. One output pair can satisfy only one reference field.
`100` is not `1000`, and `12.34` is not `1234`.

The shared path is `evaluation/keyvalue_leg.py` → `kv_value_judge.py`; live runs,
saved-run scoring, rescoring and future engines all use it. `--skip-judge` reads
existing confirmations but makes no new call; unresolved plausible pairs stay
pending. Failed/invalid paid replies remain journaled and are not automatically
retried. Explicit no-match decisions are cached. No separate synonym batch is
needed; the old key-only batch remains archived and is not active.

KV headline remains `(correct + blank_ok) / selected fields`. Extra pairs are
visible but do not directly penalize incomplete-reference pages. Native Textract
uses FORMS. All structured reference fields participate (including array leaves);
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
Coarse layout blocks have separate diagnostics, not word scores.

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
