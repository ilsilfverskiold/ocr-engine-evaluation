# Changelog

Notable changes to the scoring code. The reference run is never modified;
every change is rescored from stored outputs (`--rescore`) and the run id
is given here.

## 2026-10-02 — audit corrections, publication still pending

- Field scorer v10 treats the independent nonresident-spouse election
  separately from the five filing statuses, and rejects unknown group values
  as evidence of unchecked options. Replay changes three KV scores; explicit
  states and recognized selections remain supported.

- Field scorer v9 preserves numeric label identifiers and explicit field
  ownership. Missing array cells cannot borrow scalar summaries. Isolated
  replay changes four schema scores; all 220 KV scores stay unchanged. It
  also corrects 366 missing fields previously diagnosed with unrelated values.

- Geometry parser v4 recovers a missing opening box-array bracket only when
  all four coordinates are explicitly present, recording each repair. This
  corrects one new diagnostic score (65/222 → 79/222); none of the 145 archived
  geometry scores or parsed-box counts change.

- Approved geometry diagnostic run has 13 audited responses, one timeout and
  seven unattempted calls. Preserve actual boxes, request frames and verdicts.
  Pixel/dialect prompts do not establish a general fix on the completed pages;
  the comparison is partial and does not replace historical benchmark scores.

- Geometry scoring accepts explicitly converted coordinate frames without
  applying a second scale heuristic; out-of-frame boxes remain evidence.
  Existing legacy-call behavior is unchanged.

- Field scorer v8 preserves numeric signs and exact precision; malformed
  decimal punctuation cannot fall through to punctuation-stripped equality.
  Preserve legitimate identifier/count/printed-amount formatting. Isolated
  replay changes none of the 675 saved KV/schema scores or verdicts.

- Source-image review corrects chart_003's text reference (inferred bar values
  are not printed text) and excludes its unverified exact numeric series from
  schema targets. Six scanned references no longer require unprinted source
  citations. Isolated replay changes 98 text and seven schema scores; all 11
  chart Markdown headlines stay unchanged. Historical sidecars stay intact.

- Marker converters receive isolated configuration copies: OCRConverter's
  mutation can no longer leak into a later document/table call in a persistent
  worker. Preserve the full OCR renderer tree and exception traceback. An
  explicit, unselected all-blocks probe disables the OCR builder's skipped
  block classes; its quality has not been measured.

- Reproduced GLM SDK's hosted-mode failure using glmocr 0.1.5's actual
  config loader. Pin the SDK and use its accepted self-hosted arguments,
  reject unexpected endpoint overrides, retain raw SDK regions and mark
  their boxes as layout blocks. Default registry routing is unchanged;
  the explicit pipeline probe still needs deployment and quality validation.

- Report explanations now use primary per-leg verdicts. The previous helper
  could claim most Markdown blocks were absent despite zero absent checklist
  items (145 archived results). Geometry distinguishes unmatched text units
  from matched boxes failing containment; text metrics no longer assert an
  unverified recognition failure or repetition loop as the cause.

- Geometry-v3 keeps every parsed and scored box, including unmatched and
  out-of-frame predictions, and preserves actual returned text for fuzzy
  line matches. Known native xyxy coordinates no longer undergo the legacy
  Gemini axis heuristic. Checked all 145 scores and overlays on all 17 pages.

- Field v7 accepts source-verified trailing lease unit labels, preserves
  explicit field attribution, and avoids assigning an initial-term value
  to an ambiguous blank renewal field. Isolated replay changes 11 of 675
  field scores across four engines; historical scores remain untouched.

- Textract's adapter now reads native MERGED_CELL relationships and retains
  elementary cells alongside merged groups. Stored calls are readapted from
  provider blocks. Nine of 97 table/KV scores improve in an isolated replay;
  source response files remain untouched.

- Table code-v5 distinguishes malformed OCR amounts from valid currency
  formatting. Field v6 adds quantity-scoped Units and unambiguous US date
  formatting, while disallowing numeric shared-label substring credit.

- Geometry-v2 assigns repeated words globally, preventing an early missing
  occurrence from consuming a later correct box. All 145 saved rows replayed;
  27 improve. Future details retain every scored reference/prediction box.

- Table code-v4 rejects amounts under the wrong named row or year, preserves
  HTML spans, and uses unsampled header context. Corrected four receipt
  header coordinates from the source image without changing the saved cache.
  Offline replay changes 125 of 504 table scores; archived results unchanged.

- Separate nullable inference schemas with explicit targets repair conflicting
  null instructions and open dictionaries. Nullable nested GLM templates keep
  their structure; exact request schemas are persisted for future calls.
- Versioned loader correction repairs fintab_001's merged-header column
  mapping and removes unrequested CSV targets from five financial references.
  An isolated replay changes 28/35 financial schema scores. Original raw
  sidecars, calls and archived scores remain preserved.

- Field scorer v5 preserves primitive-array positions, prioritizes an exact
  schema path over sibling leaf-name matches, and accepts RM currency
  formatting. Offline candidate replay changes 32 field scores; archived
  v4 results are unchanged and no corrected final ranking is claimed.
- Geometry fallback reads Qwen text_content/bbox_2d from valid objects in
  malformed/truncated replies. Three pages recover boxes; word scores remain
  zero. Regression coverage and the full 204-test suite pass.
- Causal audit found 19 GT/schema conflicts and an effective empty
  LlamaExtract schema. The prior report is provisional pending those repairs
  and targeted pipeline/prompt checks; see the handoff.

## 2026-10-02 — default prompt provenance

- Gemini and Claude now persist their actual default prompt through runner
  metadata. This repairs logging only; prompts and generation settings are
  unchanged. Four mocked request/persistence regressions pass.

## 2026-10-02 — final offline rescore `20261002T095730Z`

- `kv-schema-code-v4`: explicit semantic keys take precedence for blank
  fields; preserve the repeated-label lease blank policy without borrowing a
  dependent's blank to excuse an invented spouse surname.
- `--skip-judge` retains KV/schema; removed table-judge costs from estimates;
  reports distinguish current code verdicts from historical saved judge scores.
- Final rescore from the original run plus the two geometry sources. Exactly
  three score changes from 093008, all lease_004 KV renewalMonths. Textract's
  blank match remains ambiguous; the final handoff records the exact evidence.
- Final evidence report uses the same page in each engine comparison and
  resolves outputs through source_timestamp and pass_name. Current docs refer
  to the final run; superseded 084627/093008 run artifacts removed.
- Validated with 195 full-suite tests plus 120 targeted tests after CLI cleanup,
  successful report/cost audit and source-file hash checks. No paid calls.

## 2026-10-01 — code-v3 / kv-schema-code-v3 (superseded run `20261002T084627Z`, deleted)

- key_value: keys matched by the same words in another order; a block value
  split over several keys; a date printed as several blanks; a distinctive
  value under a key that names no other field when no key matched or the
  page's "labels" are sentence fragments; "Label: value" lines inside a
  multi-line value read as pairs; duplicate JSON keys kept; every fenced
  block of a multi-block answer read; string values compared without
  spaces. A page without
  a stage-1 file or without any printed pair is not applicable (chart_003).
- schema: array rows aligned one-to-one to the engine's rows by shared
  values; a cell judged in its own row (blank cells no longer "invented"
  from other rows); a printed dash or n/a counts as blank.
- tables: only the checklist tables that correspond to the human table
  reference are scored (`TABLE_POLICY["human_marked_tables_only"]`).
- markdown/tables: a line of several check-list items is that many list
  items; an explicit ```markdown fence is unwrapped even with a note after
  it or no closing fence; a pipe table glued to the line above it is parsed
  (`table_extractor`).
- key_value: a label that names several things ("first name and middle
  initial") allows the reference's part of the engine's value.
- geometry: marker's empty OCR passes are error rows, not 0.00.
- Why: `docs/handoff-2026-10-01.md` §3a, §3b, §3f.

## 2026-10-01 — code-v2 / kv-schema-code-v2 (run `20261001T132323Z`, deleted)

- Markdown and tables scored by code against the cached image checklists;
  key_value and schema by parsed pairs, printed labels (stage 1) and exact
  values; geometry repeated-value pairing fix, line-level number, block
  engines not applicable; `--rescore` command; retried calls are not error
  rows. Why: `docs/rescoring-plan.md`, `docs/handoff-2026-10-01.md`.
