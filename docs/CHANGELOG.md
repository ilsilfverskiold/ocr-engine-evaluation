# Changelog

Notable changes to the scoring code. The reference run is never modified;
every change is rescored from stored outputs (`--rescore`) and the run id
is given here.

## 2026-10-06 — sampled schema results

- Run 20261006T214240Z consolidates 453 scored results and two provider errors
  across seven engines and 65 pages using identical sampled schemas.
- Save request/scoring/cost audit and actual field evidence; estimated total
  $4.72786 including measured GPU startup. No new judge or repeated successful call.

## 2026-10-06 — identical bounded schema samples

- Freeze one common sample per page for all seven engines (3,316/4,251 fields).
  Request and scorer use the same keys; full gold and KV coverage are preserved.
- Add resume-unattempted to avoid paying again for previously attempted calls.
- Preserve LlamaExtract Multimodal mode with stateless one-job submission and
  correct nominal extraction cost; display actual sample/schema in the viewer.

## 2026-10-06 — API retry and usage accounting

- Disable automatic Claude/Gemini SDK retries and persist generation settings.
- Include reported Gemini thinking tokens in billed-output cost.
- Keep existing generation caps; document full-corpus offline sizing before
  changing caps or submitting schema calls.

## 2026-10-06 — exact dry-run request plans

- Export task prompts/schemas, source/configuration hashes and Modal settings
  with --dry-run --plan-output. Count only eligible page/pass combinations using
  the same rule as live inference; forbid combining dry-run with execution.
- Schema rerun preparation is 455 calls, not authorization or completed inference.

## 2026-10-06 — reused 57 Markdown catalogues; added 35 missing

- Existing cache and its 627 scores are unchanged. New image-only generation
  cost $0.64341; accepted 35 source-reviewed catalogues, retaining corrections,
  exclusions and original responses. newspaper_000 returned no catalogue.
- Free rescore 20261006T202641Z: 908 results scored across 92 pages, zero errors.
  No additional engine inference; new checklist creation is cached permanently.
- Fix provider response-schema rejection; validate one request before batch
  submission, preserve existing caches at execution time, report no-content stops.

## 2026-10-06 — reviewed catalogues for missing Markdown pages

- Prepare exact image-only catalogue requests and keep generated proposals
  inactive until source-image review. Preserve source hashes and paid journals.
- Allow reviewed structural catalogues without a full Markdown sidecar; do not
  derive CER/WER or full-page block statistics from a sampled catalogue.
- Remove obsolete KV execution plans from the current follow-up and align
  README scoring formulas with the implemented scorers. No paid generation yet.

## 2026-10-06 — completed KV rescore

- `20261006T195826Z`: 260 scored results / 65 pages, four engines, no pending
  KV decisions or scoring errors. Completed cache replay makes no new calls.
- 342 value-assisted calls cost $1.4755785. Preserve all request/response
  journals, active caches and a hashed audit ledger. Keep final results only;
  intermediate runs remain in local scratch. Schema/other legs remain separate.

## 2026-10-06 — value-assisted KV evaluation during scoring

- Replace separate key-only synonym generation with a shared, incremental KV
  evaluator: match known keys, nominate unmatched pairs by complete value, judge
  field identity only when needed, and cache confirmations/rejections.
- Keep contextual/repeated-label confirmations tied to their key/value pair.
  Check every value locally after key matching; all selected shared reference
  fields participate. Schema evaluation is unchanged.
- Add bounded spending, journal recovery, concurrent page-safe caching,
  cached-only replay and KV-only rescore filtering. Historical key-only calls
  remain archived and inactive.

## 2026-10-06 — bounded compact key-only preparation

- Share the tested compact JSON request builder between controls and corpus
  preparation. Execution journals and held-cache policy are unchanged.
- Full replacement is prepared but on hold; review existing paid decisions for
  reuse before proposing additional spending. No new calls or active synonyms.

## 2026-10-06 — portable rescoring and current evidence viewer

- Preserve original call provenance across repeated rescoring and relocate
  document paths in fresh checkouts. Add document-type filtering and current
  scorer-specific tables, boxes, Markdown, field evidence and text metrics.
- Free rescore checkpoint retained outside git for verification; unfinished
  schema/KV/reference work remains clearly pending. Published report unchanged.

## 2026-10-06 — excluded fields do not block KV completion

- Respect explicit human field exclusions before requesting key synonyms.
  Shared schema/KV exclusion regression and the full offline suite pass.

## 2026-10-06 — keep unfinished KV mapping out of published scores

- Mark results with unreviewed keys pending instead of scoring missing synonyms
  as engine failures. Retain provisional details and explicit cached no-matches.
- Match equivalent explicit printed tax-line identifiers deterministically,
  closing the control miss without another call. 109 relevant tests pass.

## 2026-10-06 — geometry diagnostics respect annotated scope

- Add region and unreadable-area exclusions for extra-box precision/F1 and
  box-text CER/WER, while retaining all returned boxes as evidence. Invalid and
  out-of-frame predictions remain counted. Record geometry-v6 and scope limits.
- Validate with 66 geometry/loader tests; no engine or judge calls.

## 2026-10-06 — ship accepted references and preserve downloaded source identity

- Include 42 table gold, 65 shared structured gold and five original shipping
  reference files. Preserve their accepted content; drafts/history stay local.
- Download original encoded images with manifest checks and resumable sidecars,
  refusing to overwrite local edits. Validate with 24 loader/download tests.

## 2026-10-06 — record approved KV control results

- Six approved calls, no retries: all replies complete, 40/41 control decisions
  correct, no incorrect associations and one missed explicit tax-line synonym.
  Actual usage cost $0.0174635. No scores or caches activated.
- Keep experimental plan/replies/audit outside GitHub. Record the narrow miss
  and next action in the existing follow-up; no larger batch or schema calls.

## 2026-10-06 — prepare compact KV correction controls

- Add offline preparation and grading of key-only controls using saved output
  keys, compact bounded-ID replies and explicit identity/ambiguity instructions.
  Keep expected decisions outside provider requests. Reject duplicate JSON
  properties, incomplete replies and attempts to publish control subsets.
- Hold executed synonym batches for semantic review instead of activating them
  merely because their JSON validates. Preserve request failures in journals.
- Prepare six calls / 41 cases in scratch, reserve $0.0772059, unrun. Recover
  one 54-decision formatting candidate offline without overwriting paid evidence
  or approving its meanings. No scores, gold references or OCR outputs changed.
- Validation: 35 offline synonym/shared-field tests pass. No paid calls.

## 2026-10-06 — execute approved KV synonym batch; hold faulty decisions

- Attempt all 117 approved requests once. 101 replies validate, 13 truncate,
  2 contain invalid output IDs and 1 returns HTTP 503. Save exact plan,
  provider replies, usage, execution summary and bounded semantic audit.
  Recorded usage $0.6900703; retain $0.013288 reserve for the no-usage 503.
- Keep the cache inactive after finding wrong/unsupported entity assignments.
  No benchmark scores or schema/OCR outputs change. These are judge-pipeline
  problems, not engine scores. No paid retry or other paid stage ran.
- Continue independent approved requests after malformed replies, with four
  workers, exclusive request journals, unchanged model/prompts/caps and a
  prechecked total reserve. Tests cover continuing without retrying or
  publishing incomplete decisions.

## 2026-10-06 — reviewed-reference scoring implementation

- Add an explicit offline-plan/paid-execute CLI for KV key-only synonym batches.
  Validate all IDs, preserve ambiguous/no-match decisions, journal submitted
  calls to prevent accidental paid retries, retain usage/full replies and freeze
  caches for all engines together. Prepared 117 calls with a $1.55423 reserve;
  execution awaits approval. No live judge decisions in this change.

- Gold tables now use cell F1 with precision/recall, blanks, extras and disjoint
  matching. Shipping retains both gold/original comparisons and selects the
  higher whole-page F1. Spans remain in the supporting structure diagnostic.
- Markdown requires actual structural syntax and disjoint content matches;
  separate CER/WER measure its returned text. Geometry adds precision/F1,
  invalid/extra diagnostics and CER/WER from returned box text.
- Shared KV/schema gold uses key-first matching and identical field exclusions.
  Schema extras reduce its headline; JSON/contract validity stays separate.
  Old or missing saved request contracts require rerun. Synonym cache creation,
  annotation-scope handling, missing Markdown references and new scores remain
  pending. No paid calls or replacement report in this checkpoint.
- Restore the original tax_000 image identity after verifying an exact rotation;
  preserve reviewed values and record the transformation. Refresh scoring docs
  to remove obsolete current-method descriptions.

## 2026-10-06 — remove silver-document substring auto-credit

- Remove punctuation-stripped substring acceptance from the silver-document
  judge. Every sampled item receives a contextual verdict; instructions preserve
  numeric meaning and distinguish the intended field from unrelated values.
- Version new judge-result metrics. Offline regression tests use a mocked judge;
  no paid calls or historical score changes. Refreshing estimates remains pending.
- Record agreed geometry precision/F1 and CER/WER on box-associated text as
  follow-up work; no geometry scoring implementation or rerun in this change.

## 2026-10-06 — separate human structured references for schema and KV

- User accepted and activated all 65 fresh references for both legs. Preserve
  flagged exclusions and back up prior bank_000 gold. Prepare explicit fresh
  schemas without applying overrides intended for legacy open dictionaries.
  New schema extraction and KV rescoring remain unrun.

- Replace earlier machine proposals with 65 independent source-image
  extractions for human review: flat readable keys/values, shared schema/KV
  selection and uncertainty notes. Fresh JSON edits regenerate the schema;
  saved gold retains source-extraction provenance. Existing gold is preserved.

- Final review workflow uses one shared field selection: editable keys/values,
  original-to-patched differences and Include in both. For these new shared
  saves, schema and KV score the same complete selected fields, including
  array leaves. No automatic array exclusion, three-scalar minimum or checklist
  sampling. Historical scores and old saves remain unchanged until resaved.

- Add a source-verified structured gold override for shared schema/KV values,
  schema, KV labels and explicit exclusions. Old references remain the fallback;
  proposals are never automatically selected and table gold is independent.
- Add a local review/save server using the existing atomic-save and backup
  protocol. The review workspace covers 65 pages; follow-up coverage expansion
  brings provisional changes to 54 pages, including new targets on 48 pages.
- Expose scalar KV checkboxes and an additive proposal merge that preserves
  saved corrections and eligibility choices. Save original reference scope
  and new field patterns as provenance; historical scoring still needs to
  distinguish newly requested fields from corrected old targets.
- Preserve declared child fields when a reviewed schema makes an object
  nullable. No new scores or engine/judge calls; human approval and rescore
  remain pending.

## 2026-10-05 — separate human gold table references from proposals

- GroundTruthLoader uses a source-verified gold table slot when present and
  otherwise retains the original reference. Proposed patches are never an
  automatic fallback; other legs are unchanged.
- Local offline editor supports review of all 42 table pages and saved engine
  evidence, accepting old/patched references, editing, saving and version
  backups. The 24 proposed patches are separated from restored originals.
- Saves now use a local HTTP server and the existing
  `documents/table_references/gold/` folder, without browser folder permissions.
- Existing scores are unchanged. Gold-based table headline scoring and the
  uniform rescore remain pending human reference review.

## 2026-10-03 — report the actual reason for N/A

- Missing-reference rows now show their saved exclusion reason instead of a
  failure diagnosis inferred from absent metrics. Unscored outputs are labeled
  accordingly; no empty structural verdict table is shown. Scores unchanged.

## 2026-10-03 — disclose Markdown heading overcredit

- Flag bold-as-heading and reused-block credit in saved Markdown results;
  invoice_000's 16/16 does not establish correct document structure.
- CER/WER remains in text results only. Saved scores are unchanged;
  structural scorer correction remains outstanding.

## 2026-10-03 — make text evidence and metric scope visible

- Text results show the normalized comparison and CER/WER/word metrics, with
  raw formatting expandable. Label the headline as text coverage. Verify
  reproduced coverage against saved scores while building the viewer.
- Separate engine-wide audit context from page-specific observations. Explain
  Claude invoice_000's complete token coverage and column-order differences.
  Saved benchmark scores are unchanged.

## 2026-10-02 — corrected saved report delivered locally

- Deliver run `20261002T162630Z` and a reproducible browser evidence viewer.
  All 3,099 scored results have arithmetic/provenance checks; 554 scores change
  after confirmed repairs. Unrun configuration controls remain disclosed.
  Remove the superseded provisional run/export; retain original raw calls.
- Historical terminal reports now preserve saved applicability and scores in
  both display and CSVs. They no longer silently apply today's ground truth.
  Reference labels no longer imply universal engine-consensus verification.
  Nineteen report tests pass. No new inference/judge calls or push.

## 2026-10-02 — earlier audit corrections

- The sroie_003 schema reference now represents one purchased item with its
  two-line description, instead of two items and three nonexistent blanks.
  The supplied request schema stays unchanged. All seven saved schema scores
  change, including Gemini 0.9167 → 1 and DeepSeek's empty answer 0.125 → 0;
  the other 39 scored results on the page remain unchanged.

- Receipt reference repairs restore omitted address text in five pages and
  their existing full-line box labels, remove one duplicate box and exclude
  one unreadable line. Retained coordinates stay unchanged. Fix one B05/805
  structured reference from the image. Isolated replay changes 70 text,
  33 geometry and six schema scores; table/KV headlines stay unchanged.
  Partial model-review provenance and per-result exclusions are disclosed.

- Invoice references now use six source-image-reviewed transcriptions,
  replacing phantom punctuation/numbers and incorrect spacing/reading order.
  Exclude two unreadable description fields from schema exact matching,
  retain their readable text prefixes, and correct one verified slash.
  Replay changes 84 text and 14 schema scores; MD/table/KV headlines stay
  unchanged. Replacement text provenance is explicitly model-reviewed.

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
