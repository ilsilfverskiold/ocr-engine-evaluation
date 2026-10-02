# Work log

## 2026-09-30 — OCR engine expansion research

Requested scope: research the expanded OCR list and write findings for another
agent to implement. Do not build engines or run judges yet. Commit work only
inside the `ocr-benchmark` repository.

Completed:

- Read the local Modal inference engineering skill and its model research,
  architecture and benchmark case-study references.
- Inspected the engine registry, runners, Modal apps, scoring legs, corpus,
  persistence/resume/report behavior and saved reference run `20260810T130609Z`.
- Researched official model cards, repositories, service APIs and Modal GPU
  pricing. Captured eight model revision IDs without downloading weights.
- Consolidated the list into 16 integrations and documented model-specific
  pipelines, batching, inputs/outputs, candidate hardware and benchmark fit.
- Verified that the saved Paddle results used v1, not 1.6; recorded the
  historical Mistral `latest` alias ambiguity and the older DeepSeek/MinerU
  identities.
- Identified deferred-judging and reference-run extension requirements, plus
  the Apple document API's macOS 26 requirement versus this host's macOS 15.6.
- Wrote [the implementation handoff](engine-expansion-research.md), including
  unresolved checks and acceptance criteria.

Validation: inspected persisted inference identities and a Mistral provider
payload; verified model metadata/configuration, Apple availability metadata,
local OS/toolchain, and credential presence without printing secret values.
Checked Markdown links to local files and Git whitespace. No runtime tests
were needed for this documentation-only change.

No engine/scorer code, reference outputs, credentials, or deployment settings
were changed. No OCR inference, GPU deployment, paid benchmark or judge ran.
The parent repository was not staged or committed.

Next: the implementation agent should read the handoff and referenced Modal
skill, resolve the remaining dependency/asset contracts, and build within this
repository. Record changes and validation here as work progresses.

## 2026-09-30 — Benchmark-leg audit and preliminary costs

User clarification: establish which benchmark legs each candidate can enter,
how those legs are scored, and preliminary implementation/run costs. Continue
research/documentation only; another agent will implement.

- Re-read executable pass planning, registry, GT eligibility, scorers and
  no-judge CLI behavior. Enumerated all existing engines' pass plans locally.
- Computed live corpus counts: 93 total; 82 text, 57 Markdown, 42 tables,
  65 schema, 17 geometry. KV has 65 pass-eligible but 56 scoreable references.
  Confirmed no-judge GT filtering retains only 82 pages; grounded
  Markdown/table/geometry union is 63.
- Recorded stale geometry/table/KV prose and the historical DeepSeek KV
  exclusion's conflict with the stated interface-based eligibility rule.
- Checked upstream Chandra and Surya custom-prompt interfaces; expanded
  extraction candidates instead of assuming that OCR task examples define
  their entire callable interface.
- Wrote [leg/pass/cost matrix](engine-leg-and-cost-matrix.md), including all
  16 distinct engines, unresolved contracts, deterministic versus judged
  metrics, API estimates, request rounding and conditional Modal scenarios.
- Linked the original handoff to the more precise matrix and added the new
  Surya/Chandra interface findings.

Validation: read-only Python imports of the actual corpus loader and pass
planner; arithmetic checked locally; public primary-source prices rechecked;
local documentation links and Git whitespace checked. No engine/scorer code,
run outputs or deployments changed; no inference or judge calls were made.
Only this nested repository's documentation is included in the commit.

## 2026-09-30 — Targeted audit of existing benchmark flaws

- Reproduced three false-positive local schema/KV verdicts without calling a
  judge: numeric substring, decimal stripping, and wrong-field association.
- Recorded schema-conformance measurement gaps and insufficient verdict-cache
  identity, distinguishing demonstrated code behavior from unmeasured impact
  on historical results.
- Rechecked saved Mistral identities and error recovery. Five Marker geometry
  document/leg errors remain without a successful replacement; other saved
  errors were recovered. Accuracy aggregation excludes error rows.
- Added the findings to the leg/cost document. No benchmark code, cached
  verdicts or saved scores changed. Validation used read-only inspection,
  local verifier calls, and Git whitespace checks; no paid/model calls ran.

## 2026-09-30 — Judge rationale and questions before continuing

- Clarified that KV uses the shared judge pipeline: local acceptance first,
  LLM verdicts for unresolved fields, or previously cached verdicts.
- Checked documentation and available Git history for the design rationale:
  fair treatment of different field names/output formats, with a local cost
  shortcut. Historical safety and claimed match-rate estimates remain
  unverified by this audit.
- Added explicit questions covering actual incidence, verdict origin,
  acceptable normalisation, field association, measured savings, cache reuse
  and ranking impact before choosing fixes or interpreting baseline impact.
- Documentation only. Git whitespace checked; no engine/scorer edits, cache
  mutations, inference, or judge calls.

## 2026-09-30 — Implementation of the 16 expansion engines, harness changes, audit

Requested: audit the existing benchmark, implement the researched engines and
decide where each is tested. No runs, deploys or judge calls were requested.

- Built 11 Modal apps, 4 cloud runners and 1 local runner (Swift helper)
  under the shared contract in [the implementation document](engine-expansion-implementation.md);
  16 registry entries, all `default=False`; the 14 baseline engines untouched.
- Harness: `--collect-only`, `--score TS`, `--extends TS`, `--docs`, combined
  `--report A,B`; manifest merge on resume; `CPU` container rate; unknown GPU
  types raise; judged-documents section now prints; T4 default removed.
- Tests: `tests/conftest.py` strips paid credentials; membership test split
  into baseline/expansion; cache-identity defect marked `xfail(strict)`.
- Audit written up in [benchmark-audit-2026-09-30.md](benchmark-audit-2026-09-30.md):
  local-verifier incidence and bounds, 155 stale verdict sets, geometry
  adapter artefacts (deepseek/paddle 0.0), mineru markdown input change,
  silver judge on 10/14 engines, report/bookkeeping defects. A first draft
  misdescribed the markdown/table judges (which grade against Gemini's
  reading of the page image, not the human sidecars) and mislabelled an
  expected property as a defect; the document was rewritten after reading
  scoring-design.md and all four judge modules in full, and every number in
  it was reproduced in-session.
- Harness: the document judge now follows the text-serving pass (not the
  pass named `default`), and collect-only runs no longer skip that pass on
  the 11 silver-text pages — the two gates behind the 10-of-14 coverage.
- Corrections to the research documents and to existing code are recorded in
  the implementation document §7 rather than edited into the research text.

Validation: `py_compile` on every new file; the offline unit suite (152
passed, 2 xfailed). **Incident:** the first test run reached Gemini through a
pre-existing test (57 `gemini-3.6-flash` verdict calls, engine name empty);
the 57 cache files it created were deleted and the suite is now guarded. No
engine, deploy, benchmark or other judge call was made. No git operations.

Next: the user deploys and smoke-tests one app at a time, runs the
cross-section collect-only, then a full extension run with `--extends`;
scoring path and the baseline fixes are decisions listed in the
implementation document §9.

## 2026-10-01 — rescoring the reference run with code scorers (handoff in handoff-2026-10-01.md)

- Scorers: markdown and tables by code against the cached image checklists
  (`evaluation/code_judges.py`); key_value/schema by parsed pairs, printed
  labels (stage 1, `evaluation/alias_pass.py`, 55 cached files) and exact
  values (`evaluation/kv_schema_code.py`); geometry repeated-value pairing
  fix, line-level number, block engines N/A + block diagnostic, DeepSeek/
  Paddle/marker adapters read from disk. Error rows for every failed call or
  scorer (`<ts>_errors.csv`); one eligibility rule per leg; per-leg cost from
  the serving pass; gemini-3.6-flash price corrected; `--rescore` command.
- Clean rescore `20261001T121908Z` of `20260810T130609Z` + the two geometry
  extension runs, on the reference run's own pages per leg (table in the
  handoff). 148 tests pass. 110 document-judge rows copied, not re-asked.
- Two line-box engines added and run on the 17 geometry pages only:
  `paddleocr_ppocrv5` (0.816), `mineru_pipeline` (0.108 / line-level 0.501).
- Validated: every number in the handoff reproduced from the run files; the
  dry-run rescore and an interim rescore with changed page sets were deleted.
- Not run: nothing else. Paid: alias pass $2.8 (≈$1.8 wasted on pages and
  fields the scorer never reads — removed from the cache), Claude pixel-prompt
  geometry test $0.12.
- Cleanup: 14 prototype HTML pages and `docs/deterministic/` removed; the
  interim status note merged into the handoff. No git operations.

## 2026-10-01 — engine-by-engine audit of the rescored run, scorer fixes, final rescore `20261001T132323Z`

- Audited every engine on every leg from `detail.verdicts` (not just Claude
  and llamaparse). Scorer defects fixed: camelCase field names never split;
  line-number labels (`5b`) rejected; no checkbox vocabulary; radio groups
  and lists of checked items earned nothing; a printed label shared by two
  fields could never be satisfied; Textract checkbox values dropped as "";
  label containment was symmetric (a short engine key matched every field
  under it); checkbox lines typed paragraph in markdown; heading-as-row and
  list-item-as-cell wrong_type; table text cells with one OCR error wrong;
  "inside a larger cell" on bare substrings. Harness: retried calls were
  carried as 74 error rows; marker error rows once per attempt; marker read
  from its `layout_block` pass instead of its `ocr_line` pass.
- `code_judges` → code-v2, `kv_schema_code` → kv-schema-code-v2;
  `runners/textract_structured_runner.py` shares `textract_forms`;
  `tests/test_code_scorer_rules.py` (18 cases). 166 tests pass.
- Rescored three times (free, keys stripped); intermediate runs
  `20261001T122709Z` (interrupted), `125643Z`, `131614Z` deleted; final
  `20261001T132323Z` + `--report`. Page sets 82/57/42/56/65/17; 5 error rows
  (marker OCR pass crashes). Handoff §2–§3 rewritten from that run.
- Verified at the CLI: `--report 20261001T132323Z` renders, cost audit
  passes, `_errors.csv` has the 5 rows; `--rescore` of a nonexistent run
  exits with "No inference records" and writes no run files.
- Validated samples of every new rule's verdicts for false positives
  (radio-group 84 correct / 30 wrong, shared-label 77, heading-as-row 166,
  list-item-as-cell 193) and the payslip "0 for blank" case against the image.
- Not run: nothing paid, no Modal, no engine, no judge. No git operations.
- Review page (funsd_004 × Claude/Gemini/Qwen, every leg) added as
  `docs/how-scored-funsd_004.html` at the user's request; `docs/scoring-design.md` got a status note
  pointing to the code modules and the handoff.
- Follow-up the same day: the three caveats were restated with evidence in
  `docs/open-questions-2026-10-01.html` (built from the run files: lease
  pages, invoice seller/client, llamaparse/marker/document-judge holes with
  retry manifests, the three markdown policy switches with each engine's
  score per switch). Corrections found while building it: lease pages score
  0.59–0.75 for Claude, not "0.1–0.4"; the invoice block is satisfied by
  Gemini (11/12) so it is the engines' representation, not the GT; the
  unjudged engines are deepseek_ocr, glm_ocr, marker, qwen3_vl. The three
  markdown choices became switches (`code_judges.MARKDOWN_POLICY`). Handoff
  §6 corrected. No paid call made; the two retry manifests await a go.
- End of day: the owner asked for a handoff to someone else.
  `docs/handoff-2026-10-01.md` rewritten for that reader: the goal in the
  owner's words, the working method, the exact state (code-v2 /
  kv-schema-code-v2, run `20261001T132323Z`, 166 tests), the open fairness
  problems with evidence and the fix each needs (lease keys, invoice block
  for Claude/Textract/Qwen, llamaparse and marker error rows, 4 unjudged
  engines, rows not yet read one by one, the three markdown switches), what
  was verified fair, the numbers marked not final. A matcher change for the
  lease keys (word-set key match + exact-value fallback) was drafted, then
  reverted so code, tests and the run agree; it is described in the handoff
  §3a. No paid call made.

## 2026-10-01 — second pass on the handoff: key_value/schema/tables fairness fixes, rescore `20261002T084627Z`

Requested: continue from `docs/handoff-2026-10-01.md` towards a fair
comparison on text, tables, schema, key_value and geometry.

- Handoff §3a (lease pages) and §3b (invoice seller/client) implemented in
  `evaluation/kv_schema_code.py` (`kv-schema-code-v3`): keys with the same
  words in another order (`name_eq`), a distinctive value under a key that
  names no other field when no key matched or the printed label is a
  sentence fragment (`is_fragment`), a date composed from several blanks,
  the ordered concatenation of pairs whose keys contain the label, and
  "Label: value" lines inside a multi-line value read as pairs (`cleanup`).
  Textract's seller/client left as missing: its LINE blocks interleave the
  two columns, so the "following lines" fix the handoff proposed does not
  exist in the data, and pairing lines by geometry would be our extractor.
- Defects found reading the rows, fixed in the same module: duplicate JSON
  keys were dropped by `dict` parsing (Gemini's lease answers); blank array
  cells were marked `invented` from other rows' values because array
  indices were ignored (Gemini 70 and llamaparse 106 false inventions on
  the payslips) — rows are now aligned one-to-one by shared values
  (`align_rows`) and a cell is judged in its own row; a printed dash or
  n/a counts as blank; a page without a stage-1 file or without any printed
  pair is not applicable on key_value instead of 0 for every engine
  (chart_003; `evaluation/keyvalue_leg.py`).
- `evaluation/code_judges.py` (`code-v3`): `TABLE_POLICY
  ["human_marked_tables_only"]` — only checklist tables that correspond to
  the human `table_html` are scored (`human_marked_tables`, threshold 0.5);
  on the 1040 pages the income lines (103 cells, 14% in the reference) had
  been 80% of the sample. `evaluation/table_leg.py` passes the reference.
- Audit of the change: an in-process rescore of every key_value, schema and
  tables row against run `20261001T132323Z` listed 1,170 verdict flips;
  every one that got worse was read, every value-only credit was read. Two
  drafts were rejected on that evidence: a one-token label containment
  ("name" in "APP NAME", "returns" in "puerReturns") and a value fallback
  that credited true/false fields from any unselected checkbox and short
  numbers by coincidence. Handoff §3f items read against the pages: chart
  text (labels printed on the charts — engine), tax-form tables (policy
  above), payslip blanks (scorer artefact fixed; literal zeros remain
  inventions), mineru tax_000/iam_002 (engine answers), tax_000 scanned
  upside down (noted), GLM truncation (engine). Handoff §3c corrected: the
  llamaparse retries succeeded in the reference run; there are no error
  rows, the empty answers are scored.
- Tests: 10 cases added to `tests/test_structured_legs.py`, 1 to
  `tests/test_table_leg.py`; 7 existing key_value tests now supply stage-1
  aliases (the leg is not applicable without them);
  `tests/test_reference_self_consistency.py` reads the deterministic
  markdown/TEDS metrics it was written to guard (the checklist headline
  need not score the sidecar 1.0, as its own comment said). 181 pass, 2 xfail.
- Two more defects found reading the first rescore's rows: Claude's
  invoice_002 key_value answer holds two fenced JSON objects (a first try,
  a note, a correction) and only the first was salvaged — every fenced
  block is now read; reference strings with spacing slips ("WV79662",
  "RegCokeNoSugar", "JAN 1 8 1983") cost 29 verdicts for 8 engines —
  string values are compared without spaces. 183 tests pass.
- Text, geometry and markdown audited the same way at the owner's request
  (every outlier, repetition flag and hard page read; every geometry page
  × engine with its parse/rescale/axis flags; markdown verdicts by element
  type). Two more fixes: marker's three empty OCR passes (no regions, no
  text, identical 0.398 s) are error rows, not 0.00; a line of several
  check-list items ("Allergies ☑ Black Outs ☐") is read as that many list
  items (plain squares excluded). The 166 text GT words that 12+ engines
  miss are listed in the handoff §3i (not edited: published dataset).
- Sampling pass for the owner's report (one low cell at a time, verdict
  engine / scorer / GT): two more of ours — Qwen's ```markdown fence with a
  note after it left its medical_001 table as a code block (tables 0.00 →
  1.00; fence now unwrapped whether or not it closes), and the 1040 blank
  "Your first name and middle initial" = "Ruth J" against firstName =
  "Ruth" (shared-label part rule extended to a label that names several
  things; 20 verdicts, all four engines alike). 186 tests pass.
- Evidence page for the owner (how each leg is measured, what was counted,
  a good and a bad result per leg with the page and the engine's output):
  built in the session scratchpad and published as a private artifact, not
  added to the repo. Building it found one more of ours: a pipe table glued
  to the line above (no blank line) is invisible to the markdown parser —
  Gemini's tax_001 grid was there and scored 0; `table_extractor` now opens
  a block before a run of table lines (five pages moved). 187 tests pass.
- Rescored (free, keys blanked): `20261001T194341Z`, `203908Z`, `215450Z`,
  `221022Z`, `222120Z` and the final `20261002T084627Z`; earlier ones deleted. `--report` renders, cost audit passes, 5 marker geometry error
  rows, 110 document-judge rows copied. key_value Claude 0.788→0.904,
  Gemini 0.797→0.885, Qwen 0.615→0.779, Textract 0.711→0.793; schema Gemini
  0.896→0.924, llamaparse 0.814→0.836; tables Textract 0.869→0.915, Mistral
  0.766→0.805, marker 0.710→0.687. Text, markdown, geometry and judge rows
  unchanged. The superseded code-v2 run `20261001T132323Z` deleted. Handoff
  rewritten.
- Not run: nothing paid, no Modal, no engine, no judge, no stage 1 for
  chart_003 (≈ $0.03). No git operations.

## 2026-10-02 — correction: the judge caches do reproduce the reference run

Re-checked while collecting examples of judge failures. The cached markdown
and table verdict files reproduce every published score exactly. The earlier
"414/627 and 367/530 differ" (handoff §4, 2026-10-01) came from recounting
with the wrong formula: the run counts table `empty_correct` as correct and
markdown `wrong_type` as half credit (`markdown_judge.py` cached path, which
contradicts its own docstring). Handoff corrected. The judge's failures that
do hold, with examples: tables credited where the engine wrote no table (28
rows ≥ 0.9, e.g. gemini_flash fintab_002 1.0, plain lines); 6,413 of 26,466
table verdicts are free credit for empty cells (funsd_002: 51 of 60); the
same rendering judged differently per engine ("Blurred Vision ☑": glm
correct, gemini/mistral wrong_type). Nothing run; read-only on the caches.

## 2026-10-02 — final offline rescore `20261002T095730Z` and evidence handoff

- Continued the pasted cleanup/scoring session. Final result replaces
  `20261002T084627Z` and `20261002T093008Z`; their run/summary/log files
  were removed only after the replacement passed the audit. Original three
  inference sources and all raw outputs remain unchanged.
- `kv-schema-code-v4`: an explicitly named blank semantic field takes
  precedence over another field with the same printed label. The preceding
  blanket blank preference would incorrectly excuse Claude tax_000's filled
  spouse surname because a dependent surname was blank. A regression test
  reproduces that and checks the inverse. Repeated generic lease labels keep
  the blank preference. Exactly three score/verdict changes from 093008:
  lease_004 renewalMonths for Claude (0.7500→0.7917), Gemini
  (0.7083→0.7500), Textract structured (0.4583→0.5000).
- Read the lease image and all three changed output candidates. Textract's
  chosen blank is an ambiguous initial-term table pair; this is blank-credit
  policy, not proof of correct field-local extraction. The handoff says so.
- Removed the obsolete CLI rejection of KV/schema with `--skip-judge` and
  the removed table-judge charge from estimates. Report methods come from
  saved scorer metadata, preserving historical judge attribution. Updated
  stale scorer docstrings, README, scoring design and final handoff. Historical
  audit/plan documents now point to current status.
- Regenerated standalone HTML in `/private/tmp/ocr-benchmark-continuation-20261002/`
  (not in the repo). It uses 12 exact source/pass examples, two engines on
  the same page for each leg, full saved details, reference images, geometry
  overlays, sampled-reference counts and a filterable complete score table.
  The old private Claude artifact remains unchanged and is superseded.
- Pending dedicated Claude/Gemini prompt comparison: 198 calls, no judge,
  exact page/prompt/config manifest in the same scratch directory. Historical
  same-page replay proxy $1.941072; not a vendor quote or spending cap.
- Validation: 195 full-suite tests passed; following the final CLI/estimator
  cleanup, 120 targeted tests passed. Final rescore/report succeeded and Modal
  cost audit passed. All 1,258 historical text rows preserve score/applicability
  (1,148 scored plus 110 N/A); 110 document-judge rows copied unchanged except
  provenance. All 4,860 pre-existing result files matched their initial hashes
  before superseded run files were removed. Raw source calls were never changed.
- The inherited candidate-engine implementations, runner/config/registry
  changes, shared references, geometry source runs and tests are retained as
  deliverables. They predate this continuation. Offline tests do not establish
  deployment or model/API correctness; no new-engine validation is claimed.
- Commit scope is the nested ocr-benchmark Git only, following the user's
  explicit instruction. Candidate integration files are recorded separately
  from scoring/final-result work. No parent-repo write or push.
- Not run: any paid API/engine/Modal/judge/alias call, new deployment, new
  engine inference or dedicated-prompt comparison. New spend: $0. The rescore
  process disabled credentials/dotenv loading and blocked socket connections.

## 2026-10-02 — recover and persist default API prompts

- Traced Gemini lease_000 Markdown 0.6154 to eight correct elements and
  five headings returned without heading markup. The call ended with STOP;
  this was not a missing prompt or truncation. Adding only heading markers
  to the saved answer makes the diagnostic 1.0; this is a counterfactual,
  not a new engine result.
- The historical call logger omitted default Gemini/Claude prompts because
  the runners did not return `prompt_used`. Their default prompt source is
  unchanged since initial commit e649e5a. Added metadata so future calls
  persist the exact prompt actually sent; no prompts or settings changed.
- Validation: 14 prompt tests passed, including four mocked provider/call-file
  checks for default and task passes. No provider calls. Historical outputs
  remain unchanged. The report's missing-prompt wording was misleading and
  has been corrected in the local, provisional export.

## 2026-10-02 — causal audit fixes: field attribution and Qwen box recovery

- Reproduced two false-positive field matches: swapped primitive-array
  columns earned full credit; an explicitly wrong schema field could borrow
  a sibling's matching value. v5 retains terminal array positions and gives
  exact schema paths precedence. Reorderable object-array rows remain aligned.
- Corrected a false negative: RM408.45 versus 408.45. The receipt image
  confirms the currency prefix; wrong digits and amounts in prose still fail.
- Recovered complete Qwen text_content/bbox_2d objects from malformed or
  truncated answers. Three geometry pages recover 102/113/116 boxes; all
  three word headlines remain zero. This proves an adapter defect existed
  without claiming it explains the remaining localization failures.
- Offline replay of all 675 scored field rows and 145 geometry rows changes
  32 scores (28 schema, four KV); 125 rows change detail. Results are an audit
  candidate in scratch, not a replacement final run. Archived calls/results
  remain unchanged. Schema mean changes are mostly downward as false credit
  is removed. All four KV engines gain one field on photo_001.
- Validation: 204 full-suite tests passed in 177.18s, including adversarial
  array/path cases, RM normalization and truncated native Qwen boxes. No
  engine/provider/Modal/judge calls. No GT data has been changed.

## 2026-10-02 — remove obsolete schema verdict client

- Removed the unused LLM verdict client, broad substring verifier and verdict
  cache implementation from schema_judge.py. Its deterministic checklist
  classes/building/sampling behavior is retained; the legacy module name
  remains for imports. Existing checklist caches are still used by alias_pass.
- Validation: deterministic helper AST comparison before removal, current
  204-test full suite passing, and repository call-site inspection. No cache
  deletion, new reference generation, model calls or result rewriting.

## 2026-10-02 — audit evidence and publication status

- Corrected the local report's primary evidence: KV extracted pairs and
  individual field verdicts; table grids/cell verdicts; actual geometry
  boxes on source images. Default prompt evidence is recovered from source
  with its logging limitation stated. Checked 12 source resolutions, SVG
  box coordinates/counts and actual KV inputs before updating the local file.
  It was not opened or presented as a cleared replacement report.
- Audited 3,099 scored rows for exact source/pass and arithmetic; replayed
  675 field and 145 geometry scores using the archived implementation, with
  zero discrepancies. Scanned all 1,227 scores below .85 plus eight errors
  across 57 engine/leg groups. The handoff now distinguishes confirmed defects,
  observed output failures and unresolved reference/pipeline attribution.
- Found 207 GT nulls prohibited by the supplied schemas, across 19 pages.
  Eight saved LlamaExtract effective schemas contain closed empty objects
  (two at root, five financial-table value maps, one form field), despite
  SUCCESS jobs. These are task-contract defects. Inspected the upside-down
  tax page, receipt currency amount, omitted chart content and actual box
  overlays. Not a claim of complete manual GT re-annotation.
- Rechecked author recipes: GLM recommends its layout/recognition SDK;
  Qwen's official OCR notebook specifies x/y bbox_2d with a /999 plotting
  transform. /999, /1000 and original-pixel probes do not rescue the poor
  coordinates. Read primary sources; did not deploy or run models.
- README, scoring design and handoff now withdraw the earlier final-status
  claim. Original v4 numbers remain archived; the candidate v5 replay and
  exploratory evidence stay in /private/tmp/ocr-benchmark-prepush-20261002.
- Cleanup: removed the blanket /docs ignore and ignored the requested local
  browser export directory only. Retained live shared reference caches and
  all source outputs. Scanned 4,955 tracked paths: no credential-pattern
  matches, tracked local caches/.env, or files over 20 MB. This is a pattern
  scan, not a guarantee that every arbitrary secret format is detectable.
- Validation: 204 tests passed; git diff --check passed; all result/source
  files remain unchanged in Git. No paid calls or GitHub push. Full fairness
  clearance awaits the explicit remaining task/reference and inference checks
  in the handoff. Commit/log operations use the nested Git only.

## 2026-10-02 — repair extraction contracts and financial reference mapping

- Added a separate inference schema: all field types permit null as the
  prompt requires, and eight previously open dictionaries have explicit
  target names. Target names are versioned without scalar answers. The
  original schema remains the scoring coverage contract. GLM instance
  templates now preserve nullable object/array structure. Future calls save
  the full request schema as well as the prompt.
- All 65 reference answers validate against their prepared request schemas;
  this resolves the 207 forbidden-null entries on 19 pages. Explicit fields
  survive the local LlamaExtract sanitizer. Vendor acceptance still requires
  a controlled call; no claim that local validation proves remote behavior.
- Repaired fintab_001's column assignment using its annotated colspan
  headers. Inspection's 70 belongs to 31.7.17 and 336 to 31.7.16. All row
  labels and amount multisets are preserved. Removed the derived csv field
  from all five financial references: it was not requested by their schemas.
  Raw corpus sidecars remain unchanged; loader/scoring metadata versions the
  correction. Historical call schemas are not retroactively replaced.
- Isolated offline comparison with the same v5 scorer changes 28 of the
  35 financial schema scores. This is a reference-fix diagnostic, not a
  replacement run or evidence that new extraction requests were executed.
- Validation: 33 contract/loader/prompt tests and all 210 full-suite tests
  pass (102.80 seconds); git diff --check passes.
  No engine, provider, Modal, alias or judge calls; no GitHub push.

## 2026-10-02 — table association audit

- Reproduced full credit for four amounts swapped between named rows or
  year columns. code-v4 now constrains values by unique row labels and
  headers, using the full checklist for context while keeping the scored
  sample. Exact placement replaces the former one-row/column proximity
  fallback. Complete row/column reorderings and transposition remain valid.
- Retained rowspan/colspan through extraction and grid parsing. Split-row
  continuations are permitted across blank or reference continuation labels,
  not across another named data row. Verdicts report conflicting anchors.
- Inspected sroie_002's original image: the cached checklist omitted its
  blank first header column. A read-time correction places RM/Code and
  Amount/Tax in columns 2/3. Original model cache and cell values preserved.
- Replayed all 504 saved table results: 125 scores and 464 cell verdicts
  change. These remain scratch diagnostics. Inspected the largest changes:
  DeepSeek fintab_000 drops 1.0 to .5714 because many amounts occupy Notes
  or the wrong year; rows containing a Notes value remain aligned. The
  original full-credit verdict was false. DeepSeek sroie_001 splits an
  extra item field without extending its headers, shifting numeric columns.
- Visual box review now includes all nine geometry engines on funsd_000,
  sroie_000 and fintab_001, using actual saved boxes over the original page.
  Recovered Qwen boxes use current normalization, not old failed-parse flags;
  errors remain errors. The review confirms that coarse line boxes and
  misplaced word boxes must be explained separately.
- Validation: 26 targeted table tests and all 216 full-suite tests passed
  (108.17 seconds); git diff --check passed.
  No inference, judge, provider or Modal calls. Report remains provisional.

## 2026-10-02 — remove reference-order bias in word geometry

- The box audit reproduced a greedy assignment error: an early missing FAX
  could consume a distant box belonging to a later correctly located FAX.
  Word matching now solves a one-to-one assignment within each normalized
  text group, maximizing located occurrences and then IoU. SROIE line
  matching and the independent line/block diagnostics are unchanged.
- Geometry metrics/manifests identify geometry-v2. New geometry details
  include every matched/unmatched reference with the actual scored reference
  and predicted boxes, prediction text, hit and IoU, not just 25 worst words.
  Coordinates are explicitly labeled after normalization; raw calls remain
  the source of original returned coordinates.
- Replayed all 145 scored geometry rows against the previous matcher with
  identical inputs/parser: 27 scores improve; none decrease. On funsd_000,
  Textract 177 to 182 located (.7973 to .8198), PP-OCRv5 125 to 150 (.5631
  to .6757), Tesseract 130 to 151 (.5856 to .6802), Gemini 70 to 74 (.3153
  to .3333). Claude remains zero: its actual coordinates are displaced.
- Qwen fintab_001 returns repeated text and coordinates reaching 2036,
  outside the documented relative frame. The raw-coordinate overlay places
  these off-page; previously recorded /999, /1000 and pixel-frame probes
  remain necessary context rather than silently clipping or moving boxes.
- Validation: 42 geometry/report tests pass, including reference-order
  permutations, preventing box reuse, and complete coordinate evidence.
  The preceding table/contract change passed the full 216-test suite.
  No inference, provider, Modal or judge calls and no report publication.

## 2026-10-02 — strict amounts and field-scoped units

- Table code-v5 no longer strips arbitrary OCR letters from amounts: 17.5C
  cannot receive full credit for 17.50. It retains numeric formatting
  normalization, accounting negatives, RM prefixes, escaped Markdown
  currency and math-wrapped minus signs. Checking the initial replay caught
  and repaired a currency-escape regression before committing it.
- Field scorer v6 accepts `35 Units` only for count/quantity fields, handles
  unambiguous month-first hyphenated dates, and prevents shared-label rules
  from borrowing a numeric substring. Ambiguous day/month dates still need
  a documented reference convention; no locale was inferred from answers.
- Replayed all 675 field rows against v5 on identical repaired references:
  six scores and eight verdicts change, all false negatives. All four KV
  engines on photo_001 gain their correctly returned quantity; the remaining
  changes recognize dates in two saved Mistral schema answers.
- Validation: 222 full-suite tests passed (103.74 seconds), plus 27 table
  tests after the currency-escape adjustment. No raw result, reference cache
  or call file was rewritten. No inference/provider/Modal/judge calls.

## 2026-10-02 — reconstruct Textract's native merged table cells

- Audited saved Textract block graphs against AWS's documented TABLE to
  MERGED_CELL to CELL relationships. Our adapter previously ignored merged
  blocks despite claiming to support merged cells. tax_002 contains 32,
  including the split Social security number header.
- Reconstruct merged groups from their child cells in row/column order,
  retain span-covered rows, and escape cell text as HTML. Preserve elementary
  cell HTML as well: replacing it with merged groups alone would discard
  valid multiline address/subject evidence. Both representations come from
  the provider response, not a model or inferred cell splitting.
- Stored-call loading rebuilds both representations without rewriting the
  original file. Table and KV readers consume them deterministically;
  textract-merged-cells-v1 is recorded in structured output and rescore
  manifests. Added no service calls and never constructed a cloud client.
- Replayed all 97 applicable Textract table/KV rows on identical scoring
  code/references. Nine scores improve, none decrease: eight table pages
  and lease_003 KV. tax_002 and tax_003 recover .4286 to 1.0; tax_004 .4545
  to .9091. Inspected both positive and negative changes in the first
  replacement-only prototype before preserving the primitive-cell evidence.
- Validation: 224 full-suite tests passed before the representation-preserving
  adjustment; 90 table/field regressions and all 97 replay cases passed after
  it. Original source calls remain unchanged. No report release or push.

## 2026-10-02 — lease label audit

- Inspected all five source-page term sections. The cached labels described
  the preceding clause but omitted trailing year(s)/month(s) labels used by
  the returned native and prompted field pairs. Add these at read time,
  recording lease-unit-labels-v1, without overwriting model caches.
- Exact returned field paths now take precedence in KV as well as schema.
  A repeated unit label explained by a filled reference field is not also
  attributed to an ambiguous blank field. Explicit renewal values remain
  inventions; unrelated semantic fields cannot excuse them.
- Replayed all 675 field results against the previous implementation with
  identical references and adapters. Eleven scores and 14 verdicts change:
  Textract (three scores), Gemini (five), Claude (two), Qwen (one). Inspected
  the changed keys/values. Repeated-label attribution remains ambiguous when
  the output provides no positional or semantic distinction.
- Validation: all 226 offline tests passed (103.86 seconds). No provider,
  inference, Modal, alias or judge calls. No new final report or push.

## 2026-10-02 — geometry evidence audit across the complete leg

- Replayed all 145 scored geometry results. Independently checked each
  coordinate transformation, each recorded mutual-center hit, reference
  counts and score arithmetic. Visually inspected comparison overlays on
  all 17 pages, covering the nine participating engines where calls succeeded.
  Separately reconstructed 4,124 WORD boxes from 34 raw Textract responses;
  text and coordinates agree with the persisted native word data.
- Found an evidence bug: fuzzy SROIE matches labeled the prediction with
  reference spelling. Preserve the actual returned word/window text instead.
  Persist all parsed and scored boxes, including unmatched and out-of-frame
  predictions, so the report can show the complete answer rather than only
  boxes that happened to match a reference.
- Found the general axis heuristic swapped MinerU pipeline's native xyxy
  table box on fintab_003. Only the legacy Gemini dialect now opts into
  the heuristic. This fixes the displayed coordinates; no headline score
  changes in the 145-row replay relative to geometry-v2.
- Inspection distinguishes sparse Marker OCR outputs, displaced Claude/Qwen
  boxes, tokenization differences, and coarse MinerU line/table boxes.
  MinerU fintab_000/003/004 each supplies one table box, not individual word
  positions. These zeros cannot be described as proof of misplaced words.
  Finer saved spans/model detections and Marker pipeline checks remain open.
- Validation: 45 targeted geometry/report tests and all 229 offline tests
  passed (129.22 seconds). Audit images/data remain in session scratch.
  No inference, provider, Modal or judge calls; no final report release/push.

## 2026-10-02 — audit the report's explanations, not only its arithmetic

- Found `_diagnose` used supporting structural recall to describe the primary
  Markdown checklist result. For 145 archived rows it would say most blocks
  were absent while the checklist recorded zero absent elements. The quoted
  Gemini lease_000 is one: eight correct, five wrong_type, zero absent.
- Report primary verdict counts for Markdown/tables/fields and separate
  text-matched-but-misplaced geometry from units without a text match.
  Low token coverage and length flags are observations, not proof of missing
  page content, invented text, or a repetition loop. Actual cause remains
  tied to the page/call inspection recorded in the handoff.
- Exercised the helper across all 3,099 archived scored six-leg rows. No
  archived table row has zero fallback rows with a positive primary grid
  count, but that mismatch is now covered by a regression. Four new report
  regressions and all 17 report tests pass. The immediately preceding change
  passed the full 229-test suite. No scoring/output files were rewritten.
- No inference, provider, Modal, alias or judge calls, no report release/push.

## 2026-10-02 — reproduce and repair GLM SDK mode selection

- Downloaded the published glmocr 0.1.5 wheel into session scratch, verified
  SHA256 9649aec2e70eeb5a8ebed9979092c58ec5c4e075a085fb82c8fa3b04e4e40977,
  and inspected its config/API implementations. Executed only the standalone
  config loader, without importing model code or running inference. The old
  pipeline_maas_enabled/ocr_base_url kwargs are silently ignored and leave
  MaaS enabled; mode=selfhosted, ocr_api_host and ocr_api_port work.
- Pin the SDK, validate the resolved local endpoint before constructing it,
  close resources after each probe, preserve SDK JSON/generation config and
  record layout boxes as coarse regions rather than words. CPU layout avoids
  silently competing with vLLM's existing 90% A10 memory reservation.
- Default engine routing remains direct vLLM. The explicit SDK sentinel is
  a quality probe with per-call CPU layout initialization included in timing;
  it must not be represented as measured persistent-pipeline throughput.
  Remote acceptance, layout artifact pinning and persistent initialization
  remain prerequisites before adopting the pipeline as a benchmark default.
- Validation: actual pinned configuration-loader reproduction and all 16
  prompt tests pass, including fail-before-client endpoint validation and
  retaining raw coarse geometry. No local models, deployment, inference or
  judge calls. No measured score/performance improvement is claimed.

## 2026-10-02 — inspect Marker's actual pipeline and isolate converter settings

- Inspected the published marker-pdf 1.8.2 wheel, SHA256
  ecc938743eea61f47eaee8f7f8e55c70b38760e96166cd7850cbd32c3386592f.
  OCRJSONRenderer's Page/Line/Char tree is traversable by our adapter. The
  recorded sparse/empty children are upstream output, not demonstrated loss
  of a complete native line array in our renderer adapter.
- Found OCRConverter mutates the config shared by persistent worker calls.
  Deep-copy per request, retain converter settings and pass sentinels, and
  save the full renderer tree plus exception tracebacks for future probes.
- Prepared a separately selected all-blocks probe: skip_ocr_blocks=[] and
  keep_chars=True. The pinned OCR builder skips tables assuming another
  processor handles them, but OCRConverter does not run TableProcessor.
  No quality gain or explanation for the five historical crashes is claimed.
- Validation: four mocked worker regressions cover control/probe, success/
  exception, nested config isolation and line/character separation. All 20
  prompt tests pass. No model import, inference, deployment, judge or push.

## 2026-10-02 — source-page audit repairs printed text and chart references

- Visually reviewed chart_003 and all six scanned_legacy source images.
  Chart text GT contained estimated data tables absent from the image, and
  the structured extension reused visibly questionable estimates as exact
  headcounts. Replace the chart portion with printed titles/ticks/legends;
  exclude the three unverified numeric-series branches from schema targets
  and future requests. Preserve the original sidecars and request history.
- Remove only the six reviewed Source: citations absent from scanned pages.
  Printed FDA URLs and adjacent document IDs remain. Record the versioned
  repairs in loader metadata and text/schema metrics; update corpus caveats
  to stop claiming historical engine corroboration validated every value.
- Same-code isolated replay: 98/98 affected text scores and 7/7 chart schema
  scores change; 11/11 chart Markdown headlines are unchanged. Six schema
  scores rise and LlamaExtract falls, confirming this shared GT defect can
  alter engine comparisons. No combined final score release yet.
- Validation: all 241 offline tests pass (157.09 seconds), including source
  repair scope, retained genuine URLs/identifiers, request-schema targets,
  read-only corpus loading and the full reference self-consistency suite.
  No inference, provider, judge, Modal or GitHub calls. Evidence stays in
  session scratch; archived results remain untouched.

## 2026-10-02 — prepare the geometry cause comparison before paid approval

- Current Anthropic guidance explicitly recommends pixel coordinates rather
  than the normalized coordinates we requested. Documented that prompt
  mismatch without claiming it alone explains every historical zero.
  Checked provider image-frame limits and Google's y/x/1000 convention.
- Prepared 21 precisely specified Claude/Gemini geometry calls on the same
  three pages, varying output cap, image frame and prompt separately. Exact
  expanded requests, prepared images and hashed executor are in session
  scratch; the full plan/prompts/cost are in the existing handoff.
- Planning reserve $2.0414; proposed allowance $3. No retries, judges,
  Modal deployments, model upgrades or per-page best-variant selection.
  Mocked all 21 payloads, persistence, both token-usage formats and budget/
  error guards. No API clients constructed or provider requests made.
- Corrected stale registry comments that still blamed Marker's tree parser,
  contradicting the pinned-source audit. No routing or prompt change.

## 2026-10-02 — preserve numeric meaning in field comparisons

- Reproduced synthetic false credits for a lost Unicode minus, accounting
  negatives, malformed decimal punctuation and integers beyond float precision.
  Use exact Decimal values, retain signs and validate punctuation grouping.
- Preserve existing whitespace formatting, identifier separators/leading zeros,
  shared address fields, numeric counts, printed lease duplicate amounts and
  tax-rate prefixes. The first replay caught an overbroad identifier rule for
  numberOfShipments; corrected it before committing.
- Isolated replay of all 675 saved KV/schema results against v7: zero changed
  scores and zero changed verdicts. These reproduced edge cases establish a
  scorer defect, not evidence that it affected the saved run.
- Validation: 244 offline tests pass, including the two numeric regression
  cases and declared-frame geometry tests in the working tree. No judge or
  field inference calls. The separately approved geometry probe was running
  during this work; its calls/costs are logged separately.

## 2026-10-02 — keep explicit geometry frames explicit

- Add infer_scale=False for callers that already converted an explicitly
  declared pixel or 0–1000 frame. A single off-page prediction must not cause
  all returned boxes to be divided by 1000 again. Keep legacy default scoring
  unchanged and retain outliers in the evidence.
- Regression covers three valid boxes plus an outlier; 33 geometry tests and
  all 244 offline tests pass. The approved probe uses the explicit conversion
  for its pixel/yx1000 variants and records legacy heuristics for controls.
  No historical headline change is attributed to this option.

## 2026-10-02 — execute and audit the approved geometry comparison

- User approved the exact 21-call / $3 plan. Saved 13 provider responses, then
  stopped on Gemini's financial-page yx1000 read timeout. Seven receipt calls
  remain unattempted. No automatic retry, judge or Modal call. Successful
  response usage totals $0.51622775; timeout billing is unknown.
- Inspected overlays for every completed response and independently checked
  each geometry verdict. Every completed response ended normally. Raising
  token limits did not fix either completed Claude page. Pixel prompting
  yields 0/222 located on the form and 7/101 on the financial page. Gemini's
  legacy conversion yields 74/222 and 101/101; explicit dialect yields 79/222
  on the form after the syntax repair logged below, with missing words and
  incorrect word-to-box associations.
- Preserve raw replies, exact requests, analysis and four deduplicated sent
  images in the diagnostic run. Scratch overlays/HTML/scripts remain outside
  the repo. Moved the zero-request DNS preflight failure to scratch. Original
  benchmark scores and provisional report remain untouched.
- Prepared and mocked an eight-call continuation with identical request
  settings and a longer client timeout. $0.695248 additional reserve within
  original $3 allowance; requested approval because the original plan said
  stop on error and no retry. No continuation calls yet. Full findings and
  limitations are in the existing handoff; final benchmark remains uncleared.

## 2026-10-02 — recover explicit coordinates from malformed box syntax

- A saved Gemini diagnostic response contains 39 malformed box entries.
  Thirty-six have all four coordinates and a closing bracket but omit the
  opening bracket; recover that exact syntax error without inferring values.
  Three remaining entries lack coordinates and stay unusable. Preserve a
  syntax-repair marker beside each recovered box and retain the raw reply.
- Form geometry becomes 79/222 located (0.3559), rather than 65/222 when
  these recoverable boxes were dropped. This is an adapter correction, not
  new model inference. Incorrect text/box associations remain measurable.
- Validation: 34 geometry tests pass, including malformed three/five-value
  and duplicate-key examples that must not be invented into valid boxes.
  Replayed all 145 archived geometry scores: zero score or parse-count
  changes. No additional provider or judge call.

- Final validation of this partial diagnostic: all 245 offline tests pass
  (93.71 seconds). HTML validation accounts for 13 overlays and all 3,958
  displayed reference/prediction boxes, checks every internal link, and
  reconciles costs/input hashes with saved responses. Punctuation-only
  references are excluded consistently from scores and overlays. All 21
  planned requests have a success/error/unattempted disposition. No provider
  authorization headers are persisted. Only the diagnostic evidence and
  three existing audit/log documents remain to commit; scratch artifacts
  are outside the repo. No push and no continuation calls.
