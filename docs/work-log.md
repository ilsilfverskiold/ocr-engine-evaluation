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

## 2026-10-02 — prevent credit from crossing explicit field owners

- Confirmed GLM tax_003 receives two credits for fields absent from its
  truncated response, borrowing zero from line1.h. Prevent typo tolerance
  from changing numbered labels, respect explicit/qualified field ownership,
  and keep absent array cells from borrowing scalar summaries.
- Isolated replay of all 675 field results changes four schema scores and
  no KV score. Eight false credits become missing; 366 wrong verdicts become
  missing because the cited values belonged to unrelated fields. Preserve
  legitimate combined fields, bare labels, label typos and row alignment.
- Added four focused regression cases; 69 structured-leg tests pass. Updated
  scoring-design's current version labels and the existing handoff with exact
  saved examples, score deltas and remaining repeated-label/checkbox caveats.
  No new inference, judge, Modal, alias/checklist call or historical rewrite.
- Full validation: all 249 offline tests pass (78.58 seconds). Only the
  matcher, its existing test module and four existing audit/design/log docs
  changed. Scratch replay files remain outside the repo. No push.

## 2026-10-02 — distinguish filing status from an independent checkbox

- Reviewed tax_002 and tax_004 source images. The nonresident-spouse election
  is independent of the five filing-status options; tax_002 visibly checks
  both MFJ and that election. Do not infer its state from the status choice.
- Field v10 also requires a recognized group selection before inferring an
  unselected checkbox. Unknown group text cannot clear every false option.
  Preserve explicit checkbox states and legitimate named group selections.
- Full 675-result replay changes three KV scores, each by one field. Four
  other verdicts change without score changes, including two medical flags
  previously diagnosed from unrelated medication names. Exact deltas and
  attribution are in the existing handoff.
- All 251 offline tests pass (82.40 seconds), including two new regressions.
  No paid call, judge, Modal deployment, checklist regeneration or historical
  rewrite. The previously prepared geometry continuation remains unexecuted
  pending the user's answer. Only the matcher, existing tests and four
  existing docs changed; review crops/replays are in session scratch. No push.

## 2026-10-02 — repair the six invoice text references from their images

- Reviewed all six source images and existing page-transcribed Markdown.
  Replace faulty plain-text references at load time; preserve originals and
  guard the reviewed source strings with hashes. Fix phantom punctuation/
  numbers, missing row numbers, split amounts and interleaved reading order.
- Exclude unreadable replacement-glyph suffixes from readable text/table
  targets and two complete description fields from schema exact matching.
  Do not guess their characters. Correct a separate printed slash in a
  retained invoice_005 schema field. Record all exclusions and repairs.
- Apply unscorable-field exclusions before sampling while retaining complete
  deterministic caches. Update reference displays from cached checklists
  without rewriting their source files or changing primary MD/table scores.
- Isolated 288-result replay: 84 text and 14 schema scores change; 66 MD,
  72 table and 24 KV headlines stay unchanged. Claude's six invoice text
  coverages are now 1.0; its prior deductions came from bad annotations.
  This is unique-word coverage, not a claim of perfect character accuracy.
- Mark the six replacement references as model-reviewed, not independently
  human-annotated. Update existing provenance, limitations, README, scoring
  design and handoff; no new documentation file.
- 253 offline tests pass (80.21 seconds); 88 ground-truth/field tests pass
  again after provenance changes. Verified cached glyph display corrections
  normalize identically for primary MD/table scoring. All scratch images and
  replay evidence remain outside the repo; no provider/judge/Modal call or
  historical run rewrite. Geometry continuation is still awaiting approval.

## 2026-10-02 — repair receipt line references before attributing geometry failures

- Reviewed all six receipt images and targeted native-resolution crops. Five
  references omit visible address suffixes from text and full-width box labels.
  Correct the verified labels/typos without moving retained coordinates; hash
  guards reject changed annotations. Preserve raw sidecars and engine outputs.
- Count sroie_003's exact duplicate TOTAL QTY box once. Exclude sroie_002's
  faded first address box with an unreadable suffix; keep only its readable
  prefix in text. Emit reference changes/exclusions in geometry metrics.
- Fix sroie_000 structured shoe description 805 → B05 from the page. This
  removes three false credits and restores three correct answers; it does
  not unconditionally raise engine scores.
- Isolated 215-result replay: 70 text, 33 geometry and six schema scores
  change; 48 table and 20 KV scores stay unchanged. Inspect corrected-line
  overlays for all 42 geometry results and independently verify every old/new
  mutual-center verdict and score. Scratch holds crops, sheets and full replay.
- Update existing handoff, scoring-design, README, provenance and limitations.
  Five partial model-reviewed repairs are disclosed; remaining receipt
  tokenization/handwriting/registration/item-segmentation questions stay open.
- All 255 offline tests pass (81.38 seconds). No paid API/judge/Modal call,
  checklist regeneration or historical score rewrite. Geometry continuation
  still awaits the previously requested approval. Only the named source/test
  files and existing documentation/metadata are modified; no new repo artifact
  or push.

## 2026-10-02 — repair a receipt item split that created false blank credit

- Source-image review confirms sroie_003 has one purchased item with a
  two-line description, matching its existing table reference. Repair the
  structured reference's two entries to one; preserve original values and
  guard them against unexpected changes. Request schema is unchanged.
- Inspect every saved schema prompt and answer. Gemini's complete item was
  falsely penalized; DeepSeek's empty extraction received three nonexistent
  blank-field credits. The corrected target has 20 fields rather than 24.
- Replay all 46 scored results on the page: all seven schema scores change;
  14 text, nine geometry, 12 table and four KV results remain unchanged.
  Record actual per-engine causes, including split descriptions and GLM's
  duplicated populated row (not independently penalized by field coverage).
- Add loader/scoring regression coverage for the real reference, a complete
  answer and an empty answer; 91 affected tests pass and the full suite has
  256 passing tests. Update existing provenance, limitations and audit/design
  docs. Scratch holds the source crop and full replay; no new repo artifact.
- No paid engine/judge/Modal call, prompt rewrite, checklist regeneration,
  historical result rewrite or push. The geometry continuation still awaits
  the user's response to the previously presented approval request.


## 2026-10-02 — close the bounded audit and deliver corrected evidence

- Preserve saved applicability/scores consistently in terminal reports and CSVs;
  remove unsupported blanket reference-consensus claims. Reproduced the prior
  discrepancy on 26 historical chart-table rows; current rescore had none.
  Nineteen report tests pass; the preceding full suite had 256 passing tests.
- Rescore the three preserved sources to `20261002T162630Z` with `--skip-judge`:
  zero new engine, Modal, alias-building or judge calls. Retain 110 historical
  document-judge estimates unchanged. Verify arithmetic and saved-call provenance
  for all 3,099 scored six-leg rows, plus 219 shared reference checklists.
- Final differences against the provisional run are 252 text, 60 geometry,
  141 tables, 19 KV and 82 schema rows (554 total); Markdown is unchanged.
- Add `scripts/build_evidence_report.py`, a reproducible local browser export
  with all 4,615 result rows, 2,282 call entries and 93 embedded page previews.
  Geometry presents actual scored boxes; KV presents native pairs; other legs
  present their scored structures and verdicts. Prompts recovered from source
  are labeled separately from persisted prompts. Build and browser render pass.
- Update existing handoff, README and scoring design to the delivered run.
  Remove superseded generated run `20261002T095730Z`, its four CSVs and old
  ignored HTML. Retain original inference and partial geometry diagnostics.
- Honor the user's stopping scope: confirmed material repairs and checked
  evidence, with unrun controls/reference limits disclosed. No further research,
  paid reruns, tuning or judge calls. Generated final HTML stays local/ignored;
  source/results/docs are committed only in ocr-benchmark. No GitHub push.


## 2026-10-03 — explain invoice text coverage and correct its evidence display

- Inspect invoice_000's source image and reproduce Claude's 1.0000 text
  coverage: all 68 distinct normalized reference tokens occur in the answer.
  The 403 literal `&nbsp;` entities become whitespace in text normalization.
  Saved CER is 0.2636 and WER 0.2874; seller/client column order differs.
  The answer retains printed item number `1.`, omitted by reference
  normalization. A perfect coverage score is not perfect transcription.
- Show normalized output/reference and supporting text metrics directly in
  the browser report; keep exact raw markup expandable. Reproduce every
  displayed text coverage score before exporting, failing on disagreement.
- Move engine-wide notes under an explicitly scoped expandable section so
  unrelated geometry/schema findings are not presented as a text-page diagnosis.
  Add a specific, source-checked explanation for this Claude invoice result.
- Rebuild the local report without modifying saved scores or scoring policy.
  No engine, judge, Modal calls or paid rerun. No push.


## 2026-10-03 — disclose Markdown heading overcredit; keep CER/WER in text

- Reproduce invoice_000's heading flaw: the scorer promotes bold-only labels
  to headings, including Seller and Client on the same line. `**ITEMS**`
  renders as a strong span inside a paragraph, not a Markdown heading.
  Four heading targets incorrectly receive full structural credit here.
- Both reference addresses also match one interleaved output paragraph.
  The saved 16/16 does not establish correct structure. Flag this explicitly
  in the viewer and handoff; show existing supporting structural metrics.
- Follow the user's correction: CER/WER stays in the text view. Remove the
  uncommitted addition to Markdown. Saved scorer policy/scores are unchanged;
  consistent structural scorer correction/rescore remains outstanding.
- Rebuild the local viewer and check Python/JavaScript syntax and whitespace.
  No engine, judge, Modal calls, paid rerun or push.


## 2026-10-03 — explain unscored Markdown pages without inventing failures

- Trace Claude sroie_000 N/A to its saved reason `no markdown reference`.
  There is no Markdown sidecar or cached Markdown checklist for this page.
  The corpus excludes 36 pages for missing Markdown references: six each of
  receipts, scanned legacy, newspapers, legal, handwritten and reports.
- Fix `_diagnose`: explicit N/A returns the recorded exclusion before looking
  at absent metrics. Previously null element recall defaulted to zero and
  falsely said fewer than half the blocks matched even though none were scored.
- Viewer presents N/A reason and unscored saved output, with no empty verdict
  table suggesting completed evaluation. Add regression coverage for missing
  references across legs and error precedence; run the report tests and rebuild.
- Scores/eligibility/references are unchanged. Extending Markdown coverage
  needs shared references/checklists before offline scoring saved outputs.
  No new engine, judge, Modal calls, reference generation or push.


## 2026-10-03 — save the next-session checklist and stop

- At the user’s request, create `docs/follow-up-2026-10-03.md` and link it
  from the handoff. Record heading overcredit, reused paragraph/table matches,
  missing Markdown references, text-only CER/WER, completed display fixes and
  a bounded rescore/delivery plan. Distinguish confirmed issues from decisions.
- Record approval/cost preparation for future image-only reference calls;
  none are authorized or run by this documentation task. No new audit,
  engine/scoring/reference-building command, test run or push.


## 2026-10-03 — compare Sonnet invoice/bank formatting provenance

- Verify default invoice_000 and bank_001 saved provider responses exactly
  match stored raw text: respectively 403 and zero literal `&nbsp;` entities.
  Both have the same recorded model and normal end_turn completion.
- Compare runner history: no default prompt change; only prompt logging was
  added. Original calls lack persisted default prompts, so distinguish source
  evidence from per-call proof. Inspect bank source and saved output; record
  differing layout formatting and prompt follow-up in the next-session list.
- Read-only evidence inspection; no inference, scoring, judge calls or push.


## 2026-10-03 — record CER/WER as required text comparison measures

- Expand the follow-up document with plain-language CER/WER definitions,
  direction, normalization, order/reference limitations and the two verified
  Sonnet examples. Record the requested prominence in text engine comparisons;
  distinguish that pending work from existing per-page metrics.
- Recommend separate coverage/CER/WER columns, not an arbitrary average of
  overlapping error rates. No combined score or new default sort approved.
- Documentation only; no scorer/report changes, inference, judge calls or push.


## 2026-10-03 — run the requested Sonnet Markdown structure/text pilot

- User explicitly revises the earlier text-only CER/WER direction: investigate
  structure plus text fidelity within Markdown, using a few saved Sonnet docs.
- Read primary CommonMark/GFM and OmniDocBench evaluation documentation. Test
  invoice_000, invoice_001, bank_001 and lease_000 offline, same default answers
  for structure and text. Actual heading matches against sampled targets are
  0/4, 0/4, 5/7, 5/5 although all saved Markdown scores are 1.0.
- Reproduce all four old checklist scores and stored CER/WER. Check actual
  table grids and list nodes, isolate heading-only counterfactuals without
  presenting them as corrected overall scores. Record methodology, limitations,
  results and standards links in the existing follow-up document.
- Scratch contains a reproducible script, JSON evidence and small HTML viewer;
  no generated pilot artifacts inside the repo. Parser fixtures pass for bold,
  ATX/Setext/HTML headings and list items. No official result/scorer change,
  engine call, reference-building/judge call or push. No new spend.

## 2026-10-03 — record Markdown prompt fairness audit and controls

- Add the complete 11-engine prompt/route audit to the existing next-session
  checklist: 915 saved calls, 627 scored results, exact prompts, provenance
  limits, native converter routes and the three recorded truncated answers.
- Record dedicated Claude/Gemini Markdown controls, the GLM SDK pipeline
  control, and Qwen's overlooked author Markdown recipe with its table-format
  compatibility requirement. Include bounded pages and proposed instructions;
  no reruns are authorized by this documentation update.
- Keep audit scripts, raw evidence inventories and trial outputs in the
  disposable folder. Production code, scores and report unchanged. Check the
  documentation diff and whitespace; no inference, judge calls, tests or push.
- Leave these provisional notes uncommitted pending the clean final change.

## 2026-10-03 — inspect Docling bank_000 table repetition

- Trace displayed repetition to native HTML colspans, the Markdown export,
  and two identical candidate grids from those representations. Distinguish
  merged-cell expansion from independently generated repeated cells.
- Confirm merged-label overcredit in saved verdicts and the absence of an
  extra-output penalty in the headline. Record concrete follow-up checks.
- Offline diagnostics stay outside the repository; no inference, production
  scorer/report changes, commits or push.
- Reproduce 0.4483 offline: native HTML alone, duplicated HTML and doubled
  Markdown all retain the same verdict totals. Extra copies are unpenalized.

## 2026-10-03 — inventory table references and clarify judge usage

- Verify 42 HTML-reference pages against the 42 scored pages; record exact
  identifiers/provenance and distinguish current code verdicts from the old
  per-engine judge and retained model-authored checklists.
- Inspect bank_000 source image and both reference representations; confirm
  two incorrect checklist amounts and account-row association differences.
  Record pending repairs without claiming all references have been audited.
- Inventory JSON stays in scratch. Only follow-up/log documentation changed;
  no reference edits, rescore, model calls, commits or push.

## 2026-10-03 — review all table-reference source images

- Visually inspect all 42 source pages against loaded HTML reference rows.
  Record 16 usable represented-table pages, 18 partial/simplified references,
  and 8 material structure mismatches. Document review limits for dense shipping
  text and financial section spans; this does not certify separate judge caches.
- Confirm all bank references agree with reviewed source cells, unlike the
  erroneous bank_000 model checklist. Record omitted invoice summaries,
  shipping multiline-cell fragmentation, form-boundary mistakes and inconsistent
  empty-row/header treatment. Recommend deterministic targets from one reviewed
  reference, without a model rewriting known values.
- Complete the requested table-interface matrix for existing engines, including
  dedicated VLM controls, already separate Marker/GLM routes, native pipelines,
  Mistral table-format options and LlamaParse custom instructions.
- Page-by-page audit JSON remains outside the repository in the disposable
  experiment folder. Only notes/log changed; no production repairs, inference,
  judge calls, rescore, commits or push. Leave provisional notes uncommitted.

## 2026-10-04 — provisional table patches and requested comparison

- User authorized type-specific subagents to repair original table references.
  Five agents patched 24 gitignored local sidecars: invoice (6), receipt (5),
  shipping (5), form (3), tax_form (5). Each retained original backups and
  source hashes; only table HTML and table review/provenance fields changed.
- User then questioned the patches and requested exact differences. Paused
  further types; built an offline comparison for all 24 pages with embedded
  original images, rendered old/new tables, cell differences, exact JSON
  differences, uncertainty notes and hash-verified backup provenance.
- Independently verified all 24 original backup hashes, image hashes, current
  HTML against recorded patches, and non-table-field preservation. Inspected
  key source examples; flagged possible KENT II/III transcription error in
  funsd_004 and consequential grouping/scope choices. Integrity checks do not
  establish annotation correctness; patches remain provisional, not approved.
- Comparison and backups are in /private/tmp/ocr-table-gold-repairs-20261003.
  Updated follow-up to record planned gold-derived table evaluation and required
  uniform rescore after accepted repairs. Scorer, scores and report unchanged;
  no inference, judge, benchmark run, commits or push.

## 2026-10-04 — make the reference comparison browsable

- Add a plain-language explanation of the intended gold-reference evaluation
  and the purpose of each type's patches to the disposable comparison page.
  Show original image, original rendered tables and patched rendered tables;
  provide page navigation, source-image toggle and full-resolution image view.
- Verify all 24 comparison panels and explanations, regenerate from checked
  backups/current references, and syntax-check the page JavaScript. Serve the
  page locally on 127.0.0.1:8767 for browser review. No additional reference
  edits, scoring changes, inference, commits or push.

## 2026-10-05 — offline human annotation editor and gold selection

- Preserve 42 original table snapshots and 24 proposed patches in separate
  local documents/table_references folders; restore all 24 corpus sidecars
  byte-for-byte from their backups. Gold starts empty. Old editor snapshots
  include pre-existing loader repairs, with raw HTML retained for provenance.
- Build an offline editor embedding all 42 images and 504 saved engine results.
  Show flagged issues, original/proposed tables, native engine tables with
  actual spans and explicitly separate parsed-text alternatives. Add Markdown
  and HTML editing, live preview, table/row/column controls, notes and flags.
- Accept old/patched or Save gold writes gold/PAGE.json through the browser's
  granted folder access. Verify saved bytes, retain previous gold in history,
  reject stale conflicting saves. Browser drafts/export remain distinct from
  saved gold. Provide a standard-library local launcher for offline reopening.
- Add gold-slot selection to GroundTruthLoader, requiring page/image identity;
  no gold means original, never proposed patch. Override only table HTML/source
  and metadata. Empty gold explicitly removes table eligibility.
- Validation: 21 ground-truth tests pass; all 42 fallbacks match original loaded
  references; all 24 raw originals restored exactly. Browser checks with network
  disabled pass 66 old/patch round trips, accept/save/history/conflict checks,
  native span display and input rejection. Browser filesystem tests use a
  temporary profile's private storage, never the real gold folder. Visually
  inspect the rendered editor. No real gold files created by validation.
- Source deliverables: evaluation/table_references.py, loader integration,
  existing loader tests and follow-up/changelog/work-log. Offline page, launcher,
  version snapshots and README are gitignored corpus artifacts. Build scripts,
  screenshots and browser test fixtures remain in /private/tmp. No inference,
  judge, benchmark run, rescore, commit or push.

## 2026-10-05 — remove browser folder connection requirement

- Replace the browser filesystem picker with a standard-library local save
  server. Save/Accept now write directly to the existing
  documents/table_references/gold folder; no browser selection or permission.
  This supersedes the granted-folder workflow described above. The unnecessary
  empty root gold-references directory was removed. Other legs are unchanged.
- Keep atomic writes, source identity checks, revision conflict protection and
  backups in gold/history. Launch in the default browser; no internet required.
  Existing review URLs on port 8767 serve the rebuilt editor.
- Verify 26 filesystem-save and loader tests, rebuild all 42 pages / 504 engine
  results, check the running API reports the existing gold path, and confirm
  the generated page contains no folder-picker API or obsolete instructions.
  No real gold annotations created in validation. No inference or rescore.
  Provisional work remains uncommitted; no push.

## 2026-10-06 — explain current table scoring after human review

- Confirm all 42 editor manifest pages have saved nonempty gold versions.
  Read the current scoring path: gold overrides table_html, but headline
  cell targets still come from cached model-authored checklists (up to 60
  sampled items, excluding empty cells). Code assigns verdicts; no live
  table judge is called. Gold feeds table scope filtering and supporting TEDS.
- Explain current limits: tolerant text matching, no one-to-one cell assignment
  or direct extra-cell penalty in the headline, excluded blanks, and spans
  discarded by supporting TEDS. The agreed gold-target migration remains
  pending. No reference edits, evaluation, model calls or report regeneration.

## 2026-10-06 — record agreed gold table cell-F1 design

- Update follow-up status to the user's completed human review of 42 pages.
- Record gold-derived cell F1 with precision/recall, one-to-one table/row/cell
  matching, blank-cell credit and invented-value penalties, missing/extra and
  duplicate output penalties, merged-cell and parser-padding exclusions,
  examples, supporting span-aware structure evaluation and verification tasks.
- Keep implementation and uniform rescore explicitly pending. Documentation
  only: no scorer changes, gold edits, tests, inference, rescore or report build.

## 2026-10-06 — clarify schema key/reference checks

- Inspect schema_leg, schema_judge, schema_contract and kv_schema_code.
  Schema targets are built without a model from true_json/json_schema, capped
  at 80 fields. Key matching allows normalized paths/leaf names and fuzzy
  naming; current accuracy is not strict JSON Schema compliance. Omitted blank
  keys may earn blank_ok; unrelated extra keys have no direct headline penalty.
- Distinguish existing targeted schema-reference repairs from the completed
  human table review, which does not certify or replace schema references.
  No scoring, reference changes, engine calls or implementation performed.

## 2026-10-06 — record schema extra-key penalty and clarify KV references

- Add the requested direct penalty for unexpected schema keys to the follow-up
  work list; formula/implementation remain pending. Record full supplied-schema
  checks so sampled-out or contract-permitted keys are not mislabeled extras.
- Confirm KV derives scalar targets from the same structured references, then
  uses cached printed-label aliases and printed-pair eligibility. Values remain
  reference-derived. Table gold does not replace these references.
- Documentation only; no scorer/reference changes, tests, inference or rescore.

## 2026-10-06 — separate schema/KV reference audit and human review editor

- Delegate one bounded source-image audit of all 65 structured-reference pages,
  independent of table gold. Preserve effective old values, raw sidecars, source
  hashes, KV aliases and prior repairs. Proposed patches affect 29 pages: 30
  value changes, 48 alias changes and 69 nullable-schema type changes. Five
  dense shipping pages explicitly retain partial field review status.
- Independently reproduce counts, apply all 147 operations, validate resulting
  records against schemas and verify originals still match saved source hashes.
  Spot-check source images for invoice suffix, receipt total, tax blanks and
  unprinted milk price. Proposals await the user's final review.
- Build local documents/structured_references review workspace: all 65 images,
  old/proposed field views, exact changes, field/JSON/schema/KV-label editors,
  exclusions, notes, 675 saved engine answers, accept/save, browser drafts and
  export. Launch on port 8768; save directly to its own gold folder, with
  backups and stale-revision rejection. No folder picker or internet needed.
- Add evaluation/structured_references.py, scripts/structured_reference_editor.py
  and tests/test_structured_references.py. Reuse the existing save protocol via
  a validation hook. Wire structured gold into the shared loader and KV labels/
  exclusions; preserve nullable object child fields in walk_fields. Table gold
  is never used to build structured targets, and its files are untouched.
- Validation: 90 tests passed, 3 corpus-wide schema-contract tests blocked by an
  existing tax_000 table-gold image identity mismatch. Stored table-editor image
  matches its gold hash; current PNG has different pixels at the same dimensions.
  Record for later resolution without rewriting table annotations or weakening
  identity checks. All 15 dedicated structured/save tests pass. Browser checks
  verify live 65-page display plus actual save/edit/reload/history, old/patched
  acceptance, invalid JSON and conflict handling in temporary fixture folders.
- Generated pages, versions and launchers stay gitignored under documents;
  audit/build/browser scripts and screenshots stay in /private/tmp/ocr-structured-
  review-20261006. No real structured gold created in tests, no original reference
  edits, inference, judge, benchmark rescore, report regeneration, commit or push.

## 2026-10-06 — investigate sparse KV eligibility on the first review pages

- Reinspect source images bank_000–004 and trace reference fields/aliases.
  Current KV counts are 1/4/2/3/5. Most displayed fields are excluded array
  entries, but selective schemas also omit real printed pairs entirely.
  Record bank_000 missing consultant name/ID/phone/partial-share preference,
  corresponding omissions on bank_001–003, and a bank_001 alias-sampling gap.
- Correct the audit scope claim: initial review checked existing targets, not
  full-page KV completeness. Add explicit coverage flags for the first four
  pages and visible counts distinguishing arrays, scalars, KV and unmapped.
  Record the need for per-leg target scope before extending KV without silently
  changing the historical requested schema. No new target values added.
- Preserve the user's saved bank_000 structured gold byte-for-byte. Old corpus,
  table gold and scorer unchanged. No inference, judge or rescore; documentation
  and local review evidence/UI updated, with script/data checks only.

## 2026-10-06 — expand structured proposals and expose KV eligibility controls

- Expand proposals across the 65-page structured workspace following the user's
  decision to include missing fields in schema too. Parent handles bank pages;
  the existing audit agent handles the other 60. Independently apply and validate
  all final records: 54 pages have proposed changes, 48 have new schema targets,
  with 532 added field patterns across pages. Array indices are normalized for
  target counting. Bank KV proposals rise from 1/4/2/3/5 to 6/9/6/12/6.
- Preserve partial coverage flags for shipping_000–004 and photo_002. Source
  ambiguity/omissions remain in page notes for human review; no claim of fully
  certified coverage. Original structured sidecars and table gold are unchanged.
- Show scalar fields/KV checkboxes first, with all-field/KV/new-target filters
  and a direct navigation button. Unticking KV preserves the schema value.
  Unsaved pages open proposals; saved pages retain gold and can add missing
  proposals to a draft without replacing existing values or eligibility flags.
  Merge array columns only against a unique unchanged baseline row; otherwise
  report manual review. Only a user save promotes the draft to gold.
- Save original schema and old/new field patterns server-side as provenance.
  This does not implement historical scope filtering: compare actual requests
  before scoring; expanded schema targets need new requests. No scores changed.
- Validation: 89 offline tests pass, including authoritative scope metadata and
  array-column scope checks. Browser checks in temporary fixture folders verify
  additions preserve human corrections and unchecked fields, a new field can be
  unticked then saved/reloaded, history/conflicts/invalid JSON work, and original
  acceptance remains available. Inspect screenshot of the live checkbox editor.
  Validate all 65 proposed/original records against their schemas and original
  sidecar hashes. All 45 real structured/table gold/history files remain byte-
  identical. Restart local editor on port 8768 with updated save handling.
- Generated workspace remains ignored under documents/structured_references;
  build/audit/browser evidence remains in /private/tmp/ocr-structured-review-
  20261006. No inference, judges, benchmark commands, rescore, commit or push.

## 2026-10-06 — correct shared schema/KV field selection

- User rejected the separate scalar-only KV interpretation and further UI
  complexity. Replace that workflow with a shared reference: source image,
  editable keys/values, change from original, Include in both, Save. Original
  and patched choices are available at the editor; detailed version/audit
  views are collapsed. All fields, including array leaves, are included by
  default. Existing explicit exclusions apply to both legs.
- New saves record shared_field_selection=true and consistent aliases for
  every field. Server validation rejects divergent schema/KV selection. Both
  scorers use identical selected paths/expected values without sampling; KV
  no longer applies array or three-scalar eligibility restrictions to reviewed
  references. KV matches discovered pairs without requiring array-shaped
  outputs. Legacy references retain their old route and scores are unchanged.
- 92 offline tests pass. New regressions verify same fields and expected values
  in both legs, selected array leaves, wrong and invented values, exclusions,
  rejection of inconsistent selection, and all 85 selected fields retained.
  Fresh-browser fixture check verifies every field has a selected checkbox,
  array exclusion and value edits survive save/reload, and the served editor
  displays the source alongside the simple shared field table. A prior browser
  session stalled; restart with a fresh test profile completed the check.
- Saved annotations, originals and table gold are not rewritten. Changes stay
  provisional until user saves; this does not certify the proposed values as
  human gold. No engine/judge calls, benchmark run, rescore, commit or push.

## 2026-10-06 — fresh per-document JSON extraction for human approval

- User explicitly requests extraction with a subagent per document. Launch 65
  independent document tasks, three concurrently, each reading only its source
  image. Prohibit reading old references, proposals, gold and engine outputs.
  Extract flat readable key/value JSON, preserve printed formatting and blank/
  checkbox states, qualify repeated fields by source context, and flag uncertain
  readings rather than infer values. No benchmark engine, external OCR API,
  judge or Modal calls.
- Receive and integrate all 65 proposals: 4,344 fields, 93 uncertainty entries
  across 27 pages, with 14 pages marked partial by their agents. These are
  unapproved machine proposals, not a claim of human-verified correctness or
  exhaustive coverage. Uncertain fields are visibly flagged and unchecked in
  both legs pending human resolution.
- Replace the previous patched slot in documents/structured_references with
  these fresh extractions, retaining source image, original snapshot and saved
  gold. Fresh proposals open by default until a fresh version is approved;
  subsequent visits open the user's saved version. Display readable flat keys
  rather than tableData array paths. Track fresh approvals separately from old
  gold. JSON edits automatically regenerate matching schemas and shared field
  mappings, and saved records retain extraction provenance.
- Validate all files with duplicate-key detection, scalar-only values, source
  image identity, JSON Schema validation, uncertainty paths, and exact agreement
  between page data, proposals, schema fields and aliases. Verify original
  sidecar hashes and all 45 pre-existing structured/table gold/history files.
  Browser checks save only to temporary fixtures: edit value, exclude field,
  save/reload, add/delete keys in JSON, regenerate schema and preserve proposal
  provenance. No real gold created or overwritten by these checks.
- Raw agent outputs, instructions, build/integration and browser evidence live
  in /private/tmp/ocr-structured-extraction-20261006. Persistent review proposals
  and HTML stay in the existing gitignored documents/structured_references
  workspace. Docs updated in place; git diff --check clean. No new scores,
  engine reruns, commit or push. Human vetting and confirmation remain pending.

## 2026-10-06 — activate user-approved shared references

- User accepts the fresh extractions as KV/schema references. Preflight all 65
  against image identities, structured-record validation and their prepared
  inference schemas, then save through StructuredGoldStore with explicit batch
  approval provenance. All 65 now load as fresh shared gold. 4,344 fields total;
  93 flagged readings remain excluded from both. Preserve any already-saved
  fresh edits (none found); back up the earlier bank_000 gold before replacing
  it. All 44 table-gold/history files remain byte-identical.
- Fix request-schema preparation: legacy named-field overrides apply only to
  schemas containing open dictionaries, so accepted explicit flat schemas are
  not forced into old nested paths or rejected. Six focused offline tests pass;
  all 65 actual accepted reference answers validate against prepared schemas.
- Inspect saved schema request evidence and applicable schema outputs: seven
  engines have 65 schema results each, from schema passes. These require new
  extraction calls because the supplied schema changed. KV's prompt supplies
  no reference schema; saved pair answers can be rescored first. No inference,
  judge calls, benchmark commands, rescore, commit or push performed.
- Activation evidence stays in /private/tmp/ocr-structured-extraction-20261006/
  activation.json; accepted files remain in the existing structured gold folder.
  The pre-existing tax_000 table-image identity issue remains separately logged
  and must be resolved before a corpus-wide run; this work does not alter it.

## 2026-10-06 — consolidate scoring decisions and geometry follow-up

- Update the existing follow-up checklist with the accepted shared KV/schema
  references and a current remaining-work summary. Mark the earlier scalar-only
  KV workflow as historical; preserve uncertain-field and partial-coverage flags.
- Record gold-driven table cell F1 and the user's shipping exception: score
  complete gold and original references separately, keep the higher F1, retain
  both results and winning-reference provenance, and apply the rule uniformly.
- Specify batched judge mapping for unresolved KV keys, reusable context-scoped
  synonyms and subsequent code-based value checks. Mapping does not judge values
  or silently grant credit to ambiguous keys. No judge calls made.
- Research official Anthropic coordinate/vision guidance and compare it with
  the already completed geometry controls. Record a bounded tighter-word-box
  prompt plus a separate crop configuration, preserving Sonnet 4.6 and requiring
  actual box inspection. Pixels/resizing alone are not a newly discovered fix.
- Documentation only this turn. No implementation, inference, benchmark command,
  reference modification, rescore or report regeneration. Check documentation
  diff/whitespace; existing implementation edits and review artifacts remain
  separate from this documentation update.

## 2026-10-06 — geometry metric plan and silver substring fix

- Record the user's geometry follow-up: located-unit precision/F1 to expose
  extra boxes, plus CER/WER from box-associated text. Keep text fidelity distinct
  from placement; record annotated scope, repeated predictions and line/word
  units. This geometry scorer work remains pending.
- Remove `_verify_locally` and its punctuation-stripped substring auto-credit
  from the separate silver-document judge. Send every sampled item in one
  contextual verdict batch, explicitly preserving decimal/sign/identifier
  meaning and labeled-field association. No false-positive local shortcut
  remains; model verdict accuracy still requires review.
- Persist `silver-document-v2-no-substring-2026-10-06` in newly generated judge
  rows. Old verdicts are not migrated or relabeled. Document that refreshing
  estimates can increase judge tokens/calls and requires a costed execution plan.
- Validation: `tests/test_judge.py` passes all 12 offline tests, including mocked
  substring, decimal, sign, wrong-field, correct-value and mixed-batch cases.
  No provider calls, benchmark commands, reference changes or rescoring. Update
  the existing follow-up and changelog; keep unrelated pending reference/editor
  changes out of this commit. No push.

## 2026-10-06 — implement reviewed-reference scoring, before reruns

- Integrate the existing offline table and structured reference editors and
  loaders. Active schema/KV gold uses the same complete field selection and
  explicit exclusions; table gold remains independent.
- Add deterministic table cell precision/recall/F1 with blanks, extras,
  disjoint matching, native-table preference and the whole-page shipping
  gold/original maximum. Preserve spans in the separate structure diagnostic.
  Keep parser omissions/extra cells visible instead of silently padding/trimming.
- Replace permissive Markdown heading/type credit with semantic Markdown/HTML
  parsing, full-content similarity and disjoint block matching. Add separate
  CER/WER from the actual Markdown answer. Missing references remain pending.
- Add geometry precision/F1, invalid/unmatched prediction counts and CER/WER
  from box-associated text. Preserve word versus line-segment units. Annotation
  scope handling and the new Sonnet inference controls remain pending.
- Match shared structured keys before checking values. Schema uses literal
  supplied keys and penalizes unexpected/duplicate fields; raw JSON/schema
  validity is a separate diagnostic. Consume reference-versioned KV synonyms
  offline; the paid key-only synonym builder remains pending.
- Guard live/stored schema scoring against changed or missing request contracts.
  Old answers need rerun, not retrospective failure against new keys. Rescore
  manifests name current scorers; copied silver verdicts are marked historical.
- Recover canonical tax_000 image bytes from the table review's embedded PNG.
  Verify the upright image is an exact 180-degree RGB rotation, back it up under
  /private/tmp/ocr-eval-rebuild-20261006/tax-identity-backup, and restore the
  manifest hash. Update structured reference image provenance without changing
  values. Table gold unchanged; all 93 corpus pages load.
- Validation: full offline suite 332 passed; request-contract and registry tests
  26 passed after adding the guard. Test changes replace obsolete nested-key
  expectations with the accepted flat gold while preserving isolated legacy
  repair coverage. No engine inference, paid judge call, published rescore or
  report regeneration. Gold packaging, fair-prompt controls and cleanup remain.

## 2026-10-06 — prepare reproducible key-only synonym judging

- Add evaluation/kv_synonyms.py and scripts/build_kv_synonyms.py. Offline
  preparation reads saved structured KV answers, unions unresolved keys per
  page, and writes exact prompts/configuration/response schema plus reference
  fingerprints and source-call hashes. No values or engine identities enter
  the model request. Distinct labels can remain ambiguous/no-match.
- Paid execution is a separate command with an explicit allowance. Validate
  complete unique output IDs and reference IDs, record every submission before
  sending, disable SDK retries, preserve raw responses/usage, reject changed
  references, resume completed calls and refuse unresolved paid retries. Freeze
  all page caches after all batches validate; scoring remains offline.
- Prepared baseline plan: 260 source calls, 65 pages, 5,272 unresolved keys,
  117 batches at most 64 keys each. Gemini 2.5 Flash, temperature 0, thinking 0,
  output cap 4,096. Official Google pricing checked 2026-10-06: $0.30/M input,
  $2.50/M output; conservative reserve $1.55423, proposed allowance $2.
- Plan saved outside the repository under
  /private/tmp/ocr-eval-rebuild-20261006/kv-synonym-plan.json. The follow-up
  records its hash, exact commands, costs and approval boundary. No paid call
  made; no synonym cache or published scores changed.
- Validation: 10 offline tests cover key-only prompts, deterministic batching,
  malformed/incomplete/duplicate IDs, reference invalidation, atomic page
  publication, failed-call journaling/no retry, budget checks and successful
  resume with mocked provider responses. These verify mechanics, not model
  mapping accuracy; inspect live mappings before final reporting.

## 2026-10-06 — preserve saved scoring-method labels

- Full suite after the synonym CLI: 342 passed and one historical-report label
  regression failed. Updating current table labels had also relabeled saved
  code-v3 results as gold-reference F1. Fix the report to derive legacy/current
  method descriptions from the recorded scorer version; mixed versions retain
  a generic saved-headline title.
- All 20 report tests pass after the fix, including old model verdicts, old code
  checklist scores, new gold-cell F1 and mixed-version descriptions. No report
  generation, inference or paid call. This closes the remaining suite failure.

## 2026-10-06 — execute the approved first KV-synonym step

- Authorization: user said “ok run the first then” after the 117 key-only
  Gemini 2.5 Flash calls and $2 allowance were distinguished from the separate
  455 schema extraction calls. Execute only that approved key-matching plan.
- Initial startup found no key in the nested repo environment and made no API
  call. Load the existing parent project .env without displaying credentials.
- First four calls yielded three valid replies and one truncated response.
  Preserve the charged truncated reply. Improve execution to continue the
  remaining independent approved requests, at four concurrent calls, without
  resubmitting completed/invalid requests. Keep exact model, prompt, output
  cap, temperature, thinking configuration and total plan reserve unchanged.
  Exclusive request journals prevent duplicate submissions; incomplete pages
  cannot become active synonym caches.
- All 117 planned requests attempted exactly once: 101 validated replies,
  13 MAX_TOKENS/incomplete JSON responses, 2 invalid-ID replies and one 503.
  Valid replies contain 4,421 decisions: 2,768 proposed matches and 1,653 null
  matches. Exact plan, request journals and execution summary are preserved
  under results/kv_synonyms/calls/cafc0f767f3fa5b1b9b717f2da153f92f23ac5541541d9b3c4c0ea67e76ca758/.
- Recorded usage cost $0.6900703. The 503 supplied no usage: retain its
  $0.013288 reserve, rather than claiming confirmed zero billing. Full plan
  reserve $1.5542298, allowance $2. No paid retries, schema/OCR calls, Modal,
  missing-reference builds or silver verdict refreshes.
- Bounded audit: structurally validate every completed reply, generate 122
  deterministic mapped-pair examples for review across represented pages, and screen
  context/index conflicts (228 candidates, not an asserted error count).
  Confirm photo_000 ASAM JAWA 1KG U.price was incorrectly assigned to Item 1
  although the accepted gold identifies it as Item 3; Item 1 is taucu kisar.
  Indexed ETF-to-name and unqualified chart-year mappings also lack key-only
  identity evidence. Save examples and limitations in audit.json. Judge
  prompts contained no values; reference values were consulted only in the
  independent offline audit, not fed back into these calls.
- No page cache activated, no score/report changes. The first batch is not
  ready to score. Correction needs compact constrained replies and conservative
  entity/section identity handling; record this in the existing follow-up.
  Do not treat these failures as weaknesses of the OCR engines.
- Validation: all 11 offline synonym tests pass, including invalid-response
  continuation, preserved costs, no retry and no partial-cache publication.
  Diff whitespace check clean. All new results are paid-run evidence; the
  temporary audit script/output remain outside the repository. No API credential
  appears in any saved JSON. Commit only in the nested OCR benchmark repo.

## 2026-10-06 — prepare a bounded KV correction check

- Continue from the held 117-call batch without spending again. Implement a
  compact output-ID-to-reference-ID object, required properties and bounded
  integer IDs, duplicate-property rejection, and explicit instructions against
  inferring item identity from list order or an ambiguous repeated label.
  Keep Gemini 2.5 Flash; proposed control configuration uses temperature 0,
  4,096 maximum output tokens and 1,024 thinking tokens. Recheck official
  structured-output/pricing documentation; record links in the follow-up.
- Prepare six calls on bank_000, photo_000, chart_003, invoice_000,
  shipping_002 and tax_000: 41 actual saved unresolved keys, 18 positive
  mappings and 23 no-matches, with manually reviewed key-only expectations.
  Expected answers remain outside prompts. Plan/cases live only in
  /private/tmp/ocr-eval-rebuild-20261006/. Reserve $0.0772059, proposed allowance
  $0.10; this new configuration has not been submitted or approved yet.
- Grade controls for false associations, missed matches and missing replies.
  Failed/missing replies cannot masquerade as correct no-matches. A passing
  small control set is not whole-corpus validation. Prevent all automatic cache
  activation on execute and reject control subsets in freeze_page.
- Offline formatting recovery: fintab_002 contains all 54 requested IDs plus
  an extra ID 0. Remove only the extraneous row into a separate scratch candidate
  and validate completeness; do not approve its meanings, alter the original
  paid journal, or activate a cache. The other 15 failed calls remain incomplete.
- Validation: 35 offline synonym/shared-field tests pass, including SDK schema
  construction, positive/negative controls, missing replies, duplicate properties,
  no partial/control cache publication, mocked paid execution and reuse. No
  paid call, benchmark run, schema extraction, rescore, reference edit or report
  regeneration. Only the existing module, CLI, tests and three docs are changed;
  all exploratory artifacts remain outside the repo. Commit in nested repo only.

## 2026-10-06 — execute approved six-call KV controls

- Authorization: user said “yes fine” to six calls with the $0.10 allowance.
  Verify the exact prepared plan hash, unchanged reference fingerprints and
  empty request journals before submitting; use existing parent .env credentials
  without exposing them. No prompt, model, configuration or case changes.
- All six calls attempted once, zero retries. All six replies are valid JSON
  and finish with STOP. Independently decode/validate saved text and reproduce
  all 41 control verdicts from saved replies. Correct: bank 8/8, chart 4/4,
  invoice 8/8, photo 8/8, shipping 5/5, tax 7/8.
- Result: 40/41 correct, zero false associations, one missed association,
  no missing decisions. All 23 intended no-matches rejected; 17/18 positive
  synonyms accepted. Judge abstained on `b Taxable amount 4b` despite reference
  `4b — Taxable amount`. Confirm the explicit line identity from both keys;
  do not claim a hidden reason for a response with no explanation requested.
- Recompute usage from the six raw provider records, including thinking tokens:
  $0.0174635 total, within the allowance. No unknown-billing calls. Verify
  artifact files contain no API credential and no active page synonym caches.
- Preserve all nine experimental artifacts outside the repo in
  /private/tmp/ocr-eval-rebuild-20261006/kv-controls-results/ (plan, six request
  journals, execution summary, audit). Remove their generated repo directory by
  moving it intact, not deleting evidence. Do not re-execute this control plan.
- Update existing follow-up and changelog. The strict all-controls-pass gate
  failed; stop before a larger batch, with the single explicit tax-line mapping
  recorded for a reviewed correction. No further pilot is needed to discover
  this same miss. No new schema/OCR call, gold edit, active cache, rescore or
  report. No source-code changes, so no repeated unit suite; validate the actual
  saved responses and costs instead. Commit only these three docs in nested git.

## 2026-10-06 — package accepted references for clean checkouts

- Version the 42 accepted table gold files, 65 accepted shared structured gold
  files and five frozen shipping original alternatives without editing their
  contents. Keep images, proposals, history and browser exports ignored. This
  makes the accepted reference layer available to other benchmark users.
- Fix downloader to preserve underlying image bytes rather than recompressing
  PIL images, check the pinned manifest hash, resume missing source files and
  refuse to overwrite differing local sidecars. Gold is never a download target.
  Verified Image(decode=False) against Hugging Face's image-loading docs.
- All 93 local source-image hashes match the committed manifest. All accepted
  references load with image identity checks; 24 ground-truth/download tests
  pass, including exact bytes, partial download recovery and edit preservation.
- Correct README's current-status claim: the saved October 2 report predates
  these changes. No external publication, API/model call, rescore or change to
  accepted field values. Commit only in the nested repository.

## 2026-10-06 — scope geometry extra-box and text diagnostics

- Add fixed annotation-region handling before matching, precision/F1 and
  box-text CER/WER. Corpus region is the retained reference-box envelope,
  excluding reviewed unreadable regions. It is derived without engine outputs
  and does not certify every printed mark within that region.
- Preserve every parsed/converted returned box for the overlay, with excluded
  predictions and scope metadata separately inspectable. Out-of-frame/invalid
  predictions remain penalized; a displaced excluded prediction cannot satisfy
  the missing reference. A duplicate annotation repair is not a spatial exclusion.
- Version geometry-v6 and document the region rule and residual annotation
  omissions. All 66 geometry/loader tests pass after correcting the fixture's
  list-versus-tuple comparison. No inference or paid call, no reference edits.

## 2026-10-06 — distinguish pending KV decisions from engine failures

- Unreviewed keys now make the KV result explicitly pending (N/A), with its
  provisional arithmetic retained only as a diagnostic. A cached explicit
  no-match is a completed decision; changed references invalidate old decisions.
  This prevents an unfinished synonym workflow from becoming an engine ranking.
- Resolve the control's tax-line miss deterministically when the entire label
  and explicit line identifier match despite printed identifier placement:
  `b Taxable amount 4b` matches `4b — Taxable amount`. Different line numbers,
  extra qualifiers and contradictory leading letters do not match; values are
  never consulted and schema still requires literal supplied keys.
- 109 shared-field, synonym and structured-leg tests pass. No model call,
  active cache, reference mutation or score publication. This closes the one
  known control miss without another paid experiment.

## 2026-10-06 — respect reviewed exclusions during KV matching

- The full offline suite caught an excluded array field incorrectly leaving KV
  pending. Exact excluded keys now bypass synonym review in both legs, while
  selected fields and expected values remain unchanged. The existing shared
  array/exclusion regression passes; all 369 tests pass. No paid calls.

## 2026-10-06 — replay historical calls and show current scoring evidence

- Rescoring a rescore now uses each record's original call timestamp. Historical
  document paths relocate to the current checkout; original paths are retained.
- Evidence viewer filters engine/type/leg/page and recalculates equally weighted
  document-type means. Show actual table cells/spans, gold/original comparisons,
  pending KV pairs, strict Markdown verdicts, text CER/WER and geometry scope,
  precision/F1 and actual returned boxes. Gold-table diagnoses count verdicts
  instead of assuming every result has wrong values.
- Free checkpoint `20261006T171802Z`: 4,615 rows; 1,148 text, 627 Markdown,
  504 table and 145 geometry results scored. Eight saved geometry errors remain;
  455 schema contracts need rerun; unfinished KV and missing catalogues are N/A.
  The 110 silver estimates are explicitly historical. No OCR/judge was called.
- Verified every scored table's F1 and rendered cell count, every Markdown
  headline against verdicts and every geometry overlay count against parsed
  boxes. JavaScript syntax passed; headless Chrome rendered the overview and
  its screenshot was inspected. Filter interactions and all detailed views were
  not exhaustively browser-tested. All 369 offline tests pass.
- Keep provisional JSONL/CSVs under session scratch `offline-rescore/`, and the
  self-contained viewer at `/private/tmp/ocr-eval-rebuild-20261006/ocr-results.html`.
  No experimental artifacts committed and no final report replaced. The small
  excluded-field fix postdates this checkpoint; final rescore must include it.

## 2026-10-06 — prepare compact KV requests; hold spending and review reuse

- Production preparation can reuse the exact compact control recipe, bounded
  to 16 keys per request. Controls and corpus requests share one builder;
  missing/invented IDs and duplicate properties remain rejected. No values
  enter requests. The CLI can load credentials from the existing parent .env
  without exposing them. All 369 offline tests pass.
- Prepared a full replacement: 356 calls, 65 pages, 5,199 unresolved keys,
  reserve $4.7173, proposed allowance $5. The user questioned repeating KV and
  the price. **Not approved; not executed.** Correctly explain that the first
  117 calls ran and the six later calls were controls, not corpus completion.
- The initial failure/flag union covers 1,075 current keys, leaving 4,124
  potential reuse decisions. The original risk screen misses known photo/chart
  errors, so these are not certified reuse counts. A stricter exact-label
  screen finds 607 positive candidates. All review lists remain in scratch;
  no cache activated, original replies preserved and no further money spent.
- Keep the full replacement on hold and review reuse before requesting a
  smaller completion batch. Paid schema, prompt controls and catalogue work
  remain separate unfinished parts of the user goal. No new engine removed.


## 2026-10-06 — inline value-assisted KV evaluation

- User superseded the separate key-only batches and explicitly authorized rewriting
  the shared evaluator and running saved KV outputs. Known keys/cached variations
  stay local; only unmatched keys with complete-value candidates reach Gemini.
  Cache negatives and exact-pair decisions as well as reusable aliases. Values
  are always checked again. Repeated reference labels cannot become unconditional
  aliases. The same route handles live, stored and rescored calls for every engine.
- Remove the old scalar/printed-pair/minimum-three eligibility gates: KV and
  schema use all selected reference fields. Add KV-only rescore filtering,
  four workers with page locks, per-call journals, one SDK attempt, a $2 cap,
  recovery of saved replies without repaying, and replanning between batches.
  Retire the old separate key-only execution CLI. No schema or OCR calls.
- Validation: 386 offline tests passed; the added selected-leg rescore regression
  also passed with the report tests (25 passed). Tests scrub paid credentials.
- First paid pass 20261006T194648Z: 319 responses, $1.3890981. Thirty-eight
  batches contained a proposed reference with a nonmatching value. The original
  validator rejected the batch; corrected validation rejects only that pair,
  preserves the proposal as audit evidence, and recovers valid entries from the
  same saved reply. No repeated calls required for that recovery. Free replay
  20261006T195502Z applies the fixes. Final completion/results to be recorded below.
