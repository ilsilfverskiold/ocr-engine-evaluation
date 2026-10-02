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
