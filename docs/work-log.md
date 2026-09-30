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
