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
