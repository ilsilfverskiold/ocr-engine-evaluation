# Work log

## 2026-10-07 — publish one reproducible benchmark

- Consolidated the 4,505 current score records, 2,375 inference ledger entries,
  2,282 exact source calls and active Markdown/KV/silver caches into
  `results/current.zip` (34.3 MiB). Every archived file was checked byte-for-byte
  and SHA-256 verified before its loose copy was removed. Original source IDs
  remain inside the archive; readers rebase document paths after relocation.
- Removed superseded run manifests, summaries, diagnostics, completed reference
  editors/proposals and review pages, historical key-only judge tooling, and
  development tests after validation. Retained source images, current gold,
  shipping alternatives, runtime adapters, all Modal integrations and the local
  final HTML. Documents decreased from 281 MiB to 99 MiB.
- Rewrote README and consolidated integration guidance. Current scoring-design
  documentation describes each leg, prompt/reference limitations and reproduction.
- Added deterministic full-versus-low-case sensitivity reporting. Cases require
  at least three scored engines, all ≤0.25 and no failed/pending participants;
  one common exclusion set is used for all engines. Full scores remain primary.
  Counts and lost document types are shown. Current cases: tables/tax_001 and
  key_value/chart_003. No scores or reference values changed.
- Terminal reporting reads the bundle without creating logs or exports by default;
  detailed reporting adds per-engine/type tables and local CSVs. HTML includes
  full/excluded comparisons, cost/latency and sortable supporting metrics.
- Deduplicated physical calls for operational reporting: 93 Textract default
  calls had separate ledger rows for KV versus other legs. Primary/all-pass cost
  now both report $0.065/page. No quality score changed.
- Validation so far: 485 regression checks passed in the post-archive run;
  two logging-mock compatibility failures were corrected and all 30 report
  checks then passed. Five tests of the removed historical CLI were retired.
  Archive-only lookup of every file, cohort threshold/pending/empty-result cases,
  aggregate equivalence and physical-call cost deduplication also passed.
  Full offline replay and final HTML checks are in progress.
- No paid inference, judge calls, deployments or push. Historical implementation
  logs and removed development files remain available through Git history.
