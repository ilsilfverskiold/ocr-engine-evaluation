# Changelog

Notable changes to the scoring code. The reference run is never modified;
every change is rescored from stored outputs (`--rescore`) and the run id
is given here.

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
