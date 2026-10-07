# OCR Benchmark

Compare OCR engines on six jobs: text, Markdown structure, tables, key/value
extraction, supplied-schema extraction and word/line location. The corpus has
93 pages. The current comparison contains 16 engine configurations, including
two geometry-only extensions. Additional integrations remain available but unrun.

## Open the current results

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_benchmark.py --report 20261006T221526Z
```

This prints saved results without inference or API calls. For the browser report,
first download the source pages, then build the self-contained HTML:

```bash
python scripts/download_data.py
python scripts/build_evidence_report.py 20261006T221526Z \
  --status "Current rescore; 201 KV outputs await contextual review. Further paid controls deferred."
```

Open **`results/reports/ocr-benchmark-final.html`**. Filter by engine, document
type, leg and page. It shows source images, exact prompts, returned field pairs,
logical table cells, actual returned boxes and per-item verdicts. Select a leg
to see its supporting metrics in the engine overview, including CER/WER.
Generated HTML is local and ignored by Git; source records and the builder ship
with the repository.

**Current status:** 4,505 six-leg results, including N/A, pending and failed rows.
Schema has 453 scored results and two server errors. Markdown uses 92/93 cached
reference catalogues. KV has 59 settled scores, 201 outputs pending the revised
contextual judge and 28 pages without structured references. Pending results
are excluded from means, so the current KV averages are incomplete. Historical
silver-document estimates are retained in the source ledger but omitted from
the browser comparison unless `--include-historical-judge` is requested.

## Data and accepted references

Source images and original sidecars are hosted at
[ilsilfverskiold/ocr-benchmark](https://huggingface.co/datasets/ilsilfverskiold/ocr-benchmark).
The downloader checks manifest hashes and refuses to overwrite differing files.
Accepted references are versioned here:

- `documents/table_references/gold/`: 42 human-reviewed table references.
- `documents/structured_references/gold/`: 65 accepted shared KV/schema references.
- `documents/table_references/old/shipping_*.json`: five shipping alternatives.
- `documents/schema_sample.json`: the same 3,316 of 4,251 eligible fields for
  every schema engine; KV retains the full set. Forty-five pages keep all fields,
  twenty are sampled to fit the inference limits.

Local review editors, proposals and save history remain available in this
checkout but are not required by the scorer and are not published with Git.
Image licences and dataset sources are in
[documents-and-licensing.md](docs/documents-and-licensing.md).

## What the scores mean

| Leg | Main score | Supporting evidence |
| --- | --- | --- |
| `text` | Distinct reference-token coverage | CER/WER, word precision/recall/F1; coverage ignores extras and order |
| `markdown` | Correct structural elements, half credit for partial matches | Actual Markdown/HTML syntax, one-to-one block matching, rendered-text CER/WER |
| `tables` | Cell F1, including blanks and extra/duplicate cells | Cell precision/recall, spans and structure similarity; row/column association |
| `key_value` | Correct values / eligible shared reference fields | Exact/cached keys first; value overlap nominates contextual judge reviews; no automatic containment credit |
| `schema` | Correct values / (requested fields + unexpected/duplicate fields) | Exact supplied keys, same frozen sample, per-field verdicts and JSON compliance |
| `geometry` | Located reference-unit recall | Located precision/F1, box-text CER/WER and actual box overlays |

CER/WER are error rates: lower is better and extra text can make them exceed 1.
They are kept separate from structure and location. Text references cover 82
pages; unreliable silver text is excluded. Geometry references cover 17 pages
and mix dataset word and line units, disclosed in each result.

For tables, accepted gold takes precedence. Shipping is compared against each
whole-page gold/original alternative and the higher F1 is selected, identically
for every engine. No per-cell reference cherry-picking occurs. The structure
metric is a custom ordered row/cell diagnostic, not official PubTabNet TEDS.

KV and schema use the same accepted keys/values. KV allows wording differences;
schema requires the exact keys supplied in the request. Value matches and
complete-token containment produce judge candidates, including split values
and repeated amounts. The judge sees all returned pairs for context. Decisions
are cached; group approvals require complete coverage and cannot reuse an
output field. Leading/trailing currency glyphs normalize without erasing signs,
decimals or identifier digits. Without a required decision, the score is pending.

Means average pages within each document type, then types equally. Errors and
missing references are excluded, with counts shown. Compare coverage alongside
scores. Historical general-model defaults are not tuned Markdown/table-only
controls; GLM's alternative SDK pipeline and further prompt controls remain
unmeasured. See [scoring-design.md](docs/scoring-design.md) for exact rules and
[follow-up-2026-10-03.md](docs/follow-up-2026-10-03.md) for completed/deferred work.

## Reproduce scores without spending

```bash
# Replay the six current legs from saved calls; no inference or new judge calls.
python run_benchmark.py --rescore 20261002T162630Z,20261006T214240Z \
  --legs text,markdown,tables,key_value,schema,geometry --skip-judge

# Rebuild HTML using the new timestamp printed by that command.
python scripts/build_evidence_report.py NEW_TIMESTAMP

# Prepare exact unresolved KV requests and a cost reserve without sending them.
python scripts/build_kv_synonyms.py prepare-values \
  --source 20261006T221526Z --output /tmp/kv-review-plan.json
```

Later supplied runs replace earlier answers for the same engine/page/leg. A
newer failure remains an error. Dedicated passes take precedence within a run;
successful resumes replace failed attempts. Source ledgers retain full history.
Accepted reference edits invalidate affected caches and the schema sample.
Changing schema keys requires new extraction, not merely rescoring old answers.

`--skip-judge` suppresses new model judging, including KV. It does not suppress
OCR calls when used with a new inference run. Existing cache decisions can still
be used. Pending KV reviews remain pending. Without this flag, KV may call the
judge when a key/value ambiguity needs it; `OCR_KV_JUDGE_BUDGET_USD` limits its
planning reserve (default $2). Failed/invalid paid replies are journaled and are
not automatically retried.

## Run engines

For local CPU inference (install Tesseract separately):

```bash
python run_benchmark.py --engines tesseract,docling --tiers easy --skip-judge
```

For paid APIs, copy `.env.example` to `.env` and fill only the relevant keys.
Use explicit engine and leg selections:

```bash
python run_benchmark.py --check
python run_benchmark.py --engines mistral_ocr_4 --tiers easy --legs text --estimate
python run_benchmark.py --engines mistral_ocr_4 --tiers easy --legs text --skip-judge

# Separate extraction from evaluation. Extraction still costs money.
python run_benchmark.py --engines gemini_flash --legs schema --collect-only
python run_benchmark.py --score SAVED_TIMESTAMP --skip-judge
```

Self-hosted engines use Modal. Deploy only the selected app; smoke tests perform
inference unless skipped:

```bash
modal setup
python scripts/setup_modal.py
python scripts/deploy_modal.py --apps glm_ocr --skip-smoke
```

`engines.py` is the registry for names, native/prompted legs, dependencies and
Modal configuration. `evaluation/legs.py` supplies task requests. Native outputs
can serve several legs in one paid conversion; prompted tasks use separate
passes. Use `python run_benchmark.py --help` for page filters, resume and errors.
A resume/retry can spend money; it is not an offline scoring operation.

To add an engine, create a runner implementing `runners/base.py`, register it
in `engines.py` and declare supported legs. Reuse a native pipeline or prompt
interface rather than inferring missing structure with another model. Run the
same shared scorers and accepted references. The unrun expansion integrations
and their research remain in place:

- [Implementation notes](docs/engine-expansion-implementation.md)
- [Research](docs/engine-expansion-research.md)
- [Leg eligibility and cost matrix](docs/engine-leg-and-cost-matrix.md)

## Costs and limitations

Saved costs are estimates from reported usage and recorded rates, not invoices.
Modal records GPU type, measured batch intervals and startup allocation; CPU,
RAM, idle time and unreturned provider usage may be excluded. A shared inference
pass costs once, not once per scored leg. Inspect source ledgers and paid-run
audits when accounting for all attempts rather than just selected answers.

The completed shared-schema extraction recorded approximately $4.7279. Earlier
value-assisted KV judging recorded $1.4756; the missing-only Markdown catalogue
run recorded $0.6434. The new KV contextual correction is prepared but **not
executed**; its $5.56 planning reserve is not spend. Existing raw calls, active
caches and audit files are retained. No new OCR inference is needed to apply
that correction to the saved answers.

Cached model-authored catalogues are not human gold and can contain errors.
A reference shared by all engines does not guarantee neutral ranking effects.
Default prompts, missing regions, segmentation and reading order can affect
scores; inspect each leg's actual output before assigning the cause. Further
paid controls and retries are deferred, not concealed as completed work.

## Development and licence

```bash
python -m pytest -q
```

Work is logged in [work-log.md](docs/work-log.md) and
[CHANGELOG.md](docs/CHANGELOG.md). Code is MIT; document images retain their
original third-party licences.
