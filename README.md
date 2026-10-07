# OCR Benchmark

Compare OCR engines on six separate tasks: text, Markdown structure, tables,
key/value extraction, supplied-schema extraction and geometry. The corpus has
93 pages. The published comparison contains 16 engine configurations and 4,505
results, including failed and not-applicable records. Additional engine
integrations are available but have not been benchmarked.

## View the saved results

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_benchmark.py --report current

# Focus on one or several legs.
python run_benchmark.py --report current --legs tables
python run_benchmark.py --report current --legs text,markdown
```

These commands read **`results/current.zip`** without calling engines or judges.
The terminal report compares engines side by side, rather than listing each
engine separately. It opens with an engine × leg matrix, then ranks engines
within each leg. Overall, easy, medium and hard tables put full scores,
outlier-excluded scores, page counts and the leg's supporting metrics together.
Legs with fewer than 24 scored pages (currently geometry) stay in one table.
Cost/latency comparisons use the actual recorded pass for that leg, with cheapest
first and a separate speed rank. Best values in each metric column are marked `*`.
Missing values sort last; failed engines remain visible. There is no cross-leg
average that rewards engines for skipping difficult legs.

```bash
# Rank text engines by character error rate, lowest first.
python run_benchmark.py --report current --legs text --report-sort cer
# Compare geometry latency, fastest first.
python run_benchmark.py --report current --legs geometry --report-sort latency
# Rank by the score after removing the common outlier cases.
python run_benchmark.py --report current --legs tables --report-sort without-outliers
# Add document families (invoice, receipt, etc.), other metrics, diagnostics and CSVs.
python run_benchmark.py --report current --report-details
```

`--report-sort` also accepts `score` (default), `wer`, `f1`, and `cost`.
Where a leg lacks the requested quality metric, its ranking falls back to Full.
Supporting metrics use the full scored set; a parenthesized count means the
metric has fewer reference pages. Cost/latency use successful scored pages;
coverage is shown and can differ between engines. No logs/exports are created
unless detailed CSVs are requested. HTML reports are unchanged by terminal reporting.

For the browser report, download the source pages once and build:

```bash
python scripts/download_data.py
python scripts/build_evidence_report.py
```

Open **`results/reports/ocr-benchmark-final.html`**. It includes source images,
exact returned outputs, saved prompts, box overlays, reference tables and
field/cell verdicts. Filter by engine, document type, leg and page. Click metric
headings to sort highest/lowest. Individual text and Markdown results show
CER/WER. The self-contained HTML opens offline and is **ignored by Git**.
Building it makes no model calls. Downloading pages requires network access.

## What each leg measures

All headline scores below run from 0 to 1, higher being better. CER/WER are
separate error rates: lower is better, and insertions can make them exceed 1.

| Leg | Headline | Why the supporting metrics matter |
| --- | --- | --- |
| `text` | Fraction of distinct normalized reference tokens found | Coverage ignores extra/repeated words and order. CER/WER measure character/word insertions, deletions and substitutions; word precision/recall/F1 show excess or missing text. No judge. |
| `markdown` | `(correct + 0.5 × partial) / reference elements` | Locate full/partial content, then check actual Markdown/HTML types. Bold text is not a heading. Disjoint spans can share a paragraph; text cannot earn credit twice. Rendered-text CER/WER are separate from structure. |
| `tables` | Reference value credit divided by reference cells, including blanks | Exact normalized values earn 1; partial text earns similarity ≥0.5; wrong numeric values earn 0. Logical items may span adjacent rows or combined cells when labels/headers establish association. Extras do not reduce the headline; strict cell F1, extra-cell rate and structure similarity remain diagnostics. |
| `key_value` | Correct values divided by eligible printed-label reference fields | Exact/cached key mappings first. Full/partial value overlap nominates ambiguous or split-field candidates for a contextual judge; overlap alone never awards credit. Approved mappings are cached. Missing required decisions remain pending. |
| `schema` | Correct values divided by requested fields plus unexpected/duplicate fields | Each engine receives the same frozen schema field sample. Exact supplied keys, types and values matter. Changing requested keys requires new extraction, not just rescoring. |
| `geometry` | Correctly located reference units divided by reference units | Text matching and mutual-center box containment establish location. Precision/F1 expose extra boxes; box-text CER/WER measure recognition. The report shows actual returned boxes, not transcription from another pass. |

Markdown references are cached, model-authored image catalogues, with reviewed
repairs; scoring the engines against them is code-based. There are 92 catalogues
for 93 pages. Full-page block/order diagnostics require independent structural
references and can be N/A even when the checklist headline is available.

Tables use accepted gold references. For shipping, score against **both whole-page
gold and original references and take the higher value score** for every engine.
Do not mix the best individual cells from different references. Items and columns
must remain associated: an identical subtotal cannot satisfy a missing tax cell.
The span-aware structure metric is a custom diagnostic, not official PubTabNet TEDS.

KV and schema share accepted underlying keys/values, but have different task
selection: schema uses a fixed sample; KV excludes invented/unprinted labels and
uncertain label assignments. The KV review overlay does not change schema gold.
Judge-approved split-value groups require full value coverage and cannot reuse
returned fields. Currency symbols normalize without deleting decimal signs or
identifier digits. Extra KV fields are diagnostic; schema extras affect its score.

Geometry covers 17 pages, using lines for SROIE and words for FUNSD/SynFinTabs.
Layout-only outputs also compete, explicitly labeled; they are not converted
into invented word boxes. Reviewed annotated-region scope limits precision;
unannotated text within that region can still affect it.

See [scoring-design.md](docs/scoring-design.md) for precise rules and limitations.

## Full scores and consistently low-scoring pages

The primary score averages pages within each document type, then gives each
represented type equal weight. Errors, missing references and pending verdicts
are excluded, with counts shown. Different coverage can make engine averages
incomparable; inspect the same-page evidence too.

A separate sensitivity comparison removes a **page/leg case** only if at least
three engines have scored it, every scored engine is at or below **0.25**, and
there are no failed or pending participants. Engines with no applicable output
are not treated as zeros. The cases are selected using the entire saved cohort,
not the current display filters, and removed identically for every engine.
Full and remaining page counts and lost document types are disclosed. If nothing
remains, the alternative is N/A. Cost and latency are never reduced by this filter.

This does **not** prove that a page is an outlier or that its reference is wrong.
Hard images, missing reference text, faulty catalogues, prompt choices and scorer
limitations can all produce low scores. Human-accepted “gold” also can contain
mistakes. The alternative is a sensitivity check, not a replacement ranking.

## Cost, latency and known comparison limits

Costs attach to **inference calls**, not each leg that reuses a call. Reports show
successful primary-pass cost/latency per page and all-pass cost per page. Modal
figures use saved batch GPU timing/rates, with steady-state values where recorded;
they are estimates, not invoices. Startup/setup, failed-call charges and reference
or synonym judging are not included in those deployment estimates. Inspect per-pass
coverage: differently tasked calls are not interchangeable. Historical model aliases
and rates describe the saved run, not guaranteed current vendor behavior/pricing.

There are 82 usable text-reference pages; 11 unreliable silver-text pages are
excluded. Reading order, punctuation and incomplete references can raise CER/WER
without proving a recognition error. Historical silver-document judge estimates
are a separate optional facility and are not part of the six published legs.

Historical default prompts were not uniformly dedicated Markdown/table requests.
Gemini's old default did not explicitly request Markdown; Sonnet requested structure
but not a dedicated Markdown-only control. These fairness limits remain disclosed.
Sonnet's improved pixel-coordinate geometry recipe is prepared but **unrun**;
the report retains saved geometry results. Further DeepSeek/prompt/backend controls
also remain unmeasured. Scorer changes do not imply new inference or new deployments.

## Re-evaluate saved calls without spending

Download the corpus first. Then:

```bash
python run_benchmark.py --rescore current --skip-judge
# Read the NEW_TIMESTAMP printed by the rescore.
python run_benchmark.py --report NEW_TIMESTAMP
python scripts/build_evidence_report.py NEW_TIMESTAMP

# Publish the reviewed result and its exact source calls as the single bundle.
python scripts/publish_results.py NEW_TIMESTAMP
```

Publishing verifies file hashes and atomically replaces `results/current.zip`.
It also retains the shared reference catalogues and cached KV judge decisions
needed for offline scoring. It does not make model calls or delete your local
working files. After verifying publication, working `results/runs`,
`results/ocr_outputs`, logs and CSV summaries can be removed; they are ignored.
The archive retains original source-call identifiers for provenance, not multiple
public leaderboards. Source paths are rebased when the repository moves.

`--skip-judge` disables new judging, including KV, but **does not disable OCR
inference** when used in an engine run. Existing cache decisions remain usable;
new unresolved KV decisions stay pending. Without the flag, KV may pay for a
contextual review when needed; `OCR_KV_JUDGE_BUDGET_USD` limits the planning
reserve (default $2). Failed paid reviews are journaled, not blindly retried.

## Run engines

Install only the external system tools your selected engines need; Tesseract
requires its system binary. API credentials belong in `.env` (see `.env.example`).
Use explicit engine and leg selections:

```bash
python run_benchmark.py --check
python run_benchmark.py --engines mistral_ocr_4 --tiers easy --legs text --estimate
# This performs inference and can spend money:
python run_benchmark.py --engines mistral_ocr_4 --tiers easy --legs text --skip-judge

# Collect first, evaluate later:
python run_benchmark.py --engines gemini_flash --legs schema --collect-only
python run_benchmark.py --score SAVED_TIMESTAMP --skip-judge
```

For Modal, authenticate and deploy only the chosen app. Smoke tests run inference
unless explicitly skipped:

```bash
modal setup
python scripts/setup_modal.py
python scripts/deploy_modal.py --apps glm_ocr --skip-smoke
```

For an extension, collect the new engines, then combine their calls with the
published baseline in one rescore:

```bash
python run_benchmark.py --engines rapidocr --collect-only
python run_benchmark.py --rescore current,NEW_TIMESTAMP --skip-judge
```

Later supplied runs replace earlier answers for overlapping engine/page/leg
combinations, including failures. Inspect new coverage and provenance before
publishing. `--dry-run`, `--docs`, `--resume` and `--rerun-errors` are described
by `python run_benchmark.py --help`. Resuming/retrying can spend money.

## Repository layout

| Path | Purpose |
| --- | --- |
| `engines.py` | Registry of interfaces, task support and configuration |
| `runners/` | Required runtime adapters that call APIs, local engines and Modal |
| `modal_apps/` | Deployable inference applications, including unrun integrations |
| `evaluation/` | Shared task planning, reference loading, scoring and reporting |
| `scripts/` | Corpus download, schema/catalogue preparation, deployment, publishing and HTML reporting |
| `documents/` | Corpus metadata and accepted references; source images download separately |
| `results/current.zip` | One published score set, inference evidence and reusable judge/reference caches |
| `results/reports/` | Local HTML only; ignored by Git |

Accepted references include 42 table gold files, 65 shared structured gold files,
five original shipping alternatives, the frozen schema sample and KV label-review
overlay. Images and original sidecars come from
[the corpus dataset](https://huggingface.co/datasets/ilsilfverskiold/ocr-benchmark);
the downloader verifies manifest hashes and refuses conflicting overwrites.
See [sources and licences](docs/documents-and-licensing.md).

Add engines through `runners/base.py` and `engines.py`, declaring actual native or
prompted capabilities. Reuse one conversion across supported legs. Never add a
second model to infer missing structure and attribute it to OCR. Unrun integration
recipes, GPU assumptions and unresolved checks are in
[engine integrations](docs/engine-integrations.md). Development scratch,
superseded exports and historical test scripts are not part of the published tree.
