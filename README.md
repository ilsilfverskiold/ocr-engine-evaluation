# OCR Benchmark

Compare OCR engines on text, Markdown, tables, key/value extraction, supplied
schemas and geometry. **93 pages · 16 measured engine configurations.**

## View the saved results

From the repository directory, install the dependencies and display the published
comparison in your terminal:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_benchmark.py --report current
```

`current` selects the published result bundle, `results/current.zip`. This only
reads saved scores; no credentials, image downloads or model calls are needed.
To focus or sort: `--legs text`, `--report-sort cer`, `--report-sort cost`, or
`--report-sort latency`. `--report-details` adds document-family breakdowns.

The existing browser report is **`results/reports/ocr-benchmark-final.html`**;
its linked evidence page shows individual outputs, source pages and box overlays.
On a fresh clone, generate the HTML comparison and its evidence page with:

```bash
python scripts/download_data.py                 # source images, once
python scripts/build_evidence_report.py current
```

Open `results/reports/ocr-benchmark-final.html` afterwards. Images are downloaded
only to display source pages in the evidence report; the references are already
in the repository. These commands do not rerun OCR or judging.

To collect your own results, use [Run a new benchmark](#run-a-new-benchmark).

## What each leg measures

A leg is one task in the comparison. Engines enter the tasks their native outputs
or supported prompts can provide; an unsupported task is N/A. Scores are compared
within each leg, rather than combined into one overall engine score.

| Leg / CLI name | What the engine must return | How it is evaluated |
| --- | --- | --- |
| Text · `text` | Transcribed page text | Reference-token coverage, plus character error rate (CER) and word error rate (WER). Coverage measures how much reference text was recovered; CER/WER also count added, missing and changed text. Lower error rates are better. No judge on pages with reliable text references. |
| Markdown · `markdown` | Text with Markdown or HTML structure | Locate reference content, then check headings, paragraphs, lists and tables use the right structure. Bold alone is not a heading. Correct elements earn 1, partial elements 0.5. CER/WER measure text fidelity separately. Code scores against shared cached model-authored checklists. |
| Tables · `tables` | Native tables, HTML tables or Markdown tables | Value credit divided by reference cells, using reviewed gold tables. Partial text and valid table reflow can earn credit; wrong numeric values earn none. Blanks count. Extras are reported separately, alongside strict cell precision/recall/F1 and structure diagnostics. Shipping uses whichever whole-page reference scores higher: gold or original. |
| Key/value · `key_value` | Field names and their values, without a supplied schema | Match printed fields against the accepted structured references. Exact keys and cached mappings are checked first; a contextual judge can resolve wording differences and split values when needed, caching confirmed mappings. Unresolved reviews remain pending. |
| Schema · `schema` | JSON following the supplied keys and types | Check the requested keys and values against the accepted structured references, with penalties for extra or duplicate fields. Engines receive the same selected fields on each page. No judge. |
| Geometry · `geometry` | Text paired with bounding boxes | Check text matches and box placement against reference words or lines. Headline is located-reference recall; precision/F1 expose extra boxes. CER/WER measure the text returned with those boxes. No judge. |

**Judged text is shown separately:** 11 pages lack reliable text references, so
their saved text results use model judgments instead of CER/WER. These estimates
are not mixed into reference-based text accuracy. Their historical scoring
limitation is described [with the results](#judged-text).

The tables below show overall and difficulty-tier results for each leg. See
[Scoring rules](docs/scoring-design.md) for matching, reference selection and
aggregation details.

## Results

Saved run: **`20261013T081846Z`**. Rankings use the saved outputs and current scoring rules.
Each leg is sorted by overall score, highest first. Overall means give each
document family equal weight. **Easy/medium/hard are difficulty tiers.** “Pages”
counts successful scores; errors are listed separately. **—** means no score,
not zero. CER/WER are error rates: lower is better and values can exceed 1.

The “Without low” column appears where the common low-scoring-page exclusion
changes results: `tax_001` for tables and `chart_003` for KV. These are sensitivity
checks, not replacement rankings. [Scoring rules](docs/scoring-design.md).

### Text accuracy

Distinct normalized reference-token coverage. CER/WER also measure omissions,
insertions and ordering; coverage alone ignores extra text. No judge.

| Engine | Overall ↑ | Easy | Medium | Hard | Pages | CER ↓ | WER ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `claude_sonnet_4_6` | 0.9634 | 0.9664 | 0.9610 | 0.9649 | 82 | 0.1267 | 0.1799 |
| `gemini_flash` | 0.9508 | 0.9690 | 0.9489 | 0.9474 | 82 | 0.1588 | 0.2375 |
| `textract_text` | 0.9471 | 0.9625 | 0.9513 | 0.9385 | 82 | 0.1476 | 0.2094 |
| `textract_structured` | 0.9465 | 0.9625 | 0.9513 | 0.9370 | 82 | 0.1478 | 0.2102 |
| `mistral_ocr_4` | 0.9231 | 0.9560 | 0.9554 | 0.8813 | 82 | 0.1355 | 0.2034 |
| `qwen3_vl` | 0.9077 | 0.9536 | 0.8856 | 0.9168 | 82 | 0.5730 | 0.7905 |
| `glm_ocr` | 0.9039 | 0.9459 | 0.9197 | 0.8761 | 82 | 0.1698 | 0.2407 |
| `mineru` | 0.8613 | 0.9133 | 0.8926 | 0.8152 | 82 | 0.2122 | 0.2826 |
| `llamaparse` | 0.8584 | 0.9145 | 0.8294 | 0.8714 | 82 | 0.1855 | 0.2826 |
| `marker` | 0.8563 | 0.8868 | 0.8673 | 0.8365 | 82 | 0.2490 | 0.3260 |
| `paddleocr_vl` | 0.8525 | 0.9386 | 0.8057 | 0.8746 | 82 | 0.2490 | 0.2499 |
| `docling` | 0.8248 | 0.9092 | 0.8814 | 0.7441 | 82 | 0.3166 | 0.4273 |
| `deepseek_ocr` | 0.8243 | 0.8910 | 0.7837 | 0.8458 | 82 | 0.2319 | 0.3298 |
| `tesseract` | 0.7391 | 0.8682 | 0.7539 | 0.6874 | 82 | 0.3532 | 0.5126 |

### Judged text

Separate image/checklist estimates for 11 hard pages without reliable text GT.
**Historical verdicts:** the old substring shortcut may inflate these scores.
They are not combined with text accuracy or CER/WER. Engines without saved
judge verdicts are omitted from this table.

| Engine | Overall ↑ | Hard | Pages |
| --- | --- | --- | --- |
| `gemini_flash` | 0.8691 | 0.8691 | 11 |
| `mistral_ocr_4` | 0.8421 | 0.8421 | 11 |
| `claude_sonnet_4_6` | 0.8272 | 0.8272 | 11 |
| `textract_text` | 0.7878 | 0.7878 | 11 |
| `textract_structured` | 0.7799 | 0.7799 | 11 |
| `mineru` | 0.6281 | 0.6281 | 11 |
| `llamaparse` | 0.6253 | 0.6253 | 11 |
| `paddleocr_vl` | 0.5245 | 0.5245 | 11 |
| `docling` | 0.5151 | 0.5151 | 11 |
| `tesseract` | 0.2853 | 0.2853 | 11 |

### Markdown structure

Correct structural elements plus half credit for partial matches. Bold is not a
heading. Cached model-authored checklists are shared across engines; code scores
the answers. CER/WER below measure rendered text separately.

| Engine | Overall ↑ | Easy | Medium | Hard | Pages | CER ↓ | WER ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mistral_ocr_4` | 0.7634 | 0.8313 | 0.6746 | 0.8242 | 92 | 0.1355 | 0.2034 |
| `qwen3_vl` | 0.7566 | 0.8913 | 0.7464 | 0.7319 | 92 | 0.2410 | 0.3122 |
| `claude_sonnet_4_6` | 0.7062 | 0.8243 | 0.6440 | 0.7311 | 92 | 0.1267 | 0.1799 |
| `mineru` | 0.6972 | 0.8225 | 0.6624 | 0.6964 | 92 | 0.2136 | 0.2836 |
| `llamaparse` | 0.6911 | 0.8441 | 0.6663 | 0.6747 | 92 | 0.1855 | 0.2826 |
| `gemini_flash` | 0.6781 | 0.8115 | 0.5792 | 0.7314 | 92 | 0.1588 | 0.2375 |
| `docling` | 0.6520 | 0.8434 | 0.6406 | 0.6141 | 92 | 0.3166 | 0.4273 |
| `marker` | 0.6398 | 0.7972 | 0.5581 | 0.6718 | 92 | 0.2490 | 0.3260 |
| `deepseek_ocr` | 0.6391 | 0.8065 | 0.5767 | 0.6519 | 92 | 0.2941 | 0.3798 |
| `paddleocr_vl` | 0.6362 | 0.8032 | 0.5512 | 0.6689 | 92 | 0.2490 | 0.2499 |
| `glm_ocr` | 0.5173 | 0.3794 | 0.4646 | 0.5979 | 92 | 0.3209 | 0.3753 |

### Tables

Reference-cell value credit against accepted gold, including blanks and partial
text matches. Logical row/column associations matter; compatible reflow is allowed.
Extras are separate diagnostics. Shipping uses the better whole-page gold/original
reference score for every engine.

| Engine | Overall ↑ | Without low ↑ | Easy | Medium | Hard | Pages | Strict F1 ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `textract_structured` | 0.7896 | 0.7872 | 0.7883 | 0.8064 | 0.7409 | 42 | 0.5729 |
| `claude_sonnet_4_6` | 0.7552 | 0.7587 | 0.9299 | 0.6681 | 0.8422 | 42 | 0.6505 |
| `qwen3_vl` | 0.7368 | 0.7420 | 0.7459 | 0.8036 | 0.5273 | 42 | 0.6261 |
| `mistral_ocr_4` | 0.7301 | 0.7324 | 0.7309 | 0.6718 | 0.9042 | 42 | 0.5886 |
| `gemini_flash` | 0.7233 | 0.7212 | 0.8156 | 0.7216 | 0.6359 | 42 | 0.6384 |
| `paddleocr_vl` | 0.7204 | 0.7205 | 0.6781 | 0.7120 | 0.7879 | 42 | 0.5950 |
| `llamaparse` | 0.6449 | 0.6503 | 0.7793 | 0.6936 | 0.3643 | 42 | 0.5480 |
| `mineru` | 0.6304 | 0.6300 | 0.6107 | 0.6080 | 0.7175 | 42 | 0.5345 |
| `glm_ocr` | 0.6288 | 0.6283 | 0.6416 | 0.6264 | 0.6232 | 42 | 0.5563 |
| `deepseek_ocr` | 0.6118 | 0.6139 | 0.7792 | 0.5847 | 0.5256 | 42 | 0.5288 |
| `docling` | 0.5472 | 0.5472 | 0.5251 | 0.5798 | 0.4713 | 42 | 0.4967 |
| `marker` | 0.5401 | 0.5396 | 0.5276 | 0.5098 | 0.6436 | 42 | 0.3719 |

### Key/value extraction

Correct values under eligible printed labels. Exact/cached mappings come first;
ambiguous wording or split values use contextual judge decisions, cached for reuse.
Missing decisions stay pending. Blank-inclusive headline; extras are diagnostic.

| Engine | Overall ↑ | Without low ↑ | Easy | Medium | Hard | Pages |
| --- | --- | --- | --- | --- | --- | --- |
| `textract_structured` | 0.7829 | 0.7840 | 0.8502 | 0.7943 | 0.7291 | 62 |
| `claude_sonnet_4_6` | 0.7752 | 0.7902 | 0.7864 | 0.7960 | 0.7332 | 65 |
| `qwen3_vl` | 0.6196 | 0.6335 | 0.6669 | 0.5498 | 0.7182 | 65 |
| `gemini_flash` | 0.5086 | 0.5185 | 0.4923 | 0.4320 | 0.6507 | 63 |

### Supplied-schema extraction

Correct values under exact requested keys, penalizing extra/duplicate keys. Every
engine receives the same frozen sample: 3,316 fields across 65 pages. No judge.

| Engine | Overall ↑ | Easy | Medium | Hard | Pages | Errors |
| --- | --- | --- | --- | --- | --- | --- |
| `gemini_flash` | 0.9679 | 0.9843 | 0.9489 | 0.9930 | 64 | 1 |
| `claude_sonnet_4_6` | 0.9597 | 0.9858 | 0.9409 | 0.9796 | 65 | 0 |
| `llamaparse` | 0.8998 | 0.9398 | 0.9162 | 0.8512 | 65 | 0 |
| `mistral_ocr_4` | 0.8635 | 0.9371 | 0.8472 | 0.8554 | 65 | 0 |
| `qwen3_vl` | 0.8591 | 0.9140 | 0.8254 | 0.8906 | 65 | 0 |
| `glm_ocr` | 0.6660 | 0.7883 | 0.6342 | 0.6607 | 65 | 0 |
| `deepseek_ocr` | 0.1604 | 0.3236 | 0.1429 | 0.1094 | 64 | 1 |

### Geometry

Located reference-unit recall: text matches and each box contains the other’s
centre. Receipts use lines; FUNSD/financial tables use words. Coarse boxes remain
coarse. Precision/F1 penalize extra boxes; CER/WER use the returned box text.

| Engine | Overall ↑ | Easy | Medium | Hard | Pages | F1 ↑ | CER ↓ | WER ↓ | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `textract_text` | 0.9476 | 0.9936 | — | 0.9246 | 17 | 0.9369 | 0.0388 | 0.1358 | 0 |
| `textract_structured` | 0.9474 | 0.9936 | — | 0.9243 | 17 | 0.9370 | 0.0387 | 0.1355 | 0 |
| `paddleocr_ppocrv5` | 0.8507 | 0.9601 | — | 0.7959 | 17 | 0.7776 | 0.1046 | 0.7023 | 0 |
| `tesseract` | 0.7956 | 0.8286 | — | 0.7791 | 17 | 0.8166 | 0.1671 | 0.3582 | 0 |
| `gemini_flash` | 0.5884 | 0.9020 | — | 0.4315 | 17 | 0.5944 | 0.1534 | 0.2532 | 0 |
| `marker` | 0.2770 | 0.5341 | — | 0.0198 | 9 | 0.3194 | 0.5563 | 0.6088 | 8 |
| `claude_sonnet_4_6` | 0.1525 | 0.0081 | — | 0.2248 | 17 | 0.1503 | 0.1186 | 0.2064 | 0 |
| `mineru_pipeline` | 0.1132 | 0.3059 | — | 0.0169 | 17 | 0.1452 | 0.2088 | 0.3155 | 0 |
| `mineru` | 0.1130 | 0.3078 | — | 0.0157 | 17 | 0.1473 | 0.1478 | 0.2848 | 0 |
| `deepseek_ocr` | 0.0806 | 0.2262 | — | 0.0077 | 17 | 0.1161 | 0.0916 | 0.1825 | 0 |
| `paddleocr_vl` | 0.0564 | 0.1388 | — | 0.0152 | 17 | 0.0820 | 0.0813 | 0.1764 | 0 |
| `qwen3_vl` | 0.0052 | 0.0142 | — | 0.0007 | 17 | 0.0057 | 0.5342 | 0.6834 | 0 |

`mineru_pipeline` mixes line and whole-table boxes. `deepseek_ocr`, `mineru` and
`paddleocr_vl` return layout blocks; their low word-location scores are shown.

## Documents

Single-page images from public datasets. Ground-truth sidecars and accepted
references are committed under `documents/`; only images are downloaded.
Text references: **63 human + 19 model-verified** pages. The remaining **11** use
separate judged-text estimates. Counts below show reference coverage.

| Tier | Document family | Pages | Text | Judged | Markdown | Tables | KV | Schema | Geometry |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| easy | invoice | 6 | 6 | — | 6 | 6 | 6 | 6 | — |
| easy | receipt | 6 | 6 | — | 6 | 5 | 6 | 6 | 6 |
| medium | bank_statement | 5 | 5 | — | 5 | 5 | 5 | 5 | — |
| medium | lease | 5 | 5 | — | 5 | — | 5 | 5 | — |
| medium | medical | 5 | 5 | — | 5 | 1 | 5 | 5 | — |
| medium | payslip | 5 | 5 | — | 5 | 5 | 5 | 5 | — |
| medium | photo_receipt | 5 | 5 | — | 5 | 2 | 5 | 5 | — |
| medium | shipping | 5 | 5 | — | 5 | 5 | 5 | 5 | — |
| medium | tax_form | 5 | 5 | — | 5 | 5 | 5 | 5 | — |
| hard | chart | 5 | 5 | — | 5 | — | 5 | 5 | — |
| hard | financial_table | 5 | 5 | — | 5 | 5 | 5 | 5 | 5 |
| hard | form | 6 | 6 | — | 6 | 3 | 6 | 6 | 6 |
| hard | handwritten | 6 | 6 | — | 6 | — | — | — | — |
| hard | legal | 6 | 6 | — | 6 | — | — | — | — |
| hard | newspaper | 6 | — | 6 | 5 | — | — | — | — |
| hard | report | 6 | 1 | 5 | 6 | — | — | — | — |
| hard | scanned_legacy | 6 | 6 | — | 6 | — | 2 | 2 | — |

Source IDs, image hashes and selection are in `documents/manifest.json`.
[Dataset attribution and licences](docs/documents-and-licensing.md).

## Cost and latency

The rankings use the **same 82 text-scored pages per engine** and the actual pass
that supplied each text result. Prices are **estimated usage charges, not verified
invoices**. Modal figures cover GPU processing only; local CPU expense is unpriced.
Startup, idle time, storage, evaluation judges, failed calls, retries, taxes and
subscriptions are excluded. Modal seconds/page are batch-amortized throughput;
API/local timings are per-call.

### Cost and volume estimates — lowest first

Volume estimates multiply the **unrounded mean cost/page** by the page count,
assuming the same document mix, token lengths, pipeline and GPU batch throughput.
They describe one text-producing pass per page, not every benchmark leg. Dollar
amounts are before free credits, caching and batch-API discounts. GPU-only amounts
are lower bounds on hosting expense; local engines have no priced estimate.

| Rank | Engine | USD/page ↓ | 1,000 pages | 100,000 pages | 1 million pages | Seconds/page |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `glm_ocr` | 0.000540 | $0.54 | $53.97 | $539.74 | 1.76 |
| 2 | `deepseek_ocr` | 0.000749 | $0.75 | $74.92 | $749.23 | 1.08 |
| 3 | `mineru` | 0.000782 | $0.78 | $78.23 | $782.33 | 1.13 |
| 4 | `paddleocr_vl` | 0.000949 | $0.95 | $94.89 | $948.93 | 1.37 |
| 5 | `marker` | 0.001173 | $1.17 | $117.27 | $1,172.69 | 1.69 |
| 6 | `gemini_flash` | 0.001251 | $1.25 | $125.13 | $1,251.28 | 4.06 |
| 7 | `textract_text` | 0.001500 | $1.50 | $150.00 | $1,500.00 | 4.73 |
| 8 | `qwen3_vl` | 0.003734 | $3.73 | $373.45 | $3,734.50 | 5.38 |
| 9 | `llamaparse` | 0.003750 | $3.75 | $375.00 | $3,750.00 | 21.88 |
| 10 | `mistral_ocr_4` | 0.004000 | $4.00 | $400.00 | $4,000.00 | 3.85 |
| 11 | `claude_sonnet_4_6` | 0.015129 | $15.13 | $1,512.91 | $15,129.11 | 18.27 |
| 12 | `textract_structured` | 0.065000 | $65.00 | $6,500.00 | $65,000.00 | 7.08 |
| — | `docling` | Unpriced | — | — | — | 15.60 |
| — | `tesseract` | Unpriced | — | — | — | 2.38 |

### Time — fastest first

| Rank | Engine | USD/page ↓ | Seconds/page ↓ |
| --- | --- | --- | --- |
| 1 | `deepseek_ocr` | 0.000749 | 1.08 |
| 2 | `mineru` | 0.000782 | 1.13 |
| 3 | `paddleocr_vl` | 0.000949 | 1.37 |
| 4 | `marker` | 0.001173 | 1.69 |
| 5 | `glm_ocr` | 0.000540 | 1.76 |
| 6 | `tesseract` | Unpriced | 2.38 |
| 7 | `mistral_ocr_4` | 0.004000 | 3.85 |
| 8 | `gemini_flash` | 0.001251 | 4.06 |
| 9 | `textract_text` | 0.001500 | 4.73 |
| 10 | `qwen3_vl` | 0.003734 | 5.38 |
| 11 | `textract_structured` | 0.065000 | 7.08 |
| 12 | `docling` | Unpriced | 15.60 |
| 13 | `claude_sonnet_4_6` | 0.015129 | 18.27 |
| 14 | `llamaparse` | 0.003750 | 21.88 |

**Geometry-only configurations** (17 pages each; GPU-only cost, lowest first):

| Engine | USD/page | 1,000 pages | 100,000 pages | 1 million pages | Seconds/page |
| --- | --- | --- | --- | --- | --- |
| `paddleocr_ppocrv5` | 0.000268 | $0.27 | $26.82 | $268.18 | 0.88 |
| `mineru_pipeline` | 0.000430 | $0.43 | $42.98 | $429.84 | 2.62 |

### Pricing audit — checked October 13, 2026

| Tested service / pipeline | Rate used and audit finding |
| --- | --- |
| Claude Sonnet 4.6, direct Anthropic API | $3 / million input tokens + $15 / million output tokens. Saved usage reproduces the recorded costs; image tokens are included. [Pricing](https://platform.claude.com/docs/en/about-claude/pricing). |
| Gemini 3.1 Flash-Lite, Gemini API | $0.25 / million input tokens + $1.50 / million output tokens, including thinking. Saved usage reproduces the recorded costs. [Pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.1-flash-lite). |
| Mistral OCR 4.1 conversion / 4.0 annotations | $4 / 1,000 OCR pages; $5 / 1,000 annotated pages. Published results already correct the old conversion logger's $0.001/page to $0.004. Schema is a separate annotated call, not the conversion price. [OCR pricing](https://docs.mistral.ai/models/ocr-4-1). |
| LlamaParse v1 Parse page with LLM / LlamaExtract v1 Multimodal | Parse: 3 credits/page × $1.25 / 1,000 credits = $0.00375/page. Published results already correct the old logger's $0.003. Schema: 20 credits = $0.025/page for an uncached file; cached extraction can be 14 credits = $0.0175. These are list-price estimates, not captured credit charges. [Mode-specific pricing](https://developers.llamaindex.ai/llamaparse/general/pricing/). |
| Textract DetectDocumentText / AnalyzeDocument TABLES + FORMS | $0.0015/page for text; $0.015 tables + $0.05 forms = $0.065/page, with OCR included. These match the first-million-page US rates, including the template's `us-east-1`; the old calls did not persist their region. The runner's fixed rate must be checked if using another region. [Pricing](https://aws.amazon.com/textract/pricing/), [US East price list](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonTextract/current/us-east-1/index.json). |
| Modal model pipelines | GPU rate × measured batch seconds / attempted pages. A100-80GB: $0.000694/s for DeepSeek, MinerU VLM, PaddleOCR-VL, Marker and Qwen; A10: $0.000306/s for GLM and PP-OCRv5; T4: $0.000164/s for MinerU pipeline. Saved calculations match these rates. **CPU/RAM and container overhead are missing from these costs.** [Resource pricing](https://modal.com/pricing). |
| Local Docling / Tesseract | Saved `$0` means no provider charge was assigned. It excludes the machine and electricity; the tables now show **Unpriced**. |

### What changes at scale

**GPU hosting:** add CPU and RAM charges, model loading and billed container idle
time to the GPU processing estimates. A useful estimate is
`(processing seconds + startup seconds + idle seconds) × container rate`;
CPU/RAM usage was not recorded, so a complete historical bill cannot be reconstructed.
As an illustration, a 190.8-second DeepSeek startup plus the configured 30-second
shutdown delay adds about **$0.153 in GPU charges per container session**: $0.00153
per page over 100 pages, or $0.0000153 over 10,000. Multiple containers repeat that
cost. Sustained batches spread it out; isolated requests do not. Modal also bills
CPU/RAM at the higher of requested or used resources. [Billing rules](https://modal.com/pricing).

**Token-priced engines:** longer outputs and extra task prompts change cost.
Async batch endpoints offer 50% lower token prices for
[Sonnet](https://platform.claude.com/docs/en/about-claude/pricing#batch-processing)
and [Gemini](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.1-flash-lite),
and [Mistral's OCR Batch API](https://docs.mistral.ai/resources/cookbooks/mistral-ocr-batch_ocr)
offers 50% off. The benchmark uses standard API calls, so these savings are
**not applied** above; running calls concurrently does not earn a batch discount.
For the same token mix, 1 million pages would be approximately $7,564.55 for
Sonnet or $625.64 for Gemini via their batch APIs; Mistral conversion would be $2,000.
These are price projections, not tested batch runs.

**Volume tiers:** Textract's next pages after the first million per month cost
$0.0006 for text or $0.05 for Forms + Tables at the US rates used here. Thus
2 million pages would cost $2,100 or $115,000 respectively, rather than doubling
the first-million price. [AWS volume examples](https://aws.amazon.com/textract/pricing/).

**Multiple legs:** count unique inference passes, not score rows. One conversion
can supply text, Markdown and tables at no additional inference charge. Dedicated
Markdown, schema, KV or geometry calls cost extra. For example, Mistral conversion
plus a separate annotated extraction is $9 / 1,000 pages; LlamaParse plus uncached
v1 Multimodal extraction is $28.75 / 1,000. Cached vendor work can reduce charges.
Evaluation judge calls are a separate benchmark expense, not a production OCR price.

## Engines and versions tested

Versions below come from saved replies, with source pins and the dated Mistral
alias resolution identified separately. Exact prompts, task options and replies
are in `results/current.zip` and the evidence report. An engine ID is a stable
result identifier, not proof of its model version.

| Engine ID | Recorded model/version | Configuration |
| --- | --- | --- |
| `claude_sonnet_4_6` | `claude-sonnet-4-6` | Anthropic API; dedicated schema/KV/Markdown and pixel-frame geometry passes |
| `deepseek_ocr` | `deepseek-ai/DeepSeek-OCR` (v1); source pin `9f30c71f441d` | vLLM, Modal A100-80GB |
| `docling` | `2.118.1` (reconstructed environment; calls saved `docling-latest`) | Standard pipeline: Heron layout + TableFormer Accurate + RapidOCR/PP-OCRv6; see provenance below |
| `gemini_flash` | `gemini-3.1-flash-lite` | Google API |
| `glm_ocr` | `zai-org/GLM-OCR`; source pin `ca5d8b3e287e` | vLLM, Modal A10 |
| `llamaparse` | v1 Balanced / Parse with LLM, August 10 run (dated release unconfirmed); schema `openai-gpt-4-1` | Hosted Parse/Extract APIs; recovered client packages `llama-cloud-services==0.6.94`, `llama-cloud==0.1.46` |
| `marker` | `marker-pdf==1.8.2` | Modal A100-80GB; 11 conversion workers; separate OCR geometry pass |
| `mineru` | `MinerU2.5-2509-1.2B`; source pin `1aa090b41282` | vLLM native batch, Modal A100-80GB |
| `mineru_pipeline` | `mineru[pipeline]==2.5.4`; Torch `2.6.0+cu124`; weights `ed6b654c018d` | DocLayout-YOLO + PP-OCR + UniMERNet/table models, Modal T4; geometry only |
| `mistral_ocr_4` | **4.1 conversion** (dated alias resolution); **4.0 schema** (saved reply) | Mistral OCR/Annotations API; mixed versions, see below |
| `paddleocr_ppocrv5` | PP-OCRv5 server det `ca867c89` / rec `b26c3587`; PaddleOCR `3.7.0`; Paddle GPU `3.2.1` | Modal A10; geometry only |
| `paddleocr_vl` | `PaddleOCR-VL@7fa00a8c55b7`, pipeline v1 | Layout + vLLM recognition, Modal A100-80GB |
| `qwen3_vl` | `Qwen/Qwen3-VL-8B-Instruct`; source pin `0c351dd01ed8` | vLLM, Modal A100-80GB |
| `tesseract` | `5.5.2` | Local CPU |
| `textract_structured` | `AnalyzeDocumentModelVersion: 1.0` | AWS FORMS/TABLES; provider-reported version; no supplied-schema interface |
| `textract_text` | `DetectDocumentTextModelVersion: 1.0` | AWS text/word detection; provider-reported version |

**Mistral:** all 93 conversion replies from `20260810T130609Z` saved
`mistral-ocr-latest`. Mistral's [changelog](https://docs.mistral.ai/resources/changelogs)
states that this alias switched to `mistral-ocr-4-1` on July 16, 2026. Thus the
conversion results (text, judged text, Markdown and tables) correspond to 4.1
according to the published alias history, not an immutable ID in those replies.
All 65 schema replies from `20261006T205850Z` explicitly report
`mistral-ocr-4-0`. The runner pins conversion to 4.1 and schema to 4.0 separately;
it does not silently substitute a model. Mistral now lists 4.0 as retired, so
reproducing that schema run may no longer be possible through the hosted API.

**Source pins:** the four revisions marked above already appear in the initial
benchmark commit `e649e5a`; full hashes are in the corresponding
[Modal apps](modal_apps). Their old replies did not record the deployed hash,
so these establish the configured weights rather than independent per-call
verification. New replies include the revision.

<details>
<summary>Recovered Docling and LlamaParse version evidence</summary>

**Docling:** [2.118.1 was released August 7, 2026](https://github.com/docling-project/docling/releases/tag/v2.118.1).
The retained environment's package/source timestamps are August 9, before the
August 10 run, and the October 2 environment inspection recorded the same
versions: Docling `2.118.1`, docling-core `2.91.0`, docling-ibm-models `3.13.3`,
docling-parse `7.11.0`, RapidOCR `3.9.2`. This reconstructs the likely run
environment; the original per-call records do not independently verify it.

The runner used `DocumentConverter()` defaults. In that environment these select
Heron (RT-DETR/ResNet50) for layout and TableFormer **Accurate** for tables.
The model-cache refs were written during the run, August 10 at 13:13 UTC:
`docling-project/docling-layout-heron@8f39ad3c0b4c58e9c2d2c84a38465abf757272d8`
and `docling-project/docling-models@fc0f2d45e2218ea24bce5045f58a389aed16dc23`
(TableFormer release `v2.3.0`). Auto-OCR resolves to RapidOCR with Torch in the
retained environment: ocrmac, ONNX Runtime and EasyOCR are absent. Its cached
August 9 checkpoints are `PP-OCRv6_det_small.pth`, `PP-OCRv6_rec_small.pth` and
`ch_ptocr_mobile_v2.0_cls_mobile.pth`. These identify a multi-model document
pipeline, not a single Docling language model. New calls record the installed
Docling release.

**LlamaParse:** the retained client packages have August 9 installation
timestamps, but SDK versions do not identify the hosted parser model. The
original runner selected Markdown output without a tier, parser version or
vendor model. The [legacy mode documentation](https://developers.llamaindex.ai/llamaparse/parse/v1/presets_and_modes/advance_parsing_modes/)
identifies this default as **Parse with LLM / Balanced** (`parse_page_with_llm`).
The [published parser version list](https://developers.llamaindex.ai/reference/resources/parsing/methods/list_versions/)
places Cost Effective `2026-08-08` immediately before our August 10 run, with
the next release on August 11. Thus **`2026-08-08` is a date-based candidate
if legacy Balanced followed v2 Cost Effective's latest release**. The inspected
changelogs and migration documentation do not establish that mapping, so this
is not a verified version pin. Other tiers had different release dates:
Agentic `2026-07-24`, Agentic Plus `2026-07-08`, Fast `2026-06-15`.
An October 13
read-only lookup of saved job `9b19a059-c95c-49b0-a983-3120f5bb9a97` returned
only its ID and `SUCCESS`; the v2 lookup did not expose that legacy job.
The underlying parsing model remains unresolved and needs provider confirmation
against the saved job IDs. Separately, all 65 saved **LlamaExtract schema**
replies name `openai-gpt-4-1`; that does not identify the parsing model.

</details>

**Comparison limits:** historical default table/Markdown prompts vary; the saved
Markdown coverage was completed with dedicated calls where needed. Sonnet's 17
geometry calls use the updated pixel prompt. Output-cap stops and provider errors
remain recorded. Reference omissions and different output granularity can affect
scores; inspect same-page evidence before attributing a loss to recognition.

## Run a new benchmark

This workflow sends the source pages to your selected engines and creates a new
run. Hosted APIs and Modal inference can incur charges. Run commands from the
repository directory.

### 1. Install and prepare the data

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
```

The download supplies source images; committed ground-truth references are
preserved. If you already installed the environment and downloaded the images,
activate `.venv` and continue.

### 2. Configure your chosen engines

Create your local credentials file once, then edit it:

```bash
cp -n .env.example .env
```

Add only the credentials required by your chosen engines, as listed in
[.env.example](.env.example). Tesseract also needs its system binary installed.
For Modal engines, authenticate and deploy the selected app first; for example:

```bash
modal setup
python scripts/setup_modal.py
python scripts/deploy_modal.py --apps glm_ocr --skip-smoke
```

`--skip-smoke` skips test inference during deployment. Deployment builds may
still incur charges. Hosted API engines such as Mistral do not need Modal setup.

### 3. Choose engines, tasks and pages

Use comma-separated engine IDs from [Engines and versions tested](#engines-and-versions-tested).
The same selection flags apply to the preview and the actual run:

| Flag | Example | Selection |
| --- | --- | --- |
| `--engines` | `mistral_ocr_4,docling` | Engines to compare. Specify explicitly to avoid running the registry defaults. |
| `--legs` | `text,markdown,tables` | Tasks to evaluate; all supported legs if omitted. See [leg definitions](#what-each-leg-measures). |
| `--tiers` | `easy,medium` | Difficulty tiers; all three if omitted. |
| `--docs` | `invoice_000,sroie_000` | Particular page IDs within the selected tiers; all pages if omitted. |

For example, preview Mistral on text, Markdown and tables without making model calls:

```bash
python run_benchmark.py --engines mistral_ocr_4 --legs text,markdown,tables --skip-judge --dry-run
```

### 4. Run and evaluate

Remove `--dry-run` to execute that experiment:

```bash
python run_benchmark.py --engines mistral_ocr_4 --legs text,markdown,tables --skip-judge
```

This runs OCR, saves the outputs, evaluates them and **prints the engine
comparison automatically**. Keep the run timestamp printed in the terminal.

**Choosing whether to use evaluation judges:** `--skip-judge` disables new judge
calls, not OCR inference. Deterministic scores and existing cached KV decisions
are still used. New KV wording that needs review remains pending, and the 11
pages without reliable text references receive no new judged-text scores.
To include those judgments, omit `--skip-judge` and add `GOOGLE_API_KEY` to `.env`.
`OCR_KV_JUDGE_BUDGET_USD` sets the KV judge planning allowance (default `$2`);
it does not cap OCR or the separate judged-text calls.

### 5. Open the results of your new run

The terminal comparison is already printed. To reopen or sort it later, replace
`RUN_TIMESTAMP` with your run's timestamp:

```bash
python run_benchmark.py --report RUN_TIMESTAMP --report-sort cost
```

To create the browser comparison and supporting evidence for that same run:

```bash
python scripts/build_evidence_report.py RUN_TIMESTAMP
```

Open `results/reports/ocr-benchmark-final.html`. This replaces the local HTML
report with the selected run; it does not change the published `current` bundle.
Both report commands only read saved results and make no model calls.
