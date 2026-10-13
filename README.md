# OCR Benchmark

Compare OCR engines on text, Markdown, tables, key/value extraction, supplied
schemas and geometry. **93 pages · 16 measured engine configurations.**

## Open the saved report

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_benchmark.py --report current
```

This reads the saved results; no credentials, image downloads or model calls.
To focus or sort: `--legs text`, `--report-sort cer`, `--report-sort cost`, or
`--report-sort latency`. `--report-details` adds document-family breakdowns.

The existing browser report is **`results/reports/ocr-benchmark-final.html`**;
its linked evidence page shows individual outputs, source pages and box overlays.
On a fresh clone, build those normal reports with:

```bash
python scripts/download_data.py                 # source images, once
python scripts/build_evidence_report.py current
```

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

The two rankings below use the **same 82 text-scored pages per engine**, with the
actual pass used for text. Costs are recorded USD/page, excluding startup,
reference/judge work and failed calls. Local CPU cost is unpriced (shown as $0).
Modal seconds/page are batch-amortized; API/local timings are per-call. Additional
schema, KV and geometry passes have their own cost/latency in the full report.

### Cost — lowest first

| Rank | Engine | USD/page ↓ | Seconds/page ↓ |
| --- | --- | --- | --- |
| 1 | `docling` | 0.000000 | 15.60 |
| 2 | `tesseract` | 0.000000 | 2.38 |
| 3 | `glm_ocr` | 0.000540 | 1.76 |
| 4 | `deepseek_ocr` | 0.000749 | 1.08 |
| 5 | `mineru` | 0.000782 | 1.13 |
| 6 | `paddleocr_vl` | 0.000949 | 1.37 |
| 7 | `marker` | 0.001173 | 1.69 |
| 8 | `gemini_flash` | 0.001251 | 4.06 |
| 9 | `textract_text` | 0.001500 | 4.73 |
| 10 | `qwen3_vl` | 0.003734 | 5.38 |
| 11 | `llamaparse` | 0.003750 | 21.88 |
| 12 | `mistral_ocr_4` | 0.004000 | 3.85 |
| 13 | `claude_sonnet_4_6` | 0.015129 | 18.27 |
| 14 | `textract_structured` | 0.065000 | 7.08 |

### Time — fastest first

| Rank | Engine | USD/page ↓ | Seconds/page ↓ |
| --- | --- | --- | --- |
| 1 | `deepseek_ocr` | 0.000749 | 1.08 |
| 2 | `mineru` | 0.000782 | 1.13 |
| 3 | `paddleocr_vl` | 0.000949 | 1.37 |
| 4 | `marker` | 0.001173 | 1.69 |
| 5 | `glm_ocr` | 0.000540 | 1.76 |
| 6 | `tesseract` | 0.000000 | 2.38 |
| 7 | `mistral_ocr_4` | 0.004000 | 3.85 |
| 8 | `gemini_flash` | 0.001251 | 4.06 |
| 9 | `textract_text` | 0.001500 | 4.73 |
| 10 | `qwen3_vl` | 0.003734 | 5.38 |
| 11 | `textract_structured` | 0.065000 | 7.08 |
| 12 | `docling` | 0.000000 | 15.60 |
| 13 | `claude_sonnet_4_6` | 0.015129 | 18.27 |
| 14 | `llamaparse` | 0.003750 | 21.88 |

**Geometry-only configurations** (17 pages each; cost sorted):

| Engine | USD/page ↓ | Seconds/page ↓ |
| --- | --- | --- |
| `paddleocr_ppocrv5` | 0.000268 | 0.88 |
| `mineru_pipeline` | 0.000430 | 2.62 |

## Engines and versions tested

These are **recorded run versions**, not inferred from today's package releases.
Aliases such as `latest` remain unpinned where the capture did not record an
immutable version. Exact prompts, task options and responses are in
`results/current.zip` and the evidence report.

| Engine ID | Recorded model/version | Configuration |
| --- | --- | --- |
| `claude_sonnet_4_6` | `claude-sonnet-4-6` | Anthropic API; dedicated schema/KV/Markdown and pixel-frame geometry passes |
| `deepseek_ocr` | `deepseek-ai/DeepSeek-OCR` (v1) | vLLM, Modal A100-80GB |
| `docling` | `docling-latest` (exact release unrecorded) | Local CPU document pipeline |
| `gemini_flash` | `gemini-3.1-flash-lite` | Google API |
| `glm_ocr` | `zai-org/GLM-OCR` | vLLM, Modal A10 |
| `llamaparse` | `llamaparse-latest`; `llama-extract` for schema | Hosted Parse/Extract APIs |
| `marker` | `marker-pdf==1.8.2` | Modal A100-80GB; 11 conversion workers; separate OCR geometry pass |
| `mineru` | `MinerU2.5-2509-1.2B` | vLLM native batch, Modal A100-80GB |
| `mineru_pipeline` | `mineru[pipeline]==2.5.4`; Torch `2.6.0+cu124`; weights `ed6b654c018d` | DocLayout-YOLO + PP-OCR + UniMERNet/table models, Modal T4; geometry only |
| `mistral_ocr_4` | `mistral-ocr-latest` conversion; `mistral-ocr-4-0` schema | Mistral OCR/Annotations API |
| `paddleocr_ppocrv5` | PP-OCRv5 server det `ca867c89` / rec `b26c3587`; PaddleOCR `3.7.0`; Paddle GPU `3.2.1` | Modal A10; geometry only |
| `paddleocr_vl` | `PaddleOCR-VL@7fa00a8c55b7`, pipeline v1 | Layout + vLLM recognition, Modal A100-80GB; not VL 1.6 |
| `qwen3_vl` | `Qwen/Qwen3-VL-8B-Instruct` | vLLM, Modal A100-80GB |
| `tesseract` | `5.5.2` | Local CPU |
| `textract_structured` | `aws-textract-analyze-document` | AWS FORMS/TABLES; no supplied-schema interface |
| `textract_text` | `aws-textract-detect-document-text` | AWS text/word detection |

**Available but unmeasured:** TeleOCR, RapidOCR, OvisOCR2, PaddleOCR-VL 1.6,
DeepSeek-OCR-2, Surya OCR 2, MinerU 4 Standard, Mistral OCR 4.1, Google Document
OCR, Mathpix, OpenDoc, LightOnOCR 2, Apple Vision, Chandra OCR 2, dots.mocr and
Datalab. Their IDs, model pins, task routes, initial GPU choices and unresolved
runtime checks are in [the engine guide](docs/engine-integrations.md).

**Comparison limits:** historical default table/Markdown prompts vary; the saved
Markdown coverage was completed with dedicated calls where needed. Sonnet's 17
geometry calls use the updated pixel prompt. Output-cap stops and provider errors
remain recorded. Reference omissions and different output granularity can affect
scores; inspect same-page evidence before attributing a loss to recognition.

## Run the benchmark yourself

After the installation above:

```bash
cp .env.example .env
# Add credentials only for your chosen engines.
python scripts/download_data.py        # source images; preserves committed GT

# Preview the selected experiment without calling engines.
python run_benchmark.py --engines mistral_ocr_4 --legs text,markdown,tables --dry-run

# Run those engines; this can incur provider/Modal charges.
python run_benchmark.py --engines mistral_ocr_4 --legs text,markdown,tables --skip-judge

# Use the timestamp printed by the run.
python run_benchmark.py --report RUN_TIMESTAMP
python scripts/build_evidence_report.py RUN_TIMESTAMP
```

`--skip-judge` prevents new evaluation judge calls; OCR inference still runs.
For KV with new wording, cached decisions are reused and uncached reviews remain
pending. Omit the flag to enable needed contextual judging; set
`OCR_KV_JUDGE_BUDGET_USD` to your allowance.

### Choose engines and pages

```bash
# Multiple engines, selected difficulty and legs.
python run_benchmark.py --engines tesseract,docling --tiers easy --legs text,tables --skip-judge

# One specific page; inspect the plan first.
python run_benchmark.py --engines claude_sonnet_4_6,gemini_flash --docs invoice_000 --legs schema --dry-run

# Collect outputs first; evaluate them separately.
python run_benchmark.py --engines rapidocr --collect-only
python run_benchmark.py --score RUN_TIMESTAMP --skip-judge
```

Use comma-separated engine IDs from `engines.py` or the engine guide. Available
legs: `text,markdown,tables,key_value,schema,geometry`. Omitted engine/leg flags
use the registry defaults. Tesseract needs its system binary; Apple Vision needs
macOS. Cloud engines use the credentials listed in `.env.example`.

For a Modal engine, deploy its app before running it:

```bash
modal setup
python scripts/setup_modal.py
python scripts/deploy_modal.py --apps rapidocr --skip-smoke
```

Deployments and smoke inference can incur charges. `--skip-smoke` skips inference,
not the build. For engine configuration and GPU requirements, use
[the engine guide](docs/engine-integrations.md); for scoring details, use
[the scoring design](docs/scoring-design.md).
