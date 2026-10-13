# OCR Benchmark

Compare OCR engines on text, Markdown, tables, key/value extraction, supplied
schemas and geometry. **93 pages· 16 measured engine configurations.**

## Results

Saved run: **`20261013T112225Z`**. Rankings use the saved outputs and current scoring rules.
Each leg is sorted by overall score, highest first. Overall means give each
document family equal weight. **Easy/medium/hard are difficulty tiers.** “Pages”
counts successful scores; errors are listed separately. **—** means no score,
not zero. CER (character error rate) and WER (word error rate) are better when
lower; values can exceed 1.

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
| `claude_sonnet_4_6` | 0.8221 | 0.8854 | 0.7909 | 0.8336 | 92 | 0.1301 | 0.1911 |
| `gemini_flash` | 0.7724 | 0.8601 | 0.7309 | 0.7868 | 92 | 0.1915 | 0.2651 |
| `mistral_ocr_4` | 0.7634 | 0.8313 | 0.6746 | 0.8242 | 92 | 0.1355 | 0.2034 |
| `qwen3_vl` | 0.7566 | 0.8913 | 0.7464 | 0.7319 | 92 | 0.2410 | 0.3122 |
| `mineru` | 0.6972 | 0.8225 | 0.6624 | 0.6964 | 92 | 0.2136 | 0.2836 |
| `llamaparse` | 0.6911 | 0.8441 | 0.6663 | 0.6747 | 92 | 0.1855 | 0.2826 |
| `docling` | 0.6520 | 0.8434 | 0.6406 | 0.6141 | 92 | 0.3166 | 0.4273 |
| `marker` | 0.6398 | 0.7972 | 0.5581 | 0.6718 | 92 | 0.2490 | 0.3260 |
| `deepseek_ocr` | 0.6391 | 0.8065 | 0.5767 | 0.6519 | 92 | 0.2941 | 0.3798 |
| `paddleocr_vl` | 0.6362 | 0.8032 | 0.5512 | 0.6689 | 92 | 0.2490 | 0.2499 |
| `glm_ocr` | 0.5173 | 0.3794 | 0.4646 | 0.5979 | 92 | 0.3209 | 0.3753 |

### Tables

Reference-cell value credit against accepted gold, including blanks and partial
text matches. Logical row/column associations matter; compatible reflow is allowed.
Extras are separate diagnostics. Shipping uses the better whole-page gold/original
reference score for every engine. Sonnet and Gemini use separate table-extraction
calls.

| Engine | Overall ↑ | Without low ↑ | Easy | Medium | Hard | Pages | Strict F1 ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `claude_sonnet_4_6` | 0.8539 | 0.8578 | 0.9325 | 0.8072 | 0.9156 | 42 | 0.7132 |
| `gemini_flash` | 0.8225 | 0.8299 | 0.8637 | 0.8542 | 0.6865 | 42 | 0.7198 |
| `textract_structured` | 0.7896 | 0.7872 | 0.7883 | 0.8064 | 0.7409 | 42 | 0.5729 |
| `qwen3_vl` | 0.7368 | 0.7420 | 0.7459 | 0.8036 | 0.5273 | 42 | 0.6261 |
| `mistral_ocr_4` | 0.7301 | 0.7324 | 0.7309 | 0.6718 | 0.9042 | 42 | 0.5886 |
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

Based on the same **82 text-scored pages per engine**, scaled to the same document
mix. Costs cover one inference pass. **Modal startup/shutdown is excluded**;
Modal prices cover GPU processing only, and local CPU cost is unpriced.
[Pricing sources and assumptions](docs/engines-and-pricing.md#pricing-audit).

### Cost — lowest first

| Engine | USD/page | 1,000 pages | 100,000 pages | 1 million pages |
| --- | --- | --- | --- | --- |
| `glm_ocr` | $0.000540 | $0.54 | $53.97 | $539.74 |
| `deepseek_ocr` | $0.000749 | $0.75 | $74.92 | $749.23 |
| `mineru` | $0.000782 | $0.78 | $78.23 | $782.33 |
| `paddleocr_vl` | $0.000949 | $0.95 | $94.89 | $948.93 |
| `marker` | $0.001173 | $1.17 | $117.27 | $1,172.69 |
| `gemini_flash` | $0.001251 | $1.25 | $125.13 | $1,251.28 |
| `textract_text` | $0.001500 | $1.50 | $150.00 | $1,500.00 |
| `qwen3_vl` | $0.003734 | $3.73 | $373.45 | $3,734.50 |
| `llamaparse` | $0.003750 | $3.75 | $375.00 | $3,750.00 |
| `mistral_ocr_4` | $0.004000 | $4.00 | $400.00 | $4,000.00 |
| `claude_sonnet_4_6` | $0.015129 | $15.13 | $1,512.91 | $15,129.11 |
| `textract_structured` | $0.065000 | $65.00 | $6,500.00 | $65,000.00 |
| `docling` | Unpriced | — | — | — |
| `tesseract` | Unpriced | — | — | — |

### Latency — fastest first

Volume times assume **sequential API/local calls** or **one warm GPU container
at measured batch throughput**. Parallel workers can shorten elapsed time;
these are projections, not measured production runs.

| Engine | Per page | 1,000 pages | 100,000 pages | 1 million pages |
| --- | --- | --- | --- | --- |
| `deepseek_ocr` | 1.08 sec | 18 min | 1 day 6 hr | 12 days 12 hr |
| `mineru` | 1.13 sec | 19 min | 1 day 7 hr | 13 days 1 hr |
| `paddleocr_vl` | 1.37 sec | 23 min | 1 day 14 hr | 15 days 20 hr |
| `marker` | 1.69 sec | 28 min | 1 day 23 hr | 19 days 13 hr |
| `glm_ocr` | 1.76 sec | 29 min | 2 days 1 hr | 20 days 10 hr |
| `tesseract` | 2.38 sec | 40 min | 2 days 18 hr | 27 days 13 hr |
| `mistral_ocr_4` | 3.85 sec | 1 hr 4 min | 4 days 11 hr | 44 days 13 hr |
| `gemini_flash` | 4.06 sec | 1 hr 8 min | 4 days 17 hr | 47 days |
| `textract_text` | 4.73 sec | 1 hr 19 min | 5 days 11 hr | 54 days 17 hr |
| `qwen3_vl` | 5.38 sec | 1 hr 30 min | 6 days 5 hr | 62 days 7 hr |
| `textract_structured` | 7.08 sec | 1 hr 58 min | 8 days 5 hr | 81 days 23 hr |
| `docling` | 15.60 sec | 4 hr 20 min | 18 days 1 hr | 180 days 14 hr |
| `claude_sonnet_4_6` | 18.27 sec | 5 hr 4 min | 21 days 3 hr | 211 days 11 hr |
| `llamaparse` | 21.88 sec | 6 hr 5 min | 25 days 8 hr | 253 days 7 hr |

### Geometry-only pipelines

These two configurations were measured on 17 pages, separately from text OCR.

| Engine | USD/page | 1,000 pages | 100,000 pages | 1 million pages |
| --- | --- | --- | --- | --- |
| `paddleocr_ppocrv5` | $0.000268 | $0.27 | $26.82 | $268.18 |
| `mineru_pipeline` | $0.000430 | $0.43 | $42.98 | $429.84 |

| Engine | Per page | 1,000 pages | 100,000 pages | 1 million pages |
| --- | --- | --- | --- | --- |
| `paddleocr_ppocrv5` | 0.88 sec | 15 min | 1 day | 10 days 3 hr |
| `mineru_pipeline` | 2.62 sec | 44 min | 3 days 1 hr | 30 days 8 hr |

## Engines and versions tested

| Engine ID | Version / model | Pipeline |
| --- | --- | --- |
| `claude_sonnet_4_6` | Sonnet 4.6 | Anthropic API |
| `deepseek_ocr` | DeepSeek-OCR v1 · `9f30c71f441d` | vLLM · Modal A100-80GB |
| `docling` | 2.118.1 (reconstructed) | Heron + TableFormer Accurate + RapidOCR/PP-OCRv6 · CPU |
| `gemini_flash` | Gemini 3.1 Flash-Lite | Google API |
| `glm_ocr` | GLM-OCR · `ca5d8b3e287e` | vLLM · Modal A10 |
| `llamaparse` | v1 Balanced; backend unrecorded | Parse API; schema: LlamaExtract / GPT-4.1 |
| `marker` | marker-pdf 1.8.2 | 11 workers · Modal A100-80GB |
| `mineru` | MinerU2.5-2509-1.2B · `1aa090b41282` | vLLM · Modal A100-80GB |
| `mineru_pipeline` | MinerU 2.5.4 · weights `ed6b654c018d` | DocLayout-YOLO + PP-OCR + UniMERNet · Modal T4 |
| `mistral_ocr_4` | OCR 4.1 (alias history); schema: 4.0 | Mistral API |
| `paddleocr_ppocrv5` | PP-OCRv5 · PaddleOCR 3.7.0 | Server detection/recognition · Modal A10 |
| `paddleocr_vl` | PaddleOCR-VL v1 · `7fa00a8c55b7` | Layout + vLLM · Modal A100-80GB |
| `qwen3_vl` | Qwen3-VL-8B-Instruct · `0c351dd01ed8` | vLLM · Modal A100-80GB |
| `tesseract` | 5.5.2 | CPU |
| `textract_structured` | AnalyzeDocument 1.0 | AWS TABLES + FORMS |
| `textract_text` | DetectDocumentText 1.0 | AWS text detection |

[Full version evidence, dependencies and limitations](docs/engines-and-pricing.md#engine-versions-and-provenance).

## View the published results

From the repository directory, install dependencies and print the terminal comparison:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_benchmark.py --report current
```

`current` reads the included `results/current.zip`. No credentials or model calls.
Add `--legs text --report-sort cer` to sort text results by error rate, or
`--report-details` for document-family breakdowns.

For the browser report, open `results/reports/ocr-benchmark-final.html` if it
already exists. On a fresh clone, build it with the source images:

```bash
python scripts/download_data.py
python scripts/build_evidence_report.py current
```

Then open `results/reports/ocr-benchmark-final.html`; its evidence links show the
saved outputs beside source pages. Building reports does not rerun engines.

## Run a new benchmark

**1. Set up** from the repository directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp -n .env.example .env
python scripts/download_data.py
```

Add credentials for your selected engines to `.env`. The download supplies
source images; references are included in the repository.

**2. Select engines and legs, then preview** the calls without running them:

```bash
python run_benchmark.py --engines mistral_ocr_4,docling --legs text,markdown,tables --skip-judge --dry-run
```

| Option | What to choose |
| --- | --- |
| `--engines` | Comma-separated IDs from the engine table above. |
| `--legs` | `text,markdown,tables,key_value,schema,geometry`; omit for all supported legs. |
| `--tiers` | `easy,medium,hard`; omit for all tiers. |
| `--docs` | Page IDs such as `invoice_000,sroie_000`; omit for all selected pages. |

**3. Run** by removing `--dry-run`. Hosted engines incur charges:

```bash
python run_benchmark.py --engines mistral_ocr_4,docling --legs text,markdown,tables --skip-judge
```

The command saves outputs, evaluates them and prints the comparison.
`--skip-judge` skips new judging, **not OCR**: uncached KV reviews remain pending
and pages without text references receive no new judged-text scores.
To enable judging, omit it and set `GOOGLE_API_KEY`. The KV judge allowance is
`OCR_KV_JUDGE_BUDGET_USD` (default `$2`); it does not cap inference or judged text.

**4. View your run later.** Replace `RUN_TIMESTAMP` with the timestamp printed by the run:

```bash
python run_benchmark.py --report RUN_TIMESTAMP
python scripts/build_evidence_report.py RUN_TIMESTAMP
```

The second command creates `results/reports/ocr-benchmark-final.html` for that run.

### Local and Modal setup

Tesseract needs its system binary installed. For a Modal engine, deploy its app
before step 2; for example:

```bash
modal setup
python scripts/setup_modal.py
python scripts/deploy_modal.py --apps glm_ocr --skip-smoke
```

`--skip-smoke` skips test inference during deployment. Builds may still incur charges.
