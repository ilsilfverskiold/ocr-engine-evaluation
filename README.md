# OCR Benchmark

Compare OCR engines on text, Markdown, tables, key/value extraction, supplied
schemas and geometry. **93 pages· 16 measured engine configurations.**

## Results

Saved run: **`20261018T111413Z`**. All deterministic legs were replayed from the selected responses in `reconciled_outputs/`; 110 saved document-judge verdicts were reused. Three unresolved KV reviews were completed after deterministic and cached matches, for $0.0666575. Only those three rows changed from the preceding verified replay; all other 4,750 rows match, including field-level evidence. **Unresolved: 9 failed result rows. No KV judgments remain pending.**
[Complete old-versus-new comparison](results/reports/old-vs-new.html).

All nine failed cases were retried. The [Marker retry log](results/logs/20261018T093009Z.log)
records five repeated exceptions and three collected responses that contain no
text or regions. The [DeepSeek retry log](results/logs/20261018T092656Z.log)
records another HTTP 500 on `shipping_003` schema. The earlier reconciliation
kept the old error records when a retry failed; these logs retain the retry history.

The [text/Markdown output repair](docs/text-markdown-output-repair.md) rescored
LlamaParse, PaddleOCR-VL v1 and MinerU pipeline from retained calls, and Docling
from a new local conversion of the same 93 pages. PaddleOCR-VL 1.6 is a separate
later-engine integration and has no rows in this benchmark.
Each leg is sorted by overall score, highest first. Overall means give each
document family equal weight. **Easy/medium/hard are difficulty tiers.** “Pages”
counts successful scores; errors are listed separately. **—** means no score,
not zero. CER (character error rate) and WER (word error rate) are better when
lower; values can exceed 1.

The “Without low” column appears where the common low-scoring-page exclusion
changes results: `chart_003` for KV. These are sensitivity
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
| `llamaparse` | 0.8814 | 0.9474 | 0.8724 | 0.8716 | 82 | 0.1754 | 0.2757 |
| `paddleocr_vl` | 0.8749 | 0.9446 | 0.8223 | 0.9076 | 82 | 0.3137 | 0.3952 |
| `docling` | 0.8643 | 0.9112 | 0.8919 | 0.8234 | 82 | 0.2989 | 0.4184 |
| `mineru` | 0.8613 | 0.9133 | 0.8926 | 0.8152 | 82 | 0.2122 | 0.2826 |
| `marker` | 0.8563 | 0.8868 | 0.8673 | 0.8365 | 82 | 0.2490 | 0.3260 |
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
| `docling` | 0.6869 | 0.8434 | 0.6472 | 0.6826 | 92 | 0.2988 | 0.4179 |
| `paddleocr_vl` | 0.6831 | 0.8074 | 0.5888 | 0.7346 | 92 | 0.3102 | 0.3821 |
| `marker` | 0.6398 | 0.7972 | 0.5581 | 0.6718 | 92 | 0.2490 | 0.3260 |
| `deepseek_ocr` | 0.6391 | 0.8065 | 0.5767 | 0.6519 | 92 | 0.2941 | 0.3798 |
| `glm_ocr` | 0.5173 | 0.3794 | 0.4646 | 0.5979 | 92 | 0.3209 | 0.3753 |

`mineru_pipeline` has corrected text 0.8599 and Markdown 0.6348 on its 17
saved pages. That smaller geometry sample is separate from these rankings.
PaddleOCR-VL v1's recovered blocks raised text coverage, but its CER rose from
0.2490 to 0.3137; the block order and inserted content affect edit distance.

### Tables

Reference-cell value credit against accepted gold, including blanks and partial
text matches. Logical row/column associations matter; compatible reflow is allowed.
Extras are separate diagnostics. Shipping uses the better whole-page gold/original
reference score for every engine. Sonnet and Gemini use separate table-extraction
calls.

| Engine | Overall ↑ | Easy | Medium | Hard | Pages | Strict F1 ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| `claude_sonnet_4_6` | 0.8810 | 0.9432 | 0.8488 | 0.9156 | 42 | 0.7132 |
| `textract_structured` | 0.8801 | 0.8802 | 0.9184 | 0.7649 | 42 | 0.5729 |
| `gemini_flash` | 0.8558 | 0.9064 | 0.8836 | 0.7219 | 42 | 0.7198 |
| `qwen3_vl` | 0.7631 | 0.7901 | 0.8209 | 0.5627 | 42 | 0.6261 |
| `mistral_ocr_4` | 0.7469 | 0.7617 | 0.6777 | 0.9396 | 42 | 0.5886 |
| `paddleocr_vl` | 0.7295 | 0.6902 | 0.7120 | 0.8213 | 42 | 0.5950 |
| `llamaparse` | 0.6519 | 0.8082 | 0.6954 | 0.3654 | 42 | 0.5480 |
| `mineru` | 0.6509 | 0.6107 | 0.6314 | 0.7498 | 42 | 0.5323 |
| `glm_ocr` | 0.6359 | 0.6416 | 0.6264 | 0.6586 | 42 | 0.5563 |
| `deepseek_ocr` | 0.6290 | 0.7964 | 0.5958 | 0.5611 | 42 | 0.5288 |
| `docling` | 0.5604 | 0.5679 | 0.5803 | 0.4932 | 42 | 0.4967 |
| `marker` | 0.5545 | 0.5634 | 0.5108 | 0.6769 | 42 | 0.3719 |

### Key/value extraction

Correct values under eligible printed labels. Exact/cached mappings come first;
ambiguous wording or split values use contextual judge decisions, cached for reuse.
Missing decisions stay pending. Unreturned blank reference fields are excluded;
returned blanks and invented values still count. Extras are diagnostic.

| Engine | Overall ↑ | Without low ↑ | Easy | Medium | Hard | Pages |
| --- | --- | --- | --- | --- | --- | --- |
| `textract_structured` | 0.8036 | 0.8048 | 0.8502 | 0.8162 | 0.7584 | 65 |
| `claude_sonnet_4_6` | 0.7962 | 0.8112 | 0.7864 | 0.8173 | 0.7642 | 65 |
| `qwen3_vl` | 0.6574 | 0.6713 | 0.6681 | 0.5947 | 0.7616 | 65 |
| `gemini_flash` | 0.5392 | 0.5491 | 0.4923 | 0.4640 | 0.6942 | 65 |

### Supplied-schema extraction

Correct values under exact requested keys, penalizing extra/duplicate keys. Every
engine receives the same frozen sample: 3,316 fields across 65 pages. No judge.

| Engine | Overall ↑ | Easy | Medium | Hard | Pages | Errors |
| --- | --- | --- | --- | --- | --- | --- |
| `gemini_flash` | 0.9680 | 0.9843 | 0.9490 | 0.9930 | 65 | 0 |
| `claude_sonnet_4_6` | 0.9597 | 0.9858 | 0.9409 | 0.9796 | 65 | 0 |
| `llamaparse` | 0.8998 | 0.9398 | 0.9162 | 0.8512 | 65 | 0 |
| `mistral_ocr_4` | 0.8635 | 0.9371 | 0.8472 | 0.8554 | 65 | 0 |
| `qwen3_vl` | 0.8591 | 0.9140 | 0.8254 | 0.8906 | 65 | 0 |
| `glm_ocr` | 0.6660 | 0.7883 | 0.6342 | 0.6607 | 65 | 0 |
| `deepseek_ocr` | 0.1604 | 0.3236 | 0.1429 | 0.1094 | 64 | 1 |

Three fresh Gemini schema calls used the same saved prompts, field schemas and
model (`gemini-3.1-flash-lite`): `bank_002` 1.0000 → 1.0000, `payslip_003`
1.0000 → 1.0000, and `funsd_003` 1.0000 → 0.7895. The form's fresh answer shifted
table rows. All 65 scored saved Gemini schema answers match their stored provider
response text. Schema supplies exact field/row identities and credits correct
blanks; populated-field accuracy across the saved run is 0.9649.
[All validation fields and raw calls](results/reports/gemini-schema-validation.html).
These three calls cost $0.010452 and remain separate from the published scores.

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
| `claude_sonnet_4_6` | 18.27 sec | 5 hr 4 min | 21 days 3 hr | 211 days 11 hr |
| `llamaparse` | 21.88 sec | 6 hr 5 min | 25 days 8 hr | 253 days 7 hr |
| `docling` | 24.84 sec | 6 hr 54 min | 28 days 17 hr | 287 days 11 hr |

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
| `docling` | 2.118.1 (recorded local rerun) | Heron + TableFormer Accurate + RapidOCR/PP-OCRv6 · CPU |
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

`current` reads the newest complete, scored run under `results/runs/` and its
saved calls. No credentials or model calls.
Add `--legs text --report-sort cer` to sort text results by error rate, or
`--report-details` for document-family breakdowns.

All 2,857 saved call versions selected for scoring live in one folder:
`results/ocr_outputs/reconciled_outputs/`, organized by document and engine.
Different scored views are files beside each other, such as `default.call.json`
and `default.tables.call.json`. Each inference row's `source_pass_name` selects
its exact file; the original logical pass and timestamps remain recorded.
The six retained rerun responses are selected for their corresponding cases.
Earlier attempts and the three Gemini schema validation responses remain
available under distinct filenames; validation responses do not replace the
published answers.

To recalculate saved outputs with the current scorers, without OCR or new
judge calls:

```bash
python run_benchmark.py --rescore current
```

This creates a separate run and preserves historical document verdicts once.
Changes to scorers or cached KV rulings can change scores. New paid KV rulings
require `--allow-kv-judge`; that flag never refreshes document verdicts.
`python scripts/verify_published_replay.py --historical` audits the original
evaluator and judgment caches pinned in Git. It blocks network and new judge
calls and fails on any score difference. To audit the current evaluator while
checking the current publication, omit `--historical`; it requires every score,
applicability flag and error to match. `--allow-kv-changes` is only for reviewing
an intentional KV scorer change; every difference is still recorded.

For the browser report, open `results/reports/ocr-benchmark-final.html`. To
rebuild it with the source images:

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
