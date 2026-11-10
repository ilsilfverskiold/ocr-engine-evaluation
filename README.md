# OCR Benchmark

OCR engines compared on six tasks: text, Markdown, tables, key/value
extraction, supplied-schema extraction and geometry. **93 single-page
documents · 17 engine configurations · one published run.** Every call, reply
and reference behind the scores is in this repository.

- [Results](#results), then [cost and latency](#cost-and-latency)
- [How it was tested](#how-it-was-tested): engines and versions, documents, scoring
- [View the results yourself](#view-the-results-yourself): terminal report and browser viewer, or the [hosted viewer](https://ilsilfverskiold.github.io/ocr-engine-evaluation/) with no setup
- [Run the benchmark](#run-the-benchmark): with the engines here, or with an engine you add

The [hosted viewer](https://ilsilfverskiold.github.io/ocr-engine-evaluation/)
shows this run in the browser, with no setup: the comparison tables and charts,
and for every engine, page and leg the prompt sent, the full reply, the output
beside its reference and the scorer's verdict per item.

## Results

Published run `20261020T120222Z`. Each leg is a separate task with its own
reference; there is no combined score. Tables are sorted by Overall, highest
first.

- **Overall** weights every document family equally. Easy, Medium and Hard
  are difficulty tiers.
- **Pages** counts scored pages. **Errors** counts failed calls, which get no
  score rather than zero; — also means no score.
- Nine rows are inference errors: eight Marker geometry pages and one
  DeepSeek-OCR schema page.

### Text accuracy

Share of the reference's distinct words the engine returned, on the 82 pages
with reliable text references. CER/WER also count order and extra text.
[How text is scored](public_docs/scoring-design.md#text).

| Engine | Overall ↑ | Easy | Medium | Hard | Pages | CER ↓ | WER ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `claude_sonnet_4_6` | 0.9634 | 0.9664 | 0.9610 | 0.9649 | 82 | 0.1267 | 0.1799 |
| `gemini_flash` | 0.9508 | 0.9690 | 0.9489 | 0.9474 | 82 | 0.1588 | 0.2375 |
| `textract_text` | 0.9471 | 0.9625 | 0.9513 | 0.9385 | 82 | 0.1476 | 0.2094 |
| `textract_structured` | 0.9465 | 0.9625 | 0.9513 | 0.9370 | 82 | 0.1478 | 0.2102 |
| `mistral_ocr_4_1` | 0.9231 | 0.9560 | 0.9554 | 0.8813 | 82 | 0.1355 | 0.2034 |
| `qwen3_vl` | 0.9077 | 0.9536 | 0.8856 | 0.9168 | 82 | 0.5730 | 0.7905 |
| `glm_ocr` | 0.9039 | 0.9459 | 0.9197 | 0.8761 | 82 | 0.1698 | 0.2407 |
| `llamaparse` | 0.8814 | 0.9474 | 0.8724 | 0.8716 | 82 | 0.1754 | 0.2757 |
| `paddleocr_vl` | 0.8749 | 0.9446 | 0.8223 | 0.9076 | 82 | 0.3137 | 0.3952 |
| `docling` | 0.8643 | 0.9112 | 0.8919 | 0.8234 | 82 | 0.2989 | 0.4184 |
| `mineru` | 0.8613 | 0.9133 | 0.8926 | 0.8152 | 82 | 0.2122 | 0.2826 |
| `marker` | 0.8563 | 0.8868 | 0.8673 | 0.8365 | 82 | 0.2490 | 0.3260 |
| `deepseek_ocr` | 0.8243 | 0.8910 | 0.7837 | 0.8458 | 82 | 0.2319 | 0.3298 |
| `tesseract` | 0.7391 | 0.8682 | 0.7539 | 0.6874 | 82 | 0.3532 | 0.5126 |

`mineru_pipeline` ran on the 17 geometry pages only; its text and Markdown
scores there, 0.8599 and 0.6348, are not ranked against the full set.

### Judged text (estimate)

The 11 hard pages without reliable text references: a model checks each output
against a cached checklist built from the page image. This is an estimate, not
a leg, and is not combined with text accuracy. The cached verdicts may include
substring-matching errors; engines without saved verdicts are omitted.

| Engine | Overall ↑ | Hard | Pages |
| --- | --- | --- | --- |
| `gemini_flash` | 0.8691 | 0.8691 | 11 |
| `mistral_ocr_4_1` | 0.8421 | 0.8421 | 11 |
| `claude_sonnet_4_6` | 0.8272 | 0.8272 | 11 |
| `textract_text` | 0.7878 | 0.7878 | 11 |
| `textract_structured` | 0.7799 | 0.7799 | 11 |
| `mineru` | 0.6281 | 0.6281 | 11 |
| `llamaparse` | 0.6253 | 0.6253 | 11 |
| `paddleocr_vl` | 0.5245 | 0.5245 | 11 |
| `docling` | 0.5151 | 0.5151 | 11 |
| `tesseract` | 0.2853 | 0.2853 | 11 |

### Markdown structure

Correct structural elements, with half credit for partial matches; bold text
is not a heading. One cached, model-written checklist per page is shared by
all engines, and the answers are scored by code.
[How Markdown is scored](public_docs/scoring-design.md#markdown).

| Engine | Overall ↑ | Easy | Medium | Hard | Pages | CER ↓ | WER ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `claude_sonnet_4_6` | 0.8221 | 0.8854 | 0.7909 | 0.8336 | 92 | 0.1301 | 0.1911 |
| `gemini_flash` | 0.7724 | 0.8601 | 0.7309 | 0.7868 | 92 | 0.1915 | 0.2651 |
| `mistral_ocr_4_1` | 0.7634 | 0.8313 | 0.6746 | 0.8242 | 92 | 0.1355 | 0.2034 |
| `qwen3_vl` | 0.7566 | 0.8913 | 0.7464 | 0.7319 | 92 | 0.2410 | 0.3122 |
| `mineru` | 0.6972 | 0.8225 | 0.6624 | 0.6964 | 92 | 0.2136 | 0.2836 |
| `llamaparse` | 0.6911 | 0.8441 | 0.6663 | 0.6747 | 92 | 0.1855 | 0.2826 |
| `docling` | 0.6869 | 0.8434 | 0.6472 | 0.6826 | 92 | 0.2988 | 0.4179 |
| `paddleocr_vl` | 0.6831 | 0.8074 | 0.5888 | 0.7346 | 92 | 0.3102 | 0.3821 |
| `marker` | 0.6398 | 0.7972 | 0.5581 | 0.6718 | 92 | 0.2490 | 0.3260 |
| `deepseek_ocr` | 0.6391 | 0.8065 | 0.5767 | 0.6519 | 92 | 0.2941 | 0.3798 |
| `glm_ocr` | 0.5173 | 0.3794 | 0.4646 | 0.5979 | 92 | 0.3209 | 0.3753 |

### Tables

Credit per reference cell; row and column associations matter, and partial
text earns partial credit. **Numbers**: share of numeric reference cells
returned correct in place. **Invented**: share of returned numbers printed
nowhere on the page. **Strict F1**: exact cell match. Sonnet and Gemini use
a separate table-extraction call. [How tables are scored](public_docs/scoring-design.md#tables).

| Engine | Overall ↑ | Easy | Medium | Hard | Pages | Numbers ↑ | Invented ↓ | Strict F1 ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gemini_flash` | 0.9280 | 0.9503 | 0.9174 | 0.9375 | 42 | 0.9047 | 0.0002 | 0.7537 |
| `claude_sonnet_4_6` | 0.9172 | 0.9539 | 0.8944 | 0.9487 | 42 | 0.9117 | 0.0054 | 0.7296 |
| `textract_structured` | 0.9116 | 0.8960 | 0.9473 | 0.8200 | 42 | 0.8915 | 0.0010 | 0.5727 |
| `qwen3_vl` | 0.8122 | 0.7998 | 0.8412 | 0.7377 | 42 | 0.7620 | 0.0082 | 0.6256 |
| `mistral_ocr_4_1` | 0.7696 | 0.7644 | 0.7054 | 0.9674 | 42 | 0.7229 | 0.0063 | 0.5779 |
| `paddleocr_vl` | 0.7543 | 0.6920 | 0.7262 | 0.9010 | 42 | 0.7089 | 0.0339 | 0.6042 |
| `llamaparse` | 0.7003 | 0.8225 | 0.7071 | 0.5579 | 42 | 0.6362 | 0.0262 | 0.5510 |
| `glm_ocr` | 0.6899 | 0.6643 | 0.6515 | 0.8306 | 42 | 0.6768 | 0.0000 | 0.5563 |
| `mineru` | 0.6684 | 0.6107 | 0.6544 | 0.7681 | 42 | 0.6056 | 0.0042 | 0.5324 |
| `deepseek_ocr` | 0.6636 | 0.7964 | 0.6150 | 0.6764 | 42 | 0.6152 | 0.1269 | 0.5172 |
| `docling` | 0.5944 | 0.5679 | 0.5922 | 0.6278 | 42 | 0.5790 | 0.0152 | 0.4975 |
| `marker` | 0.5919 | 0.5672 | 0.5451 | 0.7572 | 42 | 0.4598 | 0.0406 | 0.3660 |

### Key/value extraction

Correct values under the page's printed labels. Exact matches are settled by
code; ambiguous wording or split values go to a judge whose cached rulings are
reused. **Without low** leaves out `chart_003`, a page every engine scored
low; it is a sensitivity check, not a ranking.
[How key/value is scored](public_docs/scoring-design.md#keyvalue).

| Engine | Overall ↑ | Without low ↑ | Easy | Medium | Hard | Pages |
| --- | --- | --- | --- | --- | --- | --- |
| `textract_structured` | 0.8036 | 0.8048 | 0.8502 | 0.8162 | 0.7584 | 65 |
| `claude_sonnet_4_6` | 0.7962 | 0.8112 | 0.7864 | 0.8173 | 0.7642 | 65 |
| `qwen3_vl` | 0.6574 | 0.6713 | 0.6681 | 0.5947 | 0.7616 | 65 |
| `mistral_ocr_4_1` | 0.6527 | 0.6613 | 0.8079 | 0.6213 | 0.6301 | 65 |
| `gemini_flash` | 0.5392 | 0.5491 | 0.4923 | 0.4640 | 0.6942 | 65 |

### Supplied-schema extraction

Correct values under the exact requested keys, with a penalty for extra or
duplicate keys. Every engine gets the same 3,316 fields across 65 pages. No
judge. [How schema is scored](public_docs/scoring-design.md#schema).

| Engine | Overall ↑ | Easy | Medium | Hard | Pages | Errors |
| --- | --- | --- | --- | --- | --- | --- |
| `gemini_flash` | 0.9680 | 0.9843 | 0.9490 | 0.9930 | 65 | 0 |
| `claude_sonnet_4_6` | 0.9597 | 0.9858 | 0.9409 | 0.9796 | 65 | 0 |
| `llamaparse` | 0.8978 | 0.9242 | 0.9162 | 0.8523 | 65 | 0 |
| `qwen3_vl` | 0.8591 | 0.9140 | 0.8254 | 0.8906 | 65 | 0 |
| `mistral_ocr_4_1` | 0.8465 | 0.9576 | 0.8125 | 0.8506 | 65 | 0 |
| `glm_ocr` | 0.6660 | 0.7883 | 0.6342 | 0.6607 | 65 | 0 |
| `deepseek_ocr` | 0.1604 | 0.3236 | 0.1429 | 0.1094 | 64 | 1 |

### Geometry

**Word**: share of reference words, or lines on receipts, located with
matching text where each box contains the other's centre. **Line**: the same
after merging both sides into visual lines, the fair number for engines that
return one box per line. **Block cov**: for engines that return layout blocks,
share of reference words inside a block that contains their text; the bracket
is the median words per block. F1 penalizes extra boxes.
[How geometry is scored](public_docs/scoring-design.md#geometry).

| Engine | Word ↑ | Line ↑ | Block cov (words/box) | Easy | Medium | Hard | Pages | F1 ↑ | CER ↓ | WER ↓ | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `textract_text` | 0.9476 | 0.9895 | — | 0.9936 | — | 0.9246 | 17 | 0.9369 | 0.0388 | 0.1358 | 0 |
| `textract_structured` | 0.9474 | 0.9895 | — | 0.9936 | — | 0.9243 | 17 | 0.9370 | 0.0387 | 0.1355 | 0 |
| `paddleocr_ppocrv5` | 0.8507 | 0.9962 | — | 0.9601 | — | 0.7959 | 17 | 0.7776 | 0.1046 | 0.7023 | 0 |
| `llamaparse_granular` | 0.8479 | 0.8404 | — | 0.8460 | — | 0.8488 | 17 | 0.8643 | 0.1989 | 0.2895 | 0 |
| `tesseract` | 0.7956 | 0.7984 | — | 0.8286 | — | 0.7791 | 17 | 0.8166 | 0.1671 | 0.3582 | 0 |
| `gemini_flash` | 0.5884 | 0.8180 | — | 0.9020 | — | 0.4315 | 17 | 0.5944 | 0.1534 | 0.2532 | 0 |
| `marker` | 0.2770 | 0.4162 | — | 0.5341 | — | 0.0198 | 9 | 0.3194 | 0.5563 | 0.6088 | 8 |
| `docling` | 0.2327 | 0.4257 | 0.9526 (34) | 0.6326 | — | 0.0327 | 17 | 0.2676 | 0.0756 | 0.2158 | 0 |
| `claude_sonnet_4_6` | 0.1525 | 0.1932 | — | 0.0081 | — | 0.2248 | 17 | 0.1503 | 0.1186 | 0.2064 | 0 |
| `mineru_pipeline` | 0.1132 | 0.4456 | — | 0.3059 | — | 0.0169 | 17 | 0.1452 | 0.2088 | 0.3155 | 0 |
| `mineru` | 0.1130 | 0.4002 | 0.9669 (34) | 0.3078 | — | 0.0157 | 17 | 0.1473 | 0.1478 | 0.2848 | 0 |
| `deepseek_ocr` | 0.0806 | 0.3218 | 0.9390 (30) | 0.2262 | — | 0.0077 | 17 | 0.1161 | 0.0916 | 0.1825 | 0 |
| `paddleocr_vl` | 0.0564 | 0.2677 | 0.9512 (37) | 0.1388 | — | 0.0152 | 17 | 0.0820 | 0.0813 | 0.1764 | 0 |
| `llamaparse` | 0.0511 | 0.1680 | 0.9190 (18) | 0.1301 | — | 0.0116 | 17 | 0.0788 | 0.4918 | 0.6101 | 0 |
| `qwen3_vl` | 0.0052 | 0.0282 | — | 0.0142 | — | 0.0007 | 17 | 0.0057 | 0.5342 | 0.6834 | 0 |

`deepseek_ocr`, `mineru`, `paddleocr_vl`, `docling` and `llamaparse` return
layout blocks rather than word boxes, so read their Block cov.
`llamaparse_granular` is LlamaParse's word and line box option, measured as
its own configuration; `mineru_pipeline` mixes line and whole-table boxes.

## Cost and latency

Per-page means over the same 82 text-scored pages, multiplied out to volume,
for one inference pass. Modal figures are GPU processing only, without
container startup or shutdown; local CPU engines are unpriced.
[Pricing sources and assumptions](public_docs/engines-and-pricing.md#pricing-audit).

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
| `mistral_ocr_4_1` | $0.004000 | $4.00 | $400.00 | $4,000.00 |
| `claude_sonnet_4_6` | $0.015129 | $15.13 | $1,512.91 | $15,129.11 |
| `textract_structured` | $0.065000 | $65.00 | $6,500.00 | $65,000.00 |
| `docling` | Unpriced | — | — | — |
| `tesseract` | Unpriced | — | — | — |

### Latency — fastest first

The volume columns run one page at a time: sequential calls, or one warm GPU
container at its measured throughput. The parallel columns run as many pages at
once as each provider's default limits allow a standard paid account with no
increase requested: AWS Textract quotas in US East, Anthropic's Start tier,
LlamaCloud's Starter plan and Modal's Starter plan of 10 GPUs. Gemini and
Mistral publish no default limits, so theirs are marked estimates. Local
engines scale with your CPU cores. Cost is the same either way.
[Limits, sources and caveats](public_docs/engines-and-pricing.md#scaling-latency-in-parallel).

| Engine | Per page | 1,000 pages | 100,000 pages | 1 million pages | Parallel limit | 1 million pages in parallel |
| --- | --- | --- | --- | --- | --- | --- |
| `deepseek_ocr` | 1.08 sec | 18 min | 1 day 6 hr | 12 days 12 hr | 10 GPUs at once | 1 day 6 hr |
| `mineru` | 1.13 sec | 19 min | 1 day 7 hr | 13 days 1 hr | 10 GPUs at once | 1 day 7 hr |
| `paddleocr_vl` | 1.37 sec | 23 min | 1 day 14 hr | 15 days 20 hr | 10 GPUs at once | 1 day 14 hr |
| `marker` | 1.69 sec | 28 min | 1 day 23 hr | 19 days 13 hr | 10 GPUs at once | 1 day 23 hr |
| `glm_ocr` | 1.76 sec | 29 min | 2 days 1 hr | 20 days 10 hr | 10 GPUs at once | 2 days 1 hr |
| `tesseract` | 2.38 sec | 40 min | 2 days 18 hr | 27 days 13 hr | Local: your CPU cores | — |
| `mistral_ocr_4_1` | 3.85 sec | 1 hr 4 min | 4 days 11 hr | 44 days 13 hr | 1 request/s, about 4 at once (estimate) | 11 days 14 hr |
| `gemini_flash` | 4.06 sec | 1 hr 8 min | 4 days 17 hr | 47 days | 150k requests/day (estimate) | 6 days 16 hr |
| `textract_text` | 4.73 sec | 1 hr 19 min | 5 days 11 hr | 54 days 17 hr | 25 requests/s, about 118 at once | 11 hr 7 min |
| `qwen3_vl` | 5.38 sec | 1 hr 30 min | 6 days 5 hr | 62 days 7 hr | 10 GPUs at once | 6 days 5 hr |
| `textract_structured` | 7.08 sec | 1 hr 58 min | 8 days 5 hr | 81 days 23 hr | 10 requests/s, about 71 at once | 1 day 4 hr |
| `claude_sonnet_4_6` | 18.27 sec | 5 hr 4 min | 21 days 3 hr | 211 days 11 hr | 400k output tokens/min, about 169 at once | 1 day 6 hr |
| `llamaparse` | 21.88 sec | 6 hr 5 min | 25 days 8 hr | 253 days 7 hr | 5 jobs at once | 50 days 16 hr |
| `docling` | 24.84 sec | 6 hr 54 min | 28 days 17 hr | 287 days 11 hr | Local: your CPU cores | — |

### Geometry-only pipelines

Measured on the 17 geometry pages.

| Engine | USD/page | 1,000 pages | 100,000 pages | 1 million pages |
| --- | --- | --- | --- | --- |
| `paddleocr_ppocrv5` | $0.000268 | $0.27 | $26.82 | $268.18 |
| `mineru_pipeline` | $0.000430 | $0.43 | $42.98 | $429.84 |
| `llamaparse_granular` | $0.003750 | $3.75 | $375.00 | $3,750.00 |

| Engine | Per page | 1,000 pages | 100,000 pages | 1 million pages | Parallel limit | 1 million pages in parallel |
| --- | --- | --- | --- | --- | --- | --- |
| `paddleocr_ppocrv5` | 0.88 sec | 15 min | 1 day | 10 days 3 hr | 10 GPUs at once | 1 day |
| `mineru_pipeline` | 2.62 sec | 44 min | 3 days 1 hr | 30 days 8 hr | 10 GPUs at once | 3 days 1 hr |
| `llamaparse_granular` | 8.59 sec | 2 hr 23 min | 9 days 22 hr | 99 days 10 hr | 5 jobs at once | 19 days 21 hr |

## How it was tested

### Engines and versions

The engine ID is the value `--engines` takes.

| Engine ID | Version / model | Pipeline |
| --- | --- | --- |
| `claude_sonnet_4_6` | Sonnet 4.6 | Anthropic API |
| `deepseek_ocr` | DeepSeek-OCR v1 · `9f30c71f441d` | vLLM · Modal A100-80GB |
| `docling` | 2.118.1 (recorded local rerun) | Heron + TableFormer Accurate + RapidOCR/PP-OCRv6 · CPU |
| `gemini_flash` | Gemini 3.1 Flash-Lite | Google API |
| `glm_ocr` | GLM-OCR · `ca5d8b3e287e` | vLLM · Modal A10 |
| `llamaparse` | v1 Balanced; backend unrecorded | Parse API; schema: LlamaExtract / GPT-4.1 with `cite_sources` on the 17 box-reference pages |
| `llamaparse_granular` | LlamaParse `latest`, `output_options.granular_bboxes=[word, line]`, cost_effective tier | Parse API; geometry only |
| `marker` | marker-pdf 1.8.2 | 11 workers · Modal A100-80GB |
| `mineru` | MinerU2.5-2509-1.2B · `1aa090b41282` | vLLM · Modal A100-80GB |
| `mineru_pipeline` | MinerU 2.5.4 · weights `ed6b654c018d` | DocLayout-YOLO + PP-OCR + UniMERNet · Modal T4 |
| `mistral_ocr_4_1` | OCR 4.1 · key_value/schema `mistral-ocr-4-1`; conversion `mistral-ocr-latest` (4.1, alias history) | Mistral API |
| `paddleocr_ppocrv5` | PP-OCRv5 · PaddleOCR 3.7.0 | Server detection/recognition · Modal A10 |
| `paddleocr_vl` | PaddleOCR-VL v1 · `7fa00a8c55b7` | Layout + vLLM · Modal A100-80GB |
| `qwen3_vl` | Qwen3-VL-8B-Instruct · `0c351dd01ed8` | vLLM · Modal A100-80GB |
| `tesseract` | 5.5.2 | CPU |
| `textract_structured` | AnalyzeDocument 1.0 | AWS TABLES + FORMS |
| `textract_text` | DetectDocumentText 1.0 | AWS text detection |

[Full version evidence, dependencies and limitations](public_docs/engines-and-pricing.md#engine-versions-and-provenance).

### Documents

93 single-page images from public datasets, hosted as the HuggingFace dataset
[`ilsilfverskiold/ocr-benchmark`](https://huggingface.co/datasets/ilsilfverskiold/ocr-benchmark).
References are committed under `documents/`. Text references are 63
human-written and 19 model-verified pages; the other 11 use the judged-text
estimate. The table shows reference coverage per leg.

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
[Dataset attribution and licences](public_docs/documents-and-licensing.md).

### Scoring

| Leg | Question | Headline score |
| --- | --- | --- |
| [Text](public_docs/scoring-design.md#text) | Did the engine read the words? | Unique reference-token coverage |
| [Markdown](public_docs/scoring-design.md#markdown) | Is the page structure right? | (correct + 0.5 × partial) / elements |
| [Tables](public_docs/scoring-design.md#tables) | Are the table cells right? | Value credit / counted reference cells |
| [Key/value](public_docs/scoring-design.md#keyvalue) | Did it pull the printed fields? | (correct + blank_ok) / counted fields |
| [Schema](public_docs/scoring-design.md#schema) | Can it fill a given schema? | (correct + blank_ok) / (fields + extra fields) |
| [Geometry](public_docs/scoring-design.md#geometry) | Are the boxes in the right place? | Located reference units / reference units |

- [Scoring design](public_docs/scoring-design.md): for each leg, what is
  asked, the reference, the scorer, normalisation and known issues; which engines take part; the retry
  policy; averaging; the judged-text estimate; limitations.
- [Engine configuration and pricing](public_docs/engines-and-pricing.md): the
  exact model or version behind each engine ID, how certain each version is,
  pricing sources and how cost and latency scale.
- [Documents, references and licensing](public_docs/documents-and-licensing.md):
  corpus layout, attribution and licences, reference provenance, and how to
  add documents.

## View the results yourself

Set up once from the repository directory. Tested with Python 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Terminal report

```bash
python run_benchmark.py --report current
```

`current` is the newest scored run under `results/runs/`; in a fresh clone
that is the published run. The report is rendered from the saved result rows:
no credentials, no model calls. Add `--report-details` for document-family
breakdowns, or `--legs text --report-sort cer` to rank one leg by another
column. The score ledger is stored gzipped to stay under GitHub's file-size
limit; the harness and the viewer read it as is.

### Browser viewer

A hosted copy of the viewer needs no setup:
<https://ilsilfverskiold.github.io/ocr-engine-evaluation/>. It is published to
GitHub Pages from this repository on every push to `main`
(`.github/workflows/pages.yml`). To run it from this repository instead, follow
the steps below.

The viewer shows the comparison tables and charts, filtered by tier, document
type or engine type (general VLM, specialised OCR model, classical pipeline,
hosted document API; `engine_type` in `engines.py`), a breakdown by document
type with the metrics you pick, the engines near each leg's
best score ranked by cost, with cost and time at a chosen page volume, and for
every engine, page and leg the prompt sent, the full reply, the output beside
its reference, the scorer's verdict per item, and the geometry boxes drawn on
the page at word, line and block level.

1. Download the page images once. The HuggingFace dataset is public, so no
   token is needed:

   ```bash
   python scripts/download_data.py
   ```

2. Serve the repository directory, the folder that holds `results/` and
   `documents/`:

   ```bash
   python -m http.server 8000
   ```

3. Open <http://localhost:8000/viewer/viewer.html>.

The viewer is a static page that reads the repository's files, which is why it
needs the web server. Without step 1 it loads but shows no page images.
Markdown and HTML tables render with `marked` and `DOMPurify` from a CDN;
offline they show as source text.

### Saved calls

Every call behind the published scores is saved as
`results/ocr_outputs/reconciled_outputs/<tier>__<type>__<stem>/<engine>/<pass>.call.json`:
the prompt or schema sent, the recorded settings, the provider reply, the
output that was scored, cost and latency.

## Run the benchmark

### With the engines here

**1. Set up** from the repository directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp -n .env.example .env
python scripts/download_data.py
```

The download fetches the page images from the HuggingFace dataset; references
are in the repository. Add API keys for the engines you plan to run to `.env`;
`.env.example` names the variable for each engine.

**2. Preview** the calls without running them:

```bash
python run_benchmark.py --engines mistral_ocr_4_1,docling --legs text,markdown,tables --skip-judge --dry-run
```

| Option | What to choose |
| --- | --- |
| `--engines` | Comma-separated IDs from the engine table above. |
| `--legs` | `text,markdown,tables,key_value,schema,geometry`; omit for all supported legs. |
| `--tiers` | `easy,medium,hard`; omit for all tiers. |
| `--docs` | Page IDs such as `invoice_000,sroie_000`; omit for all selected pages. |

`--estimate` prints the expected cost; `--check` verifies API keys and Modal
deployments.

**3. Run** by removing `--dry-run`. Hosted engines incur charges:

```bash
python run_benchmark.py --engines mistral_ocr_4_1,docling --legs text,markdown,tables --skip-judge
```

The run saves every call, scores it and prints the comparison.
`--skip-judge` skips new judge calls, not OCR: key/value items that need a
ruling stay pending and the judged-text estimate is skipped. To judge, omit it
and set `GOOGLE_API_KEY`; key/value judge spend is capped by
`OCR_KV_JUDGE_BUDGET_USD`, default `$2`.

**4. Report your run** with the timestamp it printed:

```bash
python run_benchmark.py --report RUN_TIMESTAMP
```

The browser viewer opens the newest scored run, so it shows your run once it
is scored.

**Local and Modal engines.** Tesseract needs its system binary installed. A
Modal engine needs its app deployed before step 2; deploying may incur build
charges, and `--skip-smoke` skips the test inference:

```bash
modal setup
python scripts/setup_modal.py
python scripts/deploy_modal.py --apps glm_ocr --skip-smoke
```

### Add an engine

Every command reads the engine registry in `engines.py`, so an engine is one
runner and one registry entry.

1. **Runner.** Add `runners/<name>_runner.py` with a class that subclasses
   `OCRRunner` from `runners/base.py` and returns an `OCRResult` from `run()`.
   `runners/tesseract_runner.py` is the smallest example. An engine that
   takes prompts sets `accepts_task = True` and receives an `OCRTask` with the
   leg's prompt and, for schema, the JSON schema.
2. **Registry entry.** Add an `EngineSpec` to `engines.py`: `kind` (`local`,
   `cloud` or `modal`), the runner path, `required_env` for API keys,
   `cost_per_page`, `native_legs` answered from the default output, and
   `prompted_legs` that need the leg's prompt. `default=False` keeps it out of
   the default engine list.
3. **Self-hosted models** run on Modal: add `modal_apps/<name>_app.py`
   exposing `run_ocr_batch`, modelled on an existing app, and point the entry
   at it with `runner="runners.modal_runner:ModalRunner"`, `modal_app`,
   `modal_class`, `modal_app_file`, `gpu_type` and `batch_size`. Deploy it
   with `python scripts/deploy_modal.py --apps <name>`.
4. **Run it** as above with `--engines <name>`, then report it beside the
   published engines. The combined report refuses an engine and leg that
   appear in both runs.

   ```bash
   python run_benchmark.py --report RUN_TIMESTAMP,20261020T120222Z
   ```

A newer version of a measured model is a new engine entry; changing an
existing entry's model would invalidate its published results.

## License

The code is MIT licensed ([LICENSE](LICENSE)). The documents carry their
source datasets' terms; see
[documents-and-licensing.md](public_docs/documents-and-licensing.md).
