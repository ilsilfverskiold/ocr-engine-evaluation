# Engine configuration and pricing

Configuration and cost assumptions for the benchmark results in the
[README](../README.md). [Scoring rules](scoring-design.md).

## Engine versions and provenance

| Engine ID | Recorded model/version | Configuration |
| --- | --- | --- |
| `claude_sonnet_4_6` | `claude-sonnet-4-6` | Anthropic API; default transcription for text/Markdown/tables; dedicated schema/KV and pixel-frame geometry |
| `deepseek_ocr` | `deepseek-ai/DeepSeek-OCR` (v1); source pin `9f30c71f441d` | vLLM, Modal A100-80GB |
| `docling` | `2.118.1` (reconstructed environment; calls saved `docling-latest`) | Standard pipeline: Heron layout + TableFormer Accurate + RapidOCR/PP-OCRv6 |
| `gemini_flash` | `gemini-3.1-flash-lite` | Google API |
| `glm_ocr` | `zai-org/GLM-OCR`; source pin `ca5d8b3e287e` | vLLM, Modal A10 |
| `llamaparse` | v1 Balanced / Parse with LLM, August 10 run (dated release unconfirmed); schema `openai-gpt-4-1` | Hosted Parse/Extract APIs; recovered client packages `llama-cloud-services==0.6.94`, `llama-cloud==0.1.46` |
| `marker` | `marker-pdf==1.8.2` | Modal A100-80GB; 11 conversion workers; separate OCR geometry pass |
| `mineru` | `MinerU2.5-2509-1.2B`; source pin `1aa090b41282` | vLLM native batch, Modal A100-80GB |
| `mineru_pipeline` | `mineru[pipeline]==2.5.4`; Torch `2.6.0+cu124`; weights `ed6b654c018d` | DocLayout-YOLO + PP-OCR + UniMERNet/table models, Modal T4; geometry only |
| `mistral_ocr_4` | **4.1 conversion** (dated alias resolution); **4.0 schema** (saved reply) | Mistral OCR/Annotations API |
| `paddleocr_ppocrv5` | PP-OCRv5 server det `ca867c89` / rec `b26c3587`; PaddleOCR `3.7.0`; Paddle GPU `3.2.1` | Modal A10; geometry only |
| `paddleocr_vl` | `PaddleOCR-VL@7fa00a8c55b7`, pipeline v1 | Layout + vLLM recognition, Modal A100-80GB |
| `qwen3_vl` | `Qwen/Qwen3-VL-8B-Instruct`; source pin `0c351dd01ed8` | vLLM, Modal A100-80GB |
| `tesseract` | `5.5.2` | Local CPU |
| `textract_structured` | `AnalyzeDocumentModelVersion: 1.0` | AWS FORMS/TABLES; provider-reported version; no supplied-schema interface |
| `textract_text` | `DetectDocumentTextModelVersion: 1.0` | AWS text/word detection; provider-reported version |

### Version certainty

- **Mistral:** conversion replies record `mistral-ocr-latest`; the vendor's
  [alias history](https://docs.mistral.ai/resources/changelogs) resolves the run
  to OCR 4.1. Schema replies explicitly identify 4.0. The runner pins these
  separately; retired 4.0 may be unavailable for new schema runs.
- **Docling:** 2.118.1 is reconstructed from the installed environment, not
  independently pinned in each reply. Dependencies: docling-core 2.91.0,
  docling-ibm-models 3.13.3, docling-parse 7.11.0, RapidOCR 3.9.2. The pipeline
  uses Heron layout, TableFormer Accurate and Torch RapidOCR with PP-OCRv6
  detection/recognition and a PP-OCRv2 orientation classifier.
- **LlamaParse:** the configured v1 default is Parse with LLM / Balanced. Its
  hosted parser release and underlying model are unrecorded. Client SDK
  versions do not identify them. Schema extraction separately records GPT-4.1.
- **Open models:** source revisions identify configured weights. Calls that omit
  the revision do not independently verify the deployed hash. Full pins are in
  [the Modal apps](../modal_apps).

The published Sonnet and Gemini Markdown/table scores use default transcription
calls (92 Markdown pages and 42 table pages each). Fresh runs request dedicated
Markdown/table passes. Dependency ranges in `requirements.txt` are not a frozen
reproduction environment, so installation alone does not reproduce the listed versions.

## Pricing audit

Rates checked October 13, 2026. These are usage estimates, not reconciled invoices.
All 2,472 successful published inference records match the applicable token,
page or GPU-time formula; failed calls are excluded from the comparison.

| Service | Rate used | Source |
| --- | --- | --- |
| Sonnet 4.6 | $3 / million input tokens + $15 / million output tokens | [Anthropic](https://platform.claude.com/docs/en/about-claude/pricing) |
| Gemini 3.1 Flash-Lite | $0.25 / million input tokens + $1.50 / million output and thinking tokens | [Google](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.1-flash-lite) |
| Mistral OCR | $4 / 1,000 conversion pages; $5 / 1,000 annotated pages | [Mistral](https://docs.mistral.ai/models/ocr-4-1) |
| LlamaParse v1 Parse with LLM | 3 credits/page × $1.25 / 1,000 credits = $0.00375/page | [LlamaIndex](https://developers.llamaindex.ai/llamaparse/general/pricing/) |
| LlamaExtract v1 Multimodal | 20 credits = $0.025/page; cached extract-only can be 14 credits = $0.0175 | [LlamaIndex](https://developers.llamaindex.ai/llamaparse/general/pricing/) |
| Textract text | $0.0015/page for the first million monthly pages | [AWS](https://aws.amazon.com/textract/pricing/) |
| Textract TABLES + FORMS | $0.015 + $0.05 = $0.065/page for the first million; OCR included | [AWS](https://aws.amazon.com/textract/pricing/) |
| Modal A100-80GB / A10 / T4 | $0.000694 / $0.000306 / $0.000164 per GPU-second | [Modal](https://modal.com/pricing) |
| Local Docling / Tesseract | Unpriced; machine and electricity costs not measured | Local execution |

API costs use saved tokens or the selected service's page rate. Textract rates
are verified for US East; the calls do not record region, so the estimate must
be checked for other regions. LlamaIndex credit discounts/cache charges are not
captured in the saved responses. Free credits, taxes, subscriptions, storage,
failed calls and retries are excluded.

### Scaling cost

Multiply unrounded mean cost/page by page count for the same document mix and
configuration. Count unique inference passes, not score rows: shared conversion
costs once, while separate schema/KV/Markdown/geometry requests cost extra.
Evaluation judges are separate from production OCR cost.

Modal uses processing-only batch seconds divided by attempted pages.
**Startup/shutdown is excluded from the per-page multiplier.** For a complete
hosting estimate, add actual session overhead once per container session plus
CPU/RAM charges. Those resources were not measured, so GPU costs are not a full
hosting bill. [Modal billing](https://modal.com/pricing).

The benchmark uses standard API calls. Async batch pricing offers 50% discounts
for [Sonnet](https://platform.claude.com/docs/en/about-claude/pricing#batch-processing),
[Gemini](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.1-flash-lite) and
[Mistral OCR](https://docs.mistral.ai/resources/cookbooks/mistral-ocr-batch_ocr).
Concurrency alone does not earn this discount. The README projections do not
apply it. Textract's US rates after the first million monthly pages are $0.0006
for text and $0.05 for Forms + Tables. [Volume pricing](https://aws.amazon.com/textract/pricing/).

### Scaling latency

Multiply unrounded mean seconds/page by page count. The README uses the same
82 text-scored pages per engine; geometry-only configurations use 17 pages.
API/local projections assume sequential calls. GPU projections assume one warm
container at measured batch throughput. Startup/shutdown is excluded.
Parallelism, rate limits and document mix change elapsed time; these projections
are not measured production completion times. Durations round to minutes below
one day and hours above it.
