# Engine configuration and pricing

Configuration and cost assumptions for the benchmark results in the
[README](../README.md). [Scoring rules](scoring-design.md).

## Engine versions and provenance

| Engine ID | Recorded model/version | Configuration |
| --- | --- | --- |
| `claude_sonnet_4_6` | `claude-sonnet-4-6` | Anthropic API; default text transcription; dedicated Markdown/tables/schema/KV and pixel-frame geometry |
| `deepseek_ocr` | `deepseek-ai/DeepSeek-OCR` (v1); source pin `9f30c71f441d` | vLLM, Modal A100-80GB |
| `docling` | `docling==2.118.1` (recorded in every reply; local re-conversion 2026-10-17) | Standard pipeline: Heron layout + TableFormer Accurate + RapidOCR/PP-OCRv6 |
| `gemini_flash` | `gemini-3.1-flash-lite` | Google API; dedicated Markdown/table, schema/KV and geometry passes |
| `glm_ocr` | `zai-org/GLM-OCR`; source pin `ca5d8b3e287e` | vLLM, Modal A10 |
| `llamaparse` | v1 Balanced / Parse with LLM, August 10 run (dated release unconfirmed); schema `openai-gpt-4-1` | Hosted Parse/Extract APIs; recovered client packages `llama-cloud-services==0.6.94`, `llama-cloud==0.1.46` |
| `llamaparse_granular` | LlamaParse `latest`, `output_options.granular_bboxes=[word, line]`, cost_effective tier, run 2026-10-20 (17 pages, 8.6 s/page) | Hosted Parse API; geometry only |
| `marker` | `marker-pdf==1.8.2` | Modal A100-80GB; 11 conversion workers; separate OCR geometry pass |
| `mineru` | `MinerU2.5-2509-1.2B`; source pin `1aa090b41282` | vLLM native batch, Modal A100-80GB |
| `mineru_pipeline` | `mineru[pipeline]==2.5.4`; Torch `2.6.0+cu124`; weights `ed6b654c018d` | DocLayout-YOLO + PP-OCR + UniMERNet/table models, Modal T4; geometry only |
| `mistral_ocr_4_1` | `mistral-ocr-4-1` pinned for key_value and schema (replies record it); conversion `mistral-ocr-latest`, 4.1 by dated alias resolution | Mistral OCR/Annotations API |
| `paddleocr_ppocrv5` | PP-OCRv5 server det `ca867c89` / rec `b26c3587`; PaddleOCR `3.7.0`; Paddle GPU `3.2.1` | Modal A10; geometry only |
| `paddleocr_vl` | `PaddleOCR-VL@7fa00a8c55b7`, pipeline v1 | Layout + vLLM recognition, Modal A100-80GB |
| `qwen3_vl` | `Qwen/Qwen3-VL-8B-Instruct`; source pin `0c351dd01ed8` | vLLM, Modal A100-80GB |
| `tesseract` | `5.5.2` | Local CPU |
| `textract_structured` | `AnalyzeDocumentModelVersion: 1.0` | AWS FORMS/TABLES; provider-reported version; no supplied-schema interface |
| `textract_text` | `DetectDocumentTextModelVersion: 1.0` | AWS text/word detection; provider-reported version |

### Version certainty

- **Mistral:** the 93 conversion replies (2026-08-10) record only the alias
  `mistral-ocr-latest`. Per the vendor's
  [changelog](https://docs.mistral.ai/getting-started/changelog), the alias
  pointed to OCR 4.0 from 2026-06-23 and to OCR 4.1 from 2026-07-16, so those
  calls ran on 4.1. The key_value and schema annotation replies (collected
  2026-10-19) explicitly identify `mistral-ocr-4-1`, which the runner pins on
  every pass. The earlier 4.0 schema replies were replaced, not mixed in.
- **Docling:** every reply records `docling==2.118.1`; the model weights are
  the ones that package version resolves. Dependencies: docling-core 2.91.0,
  docling-ibm-models 3.13.3, docling-parse 7.11.0, RapidOCR 3.9.2. The pipeline
  uses Heron layout, TableFormer Accurate and Torch RapidOCR with PP-OCRv6
  detection/recognition and a PP-OCRv2 orientation classifier.
- **LlamaParse:** the configured v1 default is Parse with LLM / Balanced. Its
  hosted parser release and underlying model are unrecorded; the saved replies
  (page items, layout and job metadata) name no parser version. Client SDK
  versions do not identify them. Schema extraction separately records GPT-4.1.
- **Open models:** every Modal app has passed its `MODEL_REVISION` to the
  model loader since the first commit (2026-08-11); the strings saved with calls
  before 2026-10-13 omit the hash, so a call does not verify it on its own.
  Hugging Face commit history, checked 2026-10-19: the pinned commit was the
  repository head on every collection date for MinerU (head since 2026-04-09),
  DeepSeek-OCR (since 2025-11-04), Qwen3-VL (since 2025-10-15) and PaddleOCR-VL
  (2026-08-08; recorded per call). GLM-OCR's pin was head from 2026-05-19 until
  2026-09-11, when `config.json`, `tokenizer_config.json` and a new
  `processor_config.json` changed with no weight change; the 100 GLM calls of
  2026-10-06/07 postdate that and rely on the deployment passing the pin.
  PaddleOCR-VL's layout model is fetched by the `paddleocr` pipeline at container
  start without a pin and is saved as `layout=unknown`. Full pins are in
  [the Modal apps](../modal_apps).
- **Marker:** `marker-pdf==1.8.2` resolves `surya-ocr` to 0.14.6 or 0.14.7 at
  image build (not recorded); both ship the same date-stamped checkpoints
  (detection 2025_05_07, recognition 2025_05_16, layout, table and OCR-error
  2025_02_18), so the weights are determined.
- **Hosted LLMs:** `gemini-3.1-flash-lite` is Google's stable model code with no
  dated variant published, and every reply's `model_version` echoes it;
  `claude-sonnet-4-6` is the complete Anthropic model ID with no dated snapshot,
  and every reply echoes it.
- **Tesseract:** 5.5.2 is the binary; the `eng` traineddata variant is not
  recorded.

Sonnet and Gemini use separate dedicated requests for Markdown (92 pages each)
and tables (42 pages each). Exact prompts and provider settings are saved with
each call. Dependency ranges in `requirements.txt` are not a frozen reproduction
environment, so installation alone does not reproduce the listed versions.

## Pricing audit

Rates checked October 13, 2026. These are usage estimates, not reconciled invoices.
All 2,740 successful published inference records match the applicable token,
page or GPU-time formula; failed calls are excluded from the comparison.

| Service | Rate used | Source |
| --- | --- | --- |
| Sonnet 4.6 | $3 / million input tokens + $15 / million output tokens | [Anthropic](https://platform.claude.com/docs/en/about-claude/pricing) |
| Gemini 3.1 Flash-Lite | $0.25 / million input tokens + $1.50 / million output and thinking tokens | [Google](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.1-flash-lite) |
| Mistral OCR | $4 / 1,000 conversion pages; $5 / 1,000 annotated pages | [Mistral](https://docs.mistral.ai/models/ocr-4-1) |
| LlamaParse v1 Parse with LLM | 3 credits/page × $1.25 / 1,000 credits = $0.00375/page | [LlamaIndex](https://developers.llamaindex.ai/llamaparse/general/pricing/) |
| LlamaParse granular boxes (`llamaparse_granular`) | same 3 credits/page = $0.00375/page; no extra credits listed for `granular_bboxes` |  [LlamaIndex](https://developers.llamaindex.ai/llamaparse/general/pricing/) |
| LlamaExtract v1 Multimodal | 20 credits = $0.025/page; cached extract-only can be 14 credits = $0.0175; `cite_sources` lists no extra credits | [LlamaIndex](https://developers.llamaindex.ai/llamaparse/general/pricing/) |
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
costs once, while separate Markdown/table/schema/KV/geometry requests cost extra.
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

### Scaling latency in parallel

Hosted APIs and Modal accept many pages at once, up to an account limit. The
README's parallel column and the viewer's best-per-leg panel use each
provider's default limit for a standard paid account with no increase
requested, declared as `default_limits` in `engines.py` (checked 2026-11-02):

| Engine | Default limit | Basis |
| --- | --- | --- |
| `textract_text` | 25 requests per second | AWS default quota for DetectDocumentText, US East (N. Virginia). [Quotas](https://docs.aws.amazon.com/general/latest/gr/textract.html) |
| `textract_structured` | 10 requests per second | AWS default quota for AnalyzeDocument, US East (N. Virginia). [Quotas](https://docs.aws.amazon.com/general/latest/gr/textract.html) |
| `claude_sonnet_4_6` | 1,000 requests, 2M input tokens and 400k output tokens per minute | Anthropic Start tier, Sonnet 4.x. [Rate limits](https://platform.claude.com/docs/en/api/rate-limits) |
| `llamaparse`, `llamaparse_granular` | 5 parse and 5 extract jobs at once | LlamaCloud Starter plan. [Pricing](https://www.llamaindex.ai/pricing) |
| Modal engines | 10 GPUs at once, one per container | Modal Starter plan. [Pricing](https://modal.com/pricing) |
| `gemini_flash` | Estimate: 4,000 requests and 4M input tokens per minute, 150,000 requests per day | Google shows each project's limits only in [AI Studio](https://ai.google.dev/gemini-api/docs/rate-limits). Tier 1 values for Gemini 3.1 Flash-Lite from an undated [third-party table](https://www.scriptbyai.com/gemini-api-rate-limits/); Tier 1 needs only a linked billing account |
| `mistral_ocr_4_1` | Estimate: 1 request per second | Mistral publishes no default; accounts start on the lowest limits and prepaying credits does not raise them ([help center](https://help.mistral.ai/en/articles/698531-why-am-i-hitting-api-rate-limits-and-how-do-i-increase-them)). A third-party guide reports 5 requests per second, which would cut its time to a fifth |
| `tesseract`, `docling` | Your CPU cores | Local; not estimated |

Pages per second is the tightest of these: the request rate; each
token-per-minute limit divided by the engine's mean tokens per page for that
pass, as recorded in the inference ledger; or the jobs or GPUs at once divided
by seconds per page. Time is pages divided by that rate; a daily request cap
binds only once the pages exceed one day's allowance, and then sets the time.
Cost does not change. Check your own limits in each provider's console; the
estimates in particular can be well off.

- Anthropic's Start tier also caps spend at $500 a month, about 33,000 Sonnet
  text pages. Larger volumes need the Build or Scale tier, whose rate limits
  are higher.
- Textract's limits are lower outside US East and US West (Oregon): 5 requests
  per second in Europe (Ireland) and 1 in most other regions. Its asynchronous
  operations allow more.
- Every limit can be raised on request; Modal's Team plan allows 50 GPUs.
- Ramp-up, retries and Modal container startup are excluded.
