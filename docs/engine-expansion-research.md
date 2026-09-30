# OCR engine expansion: research and integration plan

Research date: 2026-09-30. Research only: no engines built, dependencies installed,
deployments started, OCR requests made, or judges run.

This is the handoff for a separate implementation agent. Commit and log future
work in the `ocr-benchmark` Git repository only. The user explicitly deferred
judging; do not start judges as part of implementation or smoke testing.

## Scope and existing baseline

The target is this `ocr-benchmark/` repository, not the older parent project.
Read the parent's `.claude/skills/modal-inference-engineer/SKILL.md` and all
three references before researching.

The requested list reduces to **16 integrations**: RapidOCR was repeated;
LightOnOCR 2 and LightOnOCR-2-1B refer to one family/checkpoint choice. Use
`lightonai/LightOnOCR-2-1B` initially. Its bbox variants locate embedded images,
which does not establish word geometry capability.

The reference run is `20260810T130609Z`, with 14 engines over 93 page images.
Evidence is in `results/runs/*_inference.jsonl`, not just today's code:

| Existing row | Persisted identity | Consequence |
| --- | --- | --- |
| `paddleocr_vl` | 93 records: `PaddlePaddle/PaddleOCR-VL@7fa00a8c55b7 pipeline v1` | **1.6 has not been tested.** Add a distinct row. |
| `deepseek_ocr` | `deepseek-ai/DeepSeek-OCR` | OCR-2 is a new version. |
| `mineru` | `opendatalab/MinerU2.5-2509-1.2B` through `mineru-vl-utils` | The 4.0 application is a different pipeline, not a checkpoint substitution. |
| `mistral_ocr_4` | All 158 inference records say `mistral-ocr-latest`; inspected saved provider payload also says latest | Today's runner pins `mistral-ocr-4-0`, but that cannot retroactively identify the historical service revision. Add explicit 4.1; retain the old provenance. |

Corpus: receipts and invoices; bank statements, shipping documents, tax forms,
medical documents, payslips, leases, photographed receipts; forms, handwriting,
legacy scans, financial tables, charts, newspapers, legal documents, and reports.
The six legs are text, markdown, tables, key_value, schema, and geometry.
There is no dedicated formula, multilingual, SVG, or multipage-continuity score.
Those vendor capabilities are motivations for future tests, not claims this
corpus can establish. Expected strengths below are research hypotheses, not
measured rankings on our corpus.

## Implementation and benchmark fit

T/M/B/G below mean text, markdown, tables, geometry. Geometry must retain its
actual granularity: words, lines, paragraphs, or regions. Do not manufacture
word boxes from an image-region box. Eligibility remains provisional until the
adapter's output contract is checked on actual responses.

| Requested engine / proposed ID | Implementation route | Most informative existing cases / proposed legs |
| --- | --- | --- |
| TeleOCR 1.2B / `teleocr` | Official TeleOCR layout and crop-recognition pipeline, including its vLLM plugin | Photographed receipts, distorted scans, tables and charts; T/M/B, region geometry |
| RapidOCR / `rapidocr` | Persistent local `rapidocr.RapidOCR`, ONNX Runtime CPU initially | Receipts, invoices, scanned text; T/G. No native table structure or schema interface |
| OvisOCR2 0.8B / `ovisocr2` | Author's native vLLM list-batch recipe | Dense text, legal pages, reports and financial tables; T/M/B |
| PaddleOCR-VL 1.6 / `paddleocr_vl_1_6` | Reuse Paddle pipeline adapter with explicit v1.6 model, pipeline, and service name | Tables, charts, forms, distorted documents; T/M/B, spotting geometry if exposed |
| DeepSeek-OCR-2 / `deepseek_ocr_2` | Reuse serving lifecycle and grounding parser; new checkpoint/processor contract | Reading order, dense pages, tables and legacy scans; T/M/B, grounded-region geometry |
| Surya OCR 2 / `surya_ocr_2` | Current Surya predictors and inference manager attached to a local vLLM server | Forms, financial tables, newspapers; T/M/B and native block geometry |
| MinerU 4.0 / `mineru_4_standard` | Full 4.x pipeline with explicit Standard tier, Torch small models and vLLM | Financial tables, reports, newspapers and layouts; T/M/B, geometry after schema validation |
| Mistral OCR 4.1 / `mistral_ocr_4_1` | Parameterize existing Mistral runner; request `mistral-ocr-4-1` | Broad API comparison: invoices, tables, reports; T/M/B, paragraph geometry; schema through Annotations later |
| Google Enterprise Document OCR / `google_document_ocr` | Document AI processor-version `process` API | Receipts, forms, handwriting, legacy scans; T/G. Form Parser and Layout Parser are separate products |
| Mathpix OCR / `mathpix` | `/v3/text` for corpus images; `/v3/pdf` with polling for future PDFs | Handwriting, scientific content and financial tables; T/M/B/G, with separate layout/word-box configurations |
| OpenDoc-0.1B / `opendoc` | `OpenOCR(task='doc', use_layout_detection=True)` | Low-resource structured parsing: invoices, tables and reports; T/M/B, layout geometry |
| LightOnOCR-2-1B / `lighton_ocr_2` | Image-only message through supported vLLM or Transformers 5 | Receipts, forms, multi-column pages and scans; T/M/B |
| Apple OCR / `apple_vision` | Local Swift Vision helper invoked by Python | On-device receipt/scanned-text baseline; T/G on this Mac |
| Chandra OCR 2 / `chandra_ocr_2` | Official Chandra manager with vLLM and `ocr_layout` | Handwriting, checkboxes/forms, financial tables and fine print; T/M/B, layout geometry |
| dots.mocr / `dots_mocr` | Official Dots parser over vLLM, layout JSON conversion | Charts, complex layout, financial tables, document grounding; T/M/B/G |
| Datalab OCR / `datalab_ocr` | Current hosted Convert API, explicit mode | Hosted structured document parsing versus local Chandra/Surya; T/M/B, optional paid word geometry and later extraction |

## Model contracts and pitfalls

### TeleOCR

Artifact: `XingChen-AGI/TeleOCR`, Apache-2.0, BF16; Transformers path uses
remote code. Upstream retains older `StarDoc-AI`/NaviDC names in examples;
resolve to a pinned local snapshot. Use the document pipeline, not only its
text-crop demonstration. It provides detection/segmentation layout modes,
OTSL table conversion and distinct recognition tasks. The repository supports
sync and async vLLM backends and registers a plugin. Its dependency ranges are
Python 3.10–3.12, Torch 2.8, Transformers 4.57, vLLM 0.11; isolate this image.
Repository configuration uses a 16,384 engine context; the model config declares
128,000 positions. The card's 4,096 output example is a crop example, not a
universal page limit. Keep native crop scheduling; verify warm reuse and
pipeline output limits before implementation. [Model](https://huggingface.co/XingChen-AGI/TeleOCR),
[pipeline](https://github.com/caipeng328/TeleOCR),
[dependencies](https://github.com/caipeng328/TeleOCR/blob/main/pyproject.toml),
[configuration](https://github.com/caipeng328/TeleOCR/blob/main/TeleOCR/config.py).

### RapidOCR

Install the current `rapidocr` package with ONNX Runtime, not a legacy API
assumed from `rapidocr_onnxruntime`. Native results expose `boxes`, `txts`,
`scores`, and optional word results. Pin the package, detector, recognizer,
dictionary and model hashes; RapidOCR names a toolkit rather than one model.
Keep a persistent instance and record actual ONNX providers and thread counts.
Image input is the native boundary; PDF support belongs in page rendering.
There is no prompt/context budget. Preserve quadrilaterals and distinguish
line boxes from true word results. CPU is the first baseline; GPU is a later
throughput experiment. Code is Apache-2.0; retain model attribution.
[Repository](https://github.com/RapidAI/RapidOCR),
[API](https://rapidai.github.io/RapidOCRDocs/main/quickstart/).

### OvisOCR2

Artifact `ATH-MaaS/OvisOCR2`, Apache-2.0, BF16, Qwen3.5 architecture. The author's
recipe pins vLLM 0.22.1, `gdn_prefill_backend='triton'`, disables thinking,
uses temperature 0 and a 16,384 output cap, and batches a list in `generate`.
Its exact prompt requests original-language Markdown, HTML tables and LaTeX;
copy it from the card rather than substituting our general transcription prompt.
Pixel bounds are 448²–2880². Preserve image tags/raw output before any optional
cleanup; image-region coordinates are not text boxes. Config declares 262,144
positions, but reserve a practical measured context, not the entire ceiling.
Native architecture support avoids a custom model-code requirement for this
vLLM path. [Author recipe](https://huggingface.co/ATH-MaaS/OvisOCR2).

### PaddleOCR-VL 1.6

Artifact `PaddlePaddle/PaddleOCR-VL-1.6`, Apache-2.0, BF16. Use the full
`PaddleOCRVL` pipeline with `pipeline_version='v1.6'`, matching recognition
model/service name and layout model. Keep list prediction and pipeline queues.
The existing app explicitly chooses v1; changing only its package version is
insufficient. Preserve chart, table and seal output. Reuse the separate Paddle
and vLLM environments if compatibility still requires them; reserve VRAM for
both stages. Config declares 131,072 positions; crop budgets remain a separate
setting. Pin all layout/preprocessing assets, not just the VLM. Transformers
examples involve custom model code; the documented pipeline owns prompting.
[Model](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6),
[pipeline documentation](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html).

### DeepSeek-OCR-2

Artifact `deepseek-ai/DeepSeek-OCR-2`, Apache-2.0, BF16. Official Markdown prompt
is `<image>\n<|grounding|>Convert the document to markdown.`; plain OCR is a
separate task. Official processing uses a 1024 base image and up to six 768
crops, corresponding to 256 + up to 6×144 visual tokens. The config still
declares **8,192 total positions**. Calculate the output allowance after image
and prompt tokens; do not blindly reuse either v1's 2,048 benchmark cap or an
8,192 generation example. Current vLLM implements `DeepseekOCR2ForCausalLM`,
so the old custom vLLM 0.8.5 wheel is not the only route. Verify required
repetition/logits settings and preserve grounding markup before cleanup.
[Author repository](https://github.com/deepseek-ai/DeepSeek-OCR-2),
[vLLM implementation](https://docs.vllm.ai/en/v0.29.0/api/vllm/model_executor/models/deepseek_ocr2/).

### Surya OCR 2

Artifact `datalab-to/surya-ocr-2`, approximately 650M parameters. This is the
new VLM-backed Surya, not the older recognition stack used by historical Marker.
Use `SuryaInferenceManager` and `RecognitionPredictor`; attach via
`SURYA_INFERENCE_URL` to a server started in Modal's enter hook instead of
auto-launching Docker inside Modal. Default full-page and optional
layout-plus-block modes share a result schema containing ordered HTML blocks,
labels and geometry. Compare these modes on dense financial tables before
choosing one global benchmark recipe. Full HTML table recognition matters:
geometric cell intersections alone are not a transcription. Model config has
262,144 positions; current settings document an 8,192 full-page generation
budget. Code is Apache-2.0; weights have modified OpenRAIL-M terms.
[Repository](https://github.com/datalab-to/surya),
[settings](https://github.com/datalab-to/surya/blob/master/surya/settings.py).

### MinerU 4.0

4.0 is an application/pipeline release. Its tiers are Flash, Basic, Standard
and Advanced; Standard and Advanced share models but spend different compute.
Recommend explicit Standard initially. Use the supported 4.x SDK/stateless
conversion or V1 service, preserving the structured result and native batching.
The 2.5 helper and old endpoints are not the 4.0 integration. Explicitly select
Torch small models and vLLM; cache and verify the 4.0 model bundle and VLM.
Record the exact resolved artifacts, package release, backend and tier.
Avoid cached document-library results when timing inference, and request all
pages for future PDFs. Upstream planning guidance is at least 16 GB system RAM;
GPU needs depend on the whole pipeline. Artifact licenses, VLM revision,
processor/context limits and exact batch API still need locking during build.
[Release](https://github.com/opendatalab/MinerU/releases/tag/mineru-4.0.0-released),
[tiers](https://github.com/opendatalab/MinerU/blob/master/docs/en/usage/tiers.md),
[model setup](https://github.com/opendatalab/MinerU/blob/master/docs/en/usage/model_source.md).

### Mistral OCR 4.1

Use `mistral-ocr-4-1` with the existing `MISTRAL_API_KEY`. Parameterize the
existing runner while preserving its old identity. Request HTML tables and
`include_blocks=True`; reintegrate table placeholders so text and Markdown
do not lose cell contents. Save the entire provider reply, returned model,
usage, dimensions and paragraph boxes. Word confidence does not mean word
coordinates. Schema extraction uses Annotations, not a free-text OCR prompt.
Published list price is $4/1,000 OCR pages or $5/1,000 annotated pages; verify
account billing and keep asynchronous Batch pricing/latency separate.
Cloud internals, dtype and model context are not customer-controlled.
[Model](https://docs.mistral.ai/models/ocr-4-1),
[response contract](https://docs.mistral.ai/studio/document-processing/basic_ocr).

### Google Enterprise Document OCR

Use Document AI, with ADC/service-account authentication, project, region,
processor ID and explicit processor version. A Gemini API key is insufficient.
Send corpus images through online processing; retain text anchors, pages,
tokens, normalized polygons and confidence. Start with T/G: table cells and
schema extraction from other Document AI processors must not be attributed to
Enterprise OCR. Explicitly configure quality scores, native PDF parsing and
any premium math/checkbox options; bill those options separately. Region
availability controls which version can run. No local model or prompt budget.
[Enterprise OCR documentation](https://docs.cloud.google.com/document-ai/docs/enterprise-document-ocr).

### Mathpix

Use `app_id`/`app_key`, image multipart upload and `/v3/text`; use asynchronous
`/v3/pdf` only for actual PDFs. Preserve Mathpix Markdown, HTML and table data,
and enable line/word data where applicable. Important constraint:
`include_word_data=True` cannot be combined with `enable_document_layout=True`.
Use separate explicitly costed passes/configurations for layout and word
geometry. Avoid base64 size limits by using documented uploads. Preserve page
headers and footers, and convert formulas without deleting their content.
Strong motivation for handwriting and math, but our current benchmark has no
dedicated formula score. Pin request options and save returned service metadata;
do not invent an immutable public model version or a local token budget.
[Image API](https://docs.mathpix.com/reference/post-v3-text),
[geometry examples](https://docs.mathpix.com/guides/image-ocr).

### OpenDoc-0.1B

Use `openocr-python==0.1.5`'s full document path. It combines PP-DocLayoutV2
with the table-capable UniRec-0.1B recognizer. The 0.1B label does not describe
the whole resident pipeline. Supports Chinese/English and image/PDF parsing;
save native JSON and Markdown. The source recipe specifies Torch 2.6,
Transformers 4.49 and Paddle GPU 3.2/CUDA 12.6: isolate dependencies from other
engines. CPU is possible; compare CPU and L4 before deciding. Pin recognizer
and layout assets and inspect actual crop batching, output limits, dtype and
weight licenses during the build. Code metadata is Apache-2.0. Do not substitute
the generic OCR task for `task='doc'`.
[Document recipe](https://github.com/Topdu/OpenOCR/blob/main/docs/opendoc.md),
[package metadata](https://github.com/Topdu/OpenOCR/blob/main/pyproject.toml).

### LightOnOCR 2

Use `lightonai/LightOnOCR-2-1B`, Apache-2.0, BF16 on CUDA. The official message
contains **only the image**, with the processor chat template. Transformers 5
and vLLM are supported paths. Recommended rendering is 200 DPI, longest edge
1540, keeping aspect ratio. This model-specific preprocessing is different
from arbitrary global downsampling. The vLLM example disables prefix and
multimodal processor caching, uses one image, 4,096 output tokens, temperature
0.2 and top-p 0.9. Record that recipe before testing deterministic alternatives.
Text config declares 16,384 positions. Check dense-page truncation and fine
print rather than treating a short example output cap as a hard limit.
[Model and recipe](https://huggingface.co/lightonai/LightOnOCR-2-1B).

### Apple OCR

Use local Vision `VNRecognizeTextRequest` with accurate recognition, explicit
languages/correction policy and word-range bounding boxes. Flip Apple's
bottom-left coordinate origin into the benchmark's top-left convention.
Record OS build, hardware, Vision revision and options. A Swift helper can
return JSON to the Python runner without a cloud service. No API charge does
not imply zero hardware cost. The machine inspected is arm64 macOS 15.6.
`RecognizeDocumentsRequest` adds paragraphs, lists and tables but Apple's
availability metadata requires macOS 26.0; it would be a separately identified
configuration, unavailable on this host. No OS upgrade is proposed here.
[Vision](https://developer.apple.com/documentation/vision/vnrecognizetextrequest),
[document API](https://developer.apple.com/documentation/vision/recognizedocumentsrequest).

### Chandra OCR 2

Artifact `datalab-to/chandra-ocr-2`, 4B, Qwen3.5 architecture. Use official
`InferenceManager(method='vllm')`, `BatchInputItem(prompt_type='ocr_layout')`
and `manager.generate(batch)`; reuse server lifecycle code, not Surya's parser.
Preserve HTML, Markdown, block boxes and image captions separately from
transcribed text. Enable header/footer inclusion: the CLI otherwise excludes
them. Settings use 192 DPI and 12,384 maximum output tokens; model config
declares 262,144 positions, not a sensible default KV reservation.
Use original precision for the baseline. Code is Apache-2.0, while weights
carry modified OpenRAIL-M terms distinct from Surya's. Pin the model-specific
runtime stack and processor settings before deployment.
[Model usage](https://huggingface.co/datalab-to/chandra-ocr-2),
[repository](https://github.com/datalab-to/chandra),
[settings](https://github.com/datalab-to/chandra/blob/master/chandra/settings.py).

### dots.mocr

Canonical HF location now resolves to `dots-studio/dots.mocr` (3B, MIT);
older examples use `rednote-hilab/dots.mocr`. Choose the document model, not
the separate SVG variant. Use official `prompt_layout_all_en` and parser:
ordered layout JSON carries Markdown text, HTML tables, formulas and boxes.
The author recommends vLLM with string chat-template content and remote code;
preserve the custom processor. Config declares 131,072 positions and the HF
example allows 24,000 generated tokens; measure visual-token usage before
choosing deployment caps. Keep headers/footers rather than adopting their
benchmark-specific deletion. SVG/chart understanding is additional capability;
our chart schema task must still ask for the benchmark's fields.
[Model](https://huggingface.co/dots-studio/dots.mocr),
[parser repository](https://github.com/rednote-hilab/dots.mocr).

### Datalab hosted OCR

Recommend the current `/api/v1/convert` API, not deprecated `/ocr` or `/marker`.
Make the mode explicit; start with Balanced and identify Accurate separately
if evaluated. Use `DATALAB_API_KEY`, upload, persist the job ID, poll with a
deadline, and inspect `success` even when status is complete. Follow signed
`result_url` downloads without forwarding the API key. Preserve HTML/JSON and
request Markdown as appropriate; word boxes require HTML and a paid option.
Keep headers/footers; record mode, extras, failed pages, and cost breakdown
(documented in cents). Do not request vendor eval rubrics. Hosted models are
not assumed identical to public Chandra/Surya checkpoints. Live service model
revision/pinning remains an explicit reproducibility limitation.
[Current conversion guide](https://documentation.datalab.to/docs/recipes/conversion/conversion-api-overview),
[API contract](https://documentation.datalab.to/api-reference/convert-document).

## Snapshot identities and unresolved build decisions

HF API revisions observed on the research date (metadata only; weights were
not downloaded). Resolve code/processor/config from the same revision at build:

| Artifact | Revision |
| --- | --- |
| XingChen-AGI/TeleOCR | `e92585356c0d0b7b7a65938f3da035c6593cc9a6` |
| ATH-MaaS/OvisOCR2 | `1fc9221b7823a371d6e97f92d527cc847e24e107` |
| PaddlePaddle/PaddleOCR-VL-1.6 | `c5630abae1d940eafe0697512a0325494b02ab42` |
| deepseek-ai/DeepSeek-OCR-2 | `aaa02f3811945a91062062994c5c4a3f4c0af2b0` |
| lightonai/LightOnOCR-2-1B | `c97bd377f04481830395218fa8951df9deaba756` |
| datalab-to/surya-ocr-2 | `3b3d4cdf88d6928b0acdc75181b13206ea67c4a3` |
| datalab-to/chandra-ocr-2 | `af93b47dba1b47b6640c86ccf487ed2260ab9a09` |
| dots-studio/dots.mocr | `e539fbb52280393adc081b289ec597430a0f9031` |

Config ceilings above came from each repository's `config.json`. They are
architecture declarations, not validated simultaneous image/output capacity.
Before building, finish each dependency lock, pipeline/model asset hashes,
custom-code review, processor pixel/token limits and supported stop/repetition
settings. Anything not established above is unresolved, not an invitation to
copy a similarly named model's defaults.

## GPU and measurement plan

These are **candidate experiments**, not measured hardware recommendations:

| Family | First candidates | Reason |
| --- | --- | --- |
| OvisOCR2, LightOn, Surya 2, TeleOCR | L4 / A10; L40S if useful batches need it | Compact weights; vision tensors, crops and long outputs determine actual fit |
| DeepSeek 2, dots.mocr | L4 / A10 at batch 1, then L40S | 3B weights may fit 24 GB, but dense-page decode and batching need measurement |
| Chandra 2 | A10 at batch 1, then L40S | 4B plus long structured output; assess full input distribution |
| Paddle 1.6, MinerU 4 | A10 fit probe, then L40S; A100 only if justified | Several stages/models share memory; profile CPU and layout work too |
| OpenDoc | CPU, then L4 | Small recognition model; layout and crop orchestration may dominate |
| RapidOCR, Apple | Local CPU / Apple Vision | Establish useful local baseline without Modal GPU allocation |

Approximate BF16 weight storage alone is 2 bytes/parameter: 1.2B ≈2.4 GB,
3B ≈6 GB, 4B ≈8 GB. These are lower bounds excluding KV cache, vision/runtime
allocations and other pipeline models. Peak memory must be measured.

Start one replica, one active request/native batch item, original precision.
Warm models and servers before timing; sweep 1/2/4/8 only where native APIs
support it. vLLM owns continuous batching; native pipelines own stage queues.
One explicit Modal batch call must return its occupied GPU interval, attempted
and successful counts, actual GPU and batch ID. Count retries/failures and
record startup separately. Use cost/successful result, p95 request latency and
quality together. Never use multi-replica wall time as GPU cost.

Observed Modal GPU-only rates: L4 $0.000222/s, A10 $0.000306/s, L40S
$0.000542/s, A100-80GB $0.000694/s. CPU/RAM, startup and idle may add charges;
no per-page cost or speedup is predicted here. [Modal pricing](https://modal.com/pricing).

## Changes needed when implementation begins

1. Add specs to `engines.py`, runners under `runners/`, and GPU apps under
   `modal_apps/`. Preserve old engine IDs and deployments. Share lifecycle,
   transport, batching metrics, image handling and output adapters; retain
   separate dependency images and task recipes where required.
2. Parameterize Mistral versions. Reuse Paddle orchestration with explicit
   pipeline versions and DeepSeek's serving/grounding utilities. Share vLLM
   infrastructure across eligible models, not a universal OCR prompt or cap.
3. Preserve native responses, HTML tables and geometry before deriving text.
   Test table placeholders, coordinate transforms, spans, block order,
   headers/footers, empty output, truncation and partial failures.
4. Add a collection-only mode if schema/key_value outputs should be obtained
   now. Current `--skip-judge` **drops those inference legs**; it is not merely
   a scoring toggle. Add explicit later scoring from stored outputs: current
   `--report` only aggregates stored scores and `--resume` skips completed
   inference, so neither supplies that deferred-judge workflow.
5. Preserve `20260810T130609Z` and create an extension run referencing it.
   A comparison/merge must retain source-run IDs, model/configuration identity,
   corpus hashes and scoring method. Directly resuming the published baseline
   currently overwrites its manifest with the new engine/leg selection.
   Do not mix new deterministic Markdown/TEDS scores with old judged scores
   as though they were the same metric.
6. Corpus selection needs care: the 11 silver-text pages have no trustworthy
   deterministic text reference. Collect their OCR output even while their
   text quality remains unscored. Do not let GT filtering silently remove
   difficult pages from a supposed 93-page collection.

The parent `.env` contains a Mistral key. No Mathpix or Datalab keys or Google
Document AI configuration were found there; this does not rule out ADC or
credentials configured elsewhere. Add missing configuration to `.env.example`
without copying secrets. Apple structured-document testing needs a macOS 26
host; the ordinary Vision OCR baseline can run on the current Mac.

Proposed build order: shared integration/collection foundations; Mistral 4.1,
RapidOCR and Apple; OvisOCR2/LightOn/DeepSeek 2; Paddle/TeleOCR/OpenDoc/MinerU;
Surya/Chandra/dots; remaining cloud adapters. Validate with the same small
cross-category set (receipt, photographed receipt, dense financial table,
form, handwriting, newspaper/report, legal fine print and chart), then collect
the unchanged corpus. No judge runs are part of this phase. GPU sweeps and
paid corpus runs remain future work after the implementation is reviewed.

## Implementation acceptance checklist

- All 16 requested integrations have distinct, reproducible identities; the
  two duplicate list entries do not create duplicate inference calls.
- New adapters use the existing registry and result schema. Optional local
  packages, credentials, OS requirements and Modal deployments are checked
  without preventing unrelated engines from running.
- Offline contract tests cover native table extraction, geometry transforms,
  asynchronous API failure/polling behavior, and batch result ordering.
- Collection and deferred scoring are separate operations. A no-judge test
  must fail on any attempted benchmark-judge call, including cached/missing-GT
  paths. Vendor extraction used as an engine is recorded separately.
- Existing reference results and manifests remain intact. A combined report
  retains configuration and scoring provenance and never silently averages
  judged scores with deterministic fallback scores.
- The handoff records exact files/settings, validation actually performed,
  unresolved runtime issues, and build/deploy/smoke-test/benchmark commands.
  Candidate GPUs become recommendations only after measured probes.
