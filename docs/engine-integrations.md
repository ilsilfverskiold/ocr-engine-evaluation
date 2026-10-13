# Additional engine integrations

The registry has 18 opt-in engines. The 16 in the table below are implemented
but have no measurements in the saved comparison in the published comparison.
The other two, `paddleocr_ppocrv5` and `mineru_pipeline`, already have 17 geometry
results each in `results/current.zip`, with no inference-error rows. They have
not been scored on their other eligible legs. The saved PP-OCRv5 calls return
word boxes; MinerU's pipeline calls mix text lines and table regions, labeled
`mixed_line_block` by the current scorer.

This guide describes source configuration, not verified runtime performance.
`engines.py` and each app/runner are authoritative for model revisions, prompts,
dependencies, task routes and initial hardware. Presence of an app file does
not establish that it is deployed. Check your Modal deployment before running an app.

Use the same six scorers and accepted references as the published baseline.
First inspect `--dry-run` and `--estimate`, then explicitly choose the engine and
pages for inference. A Modal deployment or smoke test can spend money.

```bash
python scripts/deploy_modal.py --apps rapidocr --skip-smoke
python run_benchmark.py --engines rapidocr --collect-only
python run_benchmark.py --rescore current,NEW_TIMESTAMP --skip-judge
```

Publishing a new combined result is separate: `python scripts/publish_results.py
SCORED_TIMESTAMP`. Offline rescoring does not validate an unrun integration.

## Initial hardware and task routes

GPU choices below are initial fit probes, not established optimal hardware or
throughput. Test the largest input and full output budget at declared concurrency;
then change one variable at a time. Capture startup, steady-state time, peak memory,
finish reason, exact model/revision and prompt. Costs must use actual batch timing
and GPU allocation. An unmeasured zero in the registry is not a free runtime.

| Engine | Runs on | Passes for `--legs all` | Legs entered | Geometry granularity |
| --- | --- | --- | --- | --- |
| `mistral_ocr_4_1` | Mistral API | default + schema | T M B G, schema (Annotations) | block |
| `google_document_ocr` | Document AI (ADC) | default | T G | word (tokens) |
| `mathpix` | Mathpix API | default + geometry (word-data config) | T M B, G | word |
| `datalab_ocr` | Datalab API | default + schema (Extract) | T M B, schema | none yet (see the checks below) |
| `paddleocr_vl_1_6` | Modal **A10** → L40S | default | T M B G | block |
| `opendoc` | Modal **CPU** (4 cores, 8 GiB) → L4 | default | T M B G | block |
| `deepseek_ocr_2` | Modal **L4** → A10 → L40S | default(`Free OCR.`) + grounding(M+B+G) + KV + schema | T, M B G, KV, schema | block (grounding boxes) |
| `dots_mocr` | Modal **L4** → A10 → L40S | default + KV + schema | T M B G, KV, schema | block |
| `surya_ocr_2` | Modal **L4** → A10 → L40S | default + KV + schema | T M B G, KV, schema | block |
| `chandra_ocr_2` | Modal **A10** (concurrency 1) → L40S | default + KV + schema | T M B G, KV, schema | block |
| `mineru_4_standard` | Modal **A10** (8 CPU, 32 GiB) → L40S | default | T M B G | block |
| `teleocr` | Modal **A10** (8 CPU, 16 GiB) → L4 → L40S | default | T M B G | block |
| `rapidocr` | Modal **CPU** (4 cores, 8 GiB) | default | T G | word, else line |
| `ovisocr2` | Modal **A10** → L4 → L40S | default + KV + schema | T M B, KV, schema | none |
| `lighton_ocr_2` | Modal **A10** → L4 | default | T M B | none |
| `apple_vision` | this Mac (opt-in) | default | T G | word |

T/M/B/G = text / markdown / tables / geometry. Leg decisions that were
judgement calls, with the reasoning recorded beside each registry entry:

- **`deepseek_ocr_2` key_value is enabled.** v1 dropped it after 0/65 JSON
  replies; the leg matrix said not to inherit that. The app forwards the
  prompt, so under the interface rule in [scoring-design.md](scoring-design.md)
  the engine competes and its score is its score. Expectation is low (the
  paper describes no instruction stage); that is a result, not a reason to
  skip. Cost: 65 extra L4 passes.
- **`ovisocr2` has no geometry leg** although the matrix listed a prompted
  candidate: neither the card nor the report documents text grounding, so a
  word-box prompt would be an invented ask. Its `<img>` boxes locate figures.
- **`chandra_ocr_2` extraction runs without constrained decoding.** Its
  vLLM route forwards the prompt but sends no `response_format`; Surya's does.
  Recorded per item as `response_format_applied=False`. The two are not
  equivalent extraction routes.
- **`mathpix` geometry is a second billed call** because `include_word_data`
  cannot be combined with `enable_document_layout`. The registry prompt
  `__mathpix_word_data__` is a configuration selector (the same device marker
  uses), never sent to the API.
- **`datalab_ocr` geometry is withheld** until two paid conversions confirm the
  JSON block shape; the runner already parses it into `text_regions`.
- **key_value is not offered** for `mistral_ocr_4_1` and `datalab_ocr`: both
  vendors' extraction needs a caller-supplied schema, and inventing a generic
  one would be our construction. Schema (their documented feature) is offered.
- **`apple_vision` is local inference on this machine.** It exists because it
  was requested; it stays opt-in and off by default. Whether it runs is your
  decision.
- **`rapidocr` and `opendoc` run on Modal CPU**, in the declared pipeline. `GPU_TYPE="CPU"` bills 4 cores + 8 GiB per occupied
  second (`modal_apps/_shared.py`); an app declaring `CPU` must request
  exactly that allocation.

## Checks before trusting a new deployment

Condensed from the per-engine handoffs; each app's docstring has the full
list with the probe for it. Anything not listed as verified was read from
source, not observed.

- **Every Modal app**: image build resolves (pins chosen from each project's
  lock or CI file, but never installed together here); server starts and
  warm-up passes on the declared GPU; peak memory at the declared concurrency
  on the largest intended input, not only the smoke-test page.
- **ovisocr2**: the nested `images_kwargs` pixel bounds are honoured by vLLM
  0.22.1 (check `generation.prompt_tokens` ≈ 8,152 on a 2480×3508 page);
  `gdn_prefill_backend=triton` on an A10; whether `schema_render` should be
  `instance_template` as GLM needed.
- **lighton_ocr_2**: HTML tables actually appear (sets `table_format`);
  temperature 0.2 makes runs non-deterministic — measure run-to-run variance.
- **deepseek_ocr_2**: rendered prompt is `<image>\n<prompt>`; grounding labels
  in v2 (is `image` still the picture label?); ngram 20/90 vs the recipe's
  30/90.
- **dots_mocr**: server-side resize matches the author's `smart_resize`
  (decides whether boxes land); the CLI's default 200-dpi re-render for images
  is NOT applied here.
- **surya_ocr_2 / chandra_ocr_2**: pip vLLM 0.20.1 / 0.17.0 instead of the
  vendors' Docker images; both upstream recipes let max visual tokens + max
  output exceed the 18,000 context, so vLLM clamps output to ~11–12k tokens
  (recorded as `possible_truncation`); Chandra is **5.3B / 10.6 GB**, use this instead of earlier 4B estimates — validate A10 headroom at concurrency 1.
- **paddleocr_vl_1_6**: vLLM 0.21.0 serving the 1.6 checkpoint with the
  explicit chat template; A10 fit at util 0.5 beside PP-DocLayoutV3;
  `layout_detection_model_dir` accepting the pinned snapshot.
- **opendoc**: ONNX path equivalence to the source path behind the published
  OmniDocBench number is not established; ORT thread count inside the 4-core
  allocation.
- **mineru_4_standard**: which `max_model_len` vLLM picks for the VLM
  (32,768 vs 8,192 in `text_config`); the pinned-download wrapper around
  MinerU's own downloader; A10 fit with the ~6 GiB vLLM reservation.
- **teleocr**: `pip install -c requirements.txt -e .` resolution on Python
  3.10; Pillow's 72-dpi PDF default implying upscaling toward 3500 px; vLLM
  0.11 ending at `max_model_len` with finish_reason `length`.
- **mistral_ocr_4_1**: table placeholder id spelling vs `tables[].id`; block
  pixel frame vs `pages[].dimensions`; whether annotated pages bill $5 total
  or $5 + $4.
- **google_document_ocr**: `revisions[].processor` populated; `imageless_mode`
  accepted by OCR v2.1; token polygons stay in the original frame on deskewed
  pages.
- **mathpix**: three original corpus files (`medical_000`, `bank_000`,
  `shipping_000`) exceed the 5 MB request limit. The runner already attempts
  a lossless PNG re-encode before rejecting an oversized upload; original
  size alone does not establish whether that upload will fail. Do not silently
  downscale or change endpoints to bypass the limit.
- **datalab_ocr**: JSON tree shape and page bbox base (then enable geometry);
  `cost_breakdown.final_cost_cents` presence; form-string booleans.
- **rapidocr**: whether word results appear on real pages (upstream drops
  them for the page if any line lacks them → line fallback).
- **apple_vision**: compile once and run one receipt; `word_range_failures`.

## Runtime limitations

Geometry dispatch uses each adapter's declared block/line/word contract and
normalized coordinate frame. Coarse boxes compete using their actual extents;
they are not divided into invented word boxes. MinerU classic's table HTML is
rendered to text while preserving whole-table boxes. Mathpix rotated boxes with
an unverified frame produce an adapter error, not a quality score.

Before running new integrations, resolve these source-level limitations:

- Mathpix's native MMD list syntax needs a faithful structural conversion before
  interpreting Markdown list scores; oversized images and rotated frames also
  require an explicit input/coordinate policy.
- Datalab's fallback cost estimate does not apply the vendor's cent rounding;
  prefer returned billing. Extraction fees may vary.
- Google processor lookup failures can leave processor type/version unverified.
  Ensure the selected processor is Enterprise Document OCR, and validate boxes
  on transformed pages.
- MinerU classic records downloaded model revisions but still downloads mutable
  defaults. Pin the recorded assets before reproducing a deployment.
- Generic prompted KV/schema routes on OCR fine-tunes are experimental tasks;
  supported prompt transport does not establish extraction quality. Context,
  output caps and constrained-decoding differences remain recorded per call.

These limitations do not require rebuilding the accepted references. All engines
use the same shipped corpus, applicable legs and fixed reference selection.
