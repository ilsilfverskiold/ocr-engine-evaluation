"""
Modal app: OvisOCR2 (0.8B, Qwen3.5 architecture) — ATH-MaaS/OvisOCR2
Author's offline vLLM recipe (LLM.generate over a list). GPU: A10 (initial).

Sources (read 2026-09-30):
- Model card / recipe: https://huggingface.co/ATH-MaaS/OvisOCR2 (README.md @ MODEL_REVISION)
- config.json, preprocessor_config.json, chat_template.jinja, tokenizer_config.json
  at https://huggingface.co/ATH-MaaS/OvisOCR2/raw/<MODEL_REVISION>/<file>
  (no generation_config.json exists at this revision — HTTP 404)
- Technical report: https://arxiv.org/html/2607.13639 (single page->Markdown task;
  boxes only for images/charts; SFT used a 16K maximum sequence length)
- vLLM v0.22.1 source: vllm/engine/arg_utils.py (gdn_prefill_backend),
  vllm/sampling_params.py (StructuredOutputsParams), vllm/inputs/llm.py
  (per-prompt mm_processor_kwargs), vllm/entrypoints/offline_utils.py
  (failed add_request aborts the already-added requests),
  requirements/test/cuda.txt (CI lock: transformers==5.5.3).

Backend decision:
- The card documents OFFLINE vLLM: `LLM(...)` + `self.model.generate(list, SamplingParams)`.
  It does not document an OpenAI server. So the batching owner here is vLLM's own
  scheduler behind one `LLM.generate(list)` call per run_ocr_batch — no thread
  pool, no HTTP server, no Modal fan-out.
- Per-item prompts and per-item SamplingParams (structured outputs) are passed as
  parallel lists to the same generate call, so a tagged batch mixing the
  conversion pass with key_value/schema passes is still ONE scheduler batch.

Task boundary:
- The model card and report document exactly ONE task: page image -> Markdown
  (HTML tables, LaTeX formulas, <img> tags with 0-1000 boxes for charts/images).
  No text grounding / spotting / word-box output is documented anywhere in the
  card or report. The <img src="images/bbox_..."> tags locate VISUAL regions,
  not text; they are kept in structured_output["image_regions"] and are never
  emitted as words/text_regions.
- The chat template is the general Qwen3.5 template (text + image content), so a
  caller prompt reaches the model unchanged. Instruction following beyond the
  documented OCR prompt is NOT claimed by the authors — key_value/schema passes
  are measured as "prompted candidates", not documented capabilities.
"""

import modal

GPU_TYPE = "A10"
VLLM_VERSION = "0.22.1"  # pinned by the model card: pip install "vllm==0.22.1" pillow
# vLLM 0.22.1 allows transformers>=4.56 except 5.0-5.4 and 5.5.0
# (requirements/common.txt). An unpinned resolve today picks a release published
# after vLLM 0.22.1. Pin the version vLLM's own CI lock used for this tag
# (requirements/test/cuda.txt @ v0.22.1: transformers==5.5.3).
TRANSFORMERS_VERSION = "5.5.3"
# Same CI lock (requirements/test/cuda.txt @ v0.22.1): pillow==10.4.0,
# mistral-common==1.11.2 (a vLLM dependency, pinned for reproducibility).
PILLOW_VERSION = "10.4.0"
MISTRAL_COMMON_VERSION = "1.11.2"
MODEL_ID = "ATH-MaaS/OvisOCR2"
# https://huggingface.co/api/models/ATH-MaaS/OvisOCR2 -> sha (lastModified 2026-08-28),
# pinned for reproducible inference.
MODEL_REVISION = "1fc9221b7823a371d6e97f92d527cc847e24e107"

# --- Exact card prompt (README.md @ MODEL_REVISION, OvisOCR2Parser.__init__) ----
# The card builds the img-tag example by string concatenation ('<' + 'img src=...');
# the resulting string is reproduced here verbatim, INCLUDING the leading "\n".
OCR_PROMPT = (
    '\nExtract all readable content from the image in natural human reading order '
    'and output the result as a single Markdown document. For charts or images, '
    'represent them using an HTML image tag: <img src="images/bbox_{left}_{top}_'
    '{right}_{bottom}.jpg" />, where left, top, right, bottom are bounding box '
    'coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as '
    'HTML: <table>...</table>. Transcribe all other text as standard Markdown. '
    'Preserve the original text without translation or paraphrasing.'
)

# --- Card sampling / processor settings -----------------------------------------
TEMPERATURE = 0.0          # card: SamplingParams(temperature=0.0)
MAX_TOKENS = 16384         # card: SamplingParams(max_tokens=16384)
MIN_PIXELS = 448 * 448     # card: mm_processor_kwargs.images_kwargs.min_pixels
MAX_PIXELS = 2880 * 2880   # card: mm_processor_kwargs.images_kwargs.max_pixels
GPU_MEMORY_UTILIZATION = 0.8   # card: LLM(gpu_memory_utilization=0.8)
GDN_PREFILL_BACKEND = "triton"  # card: LLM(gdn_prefill_backend="triton")

# --- Context arithmetic (explicit; see config.json / preprocessor_config.json) ---
# Vision: patch_size=16, spatial_merge_size=2 -> one LLM visual token per 32x32 px.
# The card's max_pixels bound (2880^2 = 8,294,400 px) caps visual tokens at
#     8,294,400 / (32*32) = 8,100 tokens.
# Large corpus page 2480x3508 = 8,699,840 px > max_pixels, so Qwen2-VL
# smart_resize (factor 32) scales by sqrt(8,699,840/8,294,400)=1.0242:
#     h = floor(3508/1.0242/32)*32 = 3424, w = floor(2480/1.0242/32)*32 = 2400
#     -> (3424/32)*(2400/32) = 107*75 = 8,025 visual tokens.
# Text: the rendered chat template + OCR_PROMPT is 128 tokens including the single
# <|image_pad|> placeholder (counted offline with tokenizer.json @ MODEL_REVISION),
# i.e. 127 text tokens; the generic key_value prompt renders to 55. The exact
# per-item total is recorded as generation.prompt_tokens.
# Budget: 8,100 (max visual) + 127 (template+prompt) + 16,384 (card output cap)
#       = 24,611  <=  MAX_MODEL_LEN 26,624  (2,013 headroom, absorbs longer
#         key_value/schema prompts; a schema prompt beyond that still runs but can
#         end at the context limit -> finish_reason "length", visible per item)
#       <=  262,144 (text_config.max_position_embeddings).
# If the nested images_kwargs were ignored, the processor default longest_edge
# 16,777,216 px (=4096^2) would allow up to 16,384 visual tokens; a 2480x3508 page
# would then be ~8,496 tokens and still fit. prompt_tokens makes this checkable.
# The card itself sets no max_model_len (vLLM would default to 262,144); a bounded
# value only limits KV reservation and cannot change any output within the
# 16,384 cap for prompts <= 10,240 tokens.
MAX_VISUAL_TOKENS = MAX_PIXELS // (32 * 32)   # 8,100
MAX_MODEL_LEN = 26624

# KV memory per token (bf16): only 6 of 24 layers are full attention
# (full_attention_interval=4), num_key_value_heads=2, head_dim=256:
#     6 layers * 2 (K,V) * 2 heads * 256 * 2 B = 12,288 B = 12 KiB/token.
# Worst case 24 concurrent sequences at MAX_MODEL_LEN: 24 * 26,624 * 12 KiB = 7.3 GiB.
# Gated-DeltaNet state (18 linear-attention layers, 16 heads, 128x128, fp32
# mamba_ssm_dtype) ~= 18*16*128*128*4 B = 18.9 MB/sequence -> ~0.45 GB for 24.
# Weights ~0.9B params bf16 ~= 1.8 GB. A10 (24 GB) * 0.8 = 19.2 GB budget:
# 1.8 + 7.3 + 0.45 + vision-encoder activations for one 8,100-token image
# (32,400 pre-merge patches, 12-layer ViT width 768) + CUDA graphs fits with
# several GB spare. Next GPU to probe: L4 (same 24 GB, lower rate, half the
# memory bandwidth), then L40S only if the A10 batch is compute-bound.

app = modal.App("ocr-bench-ovisocr2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        f"transformers=={TRANSFORMERS_VERSION}",
        f"Pillow=={PILLOW_VERSION}",
        f"mistral-common[image]=={MISTRAL_COMMON_VERSION}",
    )
    # Bake the pinned weights into the image so every container runs the same
    # bytes and startup does not depend on the Hub.
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{MODEL_ID}', revision='{MODEL_REVISION}')\""
    )
    .env({
        "HF_HUB_OFFLINE": "1",
        # Modal slim images include CUDA runtime libraries, not nvcc. FlashInfer's
        # sampler may JIT-compile at warmup, so use vLLM's native sampler
        # (same setting as the other vLLM apps in this repo).
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    })
)

_IMG_TAG_RE = r'<img src="images/bbox_(\d+)_(\d+)_(\d+)_(\d+)\.jpg" />'
_MODEL_VERSION = (
    f"{MODEL_ID}@{MODEL_REVISION[:12]} vllm=={VLLM_VERSION} offline LLM.generate "
    f"bf16 gdn_prefill={GDN_PREFILL_BACKEND} thinking=off T={TEMPERATURE} "
    f"max_tokens={MAX_TOKENS} pixels={MIN_PIXELS}-{MAX_PIXELS} max_model_len={MAX_MODEL_LEN}"
)


def _load_image(image_bytes: bytes):
    import io

    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.load()
    return img


def _clean_truncated_repeats(
    text: str,
    min_text_len: int = 8000,
    max_period: int = 200,
    min_period: int = 1,
    min_repeat_chars: int = 100,
    min_repeat_times: int = 5,
) -> str:
    """Verbatim port of OvisOCR2Parser._clean_truncated_repeats from the card."""
    n = len(text)
    if n < min_text_len:
        return text

    max_period = min(max_period, n - 1)
    for unit_len in range(min_period, max_period + 1):
        if text[n - 1] != text[n - 1 - unit_len]:
            continue

        match_len = 1
        idx = n - 2
        while idx >= unit_len and text[idx] == text[idx - unit_len]:
            match_len += 1
            idx -= 1

        total_len = match_len + unit_len
        repeat_times = total_len // unit_len
        tail_len = total_len % unit_len

        if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
            return text[: n - total_len + unit_len] + text[n - tail_len:]

    return text


def _card_postprocess(text: str) -> str:
    """The card's parse() defaults: filter_imgtags=True, then repeat cleanup."""
    text = text.strip()
    text = "\n\n".join(
        block for block in text.split("\n\n")
        if not block.strip().startswith('<img src="images/bbox_')
    )
    return _clean_truncated_repeats(text)


def _image_regions(raw: str) -> list[dict]:
    """<img> tags -> 0-1 boxes. VISUAL regions only; never text geometry.

    Coordinates are relative ([0,1000) per the prompt) so they map directly onto
    the EXIF-transposed image regardless of the processor's aspect-preserving
    resize (the card's own crop helper multiplies by the original width/height).
    """
    import re

    regions = []
    for m in re.finditer(_IMG_TAG_RE, raw):
        l, t, r, b = (int(v) / 1000.0 for v in m.groups())
        regions.append({"label": "image", "bbox": [l, t, r, b], "tag": m.group(0)})
    return regions


def _structured_outputs(response_format: dict | None):
    """Map the harness's OpenAI-style response_format onto vLLM offline params."""
    if not response_format:
        return None
    from vllm.sampling_params import StructuredOutputsParams

    kind = response_format.get("type")
    if kind == "json_object":
        return StructuredOutputsParams(json_object=True)
    if kind == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema")
        if not isinstance(schema, dict):
            raise ValueError("response_format json_schema has no 'schema' object")
        return StructuredOutputsParams(json=schema)
    raise ValueError(f"unsupported response_format type: {kind!r}")


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=3600,
    startup_timeout=1800,
    scaledown_window=30,
    secrets=[modal.Secret.from_name("huggingface-token")],
    max_containers=1,
)
class OvisOCR2Inference:
    @modal.enter()
    def load_model(self) -> None:
        import threading
        import time

        from vllm import LLM

        t0 = time.perf_counter()
        self._llm = LLM(
            model=MODEL_ID,
            revision=MODEL_REVISION,
            tokenizer_revision=MODEL_REVISION,
            tensor_parallel_size=1,                       # card
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,  # card
            gdn_prefill_backend=GDN_PREFILL_BACKEND,      # card
            dtype="bfloat16",                             # config.json text_config.dtype
            max_model_len=MAX_MODEL_LEN,                  # see arithmetic above
            # Serving-only settings, output-equivalent, not in the card:
            # one image per prompt and no video (avoids profiling a video input);
            # prefix/mm-processor caches off so a document's conversion and
            # extraction passes in the same tagged batch do not share cached
            # image work — each pass pays its own cost, as in the other vLLM apps.
            limit_mm_per_prompt={"image": 1, "video": 0},
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
        )
        self._tokenizer = self._llm.get_tokenizer()
        self._default_prompt_text = self._chat_prompt(OCR_PROMPT)
        self._warmup()
        self._startup_gpu_seconds = time.perf_counter() - t0
        self._startup_cost_applied = False
        self._startup_cost_lock = threading.Lock()

    @modal.exit()
    def shutdown(self) -> None:
        # Offline LLM owns its engine core process; dropping the reference lets
        # vLLM shut it down. Nothing else to clean up.
        self._llm = None

    # --- prompt construction (card: tokenizer.apply_chat_template, thinking off) --
    def _chat_prompt(self, text: str) -> str:
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _vllm_input(self, prompt_text: str, img) -> dict:
        return {
            "prompt": prompt_text,
            "multi_modal_data": {"image": img},
            "mm_processor_kwargs": {
                "images_kwargs": {"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}
            },
        }

    def _warmup(self) -> None:
        """Exercise the largest visual path once (a 2480x3508 page hits max_pixels)."""
        from PIL import Image
        from vllm import SamplingParams

        page = Image.new("RGB", (2480, 3508), (255, 255, 255))
        self._llm.generate(
            [self._vllm_input(self._default_prompt_text, page)],
            SamplingParams(max_tokens=8, temperature=TEMPERATURE),
            use_tqdm=False,
        )

    # --- public methods -----------------------------------------------------------
    @modal.method()
    def run_ocr(self, image_bytes: bytes, mime_type: str = "image/png",
                prompt: str | None = None, response_format: dict | None = None) -> dict:
        batch = self._run_batch_impl(
            [{"bytes": image_bytes, "mime_type": mime_type, "document_path": None}],
            prompt, response_format,
        )
        response = batch["results"][0]
        response["gpu_seconds"] = batch["batch_gpu_seconds"]
        response["gpu_type"] = GPU_TYPE
        response["startup_gpu_seconds"] = batch["startup_gpu_seconds"]
        response["is_cold_start"] = batch["is_cold_start"]
        return response

    @modal.method()
    def run_ocr_batch(self, files: list[dict], prompt: str | None = None,
                      response_format: dict | None = None) -> dict:
        """One measured LLM.generate batch on one GPU.

        Per-file "prompt"/"response_format" keys override the call arguments; a
        missing prompt means the card's conversion prompt.
        """
        return self._run_batch_impl(files, prompt, response_format)

    # --- internals ----------------------------------------------------------------
    def _run_batch_impl(self, files: list[dict], prompt: str | None,
                        response_format: dict | None) -> dict:
        import time

        started = time.perf_counter()
        try:
            results = self._generate_all(files, prompt, response_format)
        except Exception as exc:  # whole-batch failure: one error item per input
            results = [self._error_item(f.get("document_path"), exc) for f in files]
        batch_gpu_seconds = time.perf_counter() - started
        startup_seconds, is_cold_start = self._take_startup_metadata()
        return {
            "results": results,
            "batch_gpu_seconds": batch_gpu_seconds,
            "gpu_type": GPU_TYPE,
            "startup_gpu_seconds": startup_seconds,
            "is_cold_start": is_cold_start,
        }

    def _generate_all(self, files: list[dict], prompt: str | None,
                      response_format: dict | None) -> list[dict]:
        from vllm import SamplingParams

        results: list[dict | None] = [None] * len(files)
        pending: list[tuple[int, dict, SamplingParams, dict]] = []

        # 1. Per-item preparation: failures here are isolated to the item.
        for i, item in enumerate(files):
            doc = item.get("document_path")
            try:
                item_prompt = item.get("prompt", prompt)
                item_format = item.get("response_format", response_format)
                is_conversion = item_prompt is None or item_prompt == OCR_PROMPT
                text = OCR_PROMPT if is_conversion else item_prompt
                prompt_text = (self._default_prompt_text if is_conversion
                               else self._chat_prompt(text))
                # vLLM rejects a prompt longer than max_model_len for the WHOLE
                # generate call; check the bound here so one oversized prompt
                # cannot take the batch down. The template holds one image
                # placeholder token that expands to <= MAX_VISUAL_TOKENS.
                text_tokens = len(self._tokenizer.encode(prompt_text))
                if text_tokens + MAX_VISUAL_TOKENS >= MAX_MODEL_LEN:
                    raise ValueError(
                        f"prompt text ({text_tokens} tokens) + max visual tokens "
                        f"({MAX_VISUAL_TOKENS}) does not fit max_model_len {MAX_MODEL_LEN}")
                img = _load_image(item["bytes"])
                params = SamplingParams(
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    structured_outputs=_structured_outputs(item_format),
                )
                meta = {
                    "document_path": doc,
                    "prompt_used": text,
                    "is_conversion": is_conversion,
                    "response_format": item_format,
                    "image_size": list(img.size),
                }
                pending.append((i, self._vllm_input(prompt_text, img), params, meta))
            except Exception as exc:
                results[i] = self._error_item(doc, exc)

        # 2. One scheduler batch for every valid item (the card's list generate).
        if pending:
            fallback_error = None
            try:
                outputs = self._llm.generate(
                    [p[1] for p in pending], [p[2] for p in pending], use_tqdm=False)
            except Exception as exc:
                # An add_request validation error (e.g. a schema the structured
                # output backend rejects) fails the whole call; vLLM aborts the
                # already-added requests (offline_utils._render_and_add_requests).
                # Isolate the bad item by re-running items singly. The extra time
                # stays inside batch_gpu_seconds and is flagged per item.
                fallback_error = f"{type(exc).__name__}: {exc}"
                outputs = []
                for _, vin, params, _meta in pending:
                    try:
                        outputs.append(self._llm.generate([vin], params, use_tqdm=False)[0])
                    except Exception as item_exc:
                        outputs.append(item_exc)
            for (i, _vin, params, meta), out in zip(pending, outputs):
                if isinstance(out, Exception):
                    results[i] = self._error_item(meta["document_path"], out)
                else:
                    results[i] = self._to_item(out, meta)
                if fallback_error:
                    results[i]["structured_output"]["batch_fallback"] = fallback_error
        return results  # type: ignore[return-value]

    def _to_item(self, out, meta: dict) -> dict:
        import re

        completion = out.outputs[0]
        raw = completion.text or ""
        generation = {
            "finish_reason": completion.finish_reason,
            "stop_reason": completion.stop_reason,
            "prompt_tokens": len(out.prompt_token_ids or []),  # includes visual tokens
            "completion_tokens": len(completion.token_ids),
            "max_tokens": MAX_TOKENS,
            "max_model_len": MAX_MODEL_LEN,
            "temperature": TEMPERATURE,
        }
        structured: dict = {
            "generation": generation,
            "preprocessing": {
                "exif_transpose": True, "color": "RGB",
                "input_size": meta["image_size"],
                "processor_min_pixels": MIN_PIXELS,
                "processor_max_pixels": MAX_PIXELS,
                "note": "no app-side resize; the model processor's smart_resize "
                        "applies the card's pixel bounds",
            },
        }
        if meta["response_format"]:
            structured["response_format"] = meta["response_format"]

        if meta["is_conversion"]:
            # raw_text = the card's default parse() output (img-tag blocks removed,
            # truncated-repeat cleanup). The untouched generation is kept.
            text = _card_postprocess(raw)
            structured["raw_model_output"] = raw
            structured["postprocess"] = "card parse(): filter_imgtags=True + _clean_truncated_repeats"
            regions = _image_regions(raw)
            if regions:
                structured["image_regions"] = regions
            tables = [{"html": m.group(0)}
                      for m in re.finditer(r"<table\b.*?</table>", raw, flags=re.I | re.S)]
            if tables:
                structured["tables"] = tables
        else:
            # Prompted leg: the scorer parses the model's own answer, uncleaned.
            text = raw.strip()
            if text != raw:
                structured["raw_model_output"] = raw

        return {
            "raw_text": text,
            "structured_output": structured,
            "model_version": _MODEL_VERSION,
            "prompt_used": meta["prompt_used"],
            "document_path": meta["document_path"],
            "gpu_type": GPU_TYPE,
            "error": None,
        }

    def _error_item(self, document_path, exc: Exception) -> dict:
        return {
            "raw_text": "",
            "structured_output": {},
            "model_version": _MODEL_VERSION,
            "prompt_used": None,
            "document_path": document_path,
            "gpu_type": GPU_TYPE,
            "error": f"{type(exc).__name__}: {str(exc) or repr(exc)}",
        }

    def _take_startup_metadata(self) -> tuple[float, bool]:
        """Report startup separately once; never add it to inference GPU-seconds."""
        with self._startup_cost_lock:
            if not getattr(self, "_startup_cost_applied", True):
                self._startup_cost_applied = True
                return getattr(self, "_startup_gpu_seconds", 0.0), True
        return 0.0, False
