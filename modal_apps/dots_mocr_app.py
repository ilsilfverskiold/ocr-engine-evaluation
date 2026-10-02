"""
Modal app: dots.mocr (3B, MIT) — dots-studio/dots.mocr (older examples:
rednote-hilab/dots.mocr, which now redirects here). The document model, not
the separate dots.mocr-svg variant.

Served by upstream vLLM (DotsOCRForCausalLM, integrated since vLLM 0.11.0 per
the card) through its OpenAI-compatible server with the author's serve flags,
fed by a bounded thread pool so vLLM's continuous batching owns the batch.
GPU: L4 (first fit probe; see GPU_TYPE).

Two documented task routes, chosen per item:
- Document parsing (default, prompt=None): the author's `prompt_layout_all_en`
  -> ordered layout JSON [{bbox, category, text}], converted here with a
  vendored copy of the author's post-processing (bbox back-mapping and
  layoutjson2md). One pass answers text, markdown, tables and geometry.
- General QA (any other prompt): the author's `prompt_general` route — the
  caller's prompt plus the system prompt "You are a helpful assistant."
  (parser.py: "general tasks need system_prompt"). Used for key_value/schema.

Sources (read 2026-09-30):
- Model card @ e539fbb5: https://huggingface.co/dots-studio/dots.mocr
  (+ config.json, generation_config.json, preprocessor_config.json,
   chat_template.json, tokenizer.json, configuration_dots.py at that revision)
- Author repo @ 23f3e561: https://github.com/rednote-hilab/dots.mocr (now
  github.com/studio-dots-ai/dots.mocr): dots_mocr/utils/prompts.py,
  dots_mocr/model/inference.py, dots_mocr/parser.py,
  dots_mocr/utils/{layout_utils,format_transformer,image_utils,consts,
  output_cleaner,doc_utils}.py, demo/demo_vllm.py, demo/demo_vllm_general.py
- vLLM v0.21.0 source: vllm/model_executor/models/dots_ocr.py (placeholder
  "<|img|><|imgpad|><|endofimg|>", Qwen2VL processor), vllm/entrypoints/utils.py
"""

import modal

# ---------------------------------------------------------------------------
# Identity and pins
# ---------------------------------------------------------------------------
# First fit probe per the modal-inference-engineer skill. Memory estimate, BF16:
#   weights                       6.08 GB  (3,039,179,264 params)
#   KV cache per token            2 (K,V) x 28 layers x 2 kv heads x 128 dim
#                                 x 2 B = 28,672 B
#   one full MAX_MODEL_LEN seq    40,960 x 28,672 B = 1.09 GiB
#   BATCH_CONCURRENCY=4 at max    4.4 GiB
#   vision encoder peak (42-layer ViT, 1536 wide) on a max-size page of
#   57,600 patches: ~1-3 GB transient (unmeasured; vLLM profiles it at start)
#   + runtime / CUDA graphs ~1 GB
# ~13-15 GB of 0.90 x 24 GB. Next probes: A10 (2x the memory bandwidth for
# decode at a 1.38x rate), then L40S.
GPU_TYPE = "L4"
VLLM_VERSION = "0.21.0"          # >= 0.11.0 per the card; same pin as the repo's other vLLM apps
# The author's requirements.txt pins transformers==4.57.6; it also satisfies
# vLLM 0.21.0 (>=4.56.0, not 5.0-5.5.0). configuration_dots.py imports
# Qwen2_5_VLProcessor from transformers, so the version matters.
TRANSFORMERS_VERSION = "4.57.6"
MODEL_ID = "dots-studio/dots.mocr"
MODEL_REVISION = "e539fbb52280393adc081b289ec597430a0f9031"
SERVED_MODEL_NAME = "model"      # the author's --served-model-name
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8004
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/v1"
HEALTH_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/health"
BATCH_CONCURRENCY = 4

# ---------------------------------------------------------------------------
# Prompts (verbatim, dots_mocr/utils/prompts.py @ 23f3e561)
# ---------------------------------------------------------------------------
LAYOUT_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""
# parser.py _inference_with_vllm: system prompt only for prompt_general.
GENERAL_SYSTEM_PROMPT = "You are a helpful assistant."
# inference.py prefixes every prompt with the placeholder itself: "if no
# '<|img|><|imgpad|><|endofimg|>' here, vllm v1 will add '\n' here".
IMAGE_PLACEHOLDER = "<|img|><|imgpad|><|endofimg|>"

# Sampling: DotsMOCRParser defaults (parser.py): temperature=0.1, top_p=1.0.
# (demo_vllm*.py use top_p=0.9; the parser is the documented document path.)
# A fixed seed makes the 0.1-temperature sampling repeatable run to run
# without changing the distribution.
TEMPERATURE = 0.1
TOP_P = 1.0
SEED = 0

# ---------------------------------------------------------------------------
# Resolution policy and context arithmetic
# ---------------------------------------------------------------------------
# The page is sent at its own resolution (EXIF-transposed, RGB). The server's
# Qwen2-VL image processor applies smart_resize with these values
# (preprocessor_config.json; dots_mocr/utils/consts.py): both sides rounded to
# multiples of 28, total pixels clamped to [MIN_PIXELS, MAX_PIXELS]. The model's
# bboxes are pixels of THAT resized image (layout_utils.post_process_cells).
MIN_PIXELS = 3136
MAX_PIXELS = 11289600
IMAGE_FACTOR = 28
# Visual tokens = (h/28) x (w/28) after smart_resize (14 px patches, 2x2 merge):
#   max 11,289,600 / 784 = 14,400
MAX_VISUAL_TOKENS = 14400
# Text tokens of the layout request, counted with the pinned tokenizer.json:
#   "<|user|>" + placeholder markers + LAYOUT_PROMPT + "<|endofuser|><|assistant|>"
#   = 217 tokens including the single <|imgpad|> that expands to the visual
#   tokens -> 216 text tokens.
LAYOUT_PROMPT_TEXT_TOKENS = 216
# Output cap: the card's HF example uses max_new_tokens=24000 (parser.py's HF
# path too); the parser's vLLM path requests 32,768. 24,000 is used here.
MAX_NEW_TOKENS = 24000
# 14,400 + 216 + 24,000 = 38,616; +2,344 headroom for longer extraction
# prompts and the general-QA system prompt = 40,960 (config positions:
# 131,072). A long schema prompt cannot overflow: vLLM 0.21.0 get_max_tokens()
# takes min(requested, max_model_len - input_len).
MAX_MODEL_LEN = 40960

MODEL_VERSION = (
    f"{MODEL_ID}@{MODEL_REVISION[:12]} vllm=={VLLM_VERSION} bf16 "
    f"smart_resize[{MIN_PIXELS},{MAX_PIXELS}] T={TEMPERATURE} top_p={TOP_P} "
    f"max_new_tokens={MAX_NEW_TOKENS}"
)


def download_weights():
    import os

    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID, revision=MODEL_REVISION, token=os.environ.get("HF_TOKEN"))


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "Pillow>=10.0",
        "huggingface_hub>=0.23",
        f"vllm=={VLLM_VERSION}",
        f"transformers=={TRANSFORMERS_VERSION}",
    )
    .run_commands(
        # Fail the BUILD, not every container, if the resolver moved a pin.
        "python -c 'import vllm, transformers; "
        "print(\"vllm\", vllm.__version__, \"transformers\", transformers.__version__); "
        f"assert vllm.__version__.startswith(\"{VLLM_VERSION}\"); "
        f"assert transformers.__version__ == \"{TRANSFORMERS_VERSION}\"'",
    )
    .env({
        # Modal slim images include CUDA runtime libraries, not nvcc. FlashInfer's
        # sampler may JIT-compile at vLLM warmup, so use vLLM's native sampler.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    })
    .run_function(
        download_weights,
        secrets=[modal.Secret.from_name("huggingface-token")],
    )
)

app = modal.App("ocr-bench-dots-mocr")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def _normalized_png(image_bytes: bytes):
    """EXIF transpose, then the author's to_rgb(): RGBA is composited onto
    white (a plain convert("RGB") would turn transparent pixels black)."""
    import io

    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    else:
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), img.size


def _smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR,
                  min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """Verbatim logic of dots_mocr/utils/image_utils.py:smart_resize."""
    import math

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((h_bar * w_bar) / max_pixels)
            h_bar = max(factor, math.floor(h_bar / beta / factor) * factor)
            w_bar = max(factor, math.floor(w_bar / beta / factor) * factor)
    return h_bar, w_bar


# --- vendored from dots_mocr/utils/format_transformer.py @ 23f3e561 (MIT) ---
def _has_latex_markdown(text: str) -> bool:
    import re

    if not isinstance(text, str):
        return False
    patterns = [
        r'\$\$.*?\$\$', r'\$[^$\n]+?\$', r'\\begin\{.*?\}.*?\\end\{.*?\}',
        r'\\[a-zA-Z]+\{.*?\}', r'\\[a-zA-Z]+', r'\\\[.*?\\\]', r'\\\(.*?\\\)',
    ]
    return any(re.search(p, text, re.DOTALL) for p in patterns)


def _clean_latex_preamble(latex_text: str) -> str:
    import re

    for pattern in (r'\\documentclass\{[^}]+\}', r'\\usepackage\{[^}]+\}',
                    r'\\usepackage\[[^\]]*\]\{[^}]+\}', r'\\begin\{document\}',
                    r'\\end\{document\}'):
        latex_text = re.sub(pattern, '', latex_text, flags=re.IGNORECASE)
    return latex_text


def _get_formula_in_markdown(text: str) -> str:
    import re

    text = text.strip()
    if text.startswith('$$') and text.endswith('$$'):
        inner = text[2:-2].strip()
        return f"$$\n{inner}\n$$" if '$' not in inner else text
    if text.startswith('\\[') and text.endswith('\\]'):
        return f"$$\n{text[2:-2].strip()}\n$$"
    if len(re.findall(r'.*\\\[.*\\\].*', text)) > 0:
        return text
    if len(re.findall(r'\$([^$]+)\$', text)) > 0:
        return text
    if not _has_latex_markdown(text):
        return text
    if 'usepackage' in text:
        text = _clean_latex_preamble(text)
    if text and text[0] == '`' and text[-1] == '`':
        text = text[1:-1]
    return f"$$\n{text}\n$$"


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if text[:2] == '`$' and text[-2:] == '$`':
        text = text[1:-1]
    return text
# --- end vendored ---


def _plain_text(content: str) -> str:
    """Block text -> plain text for the geometry scorer (HTML tags and heading
    markers removed). The unmodified text stays in layout_blocks."""
    import html
    import re

    text = re.sub(r"<br\s*/?>", " ", content, flags=re.I)
    text = re.sub(r"</t[dh]>\s*<t[dh][^>]*>", " ", text, flags=re.I)
    text = re.sub(r"</tr>\s*<tr[^>]*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _load_cells(raw: str) -> tuple[list | None, str]:
    """The layout answer is a JSON list of cells. Returns (cells, status).

    status: "json" — parsed whole;
            "salvaged_partial" — the full parse failed (typically a truncated
              reply) and the complete leading cell objects were recovered;
            "failed" — nothing usable.
    The author's fallback (OutputCleaner) instead returns only the joined text
    of the cells with no boxes; recovering the complete cells keeps their
    boxes and is recorded as a deviation via this status.
    """
    import json

    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("\n") + 1:] if "\n" in text else text
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list):
                    data = value
                    break
        if isinstance(data, list):
            return data, "json"
    except (ValueError, TypeError):
        pass
    decoder = json.JSONDecoder()
    cells: list = []
    pos = text.find("{")
    while pos != -1:
        try:
            obj, end = decoder.raw_decode(text, pos)
        except ValueError:
            break
        if isinstance(obj, dict):
            cells.append(obj)
        pos = text.find("{", end)
    return (cells, "salvaged_partial") if cells else (None, "failed")


def _convert_layout(raw: str, orig_w: int, orig_h: int) -> tuple[str, dict]:
    """Layout JSON -> Markdown + benchmark shapes.

    Coordinates: the model's [x1, y1, x2, y2] are pixels of the server-side
    smart_resize()d image (input_w x input_h). The author maps them back with
    x * orig_w / input_w (post_process_cells); normalising to the page as sent
    is therefore x / input_w and y / input_h.

    Markdown: layoutjson2md() order and formatting, INCLUDING Page-header and
    Page-footer (the author's _nohf variant exists only for their benchmarks).
    Picture cells: the author embeds a base64 crop; a short placeholder link is
    written instead so a page result does not carry image bytes.
    """
    input_h, input_w = _smart_resize(orig_h, orig_w)
    cells, status = _load_cells(raw)
    structured: dict = {
        "layout_parse": status,
        "preprocessing": {
            "sent": "original resolution, EXIF-transposed, RGB (RGBA on white), PNG",
            "original_size": [orig_w, orig_h],
            "model_input_size": [input_w, input_h],
            "server_resize": f"smart_resize factor={IMAGE_FACTOR} "
                             f"min_pixels={MIN_PIXELS} max_pixels={MAX_PIXELS}",
            "expected_visual_tokens": (input_w // IMAGE_FACTOR) * (input_h // IMAGE_FACTOR),
            "fitz_preprocess": False,
            "bbox_mapping": "model px / model_input_size -> 0-1 of the page as sent",
        },
    }
    if cells is None:
        # Nothing parseable: return the model's own words so the text leg
        # still scores what it read. Visible via layout_parse="failed".
        return raw.strip(), structured

    layout_blocks: list[dict] = []
    text_regions: list[dict] = []
    tables: list[dict] = []
    md_items: list[str] = []
    for order, cell in enumerate(cells):
        if not isinstance(cell, dict):
            continue
        category = str(cell.get("category", ""))
        text = cell.get("text", "")
        text = text if isinstance(text, str) else ("" if text is None else str(text))
        bbox_px = cell.get("bbox")
        bbox = None
        if isinstance(bbox_px, (list, tuple)) and len(bbox_px) == 4:
            try:
                x0, y0, x1, y1 = (float(v) for v in bbox_px)
                bbox = [min(1.0, max(0.0, x0 / input_w)), min(1.0, max(0.0, y0 / input_h)),
                        min(1.0, max(0.0, x1 / input_w)), min(1.0, max(0.0, y1 / input_h))]
            except (TypeError, ValueError):
                bbox = None
        layout_blocks.append({"label": category, "bbox": bbox, "bbox_model_px": bbox_px,
                              "order": order, "content": text})

        if category == "Picture":
            md_items.append(f"![](picture_{order})")
        elif category == "Formula":
            md_items.append(_get_formula_in_markdown(text))
        else:
            md_items.append(_clean_text(text))
        if category == "Table" and "<table" in text.lower():
            tables.append({"html": text.strip(), "bbox": bbox, "order": order})
        if category != "Picture" and bbox is not None:
            plain = _plain_text(text)
            if plain:
                text_regions.append({"text": plain, "bbox": bbox, "label": category})

    structured["layout_blocks"] = layout_blocks
    if tables:
        structured["tables"] = tables
    if text_regions:
        structured["text_regions"] = text_regions
        structured["geometry_granularity"] = "block"
        structured["geometry_source"] = "prompt_layout_all_en cell bboxes (layout blocks)"
    return "\n\n".join(md_items), structured


def _post(path: str, payload: dict, timeout: float) -> dict:
    import json
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        f"{SERVER_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"dots.mocr request failed: HTTP {exc.code}: {body}") from exc


def _chat_payload(png_bytes: bytes, prompt: str, system_prompt: str | None,
                  max_tokens: int, response_format: dict | None) -> dict:
    """The author's inference_with_vllm() request shape."""
    import base64

    encoded = base64.b64encode(png_bytes).decode("ascii")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        {"type": "text", "text": f"{IMAGE_PLACEHOLDER}{prompt}"},
    ]})
    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
    }
    if response_format:
        payload["response_format"] = response_format
    return payload


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=1800,
    startup_timeout=1800,
    scaledown_window=30,
    secrets=[modal.Secret.from_name("huggingface-token")],
    max_containers=1,
)
class DotsMOCRInference:
    @modal.enter()
    def load_model(self) -> None:
        import os
        import subprocess
        import threading
        import time

        t0 = time.perf_counter()
        self._server_process = subprocess.Popen(
            [
                "vllm", "serve", MODEL_ID,
                "--revision", MODEL_REVISION,
                "--served-model-name", SERVED_MODEL_NAME,
                "--host", SERVER_HOST,
                "--port", str(SERVER_PORT),
                "--dtype", "bfloat16",
                # Author serve command: --tensor-parallel-size 1
                # --gpu-memory-utilization 0.9 --chat-template-content-format
                # string --served-model-name model --trust-remote-code
                "--tensor-parallel-size", "1",
                "--gpu-memory-utilization", "0.90",
                "--chat-template-content-format", "string",
                "--trust-remote-code",
                # Deployment caps (arithmetic at MAX_MODEL_LEN above).
                "--max-model-len", str(MAX_MODEL_LEN),
                "--max-num-seqs", str(BATCH_CONCURRENCY),
                # >= one max-size page's visual tokens + prompt, so a full page
                # prefills in one chunk.
                "--max-num-batched-tokens", "16384",
                "--limit-mm-per-prompt", '{"image": 1}',
                "--no-enable-prefix-caching",
                "--mm-processor-cache-gb", "0",
            ],
            env={**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0"},
        )
        self._wait_for_server()
        self._warmup_server()
        self._startup_gpu_seconds = time.perf_counter() - t0
        self._startup_cost_applied = False
        self._startup_cost_lock = threading.Lock()

    @modal.exit()
    def stop_server(self) -> None:
        process = getattr(self, "_server_process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except Exception:
                process.kill()

    # ------------------------------------------------------------------ API
    @modal.method()
    def run_ocr(self, image_bytes: bytes, mime_type: str = "image/png",
                prompt: str | None = None, response_format: dict | None = None) -> dict:
        response = self._run_item({"bytes": image_bytes, "mime_type": mime_type},
                                  prompt, response_format)
        startup_seconds, is_cold_start = self._take_startup_metadata()
        response["startup_gpu_seconds"] = startup_seconds
        response["is_cold_start"] = is_cold_start
        return response

    @modal.method()
    def run_ocr_batch(self, files: list[dict], prompt: str | None = None,
                      response_format: dict | None = None) -> dict:
        """One measured batch on one GPU. Per-item "prompt"/"response_format"
        keys override the call arguments."""
        import time
        from concurrent.futures import ThreadPoolExecutor

        started = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=BATCH_CONCURRENCY) as pool:
                results = list(pool.map(
                    lambda item: self._run_item(item, prompt, response_format), files))
        except Exception as exc:  # never raise out of a batch
            results = []
            for item in files:
                err = self._error_response(exc, item.get("prompt", prompt))
                err["document_path"] = item.get("document_path")
                results.append(err)
        batch_gpu_seconds = time.perf_counter() - started
        startup_seconds, is_cold_start = self._take_startup_metadata()
        return {
            "results": results,
            "batch_gpu_seconds": batch_gpu_seconds,
            "gpu_type": GPU_TYPE,
            "startup_gpu_seconds": startup_seconds,
            "is_cold_start": is_cold_start,
        }

    # ------------------------------------------------------------ internals
    def _run_item(self, item: dict, prompt: str | None, response_format: dict | None) -> dict:
        effective_prompt = item.get("prompt", prompt)
        try:
            response = self._infer(item["bytes"], effective_prompt,
                                   item.get("response_format", response_format))
        except Exception as exc:
            response = self._error_response(exc, effective_prompt or LAYOUT_PROMPT)
        response["document_path"] = item.get("document_path")
        return response

    def _infer(self, image_bytes: bytes, prompt: str | None, response_format: dict | None) -> dict:
        import time

        t0 = time.perf_counter()
        png_bytes, (width, height) = _normalized_png(image_bytes)
        layout_mode = prompt is None or prompt == LAYOUT_PROMPT
        task_prompt = LAYOUT_PROMPT if layout_mode else prompt
        system_prompt = None if layout_mode else GENERAL_SYSTEM_PROMPT
        # The layout pass is not a JSON-mode ask: its reply format is fixed by
        # the prompt, and constrained decoding would alter it.
        fmt = None if layout_mode else response_format
        payload = _chat_payload(png_bytes, task_prompt, system_prompt, MAX_NEW_TOKENS, fmt)
        data = _post("/chat/completions", payload, timeout=1500)
        choice = data["choices"][0]
        raw = choice["message"].get("content") or ""
        usage = data.get("usage") or {}

        generation = {
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "max_tokens_requested": MAX_NEW_TOKENS,
            "max_model_len": MAX_MODEL_LEN,
            "temperature": TEMPERATURE, "top_p": TOP_P, "seed": SEED,
        }
        if layout_mode:
            text, structured = _convert_layout(raw, width, height)
            structured["raw_model_output"] = raw
            structured["task"] = "prompt_layout_all_en"
        else:
            # General-QA answer (key_value/schema): returned as emitted; the
            # extraction scorer parses it.
            text = raw
            structured = {"task": "prompt_general", "system_prompt": GENERAL_SYSTEM_PROMPT}
            if fmt:
                structured["response_format_sent"] = fmt
        structured["generation"] = generation

        return {
            "raw_text": text,
            "structured_output": structured,
            "gpu_seconds": time.perf_counter() - t0,
            "gpu_type": GPU_TYPE,
            "model_version": MODEL_VERSION,
            "prompt_used": task_prompt,
            "error": None,
        }

    def _error_response(self, exc: Exception, prompt: str | None) -> dict:
        message = str(exc) or repr(exc)
        return {
            "raw_text": "",
            "structured_output": {},
            "gpu_seconds": 0.0,
            "gpu_type": GPU_TYPE,
            "model_version": MODEL_VERSION,
            "prompt_used": prompt,
            "error": f"{exc.__class__.__name__}: {message}",
        }

    def _take_startup_metadata(self) -> tuple[float, bool]:
        """Report startup separately once; never add it to inference GPU-seconds."""
        with self._startup_cost_lock:
            if not getattr(self, "_startup_cost_applied", True):
                self._startup_cost_applied = True
                return getattr(self, "_startup_gpu_seconds", 0.0), True
        return 0.0, False

    def _wait_for_server(self) -> None:
        import time
        import urllib.error
        import urllib.request

        deadline = time.time() + 900
        last_error = None
        while time.time() < deadline:
            status = self._server_process.poll()
            if status is not None:
                raise RuntimeError(f"dots.mocr vLLM server exited early with code {status}")
            try:
                with urllib.request.urlopen(HEALTH_URL, timeout=5) as response:
                    if response.status == 200:
                        with urllib.request.urlopen(f"{SERVER_URL}/models", timeout=5) as models:
                            if models.status == 200:
                                return
            except (OSError, urllib.error.URLError) as exc:
                last_error = exc
            time.sleep(5)
        raise TimeoutError(f"dots.mocr vLLM server did not become ready: {last_error}")

    def _warmup_server(self) -> None:
        """A page-sized image (1240x1754 -> 1232x1764 after smart_resize, 2,772
        visual tokens) on the layout prompt, so the vision encoder and the
        template path are exercised before the first timed batch. Raises: a
        server that cannot answer this must not be benchmarked."""
        import io

        from PIL import Image, ImageDraw

        img = Image.new("RGB", (1240, 1754), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        for row in range(40):
            draw.text((80, 80 + row * 40), f"Warm-up line {row}: 12.34 Invoice total", fill=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = _post("/chat/completions",
                     _chat_payload(buf.getvalue(), LAYOUT_PROMPT, None, 32, None), timeout=600)
        if not data.get("choices"):
            raise RuntimeError(f"dots.mocr warm-up returned no choices: {data}")
