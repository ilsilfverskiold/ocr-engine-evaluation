"""
Modal app: DeepSeek-OCR-2 (3B MoE, ~0.5B active) — deepseek-ai/DeepSeek-OCR-2.
A NEW engine (`deepseek_ocr_2`); modal_apps/deepseek_ocr_app.py (v1) is left
untouched and keeps its own historical recipe.

Served by upstream vLLM (DeepseekOCR2ForCausalLM, vLLM >= 0.16.0) through its
OpenAI-compatible server, fed by a bounded thread pool so vLLM's continuous
batching owns the batch. GPU: L4 (first fit probe; see GPU_TYPE).

Sources (read 2026-09-30):
- Model card @ aaa02f38:  https://huggingface.co/deepseek-ai/DeepSeek-OCR-2
- Author repo @ 2f3699eb: https://github.com/deepseek-ai/DeepSeek-OCR-2
    DeepSeek-OCR2-master/DeepSeek-OCR2-vllm/{config.py,run_dpsk_ocr2_image.py,
    run_dpsk_ocr2_pdf.py,run_dpsk_ocr2_eval_batch.py,process/image_process.py,
    process/ngram_norepeat.py}
- vLLM recipe:            https://recipes.vllm.ai/deepseek-ai/DeepSeek-OCR-2
- vLLM v0.21.0 source:    vllm/model_executor/models/deepseek_ocr2.py,
    vllm/model_executor/models/deepseek_ocr.py (NGramPerReqLogitsProcessor),
    vllm/transformers_utils/configs/deepseek_vl2.py (model_type ->
    "deepseek_ocr2"), vllm/transformers_utils/chat_templates/
    template_deepseek_ocr.jinja, vllm/entrypoints/utils.py (get_max_tokens)
- Paper:                  https://arxiv.org/abs/2601.20552

What is deliberately NOT inherited from v1:
- The figure deep-parse stage ("Parse the figure."). That prompt is on the v1
  README only. The v2 card, README and config.py document exactly two prompts
  (grounding markdown and "Free OCR."), so a second-stage figure call would be
  an invented task for this checkpoint.
- max_tokens=2048. v1's cap was an empirical fix for v1 loops; v2's cap is
  derived from its own context arithmetic below.
- Geometry in structured_output["words"]. Grounding boxes are layout-region
  boxes and go to "text_regions" with geometry_granularity="block". (v1 also
  stored the REGION LABEL — "text", "title", "table" — as each box's text;
  see _parse_grounding for the correct content association.)
"""

import modal

# ---------------------------------------------------------------------------
# Identity and pins
# ---------------------------------------------------------------------------
# First fit probe per the modal-inference-engineer skill (L4/A10 at small
# concurrency, then L40S). Memory estimate, BF16:
#   weights                       6.78 GB  (safetensors, 3,389,119,360 params)
#   KV cache per token            2 (K,V) x 12 layers x 10 kv heads x 128 dim
#                                 x 2 B = 61,440 B  (config: use_mla=false)
#   one full 8,192-token sequence 0.47 GiB
#   BATCH_CONCURRENCY=4 sequences 1.9 GiB
#   + SAM/Qwen2 vision encoders, runtime, CUDA graphs: a few GB (unmeasured)
# Well inside 0.90 x 24 GB. Decode is memory-bandwidth bound, so A10
# (~600 GB/s vs L4 ~300 GB/s) is the next probe, then L40S.
GPU_TYPE = "L4"
VLLM_VERSION = "0.21.0"          # >= 0.16.0 has DeepseekOCR2ForCausalLM; same pin as v1
# vLLM 0.21.0 requires transformers>=4.56.0 (excluding 5.0-5.5.0); pinned so the
# image is reproducible. UNVERIFIED for this model: probe = image build + server start.
TRANSFORMERS_VERSION = "4.57.6"
MODEL_ID = "deepseek-ai/DeepSeek-OCR-2"
MODEL_REVISION = "aaa02f3811945a91062062994c5c4a3f4c0af2b0"
SERVED_MODEL_NAME = "deepseek-ocr-2"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8003
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/v1"
HEALTH_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/health"
BATCH_CONCURRENCY = 4

# ---------------------------------------------------------------------------
# Prompts — the only two the v2 card/README/config.py document:
#   "# document: <image>\n<|grounding|>Convert the document to markdown."
#   "# without layouts: <image>\nFree OCR."
# The "<image>\n" prefix is produced by vLLM: the deepseek_ocr2 chat template
# is `bos + content`, and the string-content renderer joins the image
# placeholder and our text with "\n", giving "<image>\n<prompt>" exactly.
# ---------------------------------------------------------------------------
GROUNDING_PROMPT = "<|grounding|>Convert the document to markdown."
FREE_OCR_PROMPT = "Free OCR."
DEFAULT_PROMPT = FREE_OCR_PROMPT  # default pass = the text leg's documented task

# ---------------------------------------------------------------------------
# Context arithmetic (hard limit, not a choice)
# ---------------------------------------------------------------------------
# config.json language_config.max_position_embeddings = 8192; every author
# script uses max_model_len=8192.
MAX_MODEL_LEN = 8192
# Visual tokens, vLLM 0.21.0 DeepseekOCR2ProcessingInfo.get_num_image_tokens
# (identical to the author's image_process.py):
#   global 1024x1024 view: ceil((1024/16)/4)^2 = 16^2           = 256
#   separator                                                    = 1
#   local 768x768 crops:   (12 x w_tiles) x (12 x h_tiles), w*h <= 6 (MAX_CROPS)
#                          -> at most 6 x 144                    = 864
#   maximum                                                      = 1,121
# (card: "(0-6)x144 + 256 visual tokens"; the +1 is the view separator.)
MAX_VISUAL_TOKENS = 1121
# Text tokens, counted with the pinned tokenizer.json:
#   bos 1 + "\n<|grounding|>Convert the document to markdown." 9 = 10
#   ("\nFree OCR." is 4)
MAX_PROMPT_TEXT_TOKENS = 10
SAFETY_HEADROOM = 32
# 8192 - 1121 - 10 - 32 = 7029. The worst-case page on the grounding prompt
# still fits. The author scripts request max_tokens=8192, i.e. "until the
# context ends"; the recipe's client example uses 2048. This is the author's
# behaviour made explicit. Longer schema prompts need no special case: vLLM
# 0.21.0 get_max_tokens() takes min(requested, max_model_len - input_len).
MAX_NEW_TOKENS = MAX_MODEL_LEN - MAX_VISUAL_TOKENS - MAX_PROMPT_TEXT_TOKENS - SAFETY_HEADROOM
# v1 history, kept as a warning rather than a cap: v1 at 6144 produced three
# runaway outputs (a repeated sentence, "<td></td>" runs, dot leaders). The
# author's own PDF pipeline treats a page with no EOS as a repetition failure
# (SKIP_REPEAT). Here such pages are kept and marked: finish_reason "length"
# and generation.eos_emitted=False in structured_output.

# Repetition control. The author's single-image script (run_dpsk_ocr2_image.py)
# uses NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=90,
# whitelist_token_ids={128821, 128822}); its PDF script uses 20/50, its
# benchmark-eval script 40/90, the vLLM recipe 30/90. The single-image author
# setting matches this benchmark's input (one page image).
# Whitelist ids verified against THIS checkpoint's tokenizer.json @ aaa02f38:
#   128821 = "<td>", 128822 = "</td>" (same ids as v1's tokenizer).
NGRAM_SIZE = 20
NGRAM_WINDOW = 90
NGRAM_WHITELIST = [128821, 128822]

# Grounding output geometry: the author scales every box by x/999*width,
# y/999*height of the ORIGINAL image (run_dpsk_ocr2_image.py:draw_bounding_boxes).
GROUNDING_COORD_MAX = 999.0
# Non-text layout label: the author crops '<|ref|>image<|/ref|>' regions out as
# pictures and replaces them with an image link in the Markdown.
IMAGE_LABEL = "image"

MODEL_VERSION = (
    f"{MODEL_ID}@{MODEL_REVISION[:12]} vllm=={VLLM_VERSION} "
    f"base1024+<=6x768crops bf16 ngram{NGRAM_SIZE}/{NGRAM_WINDOW} "
    f"max_new_tokens<={MAX_NEW_TOKENS}"
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

app = modal.App("ocr-bench-deepseek-ocr-2")


# ---------------------------------------------------------------------------
# Pure helpers (no model state)
# ---------------------------------------------------------------------------
def _normalized_png(image_bytes: bytes) -> tuple[bytes, tuple[int, int]]:
    """EXIF-transpose + RGB, as the author's load_image() does. No resize: the
    vLLM processor builds the 1024 global view and 768 crops itself."""
    import io

    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), img.size


def _plain_text(content: str) -> str:
    """Region content -> plain text for the geometry scorer. Tables lose their
    tags, headings their '#'. The unmodified content stays in layout_blocks."""
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


def _parse_grounding(raw: str) -> tuple[str, dict]:
    """Author's grounding post-processing, plus geometry harvesting.

    The grounding output is a sequence of
        <|ref|>LABEL<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>\\nCONTENT
    where LABEL is a layout category (title/text/table/image/...) and CONTENT is
    the Markdown/HTML that follows the tag, up to the next tag. The author's
    re_match() uses exactly this pattern and treats group 2 as `label_type`.
    Coordinates are 0-999 relative to the original page.

    Markdown: as the author's run_dpsk_ocr2_image.py — image spans become an
    image link, every other ref/det span is deleted, and \\coloneqq/\\eqqcolon are
    rewritten. HTML tables are kept as HTML.
    """
    import ast
    import re

    pattern = re.compile(r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>", re.S)
    matches = list(pattern.finditer(raw))

    layout_blocks: list[dict] = []
    text_regions: list[dict] = []
    for order, match in enumerate(matches):
        label = match.group(1).strip()
        end = matches[order + 1].start() if order + 1 < len(matches) else len(raw)
        content = raw[match.end():end].strip()
        try:
            boxes = ast.literal_eval(match.group(2).strip())
        except (ValueError, SyntaxError):
            boxes = None
        norm_boxes = []
        if isinstance(boxes, (list, tuple)):
            for box in boxes:
                if isinstance(box, (list, tuple)) and len(box) == 4:
                    try:
                        x0, y0, x1, y1 = (float(v) / GROUNDING_COORD_MAX
                                          for v in box)
                    except (TypeError, ValueError):
                        continue
                    norm_boxes.append([x0, y0, x1, y1])
        block = {"label": label, "order": order, "content": content,
                 "bbox": None, "boxes": norm_boxes, "raw_det": match.group(2).strip()}
        if norm_boxes:
            # One ref normally carries one box. If it carries several, the
            # content cannot be split between them, so the region is their
            # union and the individual boxes stay in layout_blocks.
            block["bbox"] = [min(b[0] for b in norm_boxes), min(b[1] for b in norm_boxes),
                             max(b[2] for b in norm_boxes), max(b[3] for b in norm_boxes)]
        layout_blocks.append(block)
        if block["bbox"] and label.lower() != IMAGE_LABEL:
            text = _plain_text(content)
            if text:
                text_regions.append({"text": text, "bbox": block["bbox"], "label": label})

    markdown = raw
    image_idx = 0
    for match in matches:
        span = match.group(0)
        if match.group(1).strip().lower() == IMAGE_LABEL:
            markdown = markdown.replace(span, f"![](images/{image_idx}.jpg)\n", 1)
            image_idx += 1
        else:
            markdown = markdown.replace(span, "", 1)
    markdown = markdown.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()

    structured: dict = {"layout_blocks": layout_blocks}
    if text_regions:
        structured["text_regions"] = text_regions
        structured["geometry_granularity"] = "block"
        structured["geometry_coordinates"] = "normalized_xyxy"
        structured["geometry_source"] = (
            "grounding <|det|> boxes, 0-999 of the original page, /999; "
            "region-level layout blocks, not words or lines")
    return markdown, structured


def _find_tables(text: str) -> list[dict]:
    import re

    return [{"html": m.group(0)}
            for m in re.finditer(r"<table\b.*?</table>", text, flags=re.I | re.S)]


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
        raise RuntimeError(f"DeepSeek-OCR-2 request failed: HTTP {exc.code}: {body}") from exc


def _chat_payload(png_bytes: bytes, prompt: str, max_tokens: int,
                  response_format: dict | None) -> dict:
    import base64

    encoded = base64.b64encode(png_bytes).decode("ascii")
    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,                 # all author scripts and the recipe
        "skip_special_tokens": False,       # ref/det tokens must survive to be parsed
        "vllm_xargs": {
            "ngram_size": NGRAM_SIZE,
            "window_size": NGRAM_WINDOW,
            "whitelist_token_ids": NGRAM_WHITELIST,
        },
    }
    if response_format:
        # The harness's JSON enforcement for extraction legs (json_object or
        # json_schema). Forwarded unchanged and recorded in the result.
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
class DeepSeekOCR2Inference:
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
                # Hard limit from config.json; vLLM refuses a larger value.
                "--max-model-len", str(MAX_MODEL_LEN),
                "--gpu-memory-utilization", "0.90",
                # The active-sequence bound is the thread pool; this makes it
                # explicit in the engine as well.
                "--max-num-seqs", str(BATCH_CONCURRENCY),
                "--limit-mm-per-prompt", '{"image": 1}',
                # Flags from the vLLM recipe's serve command. No
                # --trust-remote-code: the recipe does not use it, and vLLM ships
                # its own DeepseekOCR2 model, config and processor.
                "--logits_processors",
                "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor",
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
        keys override the call arguments (the harness sends one tagged batch
        mixing passes)."""
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
        effective_prompt = item.get("prompt", prompt) or DEFAULT_PROMPT
        try:
            response = self._infer(item["bytes"], effective_prompt,
                                   item.get("response_format", response_format))
        except Exception as exc:
            response = self._error_response(exc, effective_prompt)
        response["document_path"] = item.get("document_path")
        return response

    def _infer(self, image_bytes: bytes, prompt: str, response_format: dict | None) -> dict:
        import time

        t0 = time.perf_counter()
        png_bytes, (width, height) = _normalized_png(image_bytes)
        payload = _chat_payload(png_bytes, prompt, MAX_NEW_TOKENS, response_format)
        data = _post("/chat/completions", payload, timeout=900)
        choice = data["choices"][0]
        raw = choice["message"].get("content") or ""
        usage = data.get("usage") or {}
        finish_reason = choice.get("finish_reason")
        # The EOS text only appears if the server includes stop strings; the
        # author strips it the same way.
        raw = raw.replace("<｜end▁of▁sentence｜>", "")

        structured: dict = {
            "generation": {
                "finish_reason": finish_reason,
                "eos_emitted": finish_reason == "stop",
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "max_tokens_requested": MAX_NEW_TOKENS,
                "max_model_len": MAX_MODEL_LEN,
                "temperature": 0.0,
                "ngram": {"ngram_size": NGRAM_SIZE, "window_size": NGRAM_WINDOW,
                          "whitelist_token_ids": NGRAM_WHITELIST},
            },
            "image_size": [width, height],
            "rendered_prompt": f"<image>\n{prompt}",
        }
        if response_format:
            structured["response_format_sent"] = response_format

        is_extraction = bool(response_format)
        if "<|ref|>" in raw and not is_extraction:
            text, parsed = _parse_grounding(raw)
            structured.update(parsed)
            structured["raw_model_output"] = raw
        else:
            # Free OCR and extraction answers are returned as emitted. The text
            # scorer strips Markdown/HTML itself; the extraction scorer parses
            # the JSON itself.
            text = raw.strip()
            if text != raw:
                structured["raw_model_output"] = raw
        if not is_extraction:
            tables = _find_tables(text)
            if tables:
                structured["tables"] = tables

        return {
            "raw_text": text,
            "structured_output": structured,
            "gpu_seconds": time.perf_counter() - t0,
            "gpu_type": GPU_TYPE,
            "model_version": MODEL_VERSION,
            "prompt_used": prompt,
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
                raise RuntimeError(f"DeepSeek-OCR-2 vLLM server exited early with code {status}")
            try:
                with urllib.request.urlopen(HEALTH_URL, timeout=5) as response:
                    if response.status == 200:
                        with urllib.request.urlopen(f"{SERVER_URL}/models", timeout=5) as models:
                            if models.status == 200:
                                return
            except (OSError, urllib.error.URLError) as exc:
                last_error = exc
            time.sleep(5)
        raise TimeoutError(f"DeepSeek-OCR-2 vLLM server did not become ready: {last_error}")

    def _warmup_server(self) -> None:
        """A page-sized image (crop path: 1240x1754 -> multi-crop) on the
        grounding prompt, so the vision encoders, the crop path and the logits
        processor are exercised before the first timed batch. Raises: a server
        that cannot answer this must not be benchmarked."""
        import io

        from PIL import Image, ImageDraw

        img = Image.new("RGB", (1240, 1754), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        for row in range(40):
            draw.text((80, 80 + row * 40), f"Warm-up line {row}: 12.34 Invoice total", fill=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = _post("/chat/completions",
                     _chat_payload(buf.getvalue(), GROUNDING_PROMPT, 32, None), timeout=600)
        if not data.get("choices"):
            raise RuntimeError(f"DeepSeek-OCR-2 warm-up returned no choices: {data}")
