"""Modal app: LightOnOCR-2-1B — lightonai/LightOnOCR-2-1B
Card's vLLM OpenAI-server recipe, image-only message. GPU: A10 (initial)."""

import modal

GPU_TYPE = "A10"
# The card does not pin a vLLM version. 0.22.1 registers
# LightOnOCRForConditionalGeneration (vllm/model_executor/models/registry.py @ v0.22.1)
# and matches the ovisocr2 app, so both new engines share one engine version.
VLLM_VERSION = "0.22.1"
# vLLM v0.22.1 CI lock (requirements/test/cuda.txt): transformers==5.5.3,
# pillow==10.4.0, mistral-common==1.11.2. The card requires transformers v5 for
# LightOnOCR-2 ("avaible in latest transformers release starting from v5").
TRANSFORMERS_VERSION = "5.5.3"
PILLOW_VERSION = "10.4.0"
MISTRAL_COMMON_VERSION = "1.11.2"
MODEL_ID = "lightonai/LightOnOCR-2-1B"
# https://huggingface.co/api/models/lightonai/LightOnOCR-2-1B -> sha
# (lastModified 2026-07-08), pinned for reproducible inference.
MODEL_REVISION = "c97bd377f04481830395218fa8951df9deaba756"
SERVED_MODEL_NAME = "lighton-ocr-2"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8003
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/v1"

# --- Card request settings (README "Usage with vLLM") ---------------------------
MAX_TOKENS = 4096     # card: "max_tokens": 4096
TEMPERATURE = 0.2     # card: "temperature": 0.2  (also generation_config.json)
TOP_P = 0.9           # card: "top_p": 0.9        (also generation_config.json)
# generation_config.json also sets top_k=0 (disabled); vLLM serve applies the
# model's generation_config as request defaults, so it is not re-sent here.

# --- Context arithmetic ---------------------------------------------------------
# Vision (config.json): pixtral encoder, patch_size=14, spatial_merge_size=2,
# vision_config.image_size=1540; processor_config.json size.longest_edge=1540
# (PixtralImageProcessorFast, resample=3 bicubic, downscale only, aspect kept).
# One LLM visual token per 28x28 px; LightOnOCR drops Pixtral's break/end tokens,
# so visual tokens = ceil(h/28) * ceil(w/28) after the <=1540 resize.
#   Max (1540x1540):          55 * 55 = 3,025 visual tokens.
#   Corpus page 2480x3508:    ratio 3508/1540 = 2.278 -> 1088 x 1540
#                             -> ceil(1540/28)=55 rows * ceil(1088/28)=39 cols
#                             = 2,145 visual tokens.
# Text: system+user+assistant template around the image = 14 tokens (counted
# offline with tokenizer.json @ MODEL_REVISION; jinja whitespace may add a few).
# Budget: 3,025 + ~20 + 4,096 (card max_tokens) = 7,141
#         <= MAX_MODEL_LEN 8,192 (1,051 headroom)
#         <= 16,384 (text_config.max_position_embeddings).
# Note: the paper trained with a 6,144 maximum sequence length and filtered
# completions longer than 3,100 tokens; the card's 4,096 output cap is kept as
# published. Dense pages can end with finish_reason "length" — recorded per item.
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.90  # vLLM default; the card sets none.

# --- Memory / concurrency ---------------------------------------------------------
# KV per token (bf16): 28 layers * 2 (K,V) * 8 KV heads * 128 head_dim * 2 B
#   = 114,688 B = 112 KiB/token -> worst-case sequence 7,141 tokens = 781 MiB.
# Weights ~1B params bf16 ~= 2 GB. A10 24 GB * 0.90 = ~21.6 GB budget; minus
# weights, vision activations for a 3,025-token image and CUDA graphs leaves
# roughly 16 GiB of KV: 16 worst-case sequences = 12.2 GiB, so 16 active
# requests fit without preemption. Outer queue (EngineSpec.batch_size) 24.
# Next GPU to probe: L4 (same 24 GB, lower rate); L40S only if a larger active
# batch measurably lowers GPU-seconds per page.
BATCH_CONCURRENCY = 16

app = modal.App("ocr-bench-lighton-ocr-2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        f"transformers=={TRANSFORMERS_VERSION}",
        f"Pillow=={PILLOW_VERSION}",
        f"mistral-common[image]=={MISTRAL_COMMON_VERSION}",
    )
    # Bake the pinned weights into the image.
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

_MODEL_VERSION = (
    f"{MODEL_ID}@{MODEL_REVISION[:12]} vllm=={VLLM_VERSION} openai-server bf16 "
    f"image-only T={TEMPERATURE} top_p={TOP_P} max_tokens={MAX_TOKENS} "
    f"max_model_len={MAX_MODEL_LEN} longest_edge=1540(processor)"
)


def _png_b64_and_size(image_bytes: bytes) -> tuple[str, list[int]]:
    """EXIF-transpose + RGB, re-encode as PNG. No app-side resize: the model's
    own processor applies the card's longest-edge-1540 policy."""
    import base64
    import io

    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii"), list(img.size)


def _expected_resize(width: int, height: int, bound: int = 1540) -> list[int]:
    """Size the Pixtral processor resizes to (downscale-only, aspect kept),
    mirroring vLLM PixtralHFEncoderInfo.get_patch_grid_size. Informational."""
    import math

    ratio = max(width / bound, height / bound)
    if ratio > 1:
        return [int(math.floor(width / ratio)), int(math.floor(height / ratio))]
    return [width, height]


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=1800,
    startup_timeout=1800,
    scaledown_window=30,
    secrets=[modal.Secret.from_name("huggingface-token")],
    max_containers=1,
)
class LightOnOCR2Inference:
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
                "--tokenizer-revision", MODEL_REVISION,
                "--served-model-name", SERVED_MODEL_NAME,
                "--host", SERVER_HOST,
                "--port", str(SERVER_PORT),
                "--dtype", "bfloat16",  # config.json dtype
                # --- card flags, verbatim ---
                "--limit-mm-per-prompt", '{"image": 1}',
                "--mm-processor-cache-gb", "0",
                "--no-enable-prefix-caching",
                # --- deployment bounds (see arithmetic above) ---
                "--max-model-len", str(MAX_MODEL_LEN),
                "--gpu-memory-utilization", f"{GPU_MEMORY_UTILIZATION:.2f}",
            ],
            env={**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0"},
        )
        self._wait_for_server()
        self._warmup_server()
        self._startup_gpu_seconds = time.perf_counter() - t0
        self._startup_cost_applied = False
        self._startup_cost_lock = threading.Lock()

    @modal.exit()
    def shutdown(self) -> None:
        proc = getattr(self, "_server_process", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()

    # --- public methods -----------------------------------------------------------
    @modal.method()
    def run_ocr(self, image_bytes: bytes, mime_type: str = "image/png",
                prompt: str | None = None, response_format: dict | None = None) -> dict:
        import time

        t0 = time.perf_counter()
        response = self._run_item(
            {"bytes": image_bytes, "mime_type": mime_type, "document_path": None},
            prompt, response_format,
        )
        response["gpu_seconds"] = time.perf_counter() - t0
        response["gpu_type"] = GPU_TYPE
        startup_seconds, is_cold_start = self._take_startup_metadata()
        response["startup_gpu_seconds"] = startup_seconds
        response["is_cold_start"] = is_cold_start
        return response

    @modal.method()
    def run_ocr_batch(self, files: list[dict], prompt: str | None = None,
                      response_format: dict | None = None) -> dict:
        """One measured batch on one GPU; vLLM continuous batching over
        BATCH_CONCURRENCY in-flight requests."""
        import time
        from concurrent.futures import ThreadPoolExecutor

        started = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=BATCH_CONCURRENCY) as pool:
                results = list(pool.map(
                    lambda item: self._run_item(item, prompt, response_format), files))
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

    # --- internals ----------------------------------------------------------------
    def _run_item(self, item: dict, prompt: str | None,
                  response_format: dict | None) -> dict:
        doc = item.get("document_path")
        try:
            item_prompt = item.get("prompt", prompt)
            item_format = item.get("response_format", response_format)
            if item_prompt is not None or item_format is not None:
                # accepts_task=False: the card defines no text-prompt route, and a
                # silently ignored prompt would be scored as if it had been obeyed.
                raise ValueError(
                    "lighton_ocr_2 takes no prompt/response_format: the official "
                    "message is image-only")
            return self._convert(item["bytes"], doc)
        except Exception as exc:
            return self._error_item(doc, exc)

    def _convert(self, image_bytes: bytes, document_path, max_tokens: int = MAX_TOKENS) -> dict:
        import json
        import re
        import urllib.error
        import urllib.request

        encoded, size = _png_b64_and_size(image_bytes)
        payload = {
            "model": SERVED_MODEL_NAME,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }],
            }],
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        }
        request = urllib.request.Request(
            f"{SERVER_URL}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LightOnOCR-2 request failed: HTTP {exc.code}: {body}") from exc

        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        usage = data.get("usage") or {}
        structured: dict = {
            "generation": {
                "finish_reason": choice.get("finish_reason"),
                "stop_reason": choice.get("stop_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),  # includes visual tokens
                "completion_tokens": usage.get("completion_tokens"),
                "max_tokens": MAX_TOKENS,
                "max_model_len": MAX_MODEL_LEN,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "served_model": data.get("model"),
            },
            "preprocessing": {
                "exif_transpose": True, "color": "RGB",
                "input_size": size,
                # Card: "Render PDFs at 200 DPI ... target longest dimension of
                # 1540px ... Maintain aspect ratio". Corpus inputs are already
                # images (no PDF render step); the 1540 longest-edge resize is
                # done by the model's processor (processor_config.json).
                "processor_longest_edge": 1540,
                "expected_processor_size": _expected_resize(*size),
                "pdf_render_dpi": None,
            },
        }
        # Native HTML tables (paper: "we use HTML targets" for tables).
        tables = [{"html": m.group(0)}
                  for m in re.finditer(r"<table\b.*?</table>", text, flags=re.I | re.S)]
        if tables:
            structured["tables"] = tables
        # Figure placeholders (paper: ![image](image_N.png)); not text geometry.
        placeholders = re.findall(r"!\[image\]\(image_\d+\.png\)", text)
        if placeholders:
            structured["image_placeholders"] = placeholders

        return {
            # The model's own Markdown, unmodified.
            "raw_text": text,
            "structured_output": structured,
            "model_version": _MODEL_VERSION,
            "prompt_used": None,
            "document_path": document_path,
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

    def _wait_for_server(self) -> None:
        import time
        import urllib.error
        import urllib.request

        deadline = time.time() + 900
        last_error = None
        while time.time() < deadline:
            status = self._server_process.poll()
            if status is not None:
                raise RuntimeError(f"LightOnOCR-2 vLLM server exited early with code {status}")
            try:
                with urllib.request.urlopen(f"{SERVER_URL}/models", timeout=5) as response:
                    if response.status < 500:
                        return
            except (OSError, urllib.error.URLError) as exc:
                last_error = exc
            time.sleep(5)
        raise TimeoutError(f"LightOnOCR-2 vLLM server did not become ready: {last_error}")

    def _warmup_server(self) -> None:
        """One real request on a full-size page (exercises the 1540 resize and the
        largest visual path). Failure here fails startup instead of the batch."""
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (2480, 3508), (255, 255, 255)).save(buf, "PNG")
        result = self._convert(buf.getvalue(), None, max_tokens=8)
        if result.get("error"):
            raise RuntimeError(f"LightOnOCR-2 warm-up failed: {result['error']}")
