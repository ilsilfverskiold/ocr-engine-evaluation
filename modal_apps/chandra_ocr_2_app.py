"""
Modal app: Chandra OCR 2 — datalab-to/chandra-ocr-2. GPU: A10.

Size: the HF safetensors index reports 5,295,564,288 BF16 parameters
(10.59 GB file) — "4B" in docs/engine-expansion-research.md is wrong; Surya's
model card lists Chandra OCR 2 at 5.3B. Architecture Qwen3_5ForConditionalGeneration.

Route (all read from chandra-ocr v0.2.0, github.com/datalab-to/chandra/tree/v0.2.0;
master is 11 commits ahead with README/olmOCR-bench/hf-backend changes only):
- The official server recipe is `chandra_vllm` (chandra/scripts/vllm.py), a
  `docker run vllm/vllm-openai:v0.17.0 ...` command. Modal cannot run Docker,
  so @modal.enter starts the same `vllm serve` flags from a pip-installed
  vllm==0.17.0 and waits for /health and /v1/models.
- Inference uses the official `InferenceManager(method="vllm").generate(batch,
  ...)` (chandra/model/__init__.py) pointed at that server with the kwarg
  `vllm_api_base`. The manager's client (chandra/model/vllm.py) owns request
  shape, scale_to_fit, temperature 0 / top_p 0.1 and repeat-retries.
- Conversion: BatchInputItem(image, prompt_type="ocr_layout"), with
  include_headers_footers=True (the CLI default excludes Page-Header/Footer)
  and include_images=True (CLI default). Max output = settings.MAX_OUTPUT_TOKENS
  = 12,384. settings.IMAGE_DPI=192 governs PDF rendering only (input.py
  load_pdf_images); the corpus is images, so the image path applies:
  chandra.input.load_image upsamples any image whose width or height is below
  settings.MIN_IMAGE_DIM=1536 so the SHORT side becomes 1536 (LANCZOS).
- Extraction legs: BatchInputItem.prompt is forwarded verbatim
  (vllm.py `_generate`: `prompt = item.prompt` if set). The manager then runs
  its layout parsers over every reply, so `raw` is the answer and is kept.
  Chandra's vLLM client sends no response_format / guided decoding, so the
  harness's JSON constraint cannot be applied on this route; that is recorded
  per item (response_format_applied=False), not silently dropped.

Batching owner: vLLM continuous batching fed by the manager's own
ThreadPoolExecutor (generate_vllm max_workers). Conversion and extraction
items go to ONE manager.generate call (each BatchInputItem carries its prompt).

Licensing: code Apache-2.0; weights under a modified OpenRAIL-M licence
(free for research, personal use and startups under $2M funding/revenue;
"cannot be used competitively with our API" — HF model card). Distinct from
Surya's terms. Benchmarking is research use; no commercial deployment implied.
"""

import modal

GPU_TYPE = "A10"
MODEL_ID = "datalab-to/chandra-ocr-2"
# huggingface.co/api/models/datalab-to/chandra-ocr-2 on 2026-09-30.
MODEL_REVISION = "af93b47dba1b47b6640c86ccf487ed2260ab9a09"
CHANDRA_VERSION = "0.2.0"          # PyPI latest on 2026-09-30 (tag v0.2.0)
# chandra/scripts/vllm.py runs "vllm/vllm-openai:v0.17.0". Same version from
# PyPI; it registers Qwen3_5ForConditionalGeneration and ships its own
# qwen3_5 config class (vllm/transformers_utils/configs/qwen3_5.py).
VLLM_VERSION = "0.17.0"

# settings.VLLM_MODEL_NAME = "chandra"; chandra_vllm serves under that name.
SERVED_MODEL_NAME = "chandra"
VLLM_PORT = 8000
VLLM_BASE = f"http://127.0.0.1:{VLLM_PORT}"
VENV = "/opt/vllmenv"

# --- server flags: chandra/scripts/vllm.py (v0.2.0). get_gpu_settings("a10"):
# ratio 24/80 = 0.3 -> max_batched_tokens 2**floor(log2(2457.6)) = 2048,
# max_num_seqs max(8, (int(64*0.3)//8)*8) = 16. MTP is off by default there
# ("unstable with vLLM").
VLLM_MAX_MODEL_LEN = 18000
VLLM_GPU_MEMORY_UTILIZATION = 0.85
VLLM_MAX_NUM_SEQS = 16
VLLM_MAX_NUM_BATCHED_TOKENS = 2048
VLLM_DTYPE = "bfloat16"            # original precision (config dtype bfloat16)
MM_PROCESSOR_KWARGS = {"min_pixels": 3136, "max_pixels": 6291456}

# Active requests fed to vLLM (generate_vllm max_workers). 1 = the skill's
# first fit probe on A10: 5.3B BF16 weights leave ~7-8 GiB of the 0.85 x 22.5
# GiB pool for KV/state/activations, and a max-size page is ~6.1k visual
# tokens + up to ~11.8k output. Sweep 1/2/4/8 on one warm A10, then L40S.
BATCH_CONCURRENCY = 1

# settings.MAX_OUTPUT_TOKENS = 12384 (chandra/settings.py, README). Passing
# max_output_tokens=None lets the manager apply it; recorded per item.
# Context arithmetic: scale_to_fit caps pages at 6,291,456 px -> <= 6,144
# visual tokens (32x32 px each: patch 16 x merge 2). 6,144 + the ~0.5k-token
# OCR_LAYOUT_PROMPT + 12,384 > 18,000, and vLLM 0.17.0's get_max_tokens
# clamps max_tokens to max_model_len - input_length instead of rejecting, so
# the largest pages get an effective cap of ~11.3k. Kept at Chandra's values.
MAX_OUTPUT_TOKENS = 12384

INCLUDE_HEADERS_FOOTERS = True
INCLUDE_IMAGES = True              # CLI default (--include-images)
IMAGE_LABELS = frozenset({"Image", "Figure"})      # chandra/output.py image blocks
NON_TEXT_LABELS = IMAGE_LABELS | {"Blank-Page"}


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("python3-venv")
    .pip_install(f"chandra-ocr=={CHANDRA_VERSION}")
    # vLLM in its own venv: chandra's client stack (openai>=2.2.0) and
    # vllm 0.17.0 (openai<2.25.0, torch==2.10.0, transformers<5) are kept apart
    # so neither resolver can move the other's pins.
    .run_commands(
        f"python -m venv {VENV}",
        f"{VENV}/bin/pip install --no-cache-dir vllm=={VLLM_VERSION}",
        f"{VENV}/bin/python -c 'import vllm; print(\"vllm\", vllm.__version__)'",
    )
    .run_commands(
        f"{VENV}/bin/python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id='{MODEL_ID}', revision='{MODEL_REVISION}')\"",
        secrets=[modal.Secret.from_name("huggingface-token")],
    )
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

app = modal.App("ocr-bench-chandra-ocr-2")


# --------------------------------------------------------------------------
# Pure helpers (self-contained copy; the container cannot import repo modules)
# --------------------------------------------------------------------------

def _load_page(image_bytes: bytes, tmpdir: str, index: int):
    """EXIF-transpose + RGB, then Chandra's own image loader
    (chandra.input.load_image: RGB + short-side upsample to MIN_IMAGE_DIM)."""
    import io
    from pathlib import Path

    from PIL import Image, ImageOps
    from chandra.input import load_image

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    original = img.size
    path = Path(tmpdir) / f"{index:04d}.png"
    img.save(path, format="PNG")
    loaded = load_image(str(path))
    return loaded, original


def _html_to_text(html_text: str) -> str:
    from bs4 import BeautifulSoup

    return BeautifulSoup(html_text or "", "html.parser").get_text(" ", strip=True)


def _norm_bbox(bbox, width: float, height: float):
    """Chandra chunk bbox (pixels of the loaded image) -> 0-1.

    The loader's upsample is uniform, so page-relative coordinates are the
    same for the original image. Degenerate boxes — parse_layout substitutes
    [0, 0, 1, 1] before scaling when data-bbox fails to parse — return None.
    """
    if not bbox or not width or not height or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    clamp = lambda v: min(1.0, max(0.0, v))  # noqa: E731
    out = [clamp(x0 / width), clamp(y0 / height), clamp(x1 / width), clamp(y1 / height)]
    if out[2] - out[0] <= 0.001 or out[3] - out[1] <= 0.001:
        return None
    return out


def _visual_tokens_estimate(image) -> int | None:
    """ESTIMATE of visual tokens after Chandra's client resize (scale_to_fit)
    at 32x32 px per token (config: patch_size 16, spatial_merge_size 2)."""
    try:
        from chandra.model.util import scale_to_fit

        w, h = scale_to_fit(image).size
        return max(1, round(w / 32)) * max(1, round(h / 32))
    except Exception:
        return None


def _generation_record(out, visual_tokens) -> dict:
    """Chandra's GenerationResult carries raw/token_count/error only — no
    finish_reason — so truncation is inferred from token counts."""
    tokens = int(getattr(out, "token_count", 0) or 0)
    cap = MAX_OUTPUT_TOKENS
    if visual_tokens:
        cap = min(MAX_OUTPUT_TOKENS, VLLM_MAX_MODEL_LEN - visual_tokens)
    return {
        "finish_reason": None,
        "finish_reason_note": "not exposed by chandra generate_vllm",
        "completion_tokens": tokens,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "max_model_len": VLLM_MAX_MODEL_LEN,
        "visual_tokens_estimate": visual_tokens,
        "effective_cap_estimate": cap,
        "possible_truncation": bool(tokens and tokens >= cap - 256),
        "error": bool(getattr(out, "error", False)),
        "client_retry_policy": "chandra generate_vllm: up to MAX_VLLM_RETRIES=6 "
                               "retries on error or detected repetition; retry "
                               "temperature min(0.2*(n+1), 0.8), top_p 0.95",
    }


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=3600,
    startup_timeout=1800,
    scaledown_window=30,
    secrets=[modal.Secret.from_name("huggingface-token")],
    max_containers=1,
)
class ChandraOCR2Inference:
    @modal.enter()
    def load_model(self) -> None:
        import json
        import os
        import subprocess
        import threading
        import time

        t0 = time.perf_counter()
        self._startup_lock = threading.Lock()
        self._startup_cost_applied = False

        cmd = [
            f"{VENV}/bin/vllm", "serve", MODEL_ID,
            "--revision", MODEL_REVISION,
            "--host", "127.0.0.1", "--port", str(VLLM_PORT),
            "--served-model-name", SERVED_MODEL_NAME,
            "--no-enforce-eager",
            "--max-num-seqs", str(VLLM_MAX_NUM_SEQS),
            "--dtype", VLLM_DTYPE,
            "--max-model-len", str(VLLM_MAX_MODEL_LEN),
            "--max-num-batched-tokens", str(VLLM_MAX_NUM_BATCHED_TOKENS),
            "--gpu-memory-utilization", str(VLLM_GPU_MEMORY_UTILIZATION),
            "--enable-prefix-caching",
            "--mm-processor-kwargs", json.dumps(MM_PROCESSOR_KWARGS),
        ]
        self._server = subprocess.Popen(
            cmd,
            env={**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0",
                 "HF_HUB_OFFLINE": "1"},
        )
        self._wait_for_server()

        # chandra.settings reads the environment at import (pydantic-settings).
        os.environ.update({
            "VLLM_API_BASE": f"{VLLM_BASE}/v1",
            "VLLM_MODEL_NAME": SERVED_MODEL_NAME,
            "MODEL_CHECKPOINT": MODEL_ID,
            "MAX_OUTPUT_TOKENS": str(MAX_OUTPUT_TOKENS),
        })
        from chandra.model import InferenceManager
        from chandra.settings import settings as chandra_settings

        if int(chandra_settings.MAX_OUTPUT_TOKENS) != MAX_OUTPUT_TOKENS:
            raise RuntimeError(
                f"chandra MAX_OUTPUT_TOKENS={chandra_settings.MAX_OUTPUT_TOKENS}, "
                f"expected {MAX_OUTPUT_TOKENS}")
        self._min_image_dim = int(chandra_settings.MIN_IMAGE_DIM)
        self._manager = InferenceManager(method="vllm")
        self._config_summary = (
            f"chandra-ocr=={CHANDRA_VERSION} InferenceManager(vllm) ocr_layout "
            f"headers_footers={INCLUDE_HEADERS_FOOTERS}, vllm=={VLLM_VERSION} bf16 "
            f"ctx={VLLM_MAX_MODEL_LEN} max_tokens={MAX_OUTPUT_TOKENS} "
            f"workers={BATCH_CONCURRENCY}"
        )
        self._warmup()
        self._startup_gpu_seconds = time.perf_counter() - t0

    @modal.exit()
    def shutdown(self) -> None:
        server = getattr(self, "_server", None)
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except Exception:
                server.kill()

    # ------------------------------------------------------------------ API
    @modal.method()
    def run_ocr(self, image_bytes: bytes, mime_type: str = "image/png",
                prompt: str | None = None,
                response_format: dict | None = None) -> dict:
        batch = self._run_batch_impl([{
            "document_path": None, "bytes": image_bytes, "mime_type": mime_type,
            "prompt": prompt, "response_format": response_format,
        }])
        response = batch["results"][0]
        response["startup_gpu_seconds"] = batch["startup_gpu_seconds"]
        response["is_cold_start"] = batch["is_cold_start"]
        response["gpu_seconds"] = batch["batch_gpu_seconds"]
        response["gpu_type"] = GPU_TYPE
        return response

    @modal.method()
    def run_ocr_batch(self, files: list[dict], prompt: str | None = None,
                      response_format: dict | None = None) -> dict:
        items = []
        for f in files:
            item = dict(f)
            item.setdefault("prompt", prompt)
            item.setdefault("response_format", response_format)
            items.append(item)
        return self._run_batch_impl(items)

    # ------------------------------------------------------------ internals
    def _run_batch_impl(self, items: list[dict]) -> dict:
        import tempfile
        import time

        from chandra.model.schema import BatchInputItem

        started = time.perf_counter()
        results: list[dict | None] = [None] * len(items)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                pages, batch, batch_idx = {}, [], []
                for index, item in enumerate(items):
                    try:
                        pages[index] = _load_page(item["bytes"], tmpdir, index)
                    except Exception as exc:
                        results[index] = self._error_item(item, exc)
                        continue
                    loaded = pages[index][0]
                    if item.get("prompt"):
                        batch.append(BatchInputItem(image=loaded, prompt=item["prompt"]))
                    else:
                        batch.append(BatchInputItem(image=loaded, prompt_type="ocr_layout"))
                    batch_idx.append(index)

                if batch:
                    outputs = self._manager.generate(
                        batch,
                        include_headers_footers=INCLUDE_HEADERS_FOOTERS,
                        include_images=INCLUDE_IMAGES,
                        max_workers=BATCH_CONCURRENCY,
                        vllm_api_base=f"{VLLM_BASE}/v1",
                    )
                    if len(outputs) != len(batch):
                        raise RuntimeError(f"manager returned {len(outputs)} outputs "
                                           f"for {len(batch)} items")
                    for index, out in zip(batch_idx, outputs):
                        loaded, original = pages[index]
                        try:
                            if items[index].get("prompt"):
                                results[index] = self._extraction_response(
                                    items[index], out, loaded)
                            else:
                                results[index] = self._conversion_response(
                                    items[index], out, loaded, original)
                        except Exception as exc:
                            results[index] = self._error_item(items[index], exc)
        except Exception as exc:
            results = [self._error_item(item, exc) for item in items]

        for index, item in enumerate(items):
            if results[index] is None:
                results[index] = self._error_item(
                    item, RuntimeError("no result produced for item"))

        batch_gpu_seconds = time.perf_counter() - started
        startup_seconds, is_cold_start = self._take_startup_metadata()
        return {
            "results": results,
            "batch_gpu_seconds": batch_gpu_seconds,
            "gpu_type": GPU_TYPE,
            "startup_gpu_seconds": startup_seconds,
            "is_cold_start": is_cold_start,
        }

    def _conversion_response(self, item, out, loaded, original) -> dict:
        import re

        width, height = (out.page_box[2], out.page_box[3]) if out.page_box else loaded.size
        layout_blocks, text_regions, image_blocks = [], [], []
        for order, chunk in enumerate(out.chunks or []):
            label = chunk.get("label")
            content = chunk.get("content") or ""
            bbox = _norm_bbox(chunk.get("bbox"), width, height)
            layout_blocks.append({"label": label, "order": order, "bbox": bbox,
                                  "bbox_px": chunk.get("bbox"), "content": content})
            text = _html_to_text(content)
            if label in IMAGE_LABELS:
                # Model-written image descriptions / chart data (the prompt asks
                # for them). Kept apart from text geometry.
                image_blocks.append({"label": label, "bbox": bbox, "content": content,
                                     "text": text})
                continue
            if text and bbox and label not in NON_TEXT_LABELS:
                text_regions.append({"text": text, "bbox": bbox, "label": label})

        tables = [{"html": m.group(0)} for m in
                  re.finditer(r"<table\b.*?</table>", out.html or "", flags=re.I | re.S)]
        structured: dict = {
            "html": out.html or "",
            "layout_blocks": layout_blocks,
            "raw_model_output": out.raw or "",
            "generation": _generation_record(out, _visual_tokens_estimate(loaded)),
            "preprocessing": {
                "exif_transpose": True,
                "chandra_load_image": f"short side upsampled to {self._min_image_dim} "
                                      f"px when either side < {self._min_image_dim}",
                "original_size": list(original),
                "loaded_size": [loaded.width, loaded.height],
                "client_resize": "chandra scale_to_fit (<=6,291,456 px, 28 px grid); "
                                 "bboxes are 0-1000 page-relative, mapped by "
                                 "chandra parse_layout to loaded-image pixels",
            },
            "render_options": {"include_headers_footers": INCLUDE_HEADERS_FOOTERS,
                               "include_images": INCLUDE_IMAGES},
        }
        if tables:
            structured["tables"] = tables
        if text_regions:
            structured["text_regions"] = text_regions
            structured["geometry_granularity"] = "block"
        if image_blocks:
            structured["image_blocks"] = image_blocks
        error = None
        if out.error and not (out.raw or "").strip():
            error = "GenerationError: chandra vLLM request failed after retries"
        return {
            # Chandra's own Markdown (markdownify over its HTML), tables as HTML.
            "raw_text": out.markdown or "",
            "structured_output": structured,
            "model_version": f"{MODEL_ID}@{MODEL_REVISION[:12]} {self._config_summary}",
            "prompt_used": "ocr_layout (chandra OCR_LAYOUT_PROMPT)",
            "document_path": item.get("document_path"),
            "error": error,
        }

    def _extraction_response(self, item, out, loaded) -> dict:
        fmt = item.get("response_format")
        error = None
        if out.error and not (out.raw or "").strip():
            error = "GenerationError: chandra vLLM request failed after retries"
        return {
            "raw_text": out.raw or "",
            "structured_output": {
                "raw_model_output": out.raw or "",
                "generation": _generation_record(out, _visual_tokens_estimate(loaded)),
                "response_format_requested": fmt,
                # chandra/model/vllm.py builds chat.completions.create with
                # model/messages/max_tokens/temperature/top_p only.
                "response_format_applied": False,
            },
            "model_version": (
                f"{MODEL_ID}@{MODEL_REVISION[:12]} chandra-ocr=={CHANDRA_VERSION} "
                f"InferenceManager(vllm) BatchInputItem.prompt, vllm=={VLLM_VERSION} "
                f"bf16 max_tokens={MAX_OUTPUT_TOKENS}"
            ),
            "prompt_used": item.get("prompt"),
            "document_path": item.get("document_path"),
            "error": error,
        }

    def _error_item(self, item: dict, exc: Exception) -> dict:
        return {
            "raw_text": "",
            "structured_output": {},
            "model_version": f"{MODEL_ID}@{MODEL_REVISION[:12]} chandra-ocr=={CHANDRA_VERSION}",
            "prompt_used": item.get("prompt"),
            "document_path": item.get("document_path"),
            "error": f"{exc.__class__.__name__}: {str(exc) or repr(exc)}",
        }

    def _take_startup_metadata(self) -> tuple[float, bool]:
        with self._startup_lock:
            if not self._startup_cost_applied:
                self._startup_cost_applied = True
                return getattr(self, "_startup_gpu_seconds", 0.0), True
        return 0.0, False

    def _wait_for_server(self) -> None:
        import json
        import time
        import urllib.error
        import urllib.request

        deadline = time.time() + 1500
        last = None
        while time.time() < deadline:
            code = self._server.poll()
            if code is not None:
                raise RuntimeError(f"vLLM server exited early with code {code}")
            try:
                with urllib.request.urlopen(f"{VLLM_BASE}/health", timeout=5) as r:
                    if r.status == 200:
                        with urllib.request.urlopen(f"{VLLM_BASE}/v1/models",
                                                    timeout=5) as m:
                            ids = [d.get("id") for d in json.loads(m.read()).get("data", [])]
                            if SERVED_MODEL_NAME not in ids:
                                raise RuntimeError(f"vLLM serves {ids}, expected "
                                                   f"{SERVED_MODEL_NAME!r}")
                            return
            except (OSError, urllib.error.URLError) as exc:
                last = exc
            time.sleep(5)
        raise TimeoutError(f"vLLM server never became ready: {last}")

    def _warmup(self) -> None:
        """One ocr_layout conversion of a synthetic text page."""
        from PIL import Image, ImageDraw
        from chandra.model.schema import BatchInputItem

        page = Image.new("RGB", (1536, 2048), (255, 255, 255))
        draw = ImageDraw.Draw(page)
        draw.text((120, 120), "INVOICE No. 1042", fill=(0, 0, 0))
        for row in range(12):
            draw.text((120, 240 + 48 * row),
                      f"Item {row + 1}    Qty {row + 2}    Amount {row * 7 + 3}.00",
                      fill=(0, 0, 0))
        self._manager.generate(
            [BatchInputItem(image=page, prompt_type="ocr_layout")],
            include_headers_footers=INCLUDE_HEADERS_FOOTERS,
            max_workers=1,
            vllm_api_base=f"{VLLM_BASE}/v1",
        )
