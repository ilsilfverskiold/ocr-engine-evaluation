"""
Modal app: PaddleOCR-VL 1.6 — PaddlePaddle/PaddleOCR-VL-1.6. GPU: A10 (fit probe).

A NEW engine (`paddleocr_vl_1_6`), not a change to `paddleocr_vl` (v1). The v1
app (modal_apps/paddleocr_vl_app.py) is left untouched.

Runs the documented page-parsing pipeline, pipeline_version="v1.6":
PP-DocLayoutV3 layout detection -> per-region crops -> PaddleOCR-VL-1.6-0.9B
recognition served by vLLM on localhost. Sources (read 2026-09-30):
  - model card @ c5630ab: https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6
  - pipeline docs: https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html
  - PaddleX v3.7.2 pipeline config (layout model, VLM name, defaults):
    https://github.com/PaddlePaddle/PaddleX/blob/v3.7.2/paddlex/configs/pipelines/PaddleOCR-VL-1.6.yaml
  - PaddleX v3.7.2 genai-server vLLM defaults:
    https://github.com/PaddlePaddle/PaddleX/blob/v3.7.2/paddlex/inference/genai/configs/paddleocr_vl_09b.py
  - PaddleOCR v3.7.0 wrapper: https://github.com/PaddlePaddle/PaddleOCR/blob/v3.7.0/paddleocr/_pipelines/paddleocr_vl.py

Dependency isolation (same design as the v1 app)
------------------------------------------------
vLLM and paddlepaddle-gpu cannot share one environment: vLLM's torch/CUDA wheels
replace the CUDA 12.6 wheels paddlepaddle-gpu needs. The pipeline docs say the
same ("mixed inference methods ... usually require deploying the layout analysis
model and the VLM service in separate environments"). vLLM therefore lives in
its own venv at /opt/vllmenv; the system interpreter keeps paddlepaddle-gpu for
GPU layout detection. One GPU, one container, localhost only.

Weights are baked, not fetched at runtime
-----------------------------------------
Both halves of the pipeline are pinned by HF revision and hash-checked at image
build: the VLM (MODEL_REVISION) and PP-DocLayoutV3 (LAYOUT_REVISION). PaddleX's
own downloader resolves PaddlePaddle/PP-DocLayoutV3 at `main`, which is mutable
(its README changed on 2026-09-30); passing layout_detection_model_dir removes
that drift. Downloads use huggingface_hub only, so the build never imports
paddle (which needs libcuda.so.1, absent on Modal build machines).
"""

import modal

GPU_TYPE = "A10"  # fit probe first (skill: Paddle 1.6 "A10 fit probe, then L40S")
MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
# HF commit, confirmed via https://huggingface.co/api/models/PaddlePaddle/PaddleOCR-VL-1.6
# on 2026-09-30 (lastModified 2026-08-08). model.safetensors LFS sha256 below.
MODEL_REVISION = "c5630abae1d940eafe0697512a0325494b02ab42"
MODEL_WEIGHTS_SHA256 = "85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db"

# Layout model the v1.6 pipeline uses: PaddleX v3.7.2 PaddleOCR-VL-1.6.yaml,
# SubModules.LayoutDetection.model_name = PP-DocLayoutV3 (v1 uses PP-DocLayoutV2).
LAYOUT_MODEL_NAME = "PP-DocLayoutV3"
LAYOUT_REPO = "PaddlePaddle/PP-DocLayoutV3"
# Head on 2026-09-30. Commits after 2026-01-29 ("Upload 3 files") touched only
# README.md, so the weights are the original upload. pdiparams LFS sha256 below.
LAYOUT_REVISION = "241f8bdfc77a7c7bee915a5057aaee58c235a8d3"
LAYOUT_WEIGHTS_SHA256 = "70bd316b0582769ec968829fd1feb1a6a58b7c941b938327e551b6b12b45c137"

# The VLM name for v1.6. Source: inference.yml in the model repo
# (Global.model_name: PaddleOCR-VL-1.6-0.9B), the v1.6 pipeline yaml
# (VLRecognition.model_name), and the docs' `genai_server --model_name
# PaddleOCR-VL-1.6-0.9B`. The v1 app found this name is version-specific (a v1
# client asked for "PaddleOCR-VL-0.9B" and 404'd), so it is set on BOTH sides:
# vLLM serves it and the client requests it via vl_rec_api_model_name.
SERVED_NAME = "PaddleOCR-VL-1.6-0.9B"
PIPELINE_VERSION = "v1.6"

PADDLEOCR_VERSION = "3.7.0"     # latest on PyPI (2026-06-11); v1.6 is its default pipeline
PADDLEX_VERSION = "3.7.2"       # paddleocr 3.7.0 requires paddlex>=3.7.0,<3.8.0; pinned so it cannot drift
PADDLEPADDLE_VERSION = "3.2.1"  # model card + pipeline docs install command (cu126 index)
VLLM_VERSION = "0.21.0"         # native PaddleOCRVLForConditionalGeneration; >= PaddleX min_vllm_version 0.11.1

VLLM_PORT = 8000
VLLM_URL = f"http://127.0.0.1:{VLLM_PORT}/v1"
VENV = "/opt/vllmenv"
MODEL_DIR = "/opt/models/PaddleOCR-VL-1.6"
LAYOUT_DIR = "/opt/models/PP-DocLayoutV3"

# vLLM memory reservation. 0.5 is PaddleX's own genai-server default for vLLM
# (paddleocr_vl_09b.py get_config("vllm")), and leaves the other half of the
# card to PP-DocLayoutV3 + paddle's CUDA context + crop tensors.
# A10 arithmetic (~22.5 GiB usable): vLLM ~11.2 GiB = BF16 weights 1.92 GB
# + runtime/graphs + KV cache. KV per token = 18 layers x 2 (K,V) x 2 KV heads
# x 128 head_dim x 2 bytes = 18 KiB, so even ~6 GiB of KV holds ~350k tokens —
# far more than the pipeline's concurrent crops need. Peak must be measured.
VLLM_GPU_FRACTION = "0.5"
# Context: config.json max_position_embeddings = 131072 (hard limit). The engine
# is capped at 16384, PaddleX's genai-server default max-model-len. Per request:
# visual tokens <= max_pixels / (14*2)^2 = 1003520 / 784 = 1280 (processor
# default max_pixels, also the pipeline's ocr/table/chart/seal default), prompt
# ~<32 tokens ("OCR:", "Table Recognition:", ... + chat template), output cap
# 4096 (pipeline sets max_new_tokens=4096 when unset; pipeline.py
# _paddleocr_vl_run_vl_recognition_batches). 1280 + 32 + 4096 = 5408
# <= 16384 <= 131072. Spotting crops (not used here) would be 2048 visual.
VLLM_MAX_MODEL_LEN = "16384"
# vLLM's own PaddleOCR-VL recipe value
# (https://docs.vllm.ai/projects/recipes/en/latest/PaddlePaddle/PaddleOCR-VL.html).
# PaddleX's genai-server default is 131072; that larger prefill budget is a
# later one-variable probe, not the first A10 fit test.
VLLM_MAX_BATCHED_TOKENS = "16384"

# Truncate the stored res.json above this size (the full object also travels in
# the runner's raw_response). Pages in this corpus are expected to be far below.
PIPELINE_RESULT_MAX_BYTES = 4_000_000

# Labels whose block content is not a transcription of printed text: image crops
# carry no text, and chart content is a parsed data table, not the page's words.
_NON_TEXT_LABELS = {"image", "header_image", "footer_image", "chart"}


def _bake_weights() -> None:
    """Image-build step: fetch both pinned artifacts and verify their hashes.

    Runs with the system interpreter's huggingface_hub; never imports paddle.
    """
    import hashlib
    from pathlib import Path

    from huggingface_hub import snapshot_download

    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=MODEL_DIR)
    snapshot_download(LAYOUT_REPO, revision=LAYOUT_REVISION, local_dir=LAYOUT_DIR)
    got = _sha256(Path(MODEL_DIR) / "model.safetensors")
    if got != MODEL_WEIGHTS_SHA256:
        raise RuntimeError(f"VLM weights sha256 {got} != pinned {MODEL_WEIGHTS_SHA256}")
    got = _sha256(Path(LAYOUT_DIR) / "inference.pdiparams")
    if got != LAYOUT_WEIGHTS_SHA256:
        raise RuntimeError(f"layout weights sha256 {got} != pinned {LAYOUT_WEIGHTS_SHA256}")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libgomp1", "python3-venv")
    .pip_install(
        f"paddlepaddle-gpu=={PADDLEPADDLE_VERSION}",
        extra_index_url="https://www.paddlepaddle.org.cn/packages/stable/cu126/",
    )
    # paddleocr[doc-parser] pulls paddlex[genai-client,ocr]; paddlex is pinned
    # explicitly because paddleocr only bounds it (>=3.7.0,<3.8.0).
    # Pillow / huggingface_hub pins are reproducibility pins chosen here, not
    # upstream-prescribed versions.
    .pip_install(
        f"paddleocr[doc-parser]=={PADDLEOCR_VERSION}",
        f"paddlex[genai-client,ocr]=={PADDLEX_VERSION}",
        "Pillow==11.3.0",
        "huggingface_hub==0.36.2",
    )
    # Isolated interpreter for vLLM so its CUDA wheels cannot clobber paddle's.
    .run_commands(
        f"python -m venv {VENV}",
        f"{VENV}/bin/pip install --no-cache-dir vllm=={VLLM_VERSION}",
        f"{VENV}/bin/python -c 'import vllm; print(\"vllm\", vllm.__version__)'",
    )
    .run_function(_bake_weights, secrets=[modal.Secret.from_name("huggingface-token")])
    .env({
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "OMP_NUM_THREADS": "1",
        # Both models are local dirs; nothing should be resolved from a hub.
        # PaddleX flags from paddlex/utils/flags.py (v3.7.2).
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        "PADDLE_PDX_MODEL_SOURCE": "huggingface",
    })
)

# PaddleX still writes its cache (~/.paddlex: fonts, official_models for any
# asset not passed as a dir). Separate volume from the v1 app so the two engines
# never share or race on a cache.
paddlex_cache = modal.Volume.from_name("paddlex-model-cache-vl16", create_if_missing=True)

app = modal.App("ocr-bench-paddleocr-vl-1-6")


def _normalise_box(box, width: int, height: int):
    """Pixel box [x0,y0,x1,y1] or polygon [[x,y],...] -> [x0,y0,x1,y1] in 0-1.

    Returns None for an unrecognised shape or unknown page size, so a bad box is
    dropped rather than silently mis-scored.
    """
    if box is None or not width or not height:
        return None
    try:
        flat = list(box)
        if flat and isinstance(flat[0], (list, tuple)):       # polygon
            xs = [float(p[0]) for p in flat]
            ys = [float(p[1]) for p in flat]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        elif len(flat) == 4:
            x0, y0, x1, y1 = (float(v) for v in flat)
        else:
            return None
    except (TypeError, ValueError, IndexError):
        return None
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    clamp = lambda v: min(max(v, 0.0), 1.0)  # noqa: E731
    return [clamp(x0 / width), clamp(y0 / height), clamp(x1 / width), clamp(y1 / height)]


def _html_to_text(value: str) -> str:
    import html
    import re

    text = re.sub(r"</t[dh]>\s*<t[dh][^>]*>", "\t", value)
    text = re.sub(r"</tr>\s*<tr[^>]*>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def _write_page(image_bytes: bytes, path) -> tuple[int, int]:
    """EXIF-transpose + RGB, written as PNG at original resolution.

    Every coordinate below is relative to this image. No resize: the pipeline's
    own processor bounds each crop (min/max_pixels) internally.
    """
    import io

    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.save(path, format="PNG")
    return img.size


def _warmup_page(path) -> None:
    """A synthetic page with a title, text lines and a ruled table, so warm-up
    exercises layout detection AND the VLM (a blank page yields no crops)."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(img)
    try:
        big, small = ImageFont.load_default(size=44), ImageFont.load_default(size=26)
    except TypeError:  # Pillow without sized default font
        big = small = ImageFont.load_default()
    draw.text((120, 120), "Quarterly Report 2026", fill="black", font=big)
    for i in range(6):
        draw.text((120, 240 + 40 * i),
                  f"Line {i + 1}: revenue, costs and margin were reviewed by the team.",
                  fill="black", font=small)
    x0, y0, cw, rh = 120, 600, 300, 60
    for r in range(5):
        for c in range(3):
            draw.rectangle([x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh],
                           outline="black", width=2)
            draw.text((x0 + c * cw + 12, y0 + r * rh + 16),
                      "Item" if r == 0 else f"{r * 10 + c}", fill="black", font=small)
    img.save(path, format="PNG")


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=1800,
    startup_timeout=1800,
    scaledown_window=30,
    secrets=[modal.Secret.from_name("huggingface-token")],
    volumes={"/root/.paddlex": paddlex_cache},
    # Exactly one batching owner: predict(list_of_paths, use_queues=True) runs
    # loading, layout and VLM recognition as queued stages; the VLM stage sends
    # concurrent crop requests (vl_rec_max_concurrency, default 200) into vLLM's
    # continuous batcher. No extra thread pool, no Modal fan-out.
    max_containers=1,
)
class PaddleOCRVL16Inference:
    @modal.enter()
    def load_model(self) -> None:
        import os
        import subprocess
        import tempfile
        import time
        import urllib.error
        import urllib.request
        from pathlib import Path

        t0 = time.perf_counter()
        # vLLM first, so its memory profiling sees the card before paddle allocates.
        self._server = subprocess.Popen(
            [
                f"{VENV}/bin/vllm", "serve", MODEL_DIR,
                "--served-model-name", SERVED_NAME,
                "--host", "127.0.0.1", "--port", str(VLLM_PORT),
                "--trust-remote-code",
                # The repo's chat_template.jinja is byte-identical to PaddleX's
                # PaddleOCR-VL-1.6-0.9B.jinja ("Assistant:\n"; v1's is
                # "Assistant: "). Passed explicitly so a vLLM default cannot
                # substitute another template.
                "--chat-template", f"{MODEL_DIR}/chat_template.jinja",
                "--max-model-len", VLLM_MAX_MODEL_LEN,
                "--max-num-batched-tokens", VLLM_MAX_BATCHED_TOKENS,
                "--gpu-memory-utilization", VLLM_GPU_FRACTION,
                "--limit-mm-per-prompt", '{"image": 1}',
                # From vLLM's PaddleOCR-VL recipe.
                "--no-enable-prefix-caching",
                "--mm-processor-cache-gb", "0",
            ],
            env={**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0", "HF_HUB_OFFLINE": "1"},
        )

        deadline, last = time.time() + 900, None
        while time.time() < deadline:
            if self._server.poll() is not None:
                raise RuntimeError(f"vLLM exited early: {self._server.returncode}")
            try:
                with urllib.request.urlopen(f"{VLLM_URL}/models", timeout=5) as r:
                    body = r.read().decode("utf-8", "replace")
                    if r.status == 200 and SERVED_NAME in body:
                        break
            except (OSError, urllib.error.URLError) as exc:
                last = exc
            time.sleep(5)
        else:
            raise TimeoutError(f"vLLM never became ready: {last}")

        from paddleocr import PaddleOCRVL

        kwargs = {
            "pipeline_version": PIPELINE_VERSION,
            # Pinned layout asset (see module docstring); name must match the
            # inference.yml Global.model_name in that dir.
            "layout_detection_model_name": LAYOUT_MODEL_NAME,
            "layout_detection_model_dir": LAYOUT_DIR,
            "vl_rec_backend": "vllm-server",
            "vl_rec_server_url": VLLM_URL,
            "vl_rec_api_model_name": SERVED_NAME,
            # Same capability policy as the v1 engine: chart parsing and seal
            # recognition default to False in the pipeline config; enabled for
            # the whole corpus so no region is silently dropped, and the
            # layout detector alone decides where they fire.
            "use_chart_recognition": True,
            "use_seal_recognition": True,
            # Documented defaults, left off (same as v1). Enabling them is a
            # distinct "full preprocessing" configuration for the whole corpus.
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
        }
        last_exc = None
        for attempt in range(3):
            try:
                try:
                    self._pipeline = PaddleOCRVL(device="gpu", **kwargs)
                except TypeError:
                    self._pipeline = PaddleOCRVL(**kwargs)
                break
            except Exception as exc:
                last_exc = exc
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"PaddleOCRVL init failed: {last_exc}")

        self._layout_info = self._describe_layout_model()
        self._versions = self._package_versions()

        # Representative warm-up: text + table so both stages and the vLLM
        # scheduler are exercised before any timed batch. Failure is fatal: a
        # container that cannot parse a synthetic page must not be timed.
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "warm.png"
            _warmup_page(p)
            warm = list(self._pipeline.predict([str(p)], use_queues=True))
            if len(warm) != 1:
                raise RuntimeError(f"warm-up returned {len(warm)} results for 1 input")
            self._warmup_blocks = len(self._dig(warm[0], ("res", "parsing_res_list")) or [])

        self._startup_gpu_seconds = time.perf_counter() - t0
        self._startup_cost_applied = False

    @modal.exit()
    def shutdown(self) -> None:
        pipeline = getattr(self, "_pipeline", None)
        if pipeline is not None:
            try:
                pipeline.close()
            except Exception:
                pass
        server = getattr(self, "_server", None)
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except Exception:
                server.kill()

    # ------------------------------------------------------------------ #
    # Methods
    # ------------------------------------------------------------------ #
    @modal.method()
    def run_ocr(self, image_bytes: bytes, mime_type: str = "image/png",
                prompt: str | None = None, response_format: dict | None = None) -> dict:
        # Fixed pipeline: the documented pipeline owns its per-region prompts.
        # A caller prompt would be silently ignored, which is a benchmark lie.
        if prompt is not None or response_format is not None:
            raise ValueError("paddleocr_vl_1_6 is a fixed pipeline; it accepts no prompt "
                             "or response_format")
        batch = self._run_batch([{"bytes": image_bytes, "mime_type": mime_type,
                                  "document_path": None}])
        response = batch["results"][0]
        response["gpu_seconds"] = batch["batch_gpu_seconds"]
        response["gpu_type"] = GPU_TYPE
        response["startup_gpu_seconds"] = batch["startup_gpu_seconds"]
        response["is_cold_start"] = batch["is_cold_start"]
        return response

    @modal.method()
    def run_ocr_batch(self, files: list[dict], prompt: str | None = None,
                      response_format: dict | None = None) -> dict:
        """One queued list-predict over the whole batch on one GPU."""
        return self._run_batch(files, prompt, response_format)

    # ------------------------------------------------------------------ #
    def _run_batch(self, files, prompt=None, response_format=None) -> dict:
        import tempfile
        import time
        from pathlib import Path

        started = time.perf_counter()
        results: list = [None] * len(files)
        try:
            with tempfile.TemporaryDirectory() as td:
                runnable = []  # (index, path, (w, h))
                for index, item in enumerate(files):
                    item_prompt = item.get("prompt", prompt)
                    item_format = item.get("response_format", response_format)
                    if item_prompt is not None or item_format is not None:
                        results[index] = self._error_response(
                            ValueError("paddleocr_vl_1_6 is a fixed pipeline; it accepts "
                                       "no prompt or response_format"),
                            item.get("document_path"))
                        continue
                    path = Path(td) / f"{index:04d}.png"
                    try:
                        size = _write_page(item["bytes"], path)
                    except Exception as exc:
                        results[index] = self._error_response(exc, item.get("document_path"))
                        continue
                    runnable.append((index, str(path), size))

                if runnable:
                    rendered = list(self._pipeline.predict(
                        [path for _, path, _ in runnable], use_queues=True))
                    if len(rendered) != len(runnable):
                        raise RuntimeError(
                            f"PaddleOCR-VL returned {len(rendered)} results "
                            f"for {len(runnable)} inputs")
                    for (index, _path, size), res in zip(runnable, rendered):
                        doc = files[index].get("document_path")
                        try:
                            response = self._result_to_response(res, size)
                            response["document_path"] = doc
                        except Exception as exc:
                            response = self._error_response(exc, doc)
                        results[index] = response
        except Exception as exc:
            results = [self._error_response(exc, item.get("document_path")) for item in files]

        batch_gpu_seconds = time.perf_counter() - started
        startup_seconds, is_cold_start = self._take_startup_metadata()
        return {
            "results": results,
            "batch_gpu_seconds": batch_gpu_seconds,
            "gpu_type": GPU_TYPE,
            "startup_gpu_seconds": startup_seconds,
            "is_cold_start": is_cold_start,
        }

    def _take_startup_metadata(self) -> tuple[float, bool]:
        """Report startup separately once; never add it to inference GPU-seconds."""
        if not getattr(self, "_startup_cost_applied", True):
            startup = getattr(self, "_startup_gpu_seconds", 0.0)
            self._startup_cost_applied = True
            return startup, True
        return 0.0, False

    def _model_version(self) -> str:
        return (
            f"{MODEL_ID}@{MODEL_REVISION[:12]} pipeline {PIPELINE_VERSION} "
            f"(layout={getattr(self, '_layout_info', 'unknown')}, "
            f"paddleocr=={PADDLEOCR_VERSION}, paddlex=={PADDLEX_VERSION}, "
            f"vllm=={VLLM_VERSION} server, GPU, queued batch)"
        )

    def _error_response(self, exc: Exception, document_path=None) -> dict:
        return {
            "raw_text": "",
            "structured_output": {},
            "model_version": self._model_version(),
            "prompt_used": None,
            "document_path": document_path,
            "gpu_type": GPU_TYPE,
            "error": f"{exc.__class__.__name__}: {str(exc) or repr(exc)}",
        }

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def _describe_layout_model(self) -> str:
        """Name + source of the layout model that actually loaded.

        Reads the effective merged config first (what PaddleX was told), then
        the live predictor object (what PaddleX built), then the model dir's own
        inference.yml. All three are recorded; the pinned revision/hash covers
        the weights.
        """
        parts = []
        try:
            cfg = getattr(self._pipeline, "_merged_paddlex_config", None) or {}
            ld = (cfg.get("SubModules") or {}).get("LayoutDetection") or {}
            if ld:
                parts.append(f"cfg={ld.get('model_name')}")
        except Exception:
            pass
        try:
            inner = getattr(self._pipeline, "paddlex_pipeline", None)
            seen = set()
            stack = [inner]
            while stack:
                node = stack.pop()
                if node is None or id(node) in seen:
                    continue
                seen.add(id(node))
                model = getattr(node, "layout_det_model", None)
                if model is not None:
                    parts.append("loaded=" + str(
                        getattr(model, "model_name", None)
                        or getattr(model, "model_dir", None)
                        or type(model).__name__))
                    break
                for attr in ("_pipeline", "pipeline", "_pipelines", "_executor"):
                    child = getattr(node, attr, None)
                    if isinstance(child, (list, tuple)):
                        stack.extend(child)
                    elif child is not None:
                        stack.append(child)
        except Exception:
            pass
        try:
            import yaml
            with open(f"{LAYOUT_DIR}/inference.yml", encoding="utf-8") as f:
                name = (yaml.safe_load(f).get("Global") or {}).get("model_name")
            parts.append(f"dir={name}@{LAYOUT_REVISION[:12]}")
        except Exception:
            parts.append(f"dir=?@{LAYOUT_REVISION[:12]}")
        return ",".join(parts) or "unknown"

    @staticmethod
    def _package_versions() -> dict:
        from importlib.metadata import PackageNotFoundError, version

        out = {}
        for pkg in ("paddleocr", "paddlex", "paddlepaddle-gpu"):
            try:
                out[pkg] = version(pkg)
            except PackageNotFoundError:
                out[pkg] = None
        out["vllm"] = VLLM_VERSION  # separate venv; verified at image build
        return out

    @staticmethod
    def _dig(res, *keypaths):
        """First of `keypaths` that resolves in res.json, else None."""
        for keypath in keypaths:
            try:
                node = res.json
                for key in keypath:
                    node = node[key]
                return node
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------ #
    # Result extraction
    # ------------------------------------------------------------------ #
    def _result_to_response(self, res, image_size=(0, 0)) -> dict:
        """Everything the one pipeline call produced, in benchmark shapes.

        res.json["res"] for v1.6 (PaddleX v3.7.2 result.py _to_json) carries
        width, height, parsing_res_list (block_label/content/bbox/order/
        polygon points), layout_det_res, and spotting_res only when a spotting
        region exists. It has NO overall_ocr_res and NO table_res_list: the
        v1.6 pipeline runs no separate OCR sub-pipeline, and table_res_list is
        initialised empty and never filled (pipeline.py
        _paddleocr_vl_assemble_parsing_results).
        """
        import json

        md = res.markdown
        text = md.get("markdown_texts", "") if isinstance(md, dict) else str(md)

        page = self._dig(res, ("res",)) or {}
        page_w = page.get("width") or image_size[0]
        page_h = page.get("height") or image_size[1]

        structured: dict = {
            "preprocessing": {"exif_transpose": True, "mode": "RGB", "resize": None,
                              "coordinate_space": "original image after EXIF transpose"},
            "pipeline_config": {
                "pipeline_version": PIPELINE_VERSION,
                "layout_model": getattr(self, "_layout_info", "unknown"),
                "use_chart_recognition": True, "use_seal_recognition": True,
                "use_doc_orientation_classify": False, "use_doc_unwarping": False,
                "max_new_tokens_per_region": 4096,
                "versions": getattr(self, "_versions", {}),
            },
            # Per-region generation stats are not exposed by the pipeline result;
            # truncation is only visible indirectly (see UNVERIFIED notes).
            "generation": {"max_new_tokens_per_region": 4096,
                           "finish_reason": "not exposed by pipeline"},
        }

        # --- raw result: the full res.json, size-guarded ----------------------
        try:
            blob = json.dumps(page, ensure_ascii=False, default=str)
            if len(blob.encode("utf-8")) <= PIPELINE_RESULT_MAX_BYTES:
                structured["pipeline_result"] = json.loads(blob)
            else:
                structured["pipeline_result"] = {
                    "truncated": True, "bytes": len(blob.encode("utf-8")),
                    "keys": sorted(page.keys()) if isinstance(page, dict) else None,
                    "parsing_res_list": page.get("parsing_res_list"),
                }
        except Exception as exc:
            structured["pipeline_result"] = {"error": f"{type(exc).__name__}: {exc}"}

        # --- layout blocks ----------------------------------------------------
        parsing = page.get("parsing_res_list") if isinstance(page, dict) else None
        blocks = []
        if isinstance(parsing, list):
            for block in parsing:
                if not isinstance(block, dict):
                    continue
                raw_box = block.get("block_bbox")
                entry = {
                    "label": block.get("block_label"),
                    "bbox": _normalise_box(raw_box, page_w, page_h),
                    "bbox_px": raw_box,
                    "order": block.get("block_order"),
                    "content": block.get("block_content"),
                }
                poly = block.get("block_polygon_points")
                if poly is not None:
                    entry["poly"] = poly
                blocks.append(entry)
        if blocks:
            structured["layout_blocks"] = blocks

        # --- tables -----------------------------------------------------------
        # Primary: table_res_list (kept in case a later paddlex fills it).
        # In paddlex 3.7.x it is empty (the regression noted in the v1 app), so
        # the table-labelled parsing blocks — whose content the pipeline already
        # converted from OTSL to HTML — are the working source. table_source
        # records which path produced the tables.
        tables, table_source = [], None
        table_res = page.get("table_res_list") if isinstance(page, dict) else None
        if isinstance(table_res, list):
            for table in table_res:
                if isinstance(table, dict) and table.get("pred_html"):
                    tables.append({"html": str(table["pred_html"]),
                                   "cell_boxes": table.get("cell_box_list")})
            if tables:
                table_source = "table_res_list"
        if not tables:
            for block in blocks:
                if str(block.get("label")) == "table" and block.get("content"):
                    tables.append({"html": str(block["content"]), "bbox": block.get("bbox")})
            if tables:
                table_source = "parsing_res_list"
        if tables:
            structured["tables"] = tables
            structured["table_source"] = table_source

        # --- geometry ---------------------------------------------------------
        # Default v1.6 conversion yields only BLOCK geometry: each parsing
        # block's text is the VLM output for exactly that crop, so text and box
        # are genuinely associated. Image/chart blocks are excluded (no printed
        # transcription). Table HTML is flattened to its cell text for the
        # region text; the HTML stays in layout_blocks/tables.
        regions = []
        for block in blocks:
            label = str(block.get("label"))
            content = block.get("content")
            if label in _NON_TEXT_LABELS or not content or block.get("bbox") is None:
                continue
            region_text = _html_to_text(str(content)) if label == "table" else str(content).strip()
            if region_text:
                regions.append({"text": region_text, "bbox": block["bbox"], "label": label,
                                **({"poly": block["poly"]} if "poly" in block else {})})
        granularity = "block" if regions else None

        # Spotting output (line-level quads) exists only for regions labelled
        # "spotting", i.e. a separate use_layout_detection=False /
        # prompt_label="spotting" pass. Never produced by this configuration;
        # parsed if present so a future spotting engine can reuse this adapter.
        spotting = page.get("spotting_res") if isinstance(page, dict) else None
        if isinstance(spotting, dict) and spotting.get("rec_texts"):
            spot_regions = []
            for span, poly in zip(spotting.get("rec_texts") or [], spotting.get("rec_polys") or []):
                flat = _normalise_box(poly, page_w, page_h)
                if flat and str(span).strip():
                    spot_regions.append({"text": str(span), "bbox": flat, "poly": poly})
            if spot_regions:
                regions, granularity = spot_regions, "line"
        if regions:
            structured["text_regions"] = regions
            structured["geometry_granularity"] = granularity

        return {
            "raw_text": text,
            "structured_output": structured,
            "model_version": self._model_version(),
            "prompt_used": None,
            "gpu_type": GPU_TYPE,
            "error": None,
        }
