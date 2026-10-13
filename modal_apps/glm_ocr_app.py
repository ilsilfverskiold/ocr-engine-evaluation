"""
Modal app: GLM-OCR (0.9B) — zai-org/GLM-OCR
Served through local vLLM with the model-card OCR prompt. GPU: A10.

Backend decision:
- vLLM is intentional here. GLM-OCR is supported by upstream vLLM and the
  official recipe uses vLLM serving with MTP speculative decoding.
- The local HTTP server allows Modal input concurrency to feed vLLM's scheduler
  with overlapping requests, which is the main throughput win for warm runs.
"""

import modal

GPU_TYPE = "A10"
VLLM_VERSION = "0.21.0"
GLMOCR_VERSION = "0.1.5"
# No local downscale. GLM-OCR's own processor caps input at longest_edge=9633792 px
# (9.6MP == 12288 vision tokens), so it handles oversized pages itself.
#
# Do NOT reintroduce a small cap here. A previous 896px cap silently destroyed
# accuracy on dense documents: on bank_003 it produced invented row labels
# ("Tactical Rebalancing" -> "Temporary Restraining"), dropped leading digits
# (588.22 -> 90.92) and dropped minus signs. At 1280 the fine-print daily-balance
# grid still collapsed to the literal column header ("Amount" in every cell).
# Only full resolution transcribed the page correctly. Cost of the fix is ~1.7s/page.
MAX_IMAGE_SIDE = None
MODEL_ID = "zai-org/GLM-OCR"
# Pinned to the exact HF commit. Model repos are mutable; an unpinned
# MODEL_ID silently changes what was benchmarked.
MODEL_REVISION = "ca5d8b3e287e52589e37c28385d9655ee4372f9d"
SERVED_MODEL_NAME = "glm-ocr"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8001
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/v1"
BATCH_CONCURRENCY = 4


def _sdk_kwargs() -> dict:
    """glmocr 0.1.5's actual constructor/config contract, checked from its wheel.

    pipeline_maas_enabled and ocr_base_url are silently ignored by this SDK.
    CPU layout leaves the A10's existing vLLM memory reservation unchanged.
    This is the document-pipeline probe, not the default model-only pass.
    """
    return {"mode": "selfhosted", "ocr_api_host": SERVER_HOST,
            "ocr_api_port": SERVER_PORT, "model": SERVED_MODEL_NAME,
            "layout_device": "cpu"}


def _write_normalized_image(image_bytes: bytes, output_path: str, max_side: int | None = None) -> None:
    import io

    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    img.save(output_path, format="PNG")


VENV = "/opt/vllmenv"

# Dependency isolation, same reason as paddleocr_vl_app.
# glmocr[selfhosted] requires torch>=2.10 and transformers>=5.3; vLLM 0.21.0
# pins an older torch. In one environment pip either fails to resolve or -
# worse - upgrades torch and breaks vLLM at runtime, which shows up as garbage
# output rather than an error. vLLM therefore gets its own interpreter; the
# system interpreter keeps the SDK, which only does layout detection and
# orchestration and talks to vLLM over localhost.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "python3-venv")
    .pip_install(
        "Pillow>=10.0",
        "huggingface_hub>=0.23",
        # The official document-parsing pipeline: PP-DocLayoutV3 layout
        # detection, per-region recognition with the card's task prefixes, then
        # a formatter that assembles Markdown + structured JSON. [selfhosted]
        # pulls the layout detector so it runs against our own vLLM server
        # rather than Z.ai's hosted API.
        #
        # Before this, markdown/tables/geometry were driven by whole-page
        # prompts we invented; the card documents no markdown task at all, and
        # geometry was marked ineligible because the base model emits no boxes
        # — true of the model, false of the SDK, which returns bbox_2d per
        # region.
        f"glmocr[selfhosted]=={GLMOCR_VERSION}",
    )
    .run_commands(
        f"python -m venv {VENV}",
        # transformers>=5.3 is where GLM-OCR's architecture is recognised. vLLM
        # pins an older one, and with only `vllm==...` here the venv resolved to
        # that older version and the server died at startup with
        # "checkpoint has model type `glm_ocr` but Transformers does not
        # recognize this architecture". Installed explicitly AFTER vllm so it
        # wins, and asserted below so a bad resolve fails the BUILD instead of
        # crash-looping every container at run time.
        f"{VENV}/bin/pip install --no-cache-dir vllm=={VLLM_VERSION}",
        f"{VENV}/bin/pip install --no-cache-dir 'transformers>=5.3.0'",
        f"{VENV}/bin/python -c 'import vllm, transformers; "
        f"print(\"vllm\", vllm.__version__, \"transformers\", transformers.__version__); "
        f"from transformers import AutoConfig; "
        f"assert tuple(int(p) for p in transformers.__version__.split(\".\")[:2]) >= (5, 3), "
        f"\"transformers too old for glm_ocr\"'",
    )
    .env({
        # Modal slim images include CUDA runtime libraries, not nvcc. FlashInfer's
        # sampler may JIT-compile at vLLM warmup, so use vLLM's native sampler.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    })
)

app = modal.App("ocr-bench-glm-ocr")

# The GLM-OCR model card's own OCR prompt. Kept as the default so omitting the
# per-request `prompt` reproduces leg-1 behaviour exactly.
OCR_PROMPT = "Text Recognition:"

# Sentinel, not a prompt. GLM has two documented usage modes — direct
# base-model prompting (Text Recognition:, Table Recognition:, the Chinese KIE
# prefix) and the SDK document-parsing pipeline. The registry asks for the
# second by sending this string; the app routes it to _document_parse instead
# of a chat call. Anything else is sent to the model verbatim.
DOC_PARSE_PROMPT = "__glmocr_sdk_document_parse__"


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=1800,
    startup_timeout=1800,
    scaledown_window=30,
    secrets=[modal.Secret.from_name("huggingface-token")],
    max_containers=1,
)
class GLMOCRInference:
    @modal.enter()
    def load_model(self) -> None:
        import os
        import subprocess
        import threading
        import time

        t0 = time.perf_counter()
        server_env = {
            **os.environ,
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
        self._server_process = subprocess.Popen(
            [
                f"{VENV}/bin/vllm",
                "serve",
                MODEL_ID,
                "--revision",
                MODEL_REVISION,
                "--served-model-name",
                SERVED_MODEL_NAME,
                "--host",
                "0.0.0.0",
                "--port",
                str(SERVER_PORT),
                # Full-resolution pages cost up to 12288 vision tokens; with
                # max_tokens=4096 the old 8192 context could not hold them and was
                # the reason an image cap existed at all. 20480 covers the worst
                # page in the corpus (16384) with headroom.
                "--max-model-len",
                "20480",
                "--gpu-memory-utilization",
                "0.90",
                # Bounds prefill chunking so 4 concurrent full-res pages cannot OOM
                # the A10G. Chunked prefill is output-equivalent.
                "--max-num-batched-tokens",
                "16384",
                "--limit-mm-per-prompt",
                '{"image": 1}',
                "--no-enable-prefix-caching",
                "--mm-processor-cache-gb",
                "0",
                "--trust-remote-code",
                "--speculative-config.method",
                "mtp",
                "--speculative-config.num_speculative_tokens",
                "1",
            ],
            env=server_env,
        )
        self._wait_for_server()
        self._warmup_server()
        self._startup_gpu_seconds = time.perf_counter() - t0
        self._startup_cost_applied = False
        self._startup_cost_lock = threading.Lock()

    @modal.method()
    def run_ocr(
        self,
        image_bytes: bytes,
        mime_type: str = "image/png",
        prompt: str | None = None,
        response_format: dict | None = None,
    ) -> dict:
        try:
            response = self._run_ocr_impl(image_bytes, mime_type, prompt, response_format)
        except Exception as exc:
            response = self._error_response(exc)
        startup_seconds, is_cold_start = self._take_startup_metadata()
        response["startup_gpu_seconds"] = startup_seconds
        response["is_cold_start"] = is_cold_start
        return response

    @modal.method()
    def run_ocr_batch(self, files: list[dict], prompt: str | None = None,
                      response_format: dict | None = None) -> dict:
        """Run one measured vLLM batch on one A10 GPU.

        `prompt` (or a per-file "prompt" key) overrides OCR_PROMPT so a scoring
        leg can ask for something else. Omitting it reproduces leg-1 behaviour.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=BATCH_CONCURRENCY) as pool:
            results = list(pool.map(lambda item: self._run_batch_item(item, prompt, response_format), files))
        batch_gpu_seconds = time.perf_counter() - started
        startup_seconds, is_cold_start = self._take_startup_metadata()
        return {
            "results": results,
            "batch_gpu_seconds": batch_gpu_seconds,
            "gpu_type": GPU_TYPE,
            "startup_gpu_seconds": startup_seconds,
            "is_cold_start": is_cold_start,
        }

    def _run_batch_item(self, item: dict, prompt: str | None = None,
                        response_format: dict | None = None) -> dict:
        try:
            response = self._run_ocr_impl(
                item["bytes"],
                item.get("mime_type", "image/png"),
                item.get("prompt", prompt),
                item.get("response_format", response_format),
            )
        except Exception as exc:
            response = self._error_response(exc)
        response["document_path"] = item.get("document_path")
        return response

    def _take_startup_metadata(self) -> tuple[float, bool]:
        """Report startup separately once; never add it to inference GPU-seconds."""
        with self._startup_cost_lock:
            if not getattr(self, "_startup_cost_applied", True):
                startup_seconds = getattr(self, "_startup_gpu_seconds", 0.0)
                self._startup_cost_applied = True
                return startup_seconds, True
        return 0.0, False

    def _error_response(self, exc: Exception) -> dict:
        message = str(exc) or repr(exc)
        return {
            "raw_text": "",
            "structured_output": {},
            "gpu_seconds": 0.0,
            "gpu_type": GPU_TYPE,
            "model_version": f"{MODEL_ID} via vllm-server",
            "error": f"{exc.__class__.__name__}: {message}",
        }

    def _warmup_server(self) -> None:
        import base64
        import io
        import json
        import urllib.request
        from PIL import Image as _PILImage

        buf = io.BytesIO()
        _PILImage.new("RGB", (32, 32), (255, 255, 255)).save(buf, "PNG")
        payload = json.dumps({
            "model": SERVED_MODEL_NAME,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"}},
                {"type": "text", "text": "Text Recognition:"},
            ]}],
            "max_tokens": 1,
        }).encode()
        try:
            with urllib.request.urlopen(
                urllib.request.Request(f"{SERVER_URL}/chat/completions", data=payload,
                                       headers={"Content-Type": "application/json"}, method="POST"),
                timeout=300,
            ) as r:
                r.read()
        except Exception:
            pass

    def _wait_for_server(self) -> None:
        import time
        import urllib.error
        import urllib.request

        deadline = time.time() + 600
        models_url = f"{SERVER_URL}/models"
        last_error = None

        while time.time() < deadline:
            process_status = self._server_process.poll()
            if process_status is not None:
                raise RuntimeError(f"GLM-OCR vLLM server exited early with code {process_status}")

            try:
                with urllib.request.urlopen(models_url, timeout=5) as response:
                    if response.status < 500:
                        return
            except (OSError, urllib.error.URLError) as exc:
                last_error = exc

            time.sleep(5)

        raise TimeoutError(f"GLM-OCR vLLM server did not become ready: {last_error}")

    def _document_parse(self, img_path: str) -> dict:
        """The official SDK document-parsing pipeline, run against our vLLM."""
        from glmocr import GlmOcr
        from glmocr.config import load_config
        from importlib.metadata import version

        if version("glmocr") != GLMOCR_VERSION:
            raise RuntimeError(f"SDK version differs from pinned glmocr=={GLMOCR_VERSION}")
        kwargs = _sdk_kwargs()
        config = load_config(**kwargs)
        # Validate before GlmOcr can construct a hosted client or start models.
        api = config.pipeline.ocr_api
        if config.pipeline.maas.enabled or api.api_host != SERVER_HOST or api.api_port != SERVER_PORT:
            raise RuntimeError("GLM SDK did not resolve the required self-hosted endpoint")
        if api.api_url or api.api_path != "/v1/chat/completions" or api.model != SERVED_MODEL_NAME:
            raise RuntimeError("GLM SDK endpoint/model differs from the local vLLM contract")
        with GlmOcr(**kwargs) as ocr:
            result = ocr.parse(img_path, save_layout_visualization=False)
        if getattr(result, "_error", None):
            raise RuntimeError(f"GLM SDK pipeline failed: {result._error}")

        markdown = getattr(result, "markdown_result", "") or ""
        raw_json = getattr(result, "json_result", None) or []
        # json_result is List[List[Dict]] — one list per page.
        pages = raw_json if raw_json and isinstance(raw_json[0], list) else [raw_json]

        regions, tables = [], []
        for page in pages:
            for block in page or []:
                if not isinstance(block, dict):
                    continue
                label = str(block.get("label", ""))
                content = block.get("content")
                bbox = block.get("bbox_2d")
                # bbox_2d is [x1, y1, x2, y2] on a 0-1000 scale.
                norm = None
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    try:
                        norm = [float(v) / 1000.0 for v in bbox]
                    except (TypeError, ValueError):
                        norm = None
                regions.append({"label": label, "content": content, "bbox": norm})
                if label == "table" and content:
                    tables.append({"html": str(content)})

        structured: dict = {
            "regions": regions, "layout_blocks": regions,
            "geometry_granularity": "layout_block",
            "sdk_json_result": raw_json,
            "sdk_version": GLMOCR_VERSION,
            "sdk_config": kwargs,
            "sdk_generation_config": config.pipeline.page_loader.model_dump(),
            # This first probe initializes layout per call. Do not present its
            # latency as a warmed persistent pipeline measurement.
            "measurement_note": "quality probe; includes per-call CPU layout initialization",
        }
        if tables:
            structured["tables"] = tables
        return {"raw_text": markdown, "structured_output": structured}

    def _run_ocr_impl(
        self,
        image_bytes: bytes,
        mime_type: str = "image/png",
        prompt: str | None = None,
        response_format: dict | None = None,
    ) -> dict:
        import base64
        import json
        import tempfile
        import time
        import urllib.request
        from pathlib import Path

        t0 = time.perf_counter()

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = str(Path(tmpdir) / "input.png")
            _write_normalized_image(image_bytes, img_path, max_side=MAX_IMAGE_SIDE)

            # DOC_PARSE_PROMPT is a sentinel, not a prompt: the registry uses it
            # to ask for the SDK pipeline instead of a direct base-model call.
            if prompt == DOC_PARSE_PROMPT:
                parsed = self._document_parse(img_path)
                return {
                    **parsed,
                    "gpu_seconds": time.perf_counter() - t0,
                    "gpu_type": GPU_TYPE,
                    "model_version": f"{MODEL_ID}@{MODEL_REVISION} via glmocr=={GLMOCR_VERSION} selfhosted pipeline",
                }

            encoded = base64.b64encode(Path(img_path).read_bytes()).decode("ascii")
            payload = {
                "model": SERVED_MODEL_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                            {"type": "text", "text": prompt or OCR_PROMPT},
                        ],
                    }
                ],
                "max_tokens": 4096,
                "temperature": 0,
            }

            if response_format:
                # vLLM distinguishes the two: json_object only forces valid
                # JSON, json_schema constrains generation to the schema. The
                # schema stays in the prompt as well — constrained decoding
                # controls what CAN be emitted, the prompt says what the fields
                # mean.
                payload["response_format"] = response_format

            request = urllib.request.Request(
                f"{SERVER_URL}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=600) as response:
                data = json.loads(response.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"]

        gpu_seconds = time.perf_counter() - t0
        return {
            "raw_text": text,
            # Keep the provider's reply beside the text we scored:
            # finish_reason distinguishes "the model said this" from
            # "the model was cut off mid-JSON", which is invisible
            # once only the text survives.
            "structured_output": {"api_response": data} if "data" in dir() else {},
            "gpu_seconds": gpu_seconds,
            "gpu_type": GPU_TYPE,
            "model_version": f"{MODEL_ID} via vllm-server",
        }
