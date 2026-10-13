"""Modal app: Qwen3-VL-8B — Qwen/Qwen3-VL-8B-Instruct. GPU: A100-80GB.
Weights are pre-fetched into the image's Hugging Face cache at build time;
inference runs through a local vLLM server."""

import modal

GPU_TYPE = "A100-80GB"  # hardware-matched to published throughput figures (single A100)
VLLM_VERSION = "0.21.0"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
# Pinned to the exact HF commit. Model repos are mutable; an unpinned
# MODEL_ID silently changes what was benchmarked.
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
SERVED_MODEL_NAME = "qwen3-vl"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/v1"
BATCH_CONCURRENCY = 3


def _write_normalized_image(image_bytes: bytes, output_path: str, max_side: int | None = None) -> None:
    import io

    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    img.save(output_path, format="PNG")


def download_weights():
    from huggingface_hub import snapshot_download
    import os

    snapshot_download(MODEL_ID, revision=MODEL_REVISION, token=os.environ.get("HF_TOKEN"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "Pillow>=10.0",
        "huggingface_hub>=0.23",
        f"vllm=={VLLM_VERSION}",
    )
    .env({
        "OMP_NUM_THREADS": "1",
        # Modal slim images include CUDA runtime libraries, not nvcc. FlashInfer's
        # sampler may JIT-compile at vLLM warmup, so use vLLM's native sampler.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        # Keep CUDA graph profiling from reserving avoidable KV-cache memory.
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0",
    })
    .run_function(
        download_weights,
        secrets=[modal.Secret.from_name("huggingface-token")],
    )
)

app = modal.App("ocr-bench-qwen3-vl")

# The default-pass ask. Qwen3-VL is a general instruction-following VLM whose
# card documents no OCR-specific prompt, so this mirrors the ask the other
# general VLM runners (gemini, claude) use: one conversion that serves the
# text, markdown and tables legs. Text metrics strip markdown before scoring.
_OCR_PROMPT = (
    "Convert this document image to Markdown. Transcribe all text exactly as it "
    "appears, preserving reading order. Use # headings for section titles and "
    "GitHub-flavoured Markdown pipe tables for every table. "
    "Do not add commentary, summaries, or corrections. Output only the Markdown."
)


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=1800,
    startup_timeout=1800,
    scaledown_window=30,
    max_containers=1,
)
class Qwen3VLInference:
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
            "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0",
        }
        self._server_process = subprocess.Popen(
            [
                "vllm",
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
                # Full-resolution pages: Qwen3-VL's processor allows up to
                # 12845056 px == 16384 vision tokens, plus 4096 output.
                "--max-model-len",
                "24576",
                "--gpu-memory-utilization",
                "0.95",
                # Bounds prefill chunking so concurrent full-res pages cannot OOM.
                "--max-num-batched-tokens",
                "16384",
                "--limit-mm-per-prompt",
                '{"image": 1, "video": 0}',
                "--no-enable-prefix-caching",
                "--mm-processor-cache-gb",
                "0",
                "--trust-remote-code",
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
        """Run one measured batch on one A100 and return its exact GPU wall time.

        `prompt` (or a per-file "prompt" key) overrides the default OCR prompt so
        a scoring leg can ask for something else. Omitting it reproduces leg-1
        behaviour exactly.
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
                {"type": "text", "text": "A"},
            ]}],
            "max_tokens": 1,
            "chat_template_kwargs": {"enable_thinking": False},
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
                raise RuntimeError(f"Qwen3-VL vLLM server exited early with code {process_status}")

            try:
                with urllib.request.urlopen(models_url, timeout=5) as response:
                    if response.status < 500:
                        return
            except (OSError, urllib.error.URLError) as exc:
                last_error = exc

            time.sleep(5)

        raise TimeoutError(f"Qwen3-VL vLLM server did not become ready: {last_error}")

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
            # No local downscale — the processor caps input size itself. A prior
            # 1280px cap corrupted fine print on dense documents (see the note in
            # glm_ocr_app.py); every uncapped engine in this benchmark gets full res.
            _write_normalized_image(image_bytes, img_path, max_side=None)

            encoded = base64.b64encode(Path(img_path).read_bytes()).decode("ascii")
            payload = {
                "model": SERVED_MODEL_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                            {"type": "text", "text": prompt or _OCR_PROMPT},
                        ],
                    }
                ],
                "max_tokens": 4096,
                "temperature": 0,
                # Disable Qwen3's extended thinking — without this the model
                # emits <think>...</think> tokens before the answer, and in
                # some cases loops the entire page content until max_tokens.
                "chat_template_kwargs": {"enable_thinking": False},
                # Mild penalty to suppress document-level repetition loops
                # that occur when the model lacks a clean EOS signal.
                "repetition_penalty": 1.05,
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
        effective_prompt = prompt or _OCR_PROMPT
        structured: dict = {}
        try:
            # finish_reason is the only way to tell a short answer from a
            # truncated one after the fact.
            structured["api_response"] = data
        except NameError:
            pass

        # Qwen's documented spotting format is
        #   [{"bbox_2d": [x1, y1, x2, y2], "text_content": "..."}]
        # Qwen3-VL's README says only "relative position coordinates"; the
        # 0-1000 convention assumed below is NOT confirmed by its docs, and the
        # saved run's numbers fit no single frame (y up to 2036 on one page,
        # 704 for the last line of another). Dividing by 1000 is kept because
        # no other conversion scores better (checked 2026-09-30: /1000, /image
        # size, /resized frame all <= 0.006 mean on the 17 geometry pages); the
        # low score is the model's boxes, not this constant.
        if "bbox_2d" in effective_prompt:
            try:
                import re as _re

                body = _re.sub(r"^\s*```(?:json)?|```\s*$", "", text or "",
                               flags=_re.M).strip()
                start, end = body.find("["), body.rfind("]")
                items = json.loads(body[start:end + 1]) if start != -1 else []
                words = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    box = item.get("bbox_2d")
                    content = item.get("text_content")
                    if (isinstance(box, (list, tuple)) and len(box) == 4
                            and content and str(content).strip()):
                        words.append({
                            "text": str(content),
                            "bbox": [float(v) / 1000.0 for v in box],
                        })
                if words:
                    structured["words"] = words
                    structured["geometry_granularity"] = "word"
            except Exception as exc:
                structured["geometry_parse_error"] = f"{type(exc).__name__}: {exc}"

        return {"raw_text": text,
                "structured_output": structured,
                # Record the prompt the model actually received. None meant
                # "app default", which makes a saved result unreadable once the
                # default changes.
                "prompt_used": effective_prompt,
                "gpu_seconds": gpu_seconds, "gpu_type": GPU_TYPE,
                "model_version": f"{MODEL_ID} via vllm-server"}
