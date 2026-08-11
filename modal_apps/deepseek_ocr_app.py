"""
Modal app: DeepSeek-OCR (3B) — deepseek-ai/DeepSeek-OCR
Uses the upstream vLLM OpenAI-compatible serving path. GPU: A100-80GB.

Backend decision:
- vLLM is intentional here. DeepSeek's OCR recipe documents both offline and
  OpenAI-compatible vLLM serving, including the required NGram logits processor.
- The local HTTP server lets Modal input concurrency drive vLLM batching.
"""

import modal

GPU_TYPE = "A100-80GB"  # hardware-matched to published PaddleOCR-VL/MinerU throughput figures (single A100)
VLLM_VERSION = "0.21.0"
MODEL_ID = "deepseek-ai/DeepSeek-OCR"
# Pinned to the exact HF commit. Model repos are mutable; an unpinned
# MODEL_ID silently changes what was benchmarked.
MODEL_REVISION = "9f30c71f441d010e5429c532364a86705536c53a"
SERVED_MODEL_NAME = "deepseek-ocr"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8002
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/v1"
OCR_PROMPT = "<|grounding|>Convert the document to markdown."

# The card's second documented stage. DeepSeek-OCR's "deep parsing" is a
# SECOND call on a detected figure, not a different page-level prompt: the
# paper defines chart parsing as image-to-HTML-table on the figure crop. Page
# parsing alone therefore under-tests the model on charts — the grounding pass
# correctly reports "here is a figure" and stops, which is the right answer to
# the question it was asked.
#
# This mirrors what PaddleOCR-VL does internally when use_chart_recognition is
# on: detect the chart region, run a second recogniser over it. Giving paddle
# that path and withholding DeepSeek's would not be a like-for-like comparison.
FIGURE_PROMPT = "Parse the figure."
# Region labels the grounding pass uses for a non-text block. Matched
# case-insensitively. Every label actually seen is recorded in
# structured_output["regions"], so this set can be corrected against real
# output instead of guessed at twice.
FIGURE_LABELS = frozenset({"image", "figure", "chart", "picture", "diagram", "graph"})
# A page of figures should not turn into a dozen extra GPU calls.
MAX_FIGURE_CALLS = 3
BATCH_CONCURRENCY = 3


def _write_normalized_image(image_bytes: bytes, output_path: str) -> None:
    import io

    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.save(output_path, format="PNG")


def _read_text_artifacts(output_dir: str) -> str:
    from pathlib import Path

    output_path = Path(output_dir)
    candidates = []
    for pattern in ("*.mmd", "*.md", "*.txt"):
        candidates.extend(sorted(output_path.rglob(pattern)))

    parts = []
    for path in candidates:
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def _html_table_to_text(html_text: str) -> str:
    import html
    import re

    text = re.sub(r"<br\s*/?>", " ", html_text)
    text = re.sub(r"</t[dh]>\s*<t[dh][^>]*>", "\t", text)
    text = re.sub(r"</tr>\s*<tr[^>]*>", "\n", text)
    text = re.sub(r"</?(table|tbody|thead|tr|td|th)[^>]*>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def _crop_to_base64(image_path: str, bbox: list[float], pad: float = 0.01) -> str | None:
    """Crop a 0-1 normalised region out of the page and base64 it.

    A small pad is added because the grounding boxes hug the figure, and chart
    axis labels sit right on that edge.
    """
    import base64
    import io

    try:
        from PIL import Image

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            x0, y0, x1, y1 = bbox
            left = max(0, int((x0 - pad) * width))
            top = max(0, int((y0 - pad) * height))
            right = min(width, int((x1 + pad) * width))
            bottom = min(height, int((y1 + pad) * height))
            if right - left < 8 or bottom - top < 8:
                return None
            buf = io.BytesIO()
            img.crop((left, top, right, bottom)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _chat_completion(encoded_image: str, prompt: str, max_tokens: int = 2048) -> str:
    """One vLLM chat call against the served DeepSeek-OCR, returning its text."""
    import json
    import urllib.error
    import urllib.request

    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{encoded_image}"}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "skip_special_tokens": False,
        "vllm_xargs": {"ngram_size": 30, "window_size": 90,
                       "whitelist_token_ids": [128821, 128822]},
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
        raise RuntimeError(
            f"DeepSeek-OCR request failed: HTTP {exc.code}: {body}") from exc
    return data["choices"][0]["message"]["content"]


def _clean_deepseek_output(raw_text: str, preserve_markdown: bool = False) -> tuple[str, dict]:
    """Strip DeepSeek's grounding tokens; flatten markdown ONLY for the plain-text
    ask. The grounding pass IS the markdown answer — stripping its headings or
    replacing its tables would score our cleanup, not the model (the markdown
    scorer reads embedded HTML tables natively)."""
    import json
    import re

    tables = [{"html": match.group(0)} for match in re.finditer(r"<table\b.*?</table>", raw_text, flags=re.I | re.S)]

    # Grounding mode annotates every region it reads with its box:
    #   <|ref|>text<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
    # normalised 0-999 (github.com/deepseek-ai/DeepSeek-OCR). The cleaner below
    # deletes those spans, so they have to be harvested BEFORE it runs or the
    # model's coordinates are lost with nothing left to re-derive them from.
    # Boxes are region-level, not word-level — a ref can hold a whole line or
    # heading — so the geometry scorer's line-merging path is the right
    # comparison, not one box per word.
    words: list[dict] = []
    regions: list[dict] = []
    for match in re.finditer(
        r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>\s*(\[\[.*?\]\])\s*<\|/det\|>",
        raw_text, flags=re.S,
    ):
        label = re.sub(r"<[^>]+>", " ", match.group(1)).strip()
        try:
            boxes = json.loads(match.group(2))
        except Exception:
            continue
        for box in boxes:
            if not (isinstance(box, list) and len(box) == 4):
                continue
            try:
                x0, y0, x1, y1 = (float(v) / 999.0 for v in box)
            except (TypeError, ValueError):
                continue
            if label:
                words.append({"text": label, "bbox": [x0, y0, x1, y1]})
                # Kept whole, labels included, so the figure detector below can
                # be checked against what the model really emits.
                regions.append({"label": label, "bbox": [x0, y0, x1, y1]})

    text = raw_text
    if not preserve_markdown:
        def replace_table(match) -> str:
            return "\n" + _html_table_to_text(match.group(0)) + "\n"

        text = re.sub(r"<table\b.*?</table>", replace_table, text, flags=re.I | re.S)
    text = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", text, flags=re.S)
    text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text, flags=re.S)
    text = re.sub(r"<\|[^|]+?\|>", "", text)
    if not preserve_markdown:
        text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()

    structured_output: dict = {"tables": tables} if tables else {}
    if words:
        # The geometry scorer reads structured_output["words"] before falling
        # back to the text, so this is all that is needed to enter that leg.
        structured_output["words"] = words
    if regions:
        structured_output["regions"] = regions
    # Keep what the model actually said. This cleaner deletes every
    # <|ref|>/<|det|>/<|...|> span, so a reply made only of grounding
    # annotations comes back as an empty string — indistinguishable, from the
    # outside, from the model returning nothing. Without the pre-clean text
    # there is no way to tell whether a 0-byte result is the model's failure
    # or ours, and that question cannot be answered by re-running.
    if raw_text and raw_text.strip() != text.strip():
        structured_output["raw_model_output"] = raw_text
    return text, structured_output


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "Pillow>=10.0",
        "huggingface_hub>=0.23",
        f"vllm=={VLLM_VERSION}",
    )
    .env({
        # Modal slim images include CUDA runtime libraries, not nvcc. FlashInfer's
        # sampler may JIT-compile at vLLM warmup, so use vLLM's native sampler.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    })
)

app = modal.App("ocr-bench-deepseek-ocr")


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=1800,
    startup_timeout=1800,
    scaledown_window=30,
    secrets=[modal.Secret.from_name("huggingface-token")],
    max_containers=1,
)
class DeepSeekOCRInference:
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
                # 8192 is a HARD limit for this model, not a choice: DeepSeek-OCR's
                # config.json sets max_position_embeddings=8192 and vLLM refuses to
                # start above it. Do not raise this — it exits with a pydantic
                # ValidationError and every request fails with "function is stopped".
                "--max-model-len",
                "8192",
                "--gpu-memory-utilization",
                "0.90",
                "--logits_processors",
                "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor",
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
        """Run one measured vLLM batch on one A100.

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
                {"type": "text", "text": OCR_PROMPT},
            ]}],
            "max_tokens": 1,
            "skip_special_tokens": False,
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
                raise RuntimeError(f"DeepSeek-OCR vLLM server exited early with code {process_status}")

            try:
                with urllib.request.urlopen(models_url, timeout=5) as response:
                    if response.status < 500:
                        return
            except (OSError, urllib.error.URLError) as exc:
                last_error = exc

            time.sleep(5)

        raise TimeoutError(f"DeepSeek-OCR vLLM server did not become ready: {last_error}")

    def _run_ocr_impl(
        self,
        image_bytes: bytes,
        mime_type: str = "image/png",
        prompt: str | None = None,
        response_format: dict | None = None,
    ) -> dict:
        import base64
        import json
        import re
        import tempfile
        import time
        import urllib.error
        import urllib.request
        from pathlib import Path

        t0 = time.perf_counter()

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = str(Path(tmpdir) / "input.png")
            _write_normalized_image(image_bytes, img_path)

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
                # KEEP AT 2048. Tested 6144 (2026-07-25): gold-doc coverage moved only
                # +0.006, and three documents became degenerate loops that 2048 had been
                # truncating before they ran away:
                #   newspaper_003  9.9k -> 30.6k chars, a hallucinated sentence x272
                #                  ("The federal government spends money on a variety...")
                #   report_001     9.1k -> 27.6k chars of "<td></td>", zero whitespace
                #   tax_002        2.8k ->  5.9k chars, ". . . ." dot leaders x463
                # The ngram guard below (ngram_size=30) cannot catch these: the repeating
                # units are shorter than 30 tokens, so the blocker never trips.
                # Upstream's recipe uses 8192, which is impossible here — the model's
                # own context is 8192 total, so output must leave room for vision tokens.
                "max_tokens": 2048,
                "temperature": 0,
                "skip_special_tokens": False,
                "vllm_xargs": {
                    "ngram_size": 30,
                    "window_size": 90,
                    "whitelist_token_ids": [128821, 128822],
                },
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
            try:
                with urllib.request.urlopen(request, timeout=900) as response:
                    data = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"DeepSeek-OCR request failed: HTTP {exc.code}: {body}") from exc
            res = data["choices"][0]["message"]["content"]

            text, structured_output = _clean_deepseek_output(
                (res or "").strip(),
                preserve_markdown="<|grounding|>" in (prompt or OCR_PROMPT),
            )

            # --- documented stage two: deep-parse each detected figure --------
            # Only after a grounding pass, because only that pass reports where
            # the figures are. Each crop is a separate documented task call, not
            # a second attempt at the same one.
            figures = [
                r for r in structured_output.get("regions", [])
                if r["label"].strip().lower() in FIGURE_LABELS
            ]
            parsed_figures = []
            for region in figures[:MAX_FIGURE_CALLS]:
                crop = _crop_to_base64(img_path, region["bbox"])
                if crop is None:
                    continue
                try:
                    figure_text = _chat_completion(crop, FIGURE_PROMPT)
                except Exception as exc:                       # never fail the page
                    parsed_figures.append({"bbox": region["bbox"],
                                           "error": f"{type(exc).__name__}: {exc}"})
                    continue
                figure_text = (figure_text or "").strip()
                if not figure_text:
                    continue
                parsed_figures.append({"bbox": region["bbox"], "output": figure_text})
                # The paper defines chart parsing as image -> HTML table, so the
                # result lands in the same shape the table leg already reads.
                for match in re.finditer(r"<table\b.*?</table>", figure_text,
                                         flags=re.I | re.S):
                    structured_output.setdefault("tables", []).append(
                        {"html": match.group(0), "source": "figure_deep_parse"}
                    )
                text = f"{text}\n\n{figure_text}".strip()
            if parsed_figures:
                structured_output["figure_deep_parse"] = parsed_figures

        gpu_seconds = time.perf_counter() - t0
        return {
            "raw_text": text,
            "structured_output": structured_output,
            "gpu_seconds": gpu_seconds,
            "gpu_type": GPU_TYPE,
            "model_version": f"{MODEL_ID} via vllm-server",
        }
