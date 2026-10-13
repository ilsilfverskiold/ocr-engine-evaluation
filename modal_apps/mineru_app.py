"""
Modal app: MinerU2.5 (1.2B) — opendatalab/MinerU2.5-2509-1.2B
Uses mineru-vl-utils vLLM engine + two_step_extract().
GPU: A100-80GB.
"""

import modal

GPU_TYPE = "A100-80GB"  # hardware-matched to published PaddleOCR-VL/MinerU throughput figures (single A100)
MODEL_ID = "opendatalab/MinerU2.5-2509-1.2B"
# Pinned to the exact HF commit. Model repos are mutable; an unpinned
# MODEL_ID silently changes what was benchmarked.
MODEL_REVISION = "1aa090b41282e64fadd79c10572221f91ec21924"


def _html_table_to_text(html_text: str) -> str:
    import html
    import re

    text = re.sub(r"</t[dh]>\s*<t[dh][^>]*>", "\t", html_text)
    text = re.sub(r"</tr>\s*<tr[^>]*>", "\n", text)
    text = re.sub(r"</?(table|tbody|thead|tr|td|th)[^>]*>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def _mineru_result_to_text(result) -> tuple[str, dict]:
    """Convert MinerU layout blocks into benchmark-friendly plain text."""
    if result is None:
        return "", {}
    if isinstance(result, str):
        return result, {}
    if isinstance(result, dict):
        result = [result]
    if not isinstance(result, list):
        return str(result), {}

    # MinerU2.5 returns typed ContentBlocks: {type, bbox (normalised 0-1),
    # angle, content}. Layout detection predicts class, position, rotation and
    # reading order in one pass — that type system is the model's main output,
    # and flattening it to newline-joined text before anything saw it meant the
    # markdown leg scored plain text and geometry was called ineligible.
    lines: list[str] = []
    markdown: list[str] = []
    tables: list[dict] = []
    blocks: list[dict] = []

    # Block type -> markdown rendering. MinerU emits table content as HTML
    # already (the model generates OTSL, the wrapper converts it), so tables
    # pass through untouched.
    headings = {"title": "# ", "header": "## "}

    for block in result:
        if not isinstance(block, dict):
            if block is not None:
                lines.append(str(block))
            continue

        btype = str(block.get("type") or "")
        content = block.get("content", "")
        blocks.append({
            "label": btype,
            "bbox": block.get("bbox"),
            "angle": block.get("angle"),
            "content": content,
        })

        if content is None:
            continue
        content = str(content).strip()
        if not content:
            continue

        if btype == "table":
            tables.append({"html": content})
            lines.append(_html_table_to_text(content))
            markdown.append(content)          # already HTML; scorers read it
        elif btype in headings:
            lines.append(content)
            markdown.append(f"{headings[btype]}{content}")
        elif btype == "list_item":
            lines.append(content)
            markdown.append(f"- {content}")
        elif btype in ("page_number", "footer"):
            lines.append(content)
            markdown.append(content)
        else:
            lines.append(content)
            markdown.append(content)

    structured_output: dict = {}
    if tables:
        structured_output["tables"] = tables
    if blocks:
        structured_output["layout_blocks"] = blocks
        # bbox is already normalised 0-1, which is what the geometry scorer
        # expects. Layout-element boxes, not word boxes.
        words = [
            {"text": str(b["content"]), "bbox": list(b["bbox"])}
            for b in blocks
            if b.get("bbox") and b.get("content")
            and isinstance(b["bbox"], (list, tuple)) and len(b["bbox"]) == 4
        ]
        if words:
            structured_output["words"] = words
    structured_output["markdown"] = "\n\n".join(markdown)
    return "\n".join(lines), structured_output


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "Pillow>=10.0",
        "huggingface_hub>=0.23",
        "mineru-vl-utils[vllm]==0.2.7",
    )
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

app = modal.App("ocr-bench-mineru")


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=1800,
    startup_timeout=1800,
    scaledown_window=30,
    secrets=[modal.Secret.from_name("huggingface-token")],
    max_containers=1,
)
class MinerUInference:
    @modal.enter()
    def load_model(self) -> None:
        import os
        import time

        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

        from mineru_vl_utils import MinerUClient, MinerULogitsProcessor
        from vllm import LLM

        t0 = time.perf_counter()
        self._llm = LLM(
            model=MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            logits_processors=[MinerULogitsProcessor],
            gpu_memory_utilization=0.90,
        )
        self._client = MinerUClient(
            backend="vllm-engine",
            vllm_llm=self._llm,
        )
        # Warm up CUDA graphs inside the startup timer so the first real
        # inference doesn't carry hidden compilation overhead.
        from PIL import Image as _PILImage
        try:
            self._client.two_step_extract(_PILImage.new("RGB", (32, 32), (255, 255, 255)))
        except Exception:
            pass
        self._startup_gpu_seconds = time.perf_counter() - t0
        self._startup_cost_applied = False

    @modal.method()
    def run_ocr(self, image_bytes: bytes, mime_type: str = "image/png") -> dict:
        try:
            response = self._run_ocr_impl(image_bytes, mime_type)
        except Exception as exc:
            response = self._error_response(exc)
        startup_seconds, is_cold_start = self._take_startup_metadata()
        response["startup_gpu_seconds"] = startup_seconds
        response["is_cold_start"] = is_cold_start
        return response

    @modal.method()
    def run_ocr_batch(self, files: list[dict]) -> dict:
        """Use MinerUClient's native synchronous vLLM batch API on one A100."""
        import io
        import time
        from PIL import Image

        started = time.perf_counter()
        try:
            images = [
                Image.open(io.BytesIO(item["bytes"])).convert("RGB")
                for item in files
            ]
            extracted = self._client.batch_two_step_extract(images)
            results = []
            for item, blocks in zip(files, extracted):
                text, structured_output = _mineru_result_to_text(blocks)
                results.append({
                    "document_path": item.get("document_path"),
                    "raw_text": text,
                    "structured_output": structured_output,
                    "gpu_type": GPU_TYPE,
                    "model_version": f"{MODEL_ID}@{MODEL_REVISION} via vllm-engine native batch",
                })
            if len(results) != len(files):
                raise RuntimeError(
                    f"MinerU returned {len(results)} results for {len(files)} images"
                )
        except Exception as exc:
            results = []
            for item in files:
                response = self._error_response(exc)
                response["document_path"] = item.get("document_path")
                results.append(response)

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
            "model_version": f"{MODEL_ID}@{MODEL_REVISION} via vllm-engine",
            "error": f"{exc.__class__.__name__}: {message}",
        }

    def _run_ocr_impl(self, image_bytes: bytes, mime_type: str = "image/png") -> dict:
        import io
        import time
        from PIL import Image

        t0 = time.perf_counter()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        result = self._client.two_step_extract(img)
        text, structured_output = _mineru_result_to_text(result)

        gpu_seconds = time.perf_counter() - t0
        return {
            "raw_text": text,
            "structured_output": structured_output,
            "gpu_seconds": gpu_seconds,
            "gpu_type": GPU_TYPE,
            "model_version": f"{MODEL_ID}@{MODEL_REVISION} via vllm-engine",
        }
