"""
Modal app: Surya OCR 2 (~686M params, BF16) — datalab-to/surya-ocr-2. GPU: L4.

Route (all read from surya-ocr v0.22.1, github.com/datalab-to/surya/tree/v0.22.1):
- Surya 2 runs layout, OCR and table recognition through one VLM served by an
  OpenAI-compatible vLLM server. `SuryaInferenceManager` would normally spawn
  that server as a Docker container (surya/inference/backends/vllm.py). Modal
  containers cannot run Docker, so this app starts the SAME `vllm serve`
  command line inside @modal.enter and attaches the manager to it with
  SURYA_INFERENCE_URL (surya/inference/backends/spawn.py: attach_or_spawn,
  external-URL branch: probes <base>/health, then requires /v1/models to report
  exactly SURYA_MODEL_CHECKPOINT — hence --served-model-name below).
- Conversion uses the official `RecognitionPredictor(manager)(images,
  full_page=True)` — the path `surya_ocr` (scripts/ocr_text.py) uses: one
  HIGH_ACCURACY_BBOX_PROMPT request per page returning top-level
  <div data-label data-bbox> HTML blocks. Pages whose full-page output errors,
  is empty on a non-blank page, loops, or fails to parse fall back INSIDE the
  predictor to LayoutPredictor + per-block OCR (recognition/__init__.py).
  Layout+block mode as the default is a separate later probe (see registry
  proposal); README: "Full-page is the more accurate path".
- Extraction legs (key_value/schema) use the manager's documented caller
  prompt override: BatchInputItem.prompt "overrides the default prompt for
  prompt_type" and openai_client._generate_one sends
  `item.prompt or PROMPT_MAPPING[item.prompt_type]`. BatchInputItem.guided_json
  becomes an OpenAI `response_format` json_schema (strict). The raw generation
  is the answer; nothing is parsed out of it here.

Batching owner: vLLM continuous batching, fed by Surya's own client thread pool
(openai_client.chat_completions_batch, max_workers = SURYA_INFERENCE_PARALLEL).
There is no second pool in this app: the whole conversion list goes to ONE
predictor call, and the extraction list to ONE manager.generate call.

Dependency isolation: surya-ocr 0.22.1 pins opencv-python-headless==4.11.0.86,
while vllm 0.20.1 requires opencv-python-headless>=4.13.0 (PyPI metadata), so
vLLM lives in its own venv (same pattern as paddleocr_vl_app.py).

Licensing: Surya code Apache-2.0; weights under a modified AI Pubs OpenRAIL-M
licence (free for research/personal use and startups under $5M
funding/revenue; see the model card). Benchmarking is research use; any
commercial deployment needs its own licence review.
"""

import modal

GPU_TYPE = "L4"
MODEL_ID = "datalab-to/surya-ocr-2"
# huggingface.co/api/models/datalab-to/surya-ocr-2 on 2026-09-30.
MODEL_REVISION = "3b3d4cdf88d6928b0acdc75181b13206ea67c4a3"
SURYA_VERSION = "0.22.1"          # PyPI latest on 2026-09-30 (tag v0.22.1)
# surya/settings.py VLLM_DOCKER_IMAGE = "vllm/vllm-openai:v0.20.1" — the
# server version Surya 0.22.1 launches. Same version from PyPI.
VLLM_VERSION = "0.20.1"
# Same converter + pin chandra-ocr 0.2.0 uses for its own Markdown
# (chandra/output.py, pyproject.toml). Surya itself ships no Markdown renderer.
MARKDOWNIFY_VERSION = "1.1.0"

# Must equal surya settings.SURYA_MODEL_CHECKPOINT or attach raises
# "Model mismatch" (spawn.py). Surya's own spawn uses the same served name.
SERVED_MODEL_NAME = MODEL_ID
VLLM_PORT = 8000
VLLM_BASE = f"http://127.0.0.1:{VLLM_PORT}"
VENV = "/opt/vllmenv"

# --- vLLM server flags: copied from surya/inference/backends/vllm.py spawn_fn
# and surya/settings.py (v0.22.1). _gpu_settings("l4") -> ratio 24/24 = 1 ->
# max_num_batched_tokens 8192, max_num_seqs 32.
VLLM_MAX_MODEL_LEN = 18000          # settings.VLLM_MAX_MODEL_LEN
VLLM_GPU_MEMORY_UTILIZATION = 0.85  # settings.VLLM_GPU_MEMORY_UTILIZATION
VLLM_MAX_NUM_SEQS = 32
VLLM_MAX_NUM_BATCHED_TOKENS = 8192
VLLM_DTYPE = "bfloat16"             # settings.VLLM_DTYPE (original precision)
# vllm.py passes these to --mm-processor-kwargs; max_pixels equals
# surya.inference.util.scale_to_fit's client-side cap (3072*2048).
MM_PROCESSOR_KWARGS = {"min_pixels": 3136, "max_pixels": 6291456}
# settings.VLLM_ENABLE_MTP=True, VLLM_MTP_TOKENS=2: Surya's default server runs
# MTP speculative decoding (config.json declares mtp_num_hidden_layers=1).
MTP_TOKENS = 2

# Client-side concurrent requests (SURYA_INFERENCE_PARALLEL). The model card
# lists 8 as the default for this setting; 0.22.1's code instead auto-scales to
# max_num_seqs (32 on L4) when unset. 8 is set explicitly so the active
# concurrency is a recorded constant. Sweep 4/8/16/32 on one warm L4 before
# publishing throughput.
BATCH_CONCURRENCY = 8

# settings.SURYA_MAX_TOKENS_FULL_PAGE = 12288 in 0.22.1 (the research handoff's
# "8,192" is stale). Read back from surya.settings at runtime and recorded.
# Context arithmetic: scale_to_fit caps pages at 6,291,456 px; Qwen3.5 vision
# (patch 16, spatial merge 2) = 1 visual token per 32x32 px -> <= 6,144 visual
# tokens. 6,144 + prompt + 12,288 exceeds the 18,000 context on the largest
# pages; vLLM 0.20.1 does not reject that — entrypoints/utils.py:get_max_tokens
# clamps max_tokens to max_model_len - input_length. So on a max-size page the
# effective output cap is ~11.8k tokens. Kept at Surya's own values; the clamp
# is recorded per page (generation.effective_cap_estimate).
#
# UNVERIFIED: no extraction token budget is documented for Surya. The full-page
# budget is reused for prompted extraction so a long JSON answer is not cut
# shorter than a page transcription would be. Probe: token_count distribution
# on the 65 schema pages; any page at the cap is truncated.
EXTRACTION_MAX_TOKENS = 12288

# Canonical labels whose boxes carry no transcribed text (Surya skips OCR on
# them: SKIP_OCR_LABELS {Figure, Image, Diagram, Blank-Page} -> canonical
# Figure/Picture/Diagram/BlankPage via layout/label.py). Never used as text
# geometry.
NON_TEXT_LABELS = frozenset({"Picture", "Figure", "Diagram", "BlankPage"})


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "python3-venv")
    .pip_install(
        f"surya-ocr=={SURYA_VERSION}",
        f"markdownify=={MARKDOWNIFY_VERSION}",
    )
    .run_commands(
        f"python -m venv {VENV}",
        f"{VENV}/bin/pip install --no-cache-dir vllm=={VLLM_VERSION}",
        f"{VENV}/bin/python -c 'import vllm; print(\"vllm\", vllm.__version__)'",
    )
    # Bake the pinned weights into the image layer; the server runs with
    # HF_HUB_OFFLINE=1 so nothing can drift at container start.
    .run_commands(
        f"{VENV}/bin/python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id='{MODEL_ID}', revision='{MODEL_REVISION}')\"",
        secrets=[modal.Secret.from_name("huggingface-token")],
    )
    .env({
        # Modal slim images ship no nvcc; FlashInfer's sampler may JIT at
        # warm-up (same setting as the other vLLM apps in this repo).
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    })
)

app = modal.App("ocr-bench-surya-ocr-2")


# --------------------------------------------------------------------------
# Pure helpers (kept self-contained: the container cannot import repo modules)
# --------------------------------------------------------------------------

def _load_page(image_bytes: bytes, tmpdir: str, index: int):
    """EXIF-transpose + RGB, then Surya's own image loader.

    surya.input.load.load_image(path) is what the CLI uses for image inputs:
    Image.open(path).convert("RGB") with no resize (IMAGE_DPI settings apply
    to PDF rendering only). Surya's client later applies scale_to_fit; its
    bbox output is 0-1000 relative to the page, so no coordinate remap is
    needed for that resize.
    """
    import io
    from pathlib import Path

    from PIL import Image, ImageOps
    from surya.input.load import load_image

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    path = Path(tmpdir) / f"{index:04d}.png"
    img.save(path, format="PNG")
    images, _ = load_image(str(path))
    return images[0]


def _html_to_text(html_text: str) -> str:
    from bs4 import BeautifulSoup

    return BeautifulSoup(html_text or "", "html.parser").get_text(" ", strip=True)


def _html_to_markdown(html_text: str) -> str:
    """Deterministic HTML -> Markdown, tables kept as HTML.

    Mirrors chandra-ocr 0.2.0's Markdownify configuration (chandra/output.py:
    ATX headings, '-' bullets, math as $/$$, tables passed through as HTML) so
    both Datalab engines are rendered the same way. Adapter-side conversion:
    Surya 0.22.1 emits HTML only.
    """
    import re

    from markdownify import MarkdownConverter

    class _Md(MarkdownConverter):
        def convert_math(self, el, text, parent_tags):
            if el.has_attr("display") and el["display"] == "block":
                return "\n$$" + text.strip() + "$$\n"
            return " $" + text.strip() + "$ "

        def convert_table(self, el, text, parent_tags):
            return "\n\n" + str(el) + "\n\n"

    try:
        md = _Md(heading_style="ATX", bullets="-", escape_misc=False,
                 escape_underscores=True, escape_asterisks=True,
                 sub_symbol="<sub>", sup_symbol="<sup>").convert(html_text or "")
    except Exception:
        md = _html_to_text(html_text)
    return re.sub(r"\n{3,}", "\n\n", md).strip()


def _norm_poly(polygon, width: float, height: float):
    """Pixel polygon -> (bbox, poly) normalised 0-1 to the page as sent."""
    if not polygon or not width or not height:
        return None, None
    try:
        xs = [float(p[0]) for p in polygon]
        ys = [float(p[1]) for p in polygon]
    except (TypeError, ValueError, IndexError):
        return None, None
    bbox = [min(xs) / width, min(ys) / height,
            max(xs) / width, max(ys) / height]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None, None
    poly = [[x / width, y / height] for x, y in zip(xs, ys)]
    return bbox, poly


def _visual_tokens_estimate(image) -> int | None:
    """Estimate of the page's visual tokens after Surya's client-side resize.

    scale_to_fit output, divided into 32x32 px tokens (patch_size 16 x
    spatial_merge_size 2, config.json). An ESTIMATE: vLLM's processor applies
    its own rounding within MM_PROCESSOR_KWARGS.
    """
    try:
        from surya.inference.util import scale_to_fit

        w, h = scale_to_fit(image).size
        return max(1, round(w / 32)) * max(1, round(h / 32))
    except Exception:
        return None


def _generation_record(call: dict | None, max_tokens: int, visual_tokens) -> dict:
    """Finish/truncation evidence. Surya's client does not return
    finish_reason (openai_client.GenerationResult has raw/token_count/error/
    mean_token_prob only), so truncation is inferred from token counts."""
    rec = {
        "finish_reason": None,
        "finish_reason_note": "not exposed by surya openai_client",
        "max_tokens": max_tokens,
        "max_model_len": VLLM_MAX_MODEL_LEN,
        "visual_tokens_estimate": visual_tokens,
        "client_retry_policy": "surya openai_client: up to 3 retries on error or "
                               "detected repetition; retry temperature "
                               "min(0.2*(n+1), 0.8), top_p 0.95",
    }
    if call is None:
        return rec
    tokens = call.get("token_count") or 0
    cap = max_tokens
    if visual_tokens:
        cap = min(max_tokens, VLLM_MAX_MODEL_LEN - visual_tokens)
    rec.update({
        "completion_tokens": tokens,
        "effective_cap_estimate": cap,
        # Within 256 tokens of the (estimated) cap -> treat as truncated.
        "possible_truncation": bool(tokens and tokens >= cap - 256),
        "error": bool(call.get("error")),
        "mean_token_prob": call.get("mean_token_prob"),
    })
    return rec


class _RecordingManager:
    """Pass-through wrapper so raw generations survive the predictor.

    RecognitionPredictor only keeps parsed blocks; the model's raw HTML (and
    whether a page fell back to block mode) would otherwise be lost. This
    wrapper records every manager.generate() item and delegates everything
    else unchanged — it adds no requests and no batching.
    """

    def __init__(self, manager):
        self._manager = manager
        self.calls: list[dict] = []

    def generate(self, batch):
        outputs = self._manager.generate(batch)
        for item, out in zip(batch, outputs):
            self.calls.append({
                "prompt_type": item.prompt_type,
                "metadata": dict(item.metadata or {}),
                "max_tokens": item.max_tokens,
                "raw": out.raw,
                "token_count": out.token_count,
                "error": out.error,
                "mean_token_prob": out.mean_token_prob,
            })
        return outputs

    def __getattr__(self, name):
        return getattr(self._manager, name)


def _json_guided(response_format: dict | None):
    """Harness response_format -> Surya BatchInputItem.guided_json."""
    if not response_format:
        return None
    kind = response_format.get("type")
    if kind == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema")
        return schema if isinstance(schema, dict) else None
    if kind == "json_object":
        # UNVERIFIED: Surya only exposes json_schema (strict) guidance; the
        # schema {"type": "object"} is the JSON-Schema form of "any JSON
        # object", i.e. the json_object constraint. Probe: one key_value page,
        # confirm vLLM accepts it and output parses.
        return {"type": "object"}
    return None


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=3600,
    startup_timeout=1800,
    scaledown_window=30,
    secrets=[modal.Secret.from_name("huggingface-token")],
    max_containers=1,
)
class SuryaOCR2Inference:
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
            "--speculative-config",
            json.dumps({"method": "mtp", "num_speculative_tokens": MTP_TOKENS}),
        ]
        self._server = subprocess.Popen(
            cmd,
            env={**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0",
                 "HF_HUB_OFFLINE": "1"},
        )
        self._wait_for_server()

        # Surya reads settings from the environment at import time
        # (pydantic-settings), so these must be set before the first import.
        os.environ.update({
            "SURYA_INFERENCE_BACKEND": "vllm",
            "SURYA_INFERENCE_URL": f"{VLLM_BASE}/v1",
            "SURYA_INFERENCE_AUTOSTART": "false",
            "SURYA_INFERENCE_PARALLEL": str(BATCH_CONCURRENCY),
            "SURYA_MODEL_CHECKPOINT": MODEL_ID,
            "VLLM_GPU_TYPE": "l4",
            "DISABLE_TQDM": "true",
        })
        from surya.inference import SuryaInferenceManager
        from surya.settings import settings as surya_settings

        self._surya_settings = surya_settings
        self._manager = SuryaInferenceManager(method="vllm")
        # Attach (not spawn): raises SpawnError if /health fails or the served
        # model id differs from SURYA_MODEL_CHECKPOINT.
        self._manager.start()
        self._full_page_max_tokens = int(surya_settings.SURYA_MAX_TOKENS_FULL_PAGE)
        self._config_summary = (
            f"surya-ocr=={SURYA_VERSION} RecognitionPredictor full_page "
            f"(block fallback), vllm=={VLLM_VERSION} bf16 MTP={MTP_TOKENS} "
            f"ctx={VLLM_MAX_MODEL_LEN} max_tokens={self._full_page_max_tokens} "
            f"parallel={BATCH_CONCURRENCY}"
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
        """One measured batch. Per-item "prompt"/"response_format" keys
        override the call arguments (the harness mixes passes in one call)."""
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

        started = time.perf_counter()
        results: list[dict | None] = [None] * len(items)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                pages = {}
                for index, item in enumerate(items):
                    try:
                        pages[index] = _load_page(item["bytes"], tmpdir, index)
                    except Exception as exc:
                        results[index] = self._error_item(item, exc)

                convert_idx = [i for i in pages if not items[i].get("prompt")]
                extract_idx = [i for i in pages if items[i].get("prompt")]
                if convert_idx:
                    self._convert(items, pages, convert_idx, results)
                if extract_idx:
                    self._extract(items, pages, extract_idx, results)
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

    def _convert(self, items, pages, indices, results) -> None:
        """Official full-page conversion over the whole list in ONE call."""
        from surya.common.blank import is_blank_region
        from surya.inference.parsers import parse_full_page_html
        from surya.recognition import RecognitionPredictor, _detect_repeat_loop

        recorder = _RecordingManager(self._manager)
        predictor = RecognitionPredictor(recorder)
        predictor.disable_tqdm = True
        images = [pages[i] for i in indices]
        page_results = predictor(images, full_page=True)
        if len(page_results) != len(images):
            raise RuntimeError(
                f"RecognitionPredictor returned {len(page_results)} pages for "
                f"{len(images)} inputs")

        full_calls = {c["metadata"].get("page_idx"): c for c in recorder.calls
                      if c["prompt_type"] == "high_accuracy_bbox"}

        # Reproduce the predictor's own acceptance rule (recognition/__init__.py
        # _build_page) to know which pages fell back to layout + block OCR.
        # Fallback calls are indexed by position in the fallback list.
        rejected = []
        for local, img in enumerate(images):
            call = full_calls.get(local)
            ok = True
            if call is None or call["error"]:
                ok = False
            elif not call["raw"]:
                ok = is_blank_region(img)
            elif _detect_repeat_loop(call["raw"]):
                ok = False
            else:
                try:
                    parse_full_page_html(call["raw"])
                except Exception:
                    ok = False
            if not ok:
                rejected.append(local)
        fallback_calls: dict[int, list[dict]] = {}
        for c in recorder.calls:
            if c["prompt_type"] == "high_accuracy_bbox":
                continue
            fb_pos = c["metadata"].get("page_idx")
            if isinstance(fb_pos, int) and 0 <= fb_pos < len(rejected):
                fallback_calls.setdefault(rejected[fb_pos], []).append(c)

        for local, (index, page) in enumerate(zip(indices, page_results)):
            try:
                results[index] = self._page_response(
                    items[index], images[local], page, full_calls.get(local),
                    local in rejected, fallback_calls.get(local, []))
            except Exception as exc:
                results[index] = self._error_item(items[index], exc)

    def _page_response(self, item, img, page, full_call, fell_back,
                       fallback_calls) -> dict:
        import re

        width, height = img.size
        layout_blocks, text_regions, tables = [], [], []
        html_parts, md_parts = [], []
        for block in page.blocks:
            bbox, poly = _norm_poly(block.polygon, width, height)
            html = block.html or ""
            layout_blocks.append({
                "label": block.label,
                "raw_label": block.raw_label,
                "order": block.reading_order,
                "bbox": bbox,
                "poly": poly,
                "content": html,
                "skipped": block.skipped,
                "error": block.error,
                "confidence": block.confidence,
            })
            if block.skipped or block.error or not html:
                continue
            html_parts.append(html)
            md_parts.append(_html_to_markdown(html))
            for match in re.finditer(r"<table\b.*?</table>", html, flags=re.I | re.S):
                tables.append({"html": match.group(0), "block_label": block.label})
            text = _html_to_text(html)
            if text and bbox and block.label not in NON_TEXT_LABELS:
                text_regions.append({"text": text, "bbox": bbox, "poly": poly,
                                     "label": block.label})

        markdown = "\n\n".join(p for p in md_parts if p).strip()
        visual = _visual_tokens_estimate(img)
        structured: dict = {
            "html": "\n".join(html_parts),
            "layout_blocks": layout_blocks,
            "markdown_source": "adapter: markdownify==1.1.0 over Surya block HTML "
                               "(chandra-ocr 0.2.0 settings); tables kept as HTML",
            "raw_model_output": (full_call or {}).get("raw", ""),
            "generation": _generation_record(full_call, self._full_page_max_tokens,
                                             visual),
            "conversion_mode": "block_fallback" if fell_back else "full_page",
            "preprocessing": {
                "exif_transpose": True,
                "client_resize": "surya scale_to_fit (<=6,291,456 px, 28 px grid); "
                                 "bboxes are 0-1000 page-relative so no remap",
                "image_size": [width, height],
            },
        }
        if fell_back:
            structured["fallback_generations"] = [
                {k: c[k] for k in ("prompt_type", "metadata", "max_tokens", "raw",
                                   "token_count", "error")}
                for c in fallback_calls
            ]
        if tables:
            structured["tables"] = tables
        if text_regions:
            structured["text_regions"] = text_regions
            structured["geometry_granularity"] = "block"
            structured["geometry_coordinates"] = "normalized_xyxy"
        error = None
        if not page.blocks and full_call is not None and full_call.get("error"):
            error = "GenerationError: full-page request failed and no fallback blocks"
        return {
            "raw_text": markdown,
            "structured_output": structured,
            "model_version": f"{MODEL_ID}@{MODEL_REVISION[:12]} {self._config_summary}",
            "prompt_used": "high_accuracy_bbox (surya HIGH_ACCURACY_BBOX_PROMPT)",
            "document_path": item.get("document_path"),
            "error": error,
        }

    def _extract(self, items, pages, indices, results) -> None:
        """Caller-prompted extraction through the manager's prompt override."""
        from surya.inference.schema import (
            PROMPT_TYPE_HIGH_ACCURACY_BBOX,
            BatchInputItem,
        )

        batch = []
        for i in indices:
            batch.append(BatchInputItem(
                image=pages[i],
                # Ignored when `prompt` is set (openai_client._generate_one);
                # required by the dataclass. Page-level type for the record.
                prompt_type=PROMPT_TYPE_HIGH_ACCURACY_BBOX,
                prompt=items[i]["prompt"],
                max_tokens=EXTRACTION_MAX_TOKENS,
                guided_json=_json_guided(items[i].get("response_format")),
                metadata={"item_index": i},
            ))
        outputs = self._manager.generate(batch)
        if len(outputs) != len(batch):
            raise RuntimeError(f"manager returned {len(outputs)} outputs for "
                               f"{len(batch)} extraction items")
        for i, bitem, out in zip(indices, batch, outputs):
            call = {"raw": out.raw, "token_count": out.token_count,
                    "error": out.error, "mean_token_prob": out.mean_token_prob}
            results[i] = {
                "raw_text": out.raw or "",
                "structured_output": {
                    "raw_model_output": out.raw or "",
                    "generation": _generation_record(
                        call, EXTRACTION_MAX_TOKENS, _visual_tokens_estimate(pages[i])),
                    "guided_json": bitem.guided_json is not None,
                    "response_format_requested": items[i].get("response_format"),
                },
                "model_version": (
                    f"{MODEL_ID}@{MODEL_REVISION[:12]} surya-ocr=={SURYA_VERSION} "
                    f"manager.generate prompt override, vllm=={VLLM_VERSION} bf16 "
                    f"MTP={MTP_TOKENS} max_tokens={EXTRACTION_MAX_TOKENS}"
                ),
                "prompt_used": items[i]["prompt"],
                "document_path": items[i].get("document_path"),
                "error": ("GenerationError: surya client reported a failed request "
                          "after retries") if out.error else None,
            }

    def _error_item(self, item: dict, exc: Exception) -> dict:
        return {
            "raw_text": "",
            "structured_output": {},
            "model_version": f"{MODEL_ID}@{MODEL_REVISION[:12]} surya-ocr=={SURYA_VERSION}",
            "prompt_used": item.get("prompt"),
            "document_path": item.get("document_path"),
            "error": f"{exc.__class__.__name__}: {str(exc) or repr(exc)}",
        }

    def _take_startup_metadata(self) -> tuple[float, bool]:
        """Report startup once per container; never add it to batch seconds."""
        with self._startup_lock:
            if not self._startup_cost_applied:
                self._startup_cost_applied = True
                return getattr(self, "_startup_gpu_seconds", 0.0), True
        return 0.0, False

    def _wait_for_server(self) -> None:
        import time
        import urllib.error
        import urllib.request

        deadline = time.time() + 1200
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
                            if m.status == 200:
                                return
            except (OSError, urllib.error.URLError) as exc:
                last = exc
            time.sleep(5)
        raise TimeoutError(f"vLLM server never became ready: {last}")

    def _warmup(self) -> None:
        """Representative warm-up: one full-page conversion of a synthetic
        text page (compiles the vision path, MTP and the HTML route)."""
        from PIL import Image, ImageDraw
        from surya.recognition import RecognitionPredictor

        page = Image.new("RGB", (1240, 1754), (255, 255, 255))
        draw = ImageDraw.Draw(page)
        draw.text((100, 100), "INVOICE No. 1042", fill=(0, 0, 0))
        for row in range(12):
            draw.text((100, 200 + 40 * row),
                      f"Item {row + 1}    Qty {row + 2}    Amount {row * 7 + 3}.00",
                      fill=(0, 0, 0))
        predictor = RecognitionPredictor(self._manager)
        predictor.disable_tqdm = True
        predictor([page], full_page=True)
