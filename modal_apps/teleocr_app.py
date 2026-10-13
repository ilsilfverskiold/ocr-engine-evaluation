"""Modal app: TeleOCR 1.2B document pipeline (XingChen-AGI/TeleOCR)."""

import modal

GPU_TYPE = "A10"  # first probe; L4 has the same 24 GB at a lower rate, L40S next

MODEL_ID = "XingChen-AGI/TeleOCR"  # Apache-2.0 (HF card/license tag)
MODEL_REVISION = "e92585356c0d0b7b7a65938f3da035c6593cc9a6"
# Pipeline code: TeleOCR is not on PyPI; installed from this exact commit.
TELEOCR_REPO = "caipeng328/TeleOCR"
TELEOCR_COMMIT = "1e71f4fe792d12bbb86d2671c5ed6f3a1499b27d"
TELEOCR_SRC = "/opt/TeleOCR"
LOCAL_MODEL_DIR = "/models/TeleOCR"

# Exact versions from the author's requirements.txt lock at TELEOCR_COMMIT
# (pyproject ranges: torch>=2.8,<2.9, transformers>=4.57,<4.58, vllm>=0.11,<0.12;
# TeleOCR-vllm plugin pins vllm==0.11.0, transformers==4.57.1). The whole lock
# is applied as a pip constraints file so every transitive package matches it.
VLLM_VERSION = "0.11.0"
TORCH_VERSION = "2.8.0"
TRANSFORMERS_VERSION = "4.57.1"
PYTHON_VERSION = "3.10"  # README: conda create -n teleocr python=3.10

# Repository defaults (TeleOCR/config.py) that this app keeps and asserts.
EXPECTED_CONFIG = {
    "BACKEND": "vllm-async-engine",
    "LAYOUT_MODE": "Detection",
    "MAX_MODEL_LEN": 16384,
    "GPU_MEMORY_UTILIZATION": 0.95,
    "PDF_TOOLS": "pypdfium2",
    "MAX_PIXELS": 8000 * 8000,
}


def _download_weights() -> None:
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=LOCAL_MODEL_DIR)
    print("TeleOCR snapshot:", path)


image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("curl", "libgl1", "libglib2.0-0")
    .run_commands(
        f"curl -fsSL https://codeload.github.com/{TELEOCR_REPO}/tar.gz/{TELEOCR_COMMIT} "
        f"-o /tmp/teleocr.tgz",
        f"mkdir -p {TELEOCR_SRC} && tar xzf /tmp/teleocr.tgz -C {TELEOCR_SRC} "
        f"--strip-components=1 && rm /tmp/teleocr.tgz",
        # README: `pip install -e .`. Editable install also registers the
        # vllm.general_plugins entry point TeleOCR_vllm:register.
        f"pip install --no-cache-dir -c {TELEOCR_SRC}/requirements.txt -e {TELEOCR_SRC}",
        "python -c 'import vllm, torch, transformers; "
        "print(\"vllm\", vllm.__version__, \"torch\", torch.__version__, "
        "\"transformers\", transformers.__version__); "
        f"assert vllm.__version__ == \"{VLLM_VERSION}\"; "
        f"assert torch.__version__.split(\"+\")[0] == \"{TORCH_VERSION}\"; "
        f"assert transformers.__version__ == \"{TRANSFORMERS_VERSION}\"'",
        "python -c 'from importlib.metadata import entry_points; "
        "eps = [e.value for e in entry_points(group=\"vllm.general_plugins\")]; "
        "print(eps); assert \"TeleOCR_vllm:register\" in eps'",
    )
    .run_function(_download_weights, secrets=[modal.Secret.from_name("huggingface-token")])
    .env({"TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1"})
)

app = modal.App("ocr-bench-teleocr")


# --------------------------------------------------------------------------- #
# Output adapter
# --------------------------------------------------------------------------- #

# Region types whose recognised content is printed text (structs.py BlockType,
# post_process PARATEXT_TYPES). Equations are LaTeX; "char" is a scientific
# figure converted to a data table; image/list/equation_block regions are not
# recognised by the pipeline (prepare_for_extract skip_list).
_TEXT_TYPES = {
    "text", "title", "header", "footer", "page_number", "aside_text",
    "page_footnote", "ref_text", "phonetic", "image_caption", "image_footnote",
    "table_caption", "table_footnote", "code_caption", "code", "algorithm", "seal",
}


def _html_table_to_text(html_text: str) -> str:
    import html
    import re

    text = re.sub(r"</t[dh]>\s*<t[dh][^>]*>", "\t", html_text)
    text = re.sub(r"</tr>\s*<tr[^>]*>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def _box_from_points(points):
    """Pipeline boxes are 0-1 of the page (0-1000 model output / 1000, see
    TeleOCR_client._convert_bbox). 4 numbers are a box; more are a polygon."""
    if not points or len(points) < 4 or len(points) % 2:
        return None, None
    vals = [float(v) for v in points]
    if len(vals) == 4:
        x0, y0, x1, y1 = vals
        return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], None
    xs, ys = vals[0::2], vals[1::2]
    poly = [[x, y] for x, y in zip(xs, ys)]
    return [min(xs), min(ys), max(xs), max(ys)], poly


def _blocks_to_structured(page_blocks: list[list[dict]]) -> dict:
    layout_blocks, tables, regions = [], [], []
    for page_idx, blocks in enumerate(page_blocks):
        for order, block in enumerate(blocks):
            btype = str(block.get("type"))
            bbox, poly = _box_from_points(block.get("bbox"))
            content = block.get("content")
            entry = {"label": btype, "bbox": bbox, "order": order, "page_idx": page_idx,
                     "angle": block.get("angle"), "content": content}
            if poly:
                entry["poly"] = poly
            layout_blocks.append(entry)
            if not content or bbox is None:
                continue
            if btype == "table":
                if "<table" in content.lower():
                    tables.append({"html": content, "bbox": bbox})
                    text = _html_table_to_text(content)
                    if text:
                        regions.append({"text": text, "bbox": bbox, "label": btype,
                                        **({"poly": poly} if poly else {})})
            elif btype in _TEXT_TYPES and content.strip():
                regions.append({"text": content.strip(), "bbox": bbox, "label": btype,
                                **({"poly": poly} if poly else {})})
    structured: dict = {"layout_blocks": layout_blocks}
    if tables:
        structured["tables"] = tables
    if regions:
        structured["text_regions"] = regions
        structured["geometry_granularity"] = "block"
        structured["geometry_coordinates"] = "normalized_xyxy"
    return structured


def _normalise_input(image_bytes: bytes):
    import io

    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), img.size


def _warmup_png() -> bytes:
    """Synthetic page with a title, text lines and a ruled table so layout,
    text-crop and table-crop (OTSL) prompts all run during startup."""
    import io

    from PIL import Image, ImageDraw, ImageFont

    page = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(page)
    try:
        title_font = ImageFont.load_default(size=44)
        body_font = ImageFont.load_default(size=24)
    except TypeError:
        title_font = body_font = ImageFont.load_default()
    draw.text((100, 100), "Quarterly Operating Report", fill="black", font=title_font)
    y = 220
    for i in range(6):
        draw.text((100, y), f"Line {i + 1}: revenue, costs and margin were reviewed for "
                  f"region {i + 1}.", fill="black", font=body_font)
        y += 40
    x0, y0, cw, ch = 100, 560, 340, 56
    rows = [["Item", "Q1", "Q2"], ["Revenue", "1,200", "1,350"],
            ["Costs", "800", "870"], ["Margin", "400", "480"]]
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            box = (x0 + c * cw, y0 + r * ch, x0 + (c + 1) * cw, y0 + (r + 1) * ch)
            draw.rectangle(box, outline="black", width=2)
            draw.text((box[0] + 12, box[1] + 14), cell, fill="black", font=body_font)
    buf = io.BytesIO()
    page.save(buf, format="PNG")
    return buf.getvalue()


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    cpu=8.0,
    memory=16384,
    timeout=1800,
    startup_timeout=1800,
    scaledown_window=30,
    secrets=[modal.Secret.from_name("huggingface-token")],
    max_containers=1,
)
class TeleOCRInference:
    @modal.enter()
    def load_model(self) -> None:
        import asyncio
        import contextvars
        import threading
        import time

        t0 = time.perf_counter()
        import TeleOCR.config as CONFIG

        for key, expected in EXPECTED_CONFIG.items():
            actual = getattr(CONFIG, key)
            if actual != expected:
                raise RuntimeError(f"TeleOCR config {key}={actual!r}, expected {expected!r}")
        # Resolve the upstream "StarDoc-AI/TeleOCR" name to the pinned snapshot.
        CONFIG.model_path = LOCAL_MODEL_DIR
        self._config = CONFIG

        # One long-lived event loop owns AsyncLLM. vLLM binds its output
        # handler to the loop that first calls generate(); a fresh
        # asyncio.run() per batch would strand it on a closed loop.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever,
                                             name="teleocr-loop", daemon=True)
        self._loop_thread.start()

        # Per-page generation accounting: finish_reason and token counts are
        # read off each vLLM RequestOutput. A ContextVar set in the page task is
        # inherited by the crop tasks that aio_batch_predict creates.
        self._page_stats = contextvars.ContextVar("teleocr_page_stats", default=None)

        async def _build():
            from TeleOCR.vlm_utils.TeleOCR_model import TeleOCRMODEL_SERVICE

            return TeleOCRMODEL_SERVICE.get_model(CONFIG.BACKEND, CONFIG.model_path, None)

        self._client = asyncio.run_coroutine_threadsafe(_build(), self._loop).result()
        vlm = self._client.client
        self._max_model_len = getattr(vlm, "model_max_length", None)
        original = vlm.get_output_content
        stats_var = self._page_stats

        def _recording_get_output_content(output):
            stats = stats_var.get()
            if stats is not None:
                try:
                    choice = output.outputs[0]
                    stats["requests"] += 1
                    stats["prompt_tokens"] += len(output.prompt_token_ids or [])
                    stats["output_tokens"] += len(choice.token_ids or [])
                    stats["max_prompt_tokens"] = max(stats["max_prompt_tokens"],
                                                     len(output.prompt_token_ids or []))
                    stats["max_output_tokens"] = max(stats["max_output_tokens"],
                                                     len(choice.token_ids or []))
                    reason = str(choice.finish_reason)
                    stats["finish_reasons"][reason] = stats["finish_reasons"].get(reason, 0) + 1
                except Exception:
                    stats["unrecorded"] += 1
            return original(output)

        vlm.get_output_content = _recording_get_output_content

        warm = self._run_documents([_warmup_png()])[0]
        if isinstance(warm, BaseException):
            raise RuntimeError(f"TeleOCR warm-up failed: {warm!r}") from warm

        self._startup_gpu_seconds = time.perf_counter() - t0
        self._startup_cost_applied = False

    @modal.exit()
    def shutdown(self) -> None:
        try:
            engine = getattr(getattr(self._client, "client", None), "vllm_async_llm", None)
            if engine is not None:
                engine.shutdown()
        except Exception:
            pass
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #

    def _new_stats(self) -> dict:
        return {"requests": 0, "prompt_tokens": 0, "output_tokens": 0,
                "max_prompt_tokens": 0, "max_output_tokens": 0,
                "finish_reasons": {}, "unrecorded": 0}

    async def _parse_documents(self, png_list: list[bytes]) -> list:
        """TeleOCR aio_doc_analyze for a list of documents, sharing one semaphore.

        Per document: images_bytes_to_pdf_bytes (read_fn's image path) ->
        load_images_from_pdf (200 dpi render, 3500 px long-edge cap) -> per page
        aio_two_step_extract -> result_to_middle_json -> union_make. The page
        coroutines of every document share the semaphore sized to the client's
        max_concurrency, as aio_batch_two_step_extract does for one document.
        """
        import asyncio

        from TeleOCR.data_reader_writer import ImageDataWriter
        from TeleOCR.src.model_output_to_middle_json import result_to_middle_json
        from TeleOCR.src.vlm_middle_json_mkcontent import union_make
        from TeleOCR.tools.enum_class import ImageType
        from TeleOCR.tools.pdf_image_tools import images_bytes_to_pdf_bytes, load_images_from_pdf

        CONFIG = self._config
        semaphore = asyncio.Semaphore(self._client.max_concurrency)

        async def one_page(image, stats):
            self._page_stats.set(stats)
            return await self._client.aio_two_step_extract(image, None, semaphore, None)

        async def one_document(png: bytes):
            pdf_bytes = images_bytes_to_pdf_bytes(png)
            images_list, pdf_doc = await asyncio.to_thread(
                load_images_from_pdf, pdf_bytes, image_type=ImageType.PIL,
                threads=CONFIG.PDF_TOOLS_WORKER_MAX_NUM)
            try:
                stats = self._new_stats()
                pages = [image_dict["img_pil"] for image_dict in images_list]
                results = await asyncio.gather(
                    *(asyncio.create_task(one_page(img, stats)) for img in pages))
                # post_process has already run inside aio_two_step_extract;
                # snapshot the block dicts before the middle-JSON builder
                # rewrites bboxes into page coordinates.
                raw_blocks = [[dict(b) for b in page] for page in results]
                writer = ImageDataWriter("images")  # in-memory; crops are never saved
                middle_json = await asyncio.to_thread(
                    result_to_middle_json, results, images_list, pdf_doc, writer)
                pdf_doc = None  # closed by result_to_middle_json
                markdown = union_make(middle_json["pdf_info"], "images")
                writer.clear_buffer()
                scale = images_list[0].get("scale") if images_list else None
                return {"raw_blocks": raw_blocks, "middle_json": middle_json,
                        "markdown": markdown, "stats": stats, "render_scale": scale,
                        "render_size": list(pages[0].size) if pages else None}
            finally:
                if pdf_doc is not None:
                    try:
                        pdf_doc.close()
                    except Exception:
                        pass

        return await asyncio.gather(*(one_document(p) for p in png_list),
                                    return_exceptions=True)

    def _run_documents(self, png_list: list[bytes]) -> list:
        import asyncio

        future = asyncio.run_coroutine_threadsafe(self._parse_documents(png_list), self._loop)
        return future.result()

    # ------------------------------------------------------------------ #

    def _model_version(self) -> str:
        c = self._config
        return (f"{MODEL_ID}@{MODEL_REVISION[:12]} TeleOCR@{TELEOCR_COMMIT[:12]} "
                f"pipeline (layout={c.LAYOUT_MODE}, backend={c.BACKEND}, vllm=={VLLM_VERSION}, "
                f"max_model_len={c.MAX_MODEL_LEN})")

    def _take_startup_metadata(self) -> tuple[float, bool]:
        """Report startup separately once; never add it to inference GPU-seconds."""
        if not getattr(self, "_startup_cost_applied", True):
            self._startup_cost_applied = True
            return getattr(self, "_startup_gpu_seconds", 0.0), True
        return 0.0, False

    def _error_item(self, document_path, exc) -> dict:
        message = str(exc) or repr(exc)
        return {
            "raw_text": "",
            "structured_output": {},
            "model_version": self._model_version(),
            "prompt_used": None,
            "document_path": document_path,
            "error": f"{exc.__class__.__name__}: {message}",
        }

    def _to_item(self, document_path, parsed: dict, image_size) -> dict:
        c = self._config
        structured = _blocks_to_structured(parsed["raw_blocks"])
        structured["raw_model_output"] = {"blocks": parsed["raw_blocks"],
                                          "middle_json": parsed["middle_json"]}
        stats = parsed["stats"]
        structured["generation"] = {
            **stats,
            "truncated_requests": stats["finish_reasons"].get("length", 0),
            "max_model_len": self._max_model_len,
            "max_tokens_per_request": self._max_model_len,  # client default: model max length
        }
        structured["preprocessing"] = {
            "input": "EXIF-transposed RGB PNG written by this app",
            "pipeline": ("Pillow PDF at 72 dpi (page points == input pixels), rendered at "
                         "200 dpi with a 3500 px long-edge cap; layout input resized to "
                         "1036x1036; crops taken from the rendered page"),
            "image_size": list(image_size),
            "render_size": parsed["render_size"],
            "render_scale": parsed["render_scale"],
            "crop_max_pixels": c.MAX_PIXELS,
            "bbox_space": "normalised 0-1 of the page == the image as sent",
        }
        structured["pipeline"] = {
            "backend": c.BACKEND, "layout_mode": c.LAYOUT_MODE,
            "max_model_len": c.MAX_MODEL_LEN,
            "gpu_memory_utilization": c.GPU_MEMORY_UTILIZATION,
            "max_concurrency": self._client.max_concurrency,
        }
        return {
            "raw_text": parsed["markdown"],
            "structured_output": structured,
            "model_version": self._model_version(),
            "prompt_used": None,
            "document_path": document_path,
            "error": None,
        }

    def _run_items(self, files: list[dict]) -> list[dict]:
        results: list[dict | None] = [None] * len(files)
        pngs, sizes, live = [], [], []
        for i, item in enumerate(files):
            if item.get("prompt") is not None or item.get("response_format") is not None:
                results[i] = self._error_item(item.get("document_path"), ValueError(
                    "teleocr is a fixed pipeline; it accepts no prompt or response_format"))
                continue
            try:
                png, size = _normalise_input(item["bytes"])
                pngs.append(png)
                sizes.append(size)
                live.append(i)
            except Exception as exc:
                results[i] = self._error_item(item.get("document_path"), exc)
        parsed = self._run_documents(pngs) if pngs else []
        for i, size, outcome in zip(live, sizes, parsed):
            doc = files[i].get("document_path")
            if isinstance(outcome, BaseException):
                results[i] = self._error_item(doc, outcome)
                continue
            try:
                results[i] = self._to_item(doc, outcome, size)
            except Exception as exc:
                results[i] = self._error_item(doc, exc)
        return results  # type: ignore[return-value]

    @modal.method()
    def run_ocr_batch(self, files: list[dict], prompt: str | None = None,
                      response_format: dict | None = None) -> dict:
        import time

        started = time.perf_counter()
        try:
            if prompt is not None or response_format is not None:
                raise ValueError("teleocr is a fixed pipeline; it accepts no prompt or "
                                 "response_format")
            results = self._run_items(files)
            if len(results) != len(files):
                raise RuntimeError(f"TeleOCR returned {len(results)} results for {len(files)} inputs")
        except Exception as exc:
            results = [self._error_item(item.get("document_path"), exc) for item in files]
        batch_gpu_seconds = time.perf_counter() - started
        startup_seconds, is_cold_start = self._take_startup_metadata()
        return {
            "results": results,
            "batch_gpu_seconds": batch_gpu_seconds,
            "gpu_type": GPU_TYPE,
            "startup_gpu_seconds": startup_seconds,
            "is_cold_start": is_cold_start,
            "attempted": len(files),
            "succeeded": sum(1 for r in results if not r.get("error")),
        }

    @modal.method()
    def run_ocr(self, image_bytes: bytes, mime_type: str = "image/png",
                prompt: str | None = None, response_format: dict | None = None) -> dict:
        import time

        if prompt is not None or response_format is not None:
            raise ValueError("teleocr is a fixed pipeline; it accepts no prompt or response_format")
        started = time.perf_counter()
        item = {"document_path": None, "bytes": image_bytes, "mime_type": mime_type}
        try:
            response = self._run_items([item])[0]
        except Exception as exc:
            response = self._error_item(None, exc)
        response["gpu_seconds"] = time.perf_counter() - started
        response["gpu_type"] = GPU_TYPE
        startup_seconds, is_cold_start = self._take_startup_metadata()
        response["startup_gpu_seconds"] = startup_seconds
        response["is_cold_start"] = is_cold_start
        return response
