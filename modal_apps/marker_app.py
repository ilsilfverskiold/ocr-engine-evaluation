"""
Modal app: Marker 1.8.2 — VikParuchuri/marker
Popular Western open-source document parser. GPU: A100-80GB.
Handles PDFs directly.
"""

import modal

GPU_TYPE = "A100-80GB"  # hardware-matched to published throughput figures (single A100)
# Marker 1.8.2's own batch helper chooses floor(VRAM_GiB / 7) workers.  Modal's
# A100-80GB reports ~79 GiB to nvidia-smi, so the upstream setting is 11.
MARKER_WORKERS = 11

# Pass sentinels, not prompts. Marker 1.8.2 has a converter per task; the
# registry selects one by sending these strings. DOC_PASS is the default.
OCR_PASS = "__marker_ocr_converter__"
TABLE_PASS = "__marker_table_converter__"
DOC_PASS = "__marker_pdf_converter__"
MARKER_BATCH_CONFIG = {
    "layout_batch_size": 12,
    "detection_batch_size": 8,
    "table_rec_batch_size": 12,
    "ocr_error_batch_size": 12,
    "recognition_batch_size": 64,
    "equation_batch_size": 16,
    "detector_postprocessing_cpu_workers": 2,
    "disable_multiprocessing": True,
    "disable_tqdm": True,
}


def _blank_pdf() -> bytes:
    return (
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f\r\n0000000009 00000 n\r\n"
        b"0000000068 00000 n\r\n0000000125 00000 n\r\n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
    )


def _marker_worker_init(ready_queue, config: dict) -> None:
    """Load and warm one persistent Marker worker during Modal startup."""
    import os
    import tempfile

    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    global _marker_model_dict, _marker_config
    _marker_model_dict = create_model_dict()
    _marker_config = config

    warm_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
            handle.write(_blank_pdf())
            warm_path = handle.name
        PdfConverter(
            artifact_dict=_marker_model_dict,
            config=_marker_config,
        )(warm_path)
    except Exception:
        # The deliberately minimal PDF is only a CUDA/model warmup.  Model loading
        # succeeded if execution reached this point, even if the PDF parser rejects it.
        pass
    finally:
        if warm_path:
            try:
                os.unlink(warm_path)
            except OSError:
                pass
    ready_queue.put(os.getpid())


def _marker_page_size(path: str):
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.size
    except Exception:
        return (0, 0)


def _block_geometry(node, out: list, page_size=(0, 0)) -> None:
    """Collect {label, bbox, text} from Marker's JSON block tree.

    Marker reports PIXEL coordinates; the scorer expects 0-1. Without the page
    size the boxes come back as e.g. [0, 0, 1489, 2106] and every one misses.
    """
    width, height = page_size
    if isinstance(node, dict):
        poly = node.get("polygon")
        bbox = node.get("bbox")
        if poly or bbox:
            flat = bbox
            if not flat and poly:
                try:
                    xs = [float(p[0]) for p in poly]
                    ys = [float(p[1]) for p in poly]
                    flat = [min(xs), min(ys), max(xs), max(ys)]
                except (TypeError, ValueError, IndexError):
                    flat = None
            norm = None
            if flat and width and height:
                try:
                    x0, y0, x1, y1 = (float(v) for v in flat)
                    norm = [x0 / width, y0 / height, x1 / width, y1 / height]
                except (TypeError, ValueError):
                    norm = None
            out.append({
                "label": node.get("block_type"),
                "bbox": norm,
                "bbox_px": flat,
                "text": node.get("html") or node.get("text"),
            })
        for child in node.get("children") or []:
            _block_geometry(child, out, page_size)
    elif isinstance(node, list):
        for child in node:
            _block_geometry(child, out, page_size)


def _marker_worker_convert(task: tuple[str, str | None, str | None]) -> dict:
    """Convert one file in an already-loaded worker process.

    Marker 1.8.2 exposes a converter per task; treating it as one Markdown
    pipeline used the wrong interface for two legs and hid a third:

      OCRConverter   -> text, plus OCR line polygons
      PdfConverter   -> markdown AND layout geometry from ONE build_document
      TableConverter -> tables (its own documented table pipeline)

    Structured extraction stays out: ExtractionConverter needs an external LLM,
    which would make this a hybrid engine rather than standalone Marker.
    """
    file_path, document_path, pass_name = task
    base = {
        "document_path": document_path,
        "gpu_type": GPU_TYPE,
        "model_version": (
            f"marker-pdf==1.8.2 ({MARKER_WORKERS} persistent batch workers)"
        ),
    }
    try:
        if pass_name == OCR_PASS:
            from marker.converters.ocr import OCRConverter

            rendered = OCRConverter(artifact_dict=_marker_model_dict,
                                    config=_marker_config)(file_path)
            data = rendered.model_dump() if hasattr(rendered, "model_dump") else {}
            lines = []
            _block_geometry(data, lines, _marker_page_size(file_path))
            text = "\n".join(
                str(line["text"]) for line in lines if line.get("text"))
            # OCRConverter's own line boxes are the right geometry source for
            # this engine — layout blocks from PdfConverter are whole regions
            # (one box for an entire table), which cannot match a word or line
            # reference no matter how correct they are.
            regions = [
                {"text": str(l["text"]), "bbox": l["bbox"]}
                for l in lines if l.get("bbox") and l.get("text")
            ]
            structured = {"ocr_lines": lines}
            if regions:
                structured["text_regions"] = regions
                structured["geometry_granularity"] = "ocr_line"
            # DIAGNOSTIC: OCRConverter's renderer is OCRJSONRenderer, whose
            # schema differs from PdfConverter's block tree — the generic walk
            # found only a Page node with text=None. Dump the real shape once.
            try:
                def _shape(node, depth=0):
                    if depth > 3:
                        return type(node).__name__
                    if isinstance(node, dict):
                        return {k: _shape(v, depth + 1)
                                for k, v in list(node.items())[:30]}
                    if isinstance(node, list):
                        return [f"list[{len(node)}]",
                                _shape(node[0], depth + 1) if node else None]
                    return type(node).__name__
                structured["_debug_ocr_shape"] = _shape(data)
            except Exception as exc:
                structured["_debug_ocr_shape"] = f"{type(exc).__name__}: {exc}"
            return {**base, "raw_text": text, "structured_output": structured}

        if pass_name == TABLE_PASS:
            from marker.converters.table import TableConverter
            from marker.output import text_from_rendered

            rendered = TableConverter(artifact_dict=_marker_model_dict,
                                      config=_marker_config)(file_path)
            text, _, _ = text_from_rendered(rendered)
            return {**base, "raw_text": text, "structured_output": {}}

        # Default: the document pipeline. build_document() runs construction and
        # every processor once; markdown and JSON are two renderings of that one
        # result, so geometry costs no extra inference.
        from marker.converters.pdf import PdfConverter
        from marker.output import text_from_rendered
        from marker.renderers.json import JSONRenderer

        converter = PdfConverter(artifact_dict=_marker_model_dict,
                                 config=_marker_config)
        document = converter.build_document(file_path)
        rendered = converter.renderer(converter.config)(document)
        text, _, images = text_from_rendered(rendered)

        structured: dict = {}
        try:
            blocks: list = []
            _block_geometry(
                JSONRenderer(converter.config)(document).model_dump(), blocks,
                _marker_page_size(file_path))
            if blocks:
                structured["layout_blocks"] = blocks
                regions = [
                    {"text": str(b["text"]), "bbox": b["bbox"]}
                    for b in blocks
                    if b.get("bbox") and b.get("text")
                    and str(b.get("label")) not in ("Page",)
                ]
                if regions:
                    structured["text_regions"] = regions
                    structured["geometry_granularity"] = "layout_block"
        except Exception as exc:
            structured["geometry_error"] = f"{type(exc).__name__}: {exc}"
        # text_from_rendered returns (markdown, ext, images); the images were
        # being dropped, so every figure Marker extracted was lost and the
        # markdown left pointing at a file that no longer existed.
        if images:
            structured["images"] = sorted(images) if isinstance(images, dict) else True
        return {**base, "raw_text": text, "structured_output": structured}
    except Exception as exc:
        message = str(exc) or repr(exc)
        return {
            "document_path": document_path,
            "raw_text": "",
            "structured_output": {},
            "gpu_type": GPU_TYPE,
            "model_version": "marker-pdf==1.8.2",
            "error": f"{exc.__class__.__name__}: {message}",
        }

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "poppler-utils")
    .pip_install(
        "marker-pdf==1.8.2",
        "Pillow>=10.0",
    )
    .env({
        "MKL_DYNAMIC": "FALSE",
        "OMP_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "IN_STREAMLIT": "true",
    })
)

app = modal.App("ocr-bench-marker")


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=1800,
    startup_timeout=1800,
    scaledown_window=30,
    # Upstream Marker's folder command uses several spawned conversion workers on
    # one GPU.  We keep those workers alive so model load remains startup, outside
    # every measured batch.
    max_containers=1,
)
class MarkerInference:
    @modal.enter()
    def load_model(self) -> None:
        import queue
        import threading
        import time
        import torch.multiprocessing as mp
        from marker.utils.gpu import GPUManager

        t0 = time.perf_counter()
        # This is the same NVIDIA MPS setup used by Marker's folder conversion
        # command so its CUDA worker processes can share one GPU efficiently.
        self._gpu_manager = GPUManager(0)
        self._gpu_manager.__enter__()
        context = mp.get_context("spawn")
        self._ready_queue = context.Queue()
        self._worker_pool = context.Pool(
            processes=MARKER_WORKERS,
            initializer=_marker_worker_init,
            initargs=(self._ready_queue, MARKER_BATCH_CONFIG),
        )
        ready_pids = set()
        deadline = time.monotonic() + 1200
        while len(ready_pids) < MARKER_WORKERS:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Only {len(ready_pids)}/{MARKER_WORKERS} Marker workers became ready"
                )
            try:
                ready_pids.add(self._ready_queue.get(timeout=min(remaining, 30)))
            except queue.Empty:
                continue

        self._startup_gpu_seconds = time.perf_counter() - t0
        self._startup_cost_applied = False
        self._startup_cost_lock = threading.Lock()

    @modal.exit()
    def stop_workers(self) -> None:
        pool = getattr(self, "_worker_pool", None)
        if pool is not None:
            pool.close()
            pool.join()
        gpu_manager = getattr(self, "_gpu_manager", None)
        if gpu_manager is not None:
            gpu_manager.__exit__(None, None, None)

    @modal.method()
    def run_ocr(self, file_bytes: bytes, mime_type: str = "application/pdf",
                prompt: str | None = None) -> dict:
        batch = self._run_ocr_batch_impl([{
            "document_path": None,
            "bytes": file_bytes,
            "mime_type": mime_type,
        }], prompt)
        response = batch["results"][0]
        response["gpu_seconds"] = batch["batch_gpu_seconds"]
        response["startup_gpu_seconds"] = batch["startup_gpu_seconds"]
        response["is_cold_start"] = batch["is_cold_start"]
        return response

    @modal.method()
    def run_ocr_batch(self, files: list[dict], prompt: str | None = None) -> dict:
        return self._run_ocr_batch_impl(files, prompt)

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
            "model_version": "marker-pdf==1.8.2",
            "error": f"{exc.__class__.__name__}: {message}",
        }

    def _run_ocr_batch_impl(self, files: list[dict], prompt: str | None = None) -> dict:
        import time
        import tempfile

        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks = []
            for index, item in enumerate(files):
                suffix = ".pdf" if "pdf" in item.get("mime_type", "") else ".png"
                path = f"{temp_dir}/{index:04d}{suffix}"
                with open(path, "wb") as handle:
                    handle.write(item["bytes"])
                tasks.append((path, item.get("document_path"),
                              item.get("prompt", prompt)))
            results = self._worker_pool.map(_marker_worker_convert, tasks)

        batch_gpu_seconds = time.perf_counter() - t0
        startup_seconds, is_cold_start = self._take_startup_metadata()
        return {
            "results": results,
            "batch_gpu_seconds": batch_gpu_seconds,
            "gpu_type": GPU_TYPE,
            "startup_gpu_seconds": startup_seconds,
            "is_cold_start": is_cold_start,
        }
