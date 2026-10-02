"""
Modal app: MinerU 2.5 **pipeline backend** (`mineru -b pipeline -m ocr`) — the
small-model chain, run as its own engine beside the MinerU 2.5 VLM engine.

Why a separate engine: modal_apps/mineru_app.py runs MinerU's vlm backend
(opendatalab/MinerU2.5-2509-1.2B, two_step_extract: layout detection, then the
model reads each block), whose finest geometry is a block. The pipeline backend
is MinerU's other documented mode: layout detection (DocLayout-YOLO), then a
text-line OCR detector + recogniser, plus formula and table models. Its
middle.json carries blocks -> lines -> spans, each with a bbox — the line boxes
the geometry leg can score. Different models, different code path: a different
engine, never a swap.

Model contract (sources read 2026-10-01, tag mineru-2.5.0-released):
  * Package: mineru[pipeline]==2.5.4 (last 2.5.x on PyPI). Requires torch>=2.6
    (<3); README tested range torch 2.2–2.6 excl. 2.5, CUDA 11.8/12.4/12.6/12.8,
    "GPU: Turing architecture or later, 6GB+ VRAM", Python 3.10–3.13.
    Pinned here: torch==2.6.0 + torchvision==0.21.0 (cu124 wheels).
  * Models: downloaded by the package's own `mineru-models-download -s
    huggingface -m pipeline` (mineru/cli/models_download.py) at image build —
    the layout, OCR, formula and table weights it selects — and the HF snapshot
    commits actually fetched are recorded at build time (see
    MODEL_COMMITS_FILE) and reported in model_version. The downloader fetches
    HEAD of its repos, so the commit file is the pin.
  * API (demo/demo.py at the tag): images are turned into a one-page PDF with
    images_bytes_to_pdf_bytes (PIL "save as PDF": page size = image pixels),
    then pipeline_doc_analyze(pdf_bytes_list, lang_list, parse_method="ocr",
    formula_enable, table_enable) -> model lists; result_to_middle_json(...) ->
    middle.json with pdf_info[page]: page_size [w, h], para_blocks (level-1
    blocks; table/image blocks nest `blocks`), lines[].bbox, spans[].bbox /
    content; union_make(pdf_info, MakeMode.MM_MD, image_dir) -> markdown.
    Coordinates: bbox [x0, y0, x1, y1], origin top-left, in page units — here
    equal to image pixels — normalised by page_size.
  * Language: "en" (corpus is English; README recommends -l for non-Chinese
    documents, default "ch"). parse_method "ocr" because inputs are images.
    formula and table models left enabled (the engine's default behaviour).
  * Batching owner: pipeline_doc_analyze takes a LIST of documents and batches
    internally per stage; run_ocr_batch passes the whole chunk in one call.
  * GPU: T4 first candidate (Turing is the documented minimum; models are
    small). CPU fallback exists in MinerU but is not used here.
  * No prompt: a prompt/response_format yields an error item.
"""

import modal

GPU_TYPE = "T4"
MINERU_VERSION = "2.5.4"
TORCH_VERSION = "2.6.0"
TORCHVISION_VERSION = "0.21.0"
MODEL_ID = "opendatalab/MinerU pipeline backend (DocLayout-YOLO + PP-OCR line OCR + UniMERNet + table models)"
MODEL_REVISION = f"mineru[pipeline]=={MINERU_VERSION}"
LANG = "en"
PARSE_METHOD = "ocr"
MODEL_COMMITS_FILE = "/root/mineru_model_commits.json"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libgomp1")
    .pip_install(
        f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(f"mineru[pipeline]=={MINERU_VERSION}", "huggingface_hub>=0.23", "Pillow>=10.0")
    .env({"MINERU_MODEL_SOURCE": "huggingface", "OMP_NUM_THREADS": "1"})
    # The package's own downloader selects and fetches the pipeline weights and
    # writes ~/mineru.json pointing at them. Done at build so the image is
    # self-contained; the commits fetched are recorded for the model_version.
    .run_commands(
        "mineru-models-download -s huggingface -m pipeline",
        "python - <<'EOF'\n"
        "import json, os, glob\n"
        "root = os.path.expanduser('~/.cache/huggingface/hub')\n"
        "out = {}\n"
        "for d in sorted(glob.glob(os.path.join(root, 'models--*'))):\n"
        "    refs = os.path.join(d, 'refs', 'main')\n"
        "    out[os.path.basename(d)] = open(refs).read().strip() if os.path.exists(refs) else None\n"
        "cfg = os.path.expanduser('~/mineru.json')\n"
        "out['_mineru_json'] = json.load(open(cfg)) if os.path.exists(cfg) else None\n"
        f"json.dump(out, open('{MODEL_COMMITS_FILE}', 'w'), indent=1)\n"
        "print(json.dumps(out, indent=1))\n"
        "EOF",
    )
)

app = modal.App("ocr-bench-mineru-pipeline")


def _norm_box(box, page_w: float, page_h: float):
    try:
        x0, y0, x1, y1 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if not page_w or not page_h:
        return None
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0 / page_w, y0 / page_h, x1 / page_w, y1 / page_h]


def _walk_blocks(blocks, page_w, page_h, layout, lines_out, spans_out, tables):
    """para_blocks -> layout blocks, lines (with text) and spans, normalised."""
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        btype = str(block.get("type") or "")
        bbox = _norm_box(block.get("bbox"), page_w, page_h)
        if block.get("blocks"):                      # level-1 table/image block
            _walk_blocks(block["blocks"], page_w, page_h, layout, lines_out, spans_out, tables)
            if bbox:
                layout.append({"label": btype, "bbox": bbox, "nested": True})
            continue
        block_text = []
        for line in block.get("lines") or []:
            lbox = _norm_box(line.get("bbox"), page_w, page_h)
            parts = []
            for span in line.get("spans") or []:
                sbox = _norm_box(span.get("bbox"), page_w, page_h)
                stype = str(span.get("type") or "")
                content = span.get("content")
                if stype == "table" and span.get("html"):
                    tables.append({"html": span["html"]})
                    content = content or span.get("html")
                if content is not None and str(content).strip():
                    parts.append(str(content))
                    if sbox:
                        spans_out.append({"text": str(content), "bbox": sbox, "type": stype})
            text = " ".join(parts).strip()
            if text:
                block_text.append(text)
                if lbox:
                    lines_out.append({"text": text, "bbox": lbox, "block_type": btype})
        if bbox:
            layout.append({"label": btype, "bbox": bbox, "content": "\n".join(block_text)})


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=1800,
    startup_timeout=900,
    scaledown_window=60,
    max_containers=1,
)
class MinerUPipelineInference:
    @modal.enter()
    def load_model(self) -> None:
        import importlib.metadata as md
        import json
        import os
        import time

        t0 = time.perf_counter()
        from mineru.backend.pipeline.pipeline_analyze import doc_analyze  # noqa: F401
        from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json  # noqa: F401
        from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make  # noqa: F401

        self._versions = {}
        for pkg in ("mineru", "torch", "torchvision", "transformers", "ultralytics", "doclayout_yolo", "onnxruntime", "pypdfium2"):
            try:
                self._versions[pkg] = md.version(pkg)
            except md.PackageNotFoundError:
                pass
        if self._versions.get("mineru") != MINERU_VERSION:
            raise RuntimeError(f"mineru {self._versions.get('mineru')} != pinned {MINERU_VERSION}")
        self._model_commits = json.load(open(MODEL_COMMITS_FILE)) if os.path.exists(MODEL_COMMITS_FILE) else {}

        # Representative warm-up through the real path (layout + OCR on GPU).
        import io
        from PIL import Image, ImageDraw
        warm = Image.new("RGB", (1200, 400), "white")
        draw = ImageDraw.Draw(warm)
        for i, line in enumerate(("INVOICE 00123", "Total due: 1,234.56", "Date 2026-10-01")):
            draw.text((40, 40 + i * 110), line, fill="black", font_size=64)
        buf = io.BytesIO()
        warm.save(buf, format="PNG")
        self._parse_batch([buf.getvalue()])
        self._startup_gpu_seconds = time.perf_counter() - t0
        self._startup_cost_applied = False

    def _model_version(self) -> str:
        commits = {k[len("models--"):]: (v or "")[:12] for k, v in self._model_commits.items() if k.startswith("models--")}
        return (f"{MODEL_ID} {MODEL_REVISION} torch=={self._versions.get('torch')} "
                f"weights={commits} lang={LANG} method={PARSE_METHOD} {GPU_TYPE}")

    # ------------------------------------------------------------------ API
    @modal.method()
    def run_ocr(self, image_bytes: bytes, mime_type: str = "image/png",
                prompt: str | None = None, response_format: dict | None = None) -> dict:
        import time

        t0 = time.perf_counter()
        response = self._run_items([{"bytes": image_bytes, "mime_type": mime_type,
                                     "prompt": prompt, "response_format": response_format}])[0]
        response["gpu_seconds"] = time.perf_counter() - t0
        response["gpu_type"] = GPU_TYPE
        startup, cold = self._take_startup_metadata()
        response["startup_gpu_seconds"] = startup
        response["is_cold_start"] = cold
        return response

    @modal.method()
    def run_ocr_batch(self, files: list[dict], prompt: str | None = None,
                      response_format: dict | None = None) -> dict:
        import time

        started = time.perf_counter()
        items = []
        for item in files:
            merged = dict(item)
            merged.setdefault("prompt", prompt)
            merged.setdefault("response_format", response_format)
            items.append(merged)
        results = self._run_items(items)
        for item, response in zip(files, results):
            response["document_path"] = item.get("document_path")
        batch_seconds = time.perf_counter() - started
        startup, cold = self._take_startup_metadata()
        return {
            "results": results,
            "batch_gpu_seconds": batch_seconds,
            "gpu_type": GPU_TYPE,
            "startup_gpu_seconds": startup,
            "is_cold_start": cold,
        }

    # ------------------------------------------------------------- internals
    def _take_startup_metadata(self) -> tuple[float, bool]:
        if not getattr(self, "_startup_cost_applied", True):
            self._startup_cost_applied = True
            return getattr(self, "_startup_gpu_seconds", 0.0), True
        return 0.0, False

    def _error(self, exc: Exception) -> dict:
        return {
            "raw_text": "",
            "structured_output": {},
            "model_version": self._model_version(),
            "prompt_used": None,
            "error": f"{exc.__class__.__name__}: {str(exc) or repr(exc)}",
        }

    def _run_items(self, items: list[dict]) -> list[dict]:
        """One pipeline_doc_analyze call for the whole chunk (its native list
        input); per-item errors never take the batch down."""
        out: list[dict | None] = [None] * len(items)
        valid: list[int] = []
        for i, item in enumerate(items):
            try:
                if item.get("prompt") is not None or item.get("response_format") is not None:
                    raise ValueError("MinerU pipeline is a fixed pipeline and takes no prompt; "
                                     "engines.py must declare accepts_task=False")
                if (item.get("mime_type") or "").endswith("pdf"):
                    raise ValueError("PDF input not supported by this app; send a page image")
                valid.append(i)
            except Exception as exc:
                out[i] = self._error(exc)
        if valid:
            try:
                parsed = self._parse_batch([items[i]["bytes"] for i in valid])
                for i, result in zip(valid, parsed):
                    out[i] = result
            except Exception as exc:
                for i in valid:
                    if out[i] is None:
                        out[i] = self._error(exc)
        return [r if r is not None else self._error(RuntimeError("no result")) for r in out]

    def _parse_batch(self, images: list[bytes]) -> list[dict]:
        import copy
        import io
        import json
        import tempfile

        from PIL import Image, ImageOps
        from mineru.backend.pipeline.pipeline_analyze import doc_analyze
        from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json
        from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make
        from mineru.data.data_reader_writer import FileBasedDataWriter
        from mineru.utils.enum_class import MakeMode
        from mineru.utils.pdf_image_tools import images_bytes_to_pdf_bytes

        pdf_bytes_list, sizes = [], []
        for raw in images:
            with Image.open(io.BytesIO(raw)) as src:
                img = ImageOps.exif_transpose(src).convert("RGB")
            sizes.append(img.size)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            pdf_bytes_list.append(images_bytes_to_pdf_bytes(buf.getvalue()))

        infer_results, image_lists, pdf_docs, lang_list, ocr_enabled = doc_analyze(
            pdf_bytes_list, [LANG] * len(pdf_bytes_list), parse_method=PARSE_METHOD,
            formula_enable=True, table_enable=True)

        results = []
        with tempfile.TemporaryDirectory() as tmp:
            for idx, model_list in enumerate(infer_results):
                model_json = copy.deepcopy(model_list)
                image_dir = f"{tmp}/images_{idx}"
                import os
                os.makedirs(image_dir, exist_ok=True)
                writer = FileBasedDataWriter(image_dir)
                middle = result_to_middle_json(model_list, image_lists[idx], pdf_docs[idx], writer,
                                               lang_list[idx], ocr_enabled[idx], True)
                pdf_info = middle["pdf_info"]
                try:
                    markdown = union_make(pdf_info, MakeMode.MM_MD, "images")
                    if not isinstance(markdown, str):
                        markdown = json.dumps(markdown, ensure_ascii=False)
                except Exception as exc:
                    markdown = f"(markdown rendering failed: {type(exc).__name__}: {exc})"
                page = pdf_info[0] if pdf_info else {}
                page_w, page_h = (page.get("page_size") or [0, 0])[:2]
                layout, lines, spans, tables = [], [], [], []
                _walk_blocks(page.get("para_blocks"), page_w, page_h, layout, lines, spans, tables)
                discarded = []
                _walk_blocks(page.get("discarded_blocks"), page_w, page_h, discarded, [], [], [])
                structured = {
                    "markdown": markdown,
                    "layout_blocks": layout,
                    "discarded_blocks": discarded,
                    "text_regions": [{"text": l["text"], "bbox": l["bbox"]} for l in lines],
                    "geometry_granularity": "line",
                    "lines": lines,
                    "spans": spans,
                    "page_size": [page_w, page_h],
                    "image_size": list(sizes[idx]),
                    "lang": lang_list[idx],
                    "ocr_enabled": bool(ocr_enabled[idx]),
                    "packages": self._versions,
                    "model_commits": self._model_commits,
                    # the model-side output and the middle.json page, verbatim, for audit
                    "model_output": model_json,
                    "middle_json_page": page,
                }
                if tables:
                    structured["tables"] = tables
                results.append({
                    "raw_text": "\n".join(l["text"] for l in lines),
                    "structured_output": structured,
                    "model_version": self._model_version(),
                    "prompt_used": None,
                    "error": None,
                })
        return results
