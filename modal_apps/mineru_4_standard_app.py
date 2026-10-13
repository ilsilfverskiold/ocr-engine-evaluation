"""
Modal app: MinerU 4.0 document pipeline, explicit Standard tier.

This is a NEW engine (`mineru_4_standard`). It does not replace or modify
modal_apps/mineru_app.py, which benchmarks the MinerU2.5-2509 VLM through the
mineru-vl-utils two-step helper. MinerU 4.0 is an application/pipeline release:
tiers replace the 3.x pipeline/vlm/hybrid backends, and a tier is not a model.

What "Standard" runs (read from the mineru 4.0.10 source, parser/tier.py and
backend/analysis/pdf/window.py):
  tier "standard" -> hybrid effort "high":
    1. the image is converted to a one-page PDF and rendered back to pixels
       (docvortex PDFDocument.from_image at 200 dpi, render at 200 dpi with a
       3500 px long-edge cap),
    2. Torch small models: PP-DocLayoutV2 layout (+ table orientation check),
    3. the VLM (MinerU2.5-Pro-2605-1.2B on vLLM AsyncLLM) recognises the regions
       found by the small-model layout (`aio_batch_extract_with_layout`);
       "Text Recognition:" / "Table Recognition:" / "Formula Recognition:"
       prompts are chosen per region inside mineru-vl-utils,
    4. PP-OCRv6 detection for line filling, seal OCR, inline formula recognition,
    5. one strict MiddleJson (docvortex schema 2.0) rendered to Markdown.
  Image/chart analysis is forced off for Standard (only Advanced/xhigh sends it).

Batching owner: MinerU's own native-async runtime. `MinerUParser.parse_batch_async`
is `asyncio.gather(parse_async(p) for p in paths)`; all documents share one
AsyncLLM whose in-flight VLM requests are capped by model.vlm.max_concurrency
(default 100, shared across documents), while local small-model stages are
serialised per device by MinerU. This app issues exactly that gather, only with
return_exceptions=True so one bad page does not discard the rest of the batch.
No extra threads or Modal fan-out are added.

No document library is involved: `mineru.parser` is stateless ("it does not
create a document library"), so there is no cached-result path that could make
a timed batch skip inference. LLM-aided post-processing (title leveling,
cross-page table merge) is off by default and asserted off at startup.

Sources (read 2026-09-30):
  https://github.com/opendatalab/MinerU/releases/tag/mineru-4.0.0-released
  https://github.com/opendatalab/MinerU/tree/mineru-4.0.10-released  (code used)
    docs/en/usage/tiers.md, docs/en/usage/model_source.md, docs/en/usage/sdk_api.md,
    docs/native-async-analysis.md, docker/global/Dockerfile,
    mineru/parser/{__init__,base,mineru_parser,tier}.py, mineru/model/registry.py,
    mineru/model/download.py, mineru/model/vlm/{runtime,client,engine_utils}.py,
    mineru/backend/analysis/pdf/{pipeline,window}.py
  docvortex 0.5.7 wheel: schema.py, document/pdf/{_document,raster}.py,
    render/_internal/markdown/*.py, visualization.py
  mineru-vl-utils 2.0.5 wheel: mineru_client.py, vlm_client/vllm_async_engine_client.py
"""

import modal

GPU_TYPE = "A10"  # fit probe first; next candidate L40S (see handoff)

# VLM used by the vLLM engine for Standard (mineru/model/registry.py,
# MINERU_2_5_PRO_2605_1_2B; _VLM_MODEL_REPOS["vllm"]). Apache-2.0 on its card.
MODEL_ID = "opendatalab/MinerU2.5-Pro-2605-1.2B"
MODEL_REVISION = "bff20d4ae2bf202df9f45284b4d43681555a97ed"
# Torch small-model bundle for small_backend=torch (registry.py
# MINERU_4_MODELS_TORCH): PP-DocLayoutV2, PP-OCRv6 tiny det + small rec,
# seal det, PP-FormulaNet plus-M, SLANet+/UNet table models. Apache-2.0 tag.
SMALL_MODEL_ID = "opendatalab/MinerU-4_models_torch"
SMALL_MODEL_REVISION = "2b3afb86f4d23fa623f6b2f4b2279ed4ef93a89d"

# Package pins. mineru 4.0.10 is the latest 4.0.x patch (2026-09-29); the
# public SDK surface used here (parse_async, MinerUParser, ParseResult,
# RenderMode) is unchanged from the 4.0.0 release tag. vllm 0.21.0 is the
# version upstream's own GPU Dockerfile builds on (vllm/vllm-openai:v0.21.0);
# it pins torch==2.11.0 / torchvision==0.26.0. transformers 5.16.1 is the
# newest of the three versions upstream's resolver check exercises
# (scripts/check_transformers5_dependencies.py: 5.10.1, 5.14.0, 5.16.1) and
# satisfies both mineru[torch] (>=5.10.1,<6) and vllm 0.21.0.
MINERU_VERSION = "4.0.10"
MINERU_VL_UTILS_VERSION = "2.0.5"
DOCVORTEX_VERSION = "0.5.7"
VLLM_VERSION = "0.21.0"
TORCH_VERSION = "2.11.0"
TORCHVISION_VERSION = "0.26.0"
TRANSFORMERS_VERSION = "5.16.1"

TIER = "standard"
SMALL_BACKEND = "torch"
VLM_ENGINE = "vllm"
OCR_MODE = "auto"  # upstream default for parse(); an image-only PDF has no text layer

MINERU_HOME = "/opt/mineru_home"  # config.yaml + model base_dir live here



def _download_pinned_models() -> None:
    """Build-time model download through MinerU's own downloader, pinned.

    MinerU's downloader calls huggingface_hub.snapshot_download(repo_id) with no
    revision, so a plain `mineru-kit models download` fetches whatever HEAD is at
    build time. This wraps the two HF calls it makes (manifest listing and
    snapshot) to inject the pinned revision, then lets MinerU run its own
    completeness checks and write its own readiness markers
    (mineru/model/download.py:download_model_repo). The model set is exactly
    what `mineru-kit models download --tier standard --small-backend torch
    --vlm-engine vllm` selects (model_source.md, docker/global/Dockerfile).
    """
    import json
    import os

    os.environ["MINERU_MODEL_SOURCE"] = "huggingface"
    from huggingface_hub import HfApi

    import mineru.model.download as dl
    from mineru.model.registry import model_repos_for_tier

    pins = {MODEL_ID: MODEL_REVISION, SMALL_MODEL_ID: SMALL_MODEL_REVISION}
    original_snapshot = dl.hf_snapshot_download

    def pinned_snapshot(repo_id, **kwargs):
        if repo_id not in pins:
            raise RuntimeError(f"unpinned MinerU model repo requested: {repo_id}")
        return original_snapshot(repo_id, revision=pins[repo_id], **kwargs)

    class PinnedHfApi(HfApi):
        def list_repo_files(self, repo_id, *args, **kwargs):
            kwargs.setdefault("revision", pins[repo_id])
            return super().list_repo_files(repo_id, *args, **kwargs)

    dl.hf_snapshot_download = pinned_snapshot
    dl.HfApi = PinnedHfApi

    repos = model_repos_for_tier(TIER, small_backend=SMALL_BACKEND, vlm_engine=VLM_ENGINE)
    resolved = {}
    for repo in repos:
        root = dl.download_model_repo(repo, source="huggingface", local_as_auto=True)
        ready = dl.verify_model_repo(repo)
        if not ready.ready:
            raise RuntimeError(f"{repo.name} not ready: {ready.missing_paths}")
        repo_id = repo.repos["huggingface"]
        resolved[repo.name] = {"repo_id": repo_id, "revision": pins[repo_id], "root": str(root)}
    if set(v["repo_id"] for v in resolved.values()) != set(pins):
        raise RuntimeError(f"tier resolved unexpected repos: {resolved}")
    with open(os.path.join(MINERU_HOME, "pinned_models.json"), "w") as fh:
        json.dump(resolved, fh, indent=2)
    print(json.dumps(resolved, indent=2))


image = (
    modal.Image.debian_slim(python_version="3.12")
    # libgl1: upstream Dockerfile installs it for opencv.
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        f"mineru[torch]=={MINERU_VERSION}",
        f"mineru-vl-utils=={MINERU_VL_UTILS_VERSION}",
        f"docvortex=={DOCVORTEX_VERSION}",
        f"vllm=={VLLM_VERSION}",
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        f"transformers=={TRANSFORMERS_VERSION}",
        "Pillow>=11.0.0",
    )
    .env({
        "MINERU_HOME": MINERU_HOME,
        # Same backend selection at download and inference time
        # (model_source.md; docker/global/Dockerfile).
        "MINERU_MODEL_SMALL_BACKEND": SMALL_BACKEND,
        "MINERU_MODEL_VLM_ENGINE": VLM_ENGINE,
        "TOKENIZERS_PARALLELISM": "false",
    })
    .run_commands(
        f"mkdir -p {MINERU_HOME}",
        "python -c 'import mineru, vllm, torch, transformers; "
        "print(\"mineru\", mineru.__file__, \"vllm\", vllm.__version__, "
        "\"torch\", torch.__version__, \"transformers\", transformers.__version__)'",
    )
    # Models are baked into the image at pinned revisions. No GPU is needed:
    # "A NVIDIA / vLLM deployment can download its models on a build machine
    # without a GPU" when both backends are named explicitly (model_source.md).
    .run_function(
        _download_pinned_models,
        secrets=[modal.Secret.from_name("huggingface-token")],
    )
    # Runtime never downloads: `local` "uses ready local models and fails on
    # missing assets without downloading" (model_source.md).
    .env({"MINERU_MODEL_SOURCE": "local"})
    .run_commands(
        "mineru-kit models verify --tier standard --small-backend torch --vlm-engine vllm",
    )
)

app = modal.App("ocr-bench-mineru-4-standard")


# --------------------------------------------------------------------------- #
# Output adapter (pure Python over MinerU's typed MiddleJson)
# --------------------------------------------------------------------------- #

# Top-level block types whose content is printed text (docvortex schema.py
# BlockType). Equations are LaTeX, not the printed string, and image/chart
# bodies are pictures; neither is used as text geometry.
_TEXT_BLOCK_TYPES = {
    "text", "ref_text", "doc_title", "paragraph_title", "header", "footer",
    "page_number", "aside_text", "page_footnote", "list", "index",
}
_ANNOTATION_TYPES = {
    "image_caption", "image_footnote", "table_caption", "table_footnote",
    "chart_caption", "chart_footnote", "code_caption", "code_footnote",
}


def _html_table_to_text(html_text: str) -> str:
    import html
    import re

    text = re.sub(r"</t[dh]>\s*<t[dh][^>]*>", "\t", html_text)
    text = re.sub(r"</tr>\s*<tr[^>]*>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def _is_block(obj) -> bool:
    # Blocks (docvortex BlockBase) declare a bbox field; inline spans do not.
    return hasattr(obj, "bbox") and hasattr(obj, "type")


def _content_text(content) -> str:
    """Plain text of a block's content: str, inline spans, or child blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if content and _is_block(content[0]):
            parts = [_content_text(getattr(child, "content", None)) for child in content]
            return "\n".join(p for p in parts if p)
        parts = []
        for span in content:
            inner = getattr(span, "content", "")
            parts.append(inner if isinstance(inner, str) else _content_text(inner))
        return "".join(parts)
    return str(content)


def _bbox(block):
    box = getattr(block, "bbox", None)
    if box is None:
        return None
    return [float(v) for v in box]


def _middle_to_structured(result, markdown_default: str,
                          image_size: tuple[int, int]) -> dict:
    """Map MinerU's MiddleJson to the benchmark's structured_output shapes.

    Coordinates: MiddleJson bbox is validated as normalised [x0,y0,x1,y1] in
    0..1 of the displayed page (schema.py BlockBase._validate_bbox), y from the
    top (visualization.py _normalized_bbox_to_pdf inverts it for PDF space).
    The page here is the input image itself (from_image keeps the aspect ratio
    and adds no margin), so the boxes are already relative to the image as sent.
    """
    layout_blocks: list[dict] = []
    tables: list[dict] = []
    regions: list[dict] = []

    for page in result.pages:
        for block in page.blocks:
            btype = str(block.type)
            bbox = _bbox(block)
            content = getattr(block, "content", None)
            entry = {"label": btype, "bbox": bbox, "order": getattr(block, "index", None),
                     "page_idx": page.page_idx}

            if btype in ("table", "image", "chart", "code"):
                children = []
                for child in content or []:
                    ctype = str(child.type)
                    cbox = _bbox(child)
                    ccontent = getattr(child, "content", None)
                    if ctype == "table_body":
                        html = ccontent if isinstance(ccontent, str) else ""
                        children.append({"label": ctype, "bbox": cbox, "content": html})
                        if "<table" in html.lower():
                            tables.append({"html": html, "bbox": bbox})
                            text = _html_table_to_text(html)
                            if text and bbox:
                                regions.append({"text": text, "bbox": bbox, "label": "table"})
                    elif ctype in ("image_body", "chart_body"):
                        children.append({"label": ctype, "bbox": cbox,
                                         "content": ccontent if isinstance(ccontent, str) else None})
                    else:  # captions/footnotes, code/algorithm body
                        text = _content_text(ccontent).strip()
                        children.append({"label": ctype, "bbox": cbox, "content": text})
                        if text and cbox and (ctype in _ANNOTATION_TYPES
                                              or ctype in ("code_body", "algorithm_body")):
                            regions.append({"text": text, "bbox": cbox, "label": ctype})
                entry["children"] = children
            elif btype == "equation":
                entry["content"] = content if isinstance(content, str) else _content_text(content)
            else:
                text = _content_text(content).strip()
                entry["content"] = text
                if btype in _TEXT_BLOCK_TYPES and text and bbox:
                    regions.append({"text": text, "bbox": bbox, "label": btype})
            layout_blocks.append(entry)

    structured: dict = {
        "markdown_default_view": markdown_default,
        "layout_blocks": layout_blocks,
        "raw_model_output": result.to_dict(),
    }
    if tables:
        structured["tables"] = tables
    if regions:
        structured["text_regions"] = regions
        structured["geometry_granularity"] = "block"
        structured["geometry_coordinates"] = "normalized_xyxy"
    w, h = image_size
    # from_image writes the page at 200 dpi (w*72/200 pt); rendering at 200 dpi
    # restores w x h unless the long edge exceeds 3500 px, where it is capped.
    render_scale = min(1.0, 3500 / max(w, h)) if w and h else None
    structured["preprocessing"] = {
        "input": "EXIF-transposed RGB PNG written by this app",
        "pipeline": ("docvortex PDFDocument.from_image at 200 dpi, rendered at 200 dpi "
                     "with a 3500 px long-edge cap (official MinerU path for images)"),
        "image_size": [w, h],
        "pipeline_render_scale_vs_input": render_scale,
        "bbox_space": "normalised 0-1 of the page == the image as sent",
    }
    return structured


def _placeholder_image(block) -> str:
    # Keeps figures visible in Markdown without embedding base64 data URIs.
    return f"![{block.type}](image)"


def _write_input_png(image_bytes: bytes, path) -> tuple[int, int]:
    import io

    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.save(path, format="PNG")
    return img.size


def _warmup_page(path) -> None:
    """Synthetic page with a title, paragraphs and a ruled table, so the
    layout model, the VLM (text + table prompts) and OCR detection all run."""
    from PIL import Image, ImageDraw, ImageFont

    page = Image.new("RGB", (1654, 2339), "white")
    draw = ImageDraw.Draw(page)
    try:
        title_font = ImageFont.load_default(size=56)
        body_font = ImageFont.load_default(size=30)
    except TypeError:  # Pillow without sized default font
        title_font = body_font = ImageFont.load_default()
    draw.text((150, 150), "Quarterly Operating Report", fill="black", font=title_font)
    y = 300
    for i in range(6):
        draw.text((150, y), f"Line {i + 1}: revenue, costs and margin were reviewed for "
                  f"region {i + 1} in the period.", fill="black", font=body_font)
        y += 50
    x0, y0, cw, ch = 150, 700, 450, 70
    rows = [["Item", "Q1", "Q2"], ["Revenue", "1,200", "1,350"],
            ["Costs", "800", "870"], ["Margin", "400", "480"]]
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            box = (x0 + c * cw, y0 + r * ch, x0 + (c + 1) * cw, y0 + (r + 1) * ch)
            draw.rectangle(box, outline="black", width=2)
            draw.text((box[0] + 15, box[1] + 18), cell, fill="black", font=body_font)
    page.save(path, format="PNG")


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    # Upstream planning guidance: at least 16 GB system memory for local
    # Standard deployments (tiers.md). 32 GiB leaves room for a 24-page batch
    # rendered at up to 3500 px plus the vLLM engine-core process.
    cpu=8.0,
    memory=32768,
    timeout=1800,
    startup_timeout=1800,
    scaledown_window=30,
    secrets=[modal.Secret.from_name("huggingface-token")],
    max_containers=1,
)
class MinerU4StandardInference:
    @modal.enter()
    def load_model(self) -> None:
        import asyncio
        import json
        import os
        import tempfile
        import time
        from pathlib import Path

        t0 = time.perf_counter()
        import mineru
        from mineru.config import config
        from mineru.parser import MinerUParser

        features = config.llm_aided.features
        if features.title_leveling or features.cross_page_table_cell_merge:
            raise RuntimeError("LLM-aided post-processing is enabled; it would add an external LLM")
        if config.model.vlm.server_url:
            raise RuntimeError("model.vlm.server_url is set; this app must use the local vLLM engine")
        if (config.model.small_backend, config.model.vlm.engine) != (SMALL_BACKEND, VLM_ENGINE):
            raise RuntimeError(
                f"backend mismatch: small={config.model.small_backend} vlm={config.model.vlm.engine}")

        try:
            self._pins = json.loads(Path(MINERU_HOME, "pinned_models.json").read_text())
        except Exception as exc:
            raise RuntimeError(f"pinned model manifest missing: {exc}") from exc
        # Installed distribution version (the 4.0.10 tag's mineru/version.py
        # still reads 4.0.9, so the module attribute is not authoritative).
        try:
            from importlib.metadata import version as _dist_version
            self._mineru_version = _dist_version("mineru")
        except Exception:
            self._mineru_version = getattr(mineru, "__version__", None)
        if self._mineru_version != MINERU_VERSION:
            raise RuntimeError(f"mineru {self._mineru_version} installed, expected {MINERU_VERSION}")

        # One parser for the container. MinerUParser copies the VLM config at
        # construction; the models themselves are process-level singletons.
        self._parser = MinerUParser(tier=TIER, parse_mode=OCR_MODE, image_analysis=True)

        # Warm-up through the full Standard path: loads PP-DocLayoutV2, OCR,
        # starts AsyncLLM (vLLM engine core, CUDA graphs) and runs the VLM.
        with tempfile.TemporaryDirectory() as td:
            warm = os.path.join(td, "warmup.png")
            _warmup_page(warm)
            result = asyncio.run(self._parser.parse_async(warm))
            self._warmup_markdown_chars = len(result.markdown(image_renderer=_placeholder_image))

        self._startup_gpu_seconds = time.perf_counter() - t0
        self._startup_cost_applied = False

    @modal.exit()
    def shutdown(self) -> None:
        try:
            from mineru.model.vlm.runtime import shutdown_cached_models
            shutdown_cached_models()
        except Exception:
            pass

    # ------------------------------------------------------------------ #

    def _model_version(self) -> str:
        return (f"{MODEL_ID}@{MODEL_REVISION[:12]} "
                f"mineru=={getattr(self, '_mineru_version', None) or MINERU_VERSION} "
                f"tier={TIER} small_backend={SMALL_BACKEND}"
                f"({SMALL_MODEL_ID}@{SMALL_MODEL_REVISION[:12]}) "
                f"vlm=vllm=={VLLM_VERSION} AsyncLLM, native async batch")

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

    def _to_item(self, document_path, result, image_size) -> dict:
        from mineru.render.contracts import RenderMode

        # FULL keeps page furniture (header, footer, page_number, aside_text);
        # DEFAULT is MinerU's reading view and drops them
        # (docvortex render/_internal/markdown/renderer.py). The text leg
        # scores every printed string, so FULL is the raw_text; the default
        # view is kept alongside. For a single page the two differ only in
        # those auxiliary blocks (no cross-page merge applies).
        markdown_full = result.markdown(mode=RenderMode.FULL, image_renderer=_placeholder_image)
        markdown_default = result.markdown(image_renderer=_placeholder_image)
        structured = _middle_to_structured(result, markdown_default, image_size)
        structured["pipeline"] = {
            "tier": TIER, "hybrid_effort": "high", "ocr_mode": OCR_MODE,
            "small_backend": SMALL_BACKEND, "vlm_engine": "vllm-async-engine",
            "models": self._pins, "mineru_version": self._mineru_version,
            "markdown_mode": "full",
        }
        # MinerU does not expose per-request finish reasons or token counts
        # through the SDK result; truncation is therefore not observable here.
        structured["generation"] = {"available": False,
                                    "reason": "not exposed by mineru ParseResult"}
        return {
            "raw_text": markdown_full,
            "structured_output": structured,
            "model_version": self._model_version(),
            "prompt_used": None,
            "document_path": document_path,
            "error": None,
        }

    def _run_items(self, files: list[dict]) -> list[dict]:
        import asyncio
        import os
        import tempfile

        results: list[dict | None] = [None] * len(files)
        with tempfile.TemporaryDirectory() as td:
            paths, sizes, live = [], [], []
            for i, item in enumerate(files):
                if item.get("prompt") is not None or item.get("response_format") is not None:
                    results[i] = self._error_item(item.get("document_path"), ValueError(
                        "mineru_4_standard is a fixed pipeline; it accepts no prompt or response_format"))
                    continue
                try:
                    path = os.path.join(td, f"{i:04d}.png")
                    sizes.append(_write_input_png(item["bytes"], path))
                    paths.append(path)
                    live.append(i)
                except Exception as exc:
                    results[i] = self._error_item(item.get("document_path"), exc)

            async def _batch():
                # Same as MinerUParser.parse_batch_async (DocumentParser base:
                # asyncio.gather over parse_async), plus per-item isolation.
                return await asyncio.gather(
                    *(self._parser.parse_async(p) for p in paths), return_exceptions=True)

            parsed = asyncio.run(_batch()) if paths else []
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
                raise ValueError("mineru_4_standard is a fixed pipeline; it accepts no "
                                 "prompt or response_format")
            results = self._run_items(files)
            if len(results) != len(files):
                raise RuntimeError(f"MinerU returned {len(results)} results for {len(files)} inputs")
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
            raise ValueError("mineru_4_standard is a fixed pipeline; it accepts no "
                             "prompt or response_format")
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
