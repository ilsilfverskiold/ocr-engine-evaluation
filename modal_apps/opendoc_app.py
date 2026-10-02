"""
Modal app: OpenDoc-0.1B — openocr-python==0.1.5, OpenOCR(task='doc', use_layout_detection=True).
Device: CPU (4 cores, 8 GiB). L4 is the documented probe (flip GPU_TYPE).

Sources (read 2026-09-30):
  - recipe: https://github.com/Topdu/OpenOCR/blob/059fe2b3ec71bb137b8c8504203498b1f563fdcc/docs/opendoc.md
    (commit "release 0.1.5 fix bug", 2026-02-12; the repo has no 0.1.5 tag)
  - package source: openocr_python-0.1.5.tar.gz from PyPI
    (sha256 734f42bf6ac832d5339e153cb13c4b5e75b369d4b435e819d97554ad8dec7397):
    openocr/openocr.py, openocr/tools/infer_doc_onnx.py, openocr/tools/infer_unirec_onnx.py

What `OpenOCR(task='doc')` actually runs in 0.1.5
-------------------------------------------------
openocr.py `_init_doc_task` -> tools.infer_doc_onnx.OpenDocONNX: an ONNX Runtime
pipeline — PP-DocLayoutV2 exported to ONNX (topdu/PP_DoclayoutV2_onnx) for
layout, then UniRec-0.1B exported to ONNX (topdu/unirec_0_1b_onnx: encoder +
KV-cached decoder, greedy, FP32) on each crop. The package's dependencies are
onnxruntime, opencv, numpy, etc. — NOT torch, transformers or paddle.

The "Torch 2.6 / Transformers 4.49 / paddlepaddle-gpu 3.2.0 (cu126)" recipe in
opendoc.md belongs to the *source* path (`python tools/infer_doc.py`, PaddleX
layout + PyTorch UniRec from topdu/unirec-0.1b), which is also the path the
OmniDocBench 90.57 figure is reproduced with. This engine follows the requested
package API, so that stack is not installed. The two paths are not shown to be
output-equivalent; see the UNVERIFIED list in the hand-off.

Device choice
-------------
CPU first: the package's hard dependency is CPU `onnxruntime`; GPU needs the
optional `onnxruntime-gpu` extra and a swap of the ORT wheel. The decoder is a
Python per-token loop that passes the numpy KV cache in and out of ORT on every
step (infer_unirec_onnx.py `_infer_single_image`), so a GPU spends much of each
step on host<->device copies. The research plan also lists "CPU, then L4". The
L4 probe is a one-line change: GPU_TYPE = "L4" swaps the ORT wheel, requests the
GPU, and forces use_gpu=True; the actual ORT providers are recorded per result.
"""

import modal

GPU_TYPE = "CPU"  # probe: "L4"
ON_GPU = GPU_TYPE != "CPU"

OPENOCR_VERSION = "0.1.5"
# ORT/numpy/opencv versions current at the 0.1.5 release (2026-02-12); the
# package leaves them unpinned. Reproducibility pins, not upstream-prescribed.
ONNXRUNTIME_VERSION = "1.24.1"

# Recognizer: UniRec-0.1B ONNX export (Apache-2.0 per HF metadata).
MODEL_ID = "topdu/unirec_0_1b_onnx"
MODEL_REVISION = "e8f55f72013d544cbd6c27c576946ac63da247ed"  # head 2026-01-07
UNIREC_SHA256 = {
    "unirec_encoder.onnx": "eb6b5d38f16c4f1abc39bab47e8d4ec83103b3246f642bde599edf6634289392",
    "unirec_decoder.onnx": "334326c877856d7595614424f1e15d6b1aab6ba24c26c769f09ab38901d5021c",
    # unirec_tokenizer_mapping.json is not an LFS file (no published sha256);
    # its hash is computed at build and recorded in the manifest.
}
# Layout: PP-DocLayoutV2 ONNX export. The HF repo declares NO license; the
# source model PaddlePaddle/PP-DocLayoutV2 is Apache-2.0.
LAYOUT_ID = "topdu/PP_DoclayoutV2_onnx"
LAYOUT_REVISION = "4de0e57bf2a3f7e818f1ad0cbf2961c33818a07b"  # head 2026-02-04
LAYOUT_FILE = "PP-DoclayoutV2.onnx"
LAYOUT_SHA256 = "2009fcb35e64085ab9f6f2b27aca550edc29a040a24f7d6a0f05b74a2f804860"

ASSET_DIR = "/opt/openocr_models"
UNIREC_DIR = f"{ASSET_DIR}/unirec_0_1b_onnx"
LAYOUT_DIR = f"{ASSET_DIR}/PP_DoclayoutV2_onnx"
MANIFEST = f"{ASSET_DIR}/manifest.json"

# OpenDocONNX defaults, kept as shipped (infer_doc_onnx.py __init__/__call__):
# layout_threshold 0.5, use_chart_recognition True, merge_layout_blocks True,
# max_parallel_blocks 4 (crop-level threads over one shared ORT session),
# max_length 2048 generated tokens per crop.
MAX_LENGTH = 2048


def _bake_assets() -> None:
    """Image-build step: fetch pinned ONNX assets, verify hashes, write manifest."""
    import hashlib
    import json
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    manifest = {"unirec": {"repo": MODEL_ID, "revision": MODEL_REVISION, "files": {}},
                "layout": {"repo": LAYOUT_ID, "revision": LAYOUT_REVISION, "files": {}}}
    for name in ("unirec_encoder.onnx", "unirec_decoder.onnx", "unirec_tokenizer_mapping.json"):
        path = Path(hf_hub_download(MODEL_ID, name, revision=MODEL_REVISION, local_dir=UNIREC_DIR))
        digest = _sha256(path)
        if name in UNIREC_SHA256 and digest != UNIREC_SHA256[name]:
            raise RuntimeError(f"{name} sha256 {digest} != pinned {UNIREC_SHA256[name]}")
        manifest["unirec"]["files"][name] = digest
    path = Path(hf_hub_download(LAYOUT_ID, LAYOUT_FILE, revision=LAYOUT_REVISION, local_dir=LAYOUT_DIR))
    digest = _sha256(path)
    if digest != LAYOUT_SHA256:
        raise RuntimeError(f"{LAYOUT_FILE} sha256 {digest} != pinned {LAYOUT_SHA256}")
    manifest["layout"]["files"][LAYOUT_FILE] = digest
    Path(MANIFEST).write_text(json.dumps(manifest, indent=2))


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libgomp1")
    .pip_install(
        f"openocr-python=={OPENOCR_VERSION}",
        f"onnxruntime=={ONNXRUNTIME_VERSION}",
        "numpy==2.4.2",
        "opencv-python==4.13.0.92",
        "Pillow==11.3.0",
        "huggingface_hub==0.36.2",
    )
)
if ON_GPU:
    # CPU and GPU ORT wheels must not coexist (both provide `onnxruntime`).
    image = image.run_commands(
        "pip uninstall -y onnxruntime",
        f"pip install --no-cache-dir 'onnxruntime-gpu[cuda,cudnn]=={ONNXRUNTIME_VERSION}'",
    )
image = image.run_function(
    _bake_assets, secrets=[modal.Secret.from_name("huggingface-token")]
)

app = modal.App("ocr-bench-opendoc")

_resources = {"gpu": GPU_TYPE} if ON_GPU else {"cpu": 4.0, "memory": 8192}

# Labels OpenDocONNX treats as pictures (IMAGE_LABELS + chart when chart
# recognition is off). Their boxes carry no recognised text.
_IMAGE_LABELS = {"image", "header_image", "footer_image", "seal"}


def _base_label(label: str) -> str:
    """OpenDocONNX suffixes every label with its reading-order index (text_05)."""
    label = str(label)
    head, _, tail = label.rpartition("_")
    return head if head and tail.isdigit() else label


def _normalise_box(box, width: int, height: int):
    if box is None or not width or not height:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in list(box)[:4])
    except (TypeError, ValueError):
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
    """EXIF-transpose + RGB PNG at original resolution (coordinates refer to it).

    OpenDoc's own preprocessing (layout input 800x800; each crop resized to fit
    960x1408 for UniRec) happens inside the pipeline and its boxes are returned
    in this image's pixel space, so nothing needs mapping back.
    """
    import io

    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.save(path, format="PNG")
    return img.size


def _warmup_page(path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(img)
    try:
        big, small = ImageFont.load_default(size=44), ImageFont.load_default(size=26)
    except TypeError:
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
    # CPU pages are expected to take seconds to tens of seconds each; the
    # timeout covers an 8-page outer batch with wide margin (unmeasured).
    timeout=3600,
    startup_timeout=900,
    scaledown_window=30,
    secrets=[modal.Secret.from_name("huggingface-token")],
    # Batching owner: OpenDocONNX itself — up to 4 crops of a page recognised
    # concurrently on one shared ORT session. It has no multi-image list API
    # (a path is one image or one PDF), so pages run sequentially.
    max_containers=1,
    **_resources,
)
class OpenDocInference:
    @modal.enter()
    def load_model(self) -> None:
        import json
        import os
        import tempfile
        import time
        from pathlib import Path

        t0 = time.perf_counter()
        from openocr import OpenOCR

        self._ocr = OpenOCR(
            task="doc",
            use_layout_detection=True,
            # OpenDocONNX tests `use_gpu is True/False`; a bool, not the
            # wrapper's 'true'/'false' strings, is what selects the provider.
            use_gpu=True if ON_GPU else False,
            layout_model_path=f"{LAYOUT_DIR}/{LAYOUT_FILE}",
            unirec_encoder_path=f"{UNIREC_DIR}/unirec_encoder.onnx",
            unirec_decoder_path=f"{UNIREC_DIR}/unirec_decoder.onnx",
            tokenizer_mapping_path=f"{UNIREC_DIR}/unirec_tokenizer_mapping.json",
            auto_download=False,  # pinned, baked assets only
        )

        doc = self._ocr.model
        self._providers = {
            "layout": list(doc.layout_detector.session.get_providers()),
            "unirec_encoder": list(doc.vlm_recognizer.encoder_session.get_providers()),
            "unirec_decoder": list(doc.vlm_recognizer.decoder_session.get_providers()),
        }
        if ON_GPU and "CUDAExecutionProvider" not in self._providers["unirec_decoder"]:
            raise RuntimeError(f"GPU requested but ORT providers are {self._providers}")
        try:
            self._manifest = json.loads(Path(MANIFEST).read_text())
        except Exception:
            self._manifest = None
        try:
            affinity = len(os.sched_getaffinity(0))
        except Exception:
            affinity = None
        self._runtime = {"os_cpu_count": os.cpu_count(), "cpu_affinity": affinity}

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "warm.png"
            _warmup_page(p)
            warm = self._ocr(image_path=str(p))
            self._warmup_blocks = len(warm.get("recognition_results") or [])

        self._startup_seconds = time.perf_counter() - t0
        self._startup_reported = False

    # ------------------------------------------------------------------ #
    @modal.method()
    def run_ocr(self, image_bytes: bytes, mime_type: str = "image/png",
                prompt: str | None = None, response_format: dict | None = None) -> dict:
        # Fixed pipeline: UniRec takes no caller prompt (OpenDocONNX even
        # computes per-region prompt strings and never passes them). Accepting
        # a prompt silently would misreport what was asked.
        if prompt is not None or response_format is not None:
            raise ValueError("opendoc is a fixed pipeline; it accepts no prompt or response_format")
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
        return self._run_batch(files, prompt, response_format)

    # ------------------------------------------------------------------ #
    def _run_batch(self, files, prompt=None, response_format=None) -> dict:
        import tempfile
        import time
        from pathlib import Path

        started = time.perf_counter()
        results = []
        try:
            with tempfile.TemporaryDirectory() as td:
                for index, item in enumerate(files):
                    doc_path = item.get("document_path")
                    if item.get("prompt", prompt) is not None or \
                            item.get("response_format", response_format) is not None:
                        results.append(self._error_response(
                            ValueError("opendoc is a fixed pipeline; it accepts no prompt "
                                       "or response_format"), doc_path))
                        continue
                    try:
                        page = Path(td) / f"{index:04d}.png"
                        size = _write_page(item["bytes"], page)
                        result = self._ocr(image_path=str(page))
                        response = self._result_to_response(result, size, Path(td) / "out")
                        response["document_path"] = doc_path
                    except Exception as exc:
                        response = self._error_response(exc, doc_path)
                    results.append(response)
        except Exception as exc:
            results = [self._error_response(exc, item.get("document_path")) for item in files]

        elapsed = time.perf_counter() - started
        startup, cold = 0.0, False
        if not getattr(self, "_startup_reported", True):
            startup, cold = getattr(self, "_startup_seconds", 0.0), True
            self._startup_reported = True
        return {
            "results": results,
            # For GPU_TYPE "CPU" this is the occupied container interval; the
            # harness bills it as 4 cores + 8 GiB.
            "batch_gpu_seconds": elapsed,
            "gpu_type": GPU_TYPE,
            "startup_gpu_seconds": startup,
            "is_cold_start": cold,
        }

    def _model_version(self) -> str:
        return (
            f"{MODEL_ID}@{MODEL_REVISION[:12]} + {LAYOUT_ID}@{LAYOUT_REVISION[:12]} "
            f"openocr-python=={OPENOCR_VERSION} task=doc layout=on "
            f"(onnxruntime{'-gpu' if ON_GPU else ''}=={ONNXRUNTIME_VERSION}, FP32, {GPU_TYPE})"
        )

    def _error_response(self, exc: Exception, document_path=None) -> dict:
        return {
            "raw_text": "",
            "structured_output": {},
            "model_version": self._model_version(),
            "prompt_used": None,  # fixed pipeline (task="doc")
            "document_path": document_path,
            "gpu_type": GPU_TYPE,
            "error": f"{exc.__class__.__name__}: {str(exc) or repr(exc)}",
        }

    # ------------------------------------------------------------------ #
    def _result_to_response(self, result: dict, image_size, out_dir) -> dict:
        """Native JSON + Markdown via the package's own writers, then adapters."""
        import json
        import os

        out_dir = str(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        # Package writers: <out>/<name>/<name>.json and .md (name = input stem).
        self._ocr.save_to_json(result, out_dir)
        self._ocr.save_to_markdown(result, out_dir)
        stem = os.path.basename(result["input_path"]).rsplit(".", 1)[0]
        with open(os.path.join(out_dir, stem, f"{stem}.json"), encoding="utf-8") as f:
            native_json = json.load(f)
        with open(os.path.join(out_dir, stem, f"{stem}.md"), encoding="utf-8") as f:
            markdown = f.read()

        page_w = result.get("width") or image_size[0]
        page_h = result.get("height") or image_size[1]
        recs = result.get("recognition_results") or []

        structured: dict = {
            "native_json": native_json,
            "preprocessing": {"exif_transpose": True, "mode": "RGB", "resize": None,
                              "coordinate_space": "original image after EXIF transpose",
                              "internal": "layout input 800x800; crops fit to 960x1408 "
                                          "(boxes returned in original pixels)"},
            "pipeline_config": {
                "task": "doc", "use_layout_detection": True, "layout_threshold": 0.5,
                "use_chart_recognition": True, "merge_layout_blocks": True,
                "max_parallel_blocks": 4, "max_length": MAX_LENGTH,
                "providers": getattr(self, "_providers", None),
                "assets": getattr(self, "_manifest", None),
                "runtime": getattr(self, "_runtime", None),
            },
            # OpenDocONNX returns text only; token counts / EOS are not exposed,
            # so a crop that hit max_length is not distinguishable here.
            "generation": {"max_length_per_region": MAX_LENGTH,
                           "finish_reason": "not exposed by pipeline"},
        }

        blocks, tables, regions = [], [], []
        continuation_heads = set()
        for i, rec in enumerate(recs):
            if rec.get("is_merged_continuation") and i > 0:
                continuation_heads.add(i - 1)
        for order, rec in enumerate(recs):
            label = _base_label(rec.get("label", ""))
            bbox = _normalise_box(rec.get("bbox"), page_w, page_h)
            content = rec.get("text") or ""
            blocks.append({
                "label": label, "raw_label": rec.get("label"), "bbox": bbox,
                "bbox_px": rec.get("bbox"), "order": order, "score": rec.get("score"),
                "content": content, "is_image": bool(rec.get("is_image")),
                "is_merged_continuation": bool(rec.get("is_merged_continuation")),
            })
            if label == "table" and "<table" in content.lower():
                tables.append({"html": content.strip(), "bbox": bbox})
            # Text geometry only where the text was recognised from exactly this
            # box: skip picture blocks, empty blocks, and any merged group (the
            # head's text spans boxes it does not cover).
            merged = rec.get("is_merged_continuation") or order in continuation_heads
            if rec.get("is_image") or label in _IMAGE_LABELS or merged or bbox is None:
                continue
            region_text = _html_to_text(content) if label == "table" else content.strip()
            if region_text:
                regions.append({"text": region_text, "bbox": bbox, "label": label})

        if blocks:
            structured["layout_blocks"] = blocks
        if tables:
            structured["tables"] = tables
        if regions:
            structured["text_regions"] = regions
            structured["geometry_granularity"] = "block"

        return {
            "raw_text": markdown,
            "structured_output": structured,
            "model_version": self._model_version(),
            "prompt_used": None,  # fixed pipeline (task="doc")
            "gpu_type": GPU_TYPE,
            "error": None,
        }
