"""
Modal app: RapidOCR (rapidocr 3.9.2, ONNX Runtime CPU). CPU-only container.

Why Modal and not local: the project rule is no model inference on the user's
machine. RapidOCR is small, so it runs in a CPU container (GPU_TYPE="CPU",
cpu=4.0, memory=8192 — the allocation the harness bills for "CPU").

What runs (read from the rapidocr 3.9.2 source: rapidocr/main.py,
rapidocr/config.yaml, rapidocr/default_models.yaml; docs
rapidai.github.io/RapidOCRDocs/main/quickstart/):
  * `RapidOCR(params=...)` built once in @modal.enter, reused for every image.
  * Package-default pipeline: DB text detection (PP-OCRv6 det small) ->
    180-degree text-line classifier (ch_ppocr_mobile_v2.0 cls) -> CTC
    recognition (PP-OCRv6 rec small, multilingual; the character list is
    embedded in the ONNX metadata, not a separate dictionary file).
    All three ONNX files ship inside the 3.9.2 wheel (wheel RECORD lists
    rapidocr/models/PP-OCRv6_det_small.onnx, PP-OCRv6_rec_small.onnx,
    ch_ppocr_mobile_v2.0_cls_mobile.onnx). They are passed by explicit
    model_path and their SHA-256 checked against default_models.yaml, so no
    runtime download can substitute a different model.
  * Package-default preprocessing, recorded, not changed: Global.max_side_len
    2000 / min_side_len 30 resize, Det.limit_side_len 736 (limit_type min),
    vertical padding for very wide images, text_score 0.5 filter. The package
    maps every box back to the ORIGINAL image (map_boxes_to_original), so
    coordinates are normalised against the image as sent (after EXIF transpose).
  * `return_word_box=True` (Global option / __call__ kwarg). Word boxes are
    the toolkit's native word results: CalRecBoxes places each word inside its
    detected line quad from the recogniser's CTC column positions
    (PaddleOCR-derived, rapidocr/cal_rec_boxes/main.py). They are word
    granularity for Latin text (split on spaces and on CTC gaps > 5 columns),
    per-character for CJK. Emitted as "words" only when every kept line got
    word results; otherwise the page falls back to line boxes as
    "text_regions" with geometry_granularity "line".

Batching: one owner, a sequential loop over the files inside one container call
(RapidOCR has no batch-of-images API; ONNX Runtime's intra-op threads do the
parallel work). No prompt interface: a prompt/response_format yields an error
item rather than being silently ignored.
"""

import modal

GPU_TYPE = "CPU"
MODEL_ID = "RapidAI/RapidOCR"          # toolkit; the models are listed below
RAPIDOCR_VERSION = "3.9.2"
ONNXRUNTIME_VERSION = "1.30.0"
# The wheel is the pinned artifact; there is no HF repo/commit for it.
MODEL_REVISION = f"rapidocr=={RAPIDOCR_VERSION}"
CPU_CORES = 4.0
MEMORY_MB = 8192
ORT_INTRA_OP_THREADS = 4   # = allocated physical cores
ORT_INTER_OP_THREADS = 1

# SHA-256 from rapidocr 3.9.2 default_models.yaml (onnxruntime section), which
# also match the files in the 3.9.2 wheel (checked on a local copy 2026-09-30).
MODEL_FILES = {
    "det": ("PP-OCRv6_det_small.onnx",
            "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f"),
    "cls": ("ch_ppocr_mobile_v2.0_cls_mobile.onnx",
            "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"),
    "rec": ("PP-OCRv6_rec_small.onnx",
            "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"),
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    # opencv-python (a rapidocr dependency) needs libGL/glib at import.
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        f"rapidocr=={RAPIDOCR_VERSION}",
        f"onnxruntime=={ONNXRUNTIME_VERSION}",
        # ImageDraw.text(font_size=...) in the warm-up needs Pillow >= 10.1.
        "Pillow==12.3.0",
    )
    # Fail the build, not the first request, if the bundled models are absent.
    .run_commands(
        "python -c \"import rapidocr, pathlib; "
        "m = pathlib.Path(rapidocr.__file__).parent / 'models'; "
        "names = sorted(p.name for p in m.glob('*.onnx')); print(names); "
        "assert {'PP-OCRv6_det_small.onnx', 'PP-OCRv6_rec_small.onnx', "
        "'ch_ppocr_mobile_v2.0_cls_mobile.onnx'} <= set(names)\""
    )
)

app = modal.App("ocr-bench-rapidocr")


def _sha256(path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quad_to_box(quad, width: int, height: int):
    """4-point pixel quad -> ([x0,y0,x1,y1] normalised, normalised poly)."""
    try:
        pts = [[float(p[0]), float(p[1])] for p in quad]
    except (TypeError, ValueError, IndexError):
        return None, None
    if not pts or not width or not height:
        return None, None
    poly = [[x / width, y / height] for x, y in pts]
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)], poly


@app.cls(
    image=image,
    cpu=CPU_CORES,
    memory=MEMORY_MB,
    timeout=1800,
    startup_timeout=600,
    scaledown_window=30,
    max_containers=1,
)
class RapidOCRInference:
    @modal.enter()
    def load_model(self) -> None:
        import importlib.metadata as md
        import os
        import time
        from pathlib import Path

        t0 = time.perf_counter()
        import rapidocr
        from rapidocr import RapidOCR

        model_dir = Path(rapidocr.__file__).parent / "models"
        self._models = {}
        for stage, (fname, expected) in MODEL_FILES.items():
            path = model_dir / fname
            actual = _sha256(path)
            if actual != expected:
                raise RuntimeError(f"{fname}: sha256 {actual} != pinned {expected}")
            self._models[stage] = {"file": fname, "sha256": actual}

        params = {
            "Global.return_word_box": True,
            "Global.log_level": "warning",
            "Det.model_path": str(model_dir / MODEL_FILES["det"][0]),
            "Cls.model_path": str(model_dir / MODEL_FILES["cls"][0]),
            "Rec.model_path": str(model_dir / MODEL_FILES["rec"][0]),
            "EngineConfig.onnxruntime.intra_op_num_threads": ORT_INTRA_OP_THREADS,
            "EngineConfig.onnxruntime.inter_op_num_threads": ORT_INTER_OP_THREADS,
        }
        self._engine = RapidOCR(params=params)

        # What ONNX Runtime actually used, per stage. rapidocr only applies the
        # thread settings when 1 <= n <= os.cpu_count(), so read them back.
        self._runtime = {"os_cpu_count": os.cpu_count()}
        import hashlib
        for stage, holder in (("det", self._engine.text_det),
                              ("cls", self._engine.text_cls),
                              ("rec", self._engine.text_rec)):
            sess = holder.session.session
            opts = sess.get_session_options()
            self._runtime[stage] = {
                "providers": list(sess.get_providers()),
                "intra_op_num_threads": opts.intra_op_num_threads,
                "inter_op_num_threads": opts.inter_op_num_threads,
            }
        try:
            chars = self._engine.text_rec.session.get_character_list()
            self._models["rec"]["character_list_len"] = len(chars)
            self._models["rec"]["character_list_sha256"] = hashlib.sha256(
                "\n".join(chars).encode("utf-8")).hexdigest()
        except Exception as exc:
            self._models["rec"]["character_list_error"] = f"{type(exc).__name__}: {exc}"

        cfg = self._engine.cfg
        self._config = {
            "text_score": float(cfg.Global.text_score),
            "use_cls": bool(cfg.Global.use_cls),
            "max_side_len": int(cfg.Global.max_side_len),
            "min_side_len": int(cfg.Global.min_side_len),
            "det_limit_side_len": int(cfg.Det.limit_side_len),
            "det_limit_type": str(cfg.Det.limit_type),
            "return_word_box": True,
        }
        self._versions = {pkg: md.version(pkg) for pkg in
                          ("rapidocr", "onnxruntime", "opencv-python", "numpy", "Pillow")
                          if self._has_dist(md, pkg)}
        if self._versions.get("rapidocr") != RAPIDOCR_VERSION:
            raise RuntimeError(f"rapidocr {self._versions.get('rapidocr')} != pinned")

        # Representative warm-up: rendered text so det, cls AND rec all run.
        from PIL import Image, ImageDraw
        warm = Image.new("RGB", (1200, 400), "white")
        draw = ImageDraw.Draw(warm)
        for i, line in enumerate(("INVOICE 00123", "Total due: 1,234.56",
                                  "Date 2026-09-30")):
            draw.text((40, 40 + i * 110), line, fill="black", font_size=64)
        self._engine(warm, return_word_box=True)

        self._startup_gpu_seconds = time.perf_counter() - t0
        self._startup_cost_applied = False

    @staticmethod
    def _has_dist(md, name: str) -> bool:
        try:
            md.version(name)
            return True
        except md.PackageNotFoundError:
            return False

    def _model_version(self) -> str:
        return (f"{MODEL_ID}@{MODEL_REVISION} "
                f"det={self._models['det']['file']}[{self._models['det']['sha256'][:12]}] "
                f"cls={self._models['cls']['file']}[{self._models['cls']['sha256'][:12]}] "
                f"rec={self._models['rec']['file']}[{self._models['rec']['sha256'][:12]}] "
                f"onnxruntime=={self._versions.get('onnxruntime')} CPU")

    # ------------------------------------------------------------------ API
    @modal.method()
    def run_ocr(self, image_bytes: bytes, mime_type: str = "image/png",
                prompt: str | None = None, response_format: dict | None = None) -> dict:
        import time

        t0 = time.perf_counter()
        response = self._one({"bytes": image_bytes, "mime_type": mime_type,
                              "prompt": prompt, "response_format": response_format})
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
        results = []
        for item in files:
            merged = dict(item)
            merged.setdefault("prompt", prompt)
            merged.setdefault("response_format", response_format)
            response = self._one(merged)
            response["document_path"] = item.get("document_path")
            results.append(response)
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

    def _one(self, item: dict) -> dict:
        try:
            if item.get("prompt") is not None or item.get("response_format") is not None:
                raise ValueError("RapidOCR is a fixed pipeline and takes no prompt; "
                                 "engines.py must declare accepts_task=False")
            return self._recognise(item["bytes"])
        except Exception as exc:  # never raise out of a batch
            return self._error(exc)

    def _recognise(self, image_bytes: bytes) -> dict:
        import io

        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(image_bytes)) as src:
            img = ImageOps.exif_transpose(src).convert("RGB")
        width, height = img.size
        # A PIL image goes through rapidocr's LoadImage (RGB -> BGR).
        out = self._engine(img, return_word_box=True)

        txts = list(getattr(out, "txts", None) or [])
        boxes = getattr(out, "boxes", None)
        scores = list(getattr(out, "scores", None) or [])
        boxes = [] if boxes is None else [b.tolist() for b in boxes]

        lines = []
        for text, quad, score in zip(txts, boxes, scores):
            bbox, poly = _quad_to_box(quad, width, height)
            if bbox and str(text).strip():
                lines.append({"text": str(text), "bbox": bbox, "poly": poly,
                              "confidence": float(score)})

        # word_results: per kept line, a tuple of (text, score, 4-point box) when
        # word boxes were computed; rapidocr skips the computation for the whole
        # page if any line lacks word info, leaving non-tuple entries.
        words, lines_with_words = [], 0
        for line_words in getattr(out, "word_results", None) or ():
            got = False
            for entry in line_words or ():
                if (isinstance(entry, tuple) and len(entry) == 3
                        and entry[2] is not None and str(entry[0]).strip()):
                    bbox, poly = _quad_to_box(entry[2], width, height)
                    if bbox:
                        words.append({"text": str(entry[0]), "bbox": bbox,
                                      "poly": poly, "confidence": float(entry[1])})
                        got = True
            lines_with_words += int(got)

        structured = {
            "lines": lines,
            "image_size": [width, height],
            "pipeline_config": self._config,
            "models": self._models,
            "onnxruntime": self._runtime,
            "packages": self._versions,
            "word_box_method": "rapidocr CalRecBoxes (CTC column positions within the line quad)",
            "lines_with_word_boxes": lines_with_words,
            "elapse_det_cls_rec": list(getattr(out, "elapse_list", []) or []),
        }
        if words and lines_with_words == len(lines):
            structured["words"] = words
            structured["geometry_granularity"] = "word"
            structured["geometry_coordinates"] = "normalized_xyxy"
        elif lines:
            structured["text_regions"] = [{k: l[k] for k in ("text", "bbox", "poly")}
                                          for l in lines]
            structured["geometry_granularity"] = "line"
            structured["geometry_coordinates"] = "normalized_xyxy"
            if words:
                structured["partial_word_boxes"] = words

        return {
            # Detector order (top-to-bottom, left-to-right sort), one line each.
            "raw_text": "\n".join(l["text"] for l in lines),
            "structured_output": structured,
            "model_version": self._model_version(),
            "prompt_used": None,
            "error": None,
        }
