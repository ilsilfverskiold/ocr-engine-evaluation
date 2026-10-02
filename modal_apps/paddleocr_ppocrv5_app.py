"""
Modal app: PaddleOCR PP-OCRv5 (server det + server rec) — Paddle's classic
text-line OCR pipeline, run as its own engine beside PaddleOCR-VL.

Why a separate engine: PaddleOCR-VL (modal_apps/paddleocr_vl_app.py) is layout
detection + a 0.9B VLM that reads each layout block, so its finest geometry is
a block. PP-OCRv5 is the other Paddle product: a text-line DETECTOR, then a
recogniser per line. Its primary output is one polygon per text line, which is
what the geometry leg can score. Different weights, different code path, so a
different engine (never a swap).

Model contract (sources read 2026-10-01):
  * Detection: PaddlePaddle/PP-OCRv5_server_det, HF commit
    ca867c897ecbca8873081573a802ad70d499cb94. Card: "one of the PP-OCRv5_det
    series, the latest generation of text detection models developed by the
    PaddleOCR team"; output `dt_polys` (one 4-point polygon per text line) +
    `dt_scores`; companion `text_recognition_model_name PP-OCRv5_server_rec`.
  * Recognition: PaddlePaddle/PP-OCRv5_server_rec, HF commit
    b26c3587fda8da3c8ec0ce357214b4d661ff1558.
  * Pipeline: paddleocr==3.7.0 general OCR pipeline
    (www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html):
    PaddleOCR(text_detection_model_name=..., text_recognition_model_name=...,
    use_doc_orientation_classify, use_doc_unwarping, use_textline_orientation),
    predict(input) -> Result with rec_texts, rec_scores, rec_polys, dt_polys,
    rec_boxes ([x_min, y_min, x_max, y_max]); `return_word_box` ("Whether to
    return word-level boxes.", default False) — the fields it adds are not
    documented on that page, so the app records the whole result JSON and
    reads word boxes from whatever keys appear (see _words_from_result), and
    falls back to line boxes when none do.
  * Runtime: paddlepaddle-gpu==3.2.1 from the cu126 index — the exact
    combination modal_apps/paddleocr_vl_app.py already runs on Modal
    (paddleocr 3.7.0 pinned there for the same reproducibility reason).
  * Task boundary: one page image. The three optional preprocessors
    (document orientation, unwarping, text-line orientation) are OFF so the
    boxes are in the coordinates of the image as sent; the detector's own
    resize policy (text_det_limit_side_len / limit_type, 3.x defaults 64 /
    "min", i.e. no downscaling of these pages) is read back from the pipeline
    and recorded, not changed.
  * Batching owner: a sequential loop inside one container call. The
    pipeline's own text_rec_batch_size batches lines within a page.
  * GPU: A10 first candidate (sm_86; the Paddle cu126 wheels are known to run
    on the A100 in the VL app; T4/sm_75 is a later experiment, not assumed).
    The models are ~100 MB; memory is not the constraint.
  * No prompt: a prompt/response_format yields an error item.
"""

import modal

GPU_TYPE = "A10"
PADDLEOCR_VERSION = "3.7.0"
PADDLEPADDLE_VERSION = "3.2.1"
MODEL_ID = "PaddlePaddle/PP-OCRv5_server_det + PaddlePaddle/PP-OCRv5_server_rec"
MODELS = {
    # name -> (HF repo, pinned commit)
    "det": ("PaddlePaddle/PP-OCRv5_server_det", "ca867c897ecbca8873081573a802ad70d499cb94"),
    "rec": ("PaddlePaddle/PP-OCRv5_server_rec", "b26c3587fda8da3c8ec0ce357214b4d661ff1558"),
}
MODEL_REVISION = "det@ca867c89 rec@b26c3587"
MODEL_ROOT = "/root/.paddlex/ppocrv5"   # inside the shared paddlex volume

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libgomp1")
    .pip_install(
        f"paddlepaddle-gpu=={PADDLEPADDLE_VERSION}",
        extra_index_url="https://www.paddlepaddle.org.cn/packages/stable/cu126/",
    )
    .pip_install(f"paddleocr=={PADDLEOCR_VERSION}", "Pillow>=10.0", "huggingface_hub>=0.23")
    # Paddle cannot be imported on the GPU-less build machine (needs
    # libcuda.so.1), so the models are fetched at container start into the
    # volume, pinned by commit, and only the first container pays for it.
    .env({"OMP_NUM_THREADS": "1"})
)

# Same shared cache the PaddleOCR-VL app uses, so concurrent containers do not
# race on downloads.
paddlex_cache = modal.Volume.from_name("paddlex-model-cache", create_if_missing=True)

app = modal.App("ocr-bench-paddleocr-ppocrv5")


def _poly_to_box(poly, width: int, height: int):
    """4-point pixel polygon (or [x0,y0,x1,y1]) -> normalised bbox + poly."""
    try:
        flat = list(poly)
        if flat and isinstance(flat[0], (list, tuple)):
            pts = [[float(p[0]), float(p[1])] for p in flat]
        elif len(flat) == 4:
            x0, y0, x1, y1 = (float(v) for v in flat)
            pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        else:
            return None, None
    except (TypeError, ValueError, IndexError):
        return None, None
    if not pts or not width or not height:
        return None, None
    norm = [[x / width, y / height] for x, y in pts]
    xs = [p[0] for p in norm]
    ys = [p[1] for p in norm]
    return [min(xs), min(ys), max(xs), max(ys)], norm


def _jsonable(obj):
    """Result JSON with numpy arrays turned into lists and images dropped."""
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if not k.endswith("_img") and k != "img"}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if np is not None and isinstance(obj, np.generic):
        return obj.item()
    return obj


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/root/.paddlex": paddlex_cache},
    timeout=1800,
    startup_timeout=900,
    scaledown_window=60,
    max_containers=1,
)
class PaddleOCRv5Inference:
    @modal.enter()
    def load_model(self) -> None:
        import importlib.metadata as md
        import os
        import time

        t0 = time.perf_counter()
        from huggingface_hub import snapshot_download

        self._model_dirs = {}
        for stage, (repo, commit) in MODELS.items():
            target = os.path.join(MODEL_ROOT, repo.split("/")[-1])
            path = snapshot_download(repo, revision=commit, local_dir=target)
            self._model_dirs[stage] = path
        paddlex_cache.commit()

        from paddleocr import PaddleOCR

        self._engine = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_detection_model_dir=self._model_dirs["det"],
            text_recognition_model_name="PP-OCRv5_server_rec",
            text_recognition_model_dir=self._model_dirs["rec"],
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="gpu:0",
        )
        self._versions = {}
        for pkg in ("paddleocr", "paddlepaddle-gpu", "paddlex", "numpy", "Pillow", "opencv-python", "opencv-contrib-python"):
            try:
                self._versions[pkg] = md.version(pkg)
            except md.PackageNotFoundError:
                pass
        if self._versions.get("paddleocr") != PADDLEOCR_VERSION:
            raise RuntimeError(f"paddleocr {self._versions.get('paddleocr')} != pinned {PADDLEOCR_VERSION}")

        # Representative warm-up: rendered text so det AND rec run on the GPU.
        from PIL import Image, ImageDraw
        import numpy as np
        warm = Image.new("RGB", (1200, 400), "white")
        draw = ImageDraw.Draw(warm)
        for i, line in enumerate(("INVOICE 00123", "Total due: 1,234.56", "Date 2026-10-01")):
            draw.text((40, 40 + i * 110), line, fill="black", font_size=64)
        warm_result = list(self._engine.predict(np.array(warm)[:, :, ::-1]))
        self._config = self._read_config(warm_result)
        self._startup_gpu_seconds = time.perf_counter() - t0
        self._startup_cost_applied = False

    def _read_config(self, results) -> dict:
        """The pipeline's own settings, as it reports them, for the record."""
        cfg = {}
        try:
            res = results[0].json if results else {}
            res = res.get("res", res)
            for key in ("model_settings", "text_det_params", "text_type", "text_rec_score_thresh"):
                if key in res:
                    cfg[key] = _jsonable(res[key])
        except Exception as exc:
            cfg["error"] = f"{type(exc).__name__}: {exc}"
        return cfg

    def _model_version(self) -> str:
        return (f"{MODEL_ID} ({MODEL_REVISION}) paddleocr=={self._versions.get('paddleocr')} "
                f"paddlepaddle-gpu=={self._versions.get('paddlepaddle-gpu')} {GPU_TYPE}")

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
                raise ValueError("PP-OCRv5 is a fixed pipeline and takes no prompt; "
                                 "engines.py must declare accepts_task=False")
            if (item.get("mime_type") or "").endswith("pdf"):
                raise ValueError("PDF input not supported by this app; send a page image")
            return self._recognise(item["bytes"])
        except Exception as exc:  # never raise out of a batch
            return self._error(exc)

    @staticmethod
    def _words_from_result(res: dict, width: int, height: int) -> list[dict]:
        """Word boxes, if the pipeline returned any under the names it uses.
        Shapes seen in paddleocr 3.x sources: per line, a list of word strings
        plus a list of word boxes/polygons."""
        words = []
        texts = res.get("text_word") or res.get("rec_words") or []
        boxes = (res.get("text_word_boxes") or res.get("text_word_region")
                 or res.get("rec_word_boxes") or res.get("word_boxes") or [])
        for line_words, line_boxes in zip(texts, boxes):
            for text, box in zip(line_words or [], line_boxes or []):
                bbox, poly = _poly_to_box(box, width, height)
                if bbox and str(text).strip():
                    words.append({"text": str(text), "bbox": bbox, "poly": poly})
        return words

    def _recognise(self, image_bytes: bytes) -> dict:
        import io

        import numpy as np
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(image_bytes)) as src:
            img = ImageOps.exif_transpose(src).convert("RGB")
        width, height = img.size
        bgr = np.array(img)[:, :, ::-1]
        try:
            results = list(self._engine.predict(bgr, return_word_box=True))
            word_box_requested = True
        except TypeError:
            results = list(self._engine.predict(bgr))
            word_box_requested = False
        res = results[0].json if results else {}
        res = res.get("res", res)

        rec_texts = list(res.get("rec_texts") or [])
        scores = list(res.get("rec_scores") or [])
        polys = res.get("rec_polys") or res.get("dt_polys") or []
        lines = []
        for i, text in enumerate(rec_texts):
            poly = polys[i] if i < len(polys) else None
            bbox, npoly = _poly_to_box(poly, width, height) if poly is not None else (None, None)
            if bbox and str(text).strip():
                lines.append({"text": str(text), "bbox": bbox, "poly": npoly,
                              "confidence": float(scores[i]) if i < len(scores) else None})
        words = self._words_from_result(res, width, height)

        structured = {
            "lines": lines,
            "image_size": [width, height],
            "pipeline_config": self._config,
            "models": {k: {"repo": v[0], "commit": v[1]} for k, v in MODELS.items()},
            "packages": self._versions,
            "word_box_requested": word_box_requested,
            "result_keys": sorted(k for k in res.keys() if not k.endswith("_img")),
        }
        if words:
            structured["words"] = words
            structured["geometry_granularity"] = "word"
            structured["text_regions"] = [{k: l[k] for k in ("text", "bbox", "poly")} for l in lines]
        elif lines:
            structured["text_regions"] = [{k: l[k] for k in ("text", "bbox", "poly")} for l in lines]
            structured["geometry_granularity"] = "line"
        # The whole result (minus images) so nothing the pipeline said is lost.
        structured["result_json"] = _jsonable(res)

        return {
            "raw_text": "\n".join(l["text"] for l in lines),
            "structured_output": structured,
            "model_version": self._model_version(),
            "prompt_used": None,
            "error": None,
        }
