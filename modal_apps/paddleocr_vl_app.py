"""Modal app: PaddleOCR-VL — PaddlePaddle/PaddleOCR-VL. GPU: A100-80GB."""

import modal

GPU_TYPE = "A100-80GB"  # hardware-matched to published throughput figures (single A100)
MODEL_ID = "PaddlePaddle/PaddleOCR-VL"
# Pinned to the exact HF commit. Model repos are mutable; an unpinned
# MODEL_ID silently changes what was benchmarked.
MODEL_REVISION = "7fa00a8c55b735ba51ba49a9058f3f9c57a99a11"
# The name the pipeline client requests. It is version-specific: with
# pipeline_version="v1" the client asks vLLM for "PaddleOCR-VL-0.9B" and a
# v1.6 name returns 404 ("The model `PaddleOCR-VL-0.9B` does not exist").
SERVED_NAME = "PaddleOCR-VL-0.9B"
VLLM_PORT = 8000
VLLM_URL = f"http://127.0.0.1:{VLLM_PORT}/v1"
VENV = "/opt/vllmenv"

# vLLM is capped well below 1.0 so paddle's layout model has room on the same card.
VLLM_GPU_FRACTION = "0.55"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libgomp1", "python3-venv")
    .pip_install(
        "paddlepaddle-gpu==3.2.1",
        extra_index_url="https://www.paddlepaddle.org.cn/packages/stable/cu126/",
    )
    # Pinned. paddleocr 3.6.0 (2026-05-28) made PaddleOCR-VL v1.6 the default
    # pipeline, so an unpinned install silently pairs the v1 0.9B model with
    # v1.6 components and the result stops being reproducible "PaddleOCR-VL v1".
    # 3.7.0 is pinned rather than a pre-3.6 release because pipeline_version is
    # a documented parameter there, so v1 is selected explicitly instead of
    # being whatever the package happens to default to.
    .pip_install("paddleocr[doc-parser]==3.7.0", "Pillow>=10.0",
                 "huggingface_hub>=0.23")
    # Isolated interpreter for vLLM so its CUDA wheels cannot clobber paddle's.
    # Verified by the vllm import below: this layer shows no "Uninstalling nvidia-*"
    # lines, which is what broke libpaddle.so when vLLM shared the environment.
    .run_commands(
        f"python -m venv {VENV}",
        f"{VENV}/bin/pip install --no-cache-dir vllm==0.21.0",
        f"{VENV}/bin/python -c 'import vllm; print(\"vllm\", vllm.__version__)'",
    )
    # NOTE: paddle models cannot be baked into the image. paddlepaddle-gpu needs
    # libcuda.so.1 to import, and Modal build machines have no GPU, so any build-time
    # `from paddleocr import ...` fails. PP-DocLayoutV3 is fetched at container start
    # into the Volume below instead, so only the first container pays for it.
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "OMP_NUM_THREADS": "1"})
)

# Shared model cache. Without it, several containers starting together race to
# download PP-DocLayoutV3 and paddlex raises
# "No available model hosting platforms detected".
paddlex_cache = modal.Volume.from_name("paddlex-model-cache", create_if_missing=True)

app = modal.App("ocr-bench-paddleocr-vl")


def _normalise_box(box, width: int, height: int):
    """Paddle box or polygon in PIXELS -> [x0, y0, x1, y1] normalised 0-1.

    Accepts [x0,y0,x1,y1] and 4-point polygons ([[x,y], ...]), which is what
    rec_polys/dt_polys give. Returns None if the shape is unrecognised or the
    page size is unknown, so a bad box is dropped rather than silently scored.
    """
    if not box or not width or not height:
        return None
    try:
        flat = list(box)
        if flat and isinstance(flat[0], (list, tuple)):       # polygon
            xs = [float(p[0]) for p in flat]
            ys = [float(p[1]) for p in flat]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        elif len(flat) == 4:
            x0, y0, x1, y1 = (float(v) for v in flat)
        else:
            return None
    except (TypeError, ValueError, IndexError):
        return None
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0 / width, y0 / height, x1 / width, y1 / height]


def _page_size(image_path: str):
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return (0, 0)


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=1800,
    startup_timeout=1800,
    scaledown_window=30,
    secrets=[modal.Secret.from_name("huggingface-token")],
    volumes={"/root/.paddlex": paddlex_cache},
    # PaddleOCR-VL owns its batching: predict(list_of_paths, use_queues=True)
    # pipelines loading, layout detection, and VLM recognition inside one process.
    max_containers=1,
)
class PaddleOCRVLInference:
    @modal.enter()
    def load_model(self) -> None:
        import os
        import subprocess
        import time
        import urllib.error
        import urllib.request

        t0 = time.perf_counter()
        self._server = subprocess.Popen(
            [
                f"{VENV}/bin/vllm", "serve", MODEL_ID,
                "--revision", MODEL_REVISION,
                "--served-model-name", SERVED_NAME,
                "--host", "127.0.0.1", "--port", str(VLLM_PORT),
                "--max-model-len", "16384",
                "--gpu-memory-utilization", VLLM_GPU_FRACTION,
                "--limit-mm-per-prompt", '{"image": 1}',
                "--no-enable-prefix-caching",
                "--mm-processor-cache-gb", "0",
                "--trust-remote-code",
            ],
            env={**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0"},
        )

        deadline, last = time.time() + 900, None
        while time.time() < deadline:
            if self._server.poll() is not None:
                raise RuntimeError(f"vLLM exited early: {self._server.returncode}")
            try:
                with urllib.request.urlopen(f"{VLLM_URL}/models", timeout=5) as r:
                    if r.status < 500:
                        break
            except (OSError, urllib.error.URLError) as exc:
                last = exc
            time.sleep(5)
        else:
            raise TimeoutError(f"vLLM never became ready: {last}")

        from paddleocr import PaddleOCRVL

        # use_chart_recognition defaults to False, which is why every chart in
        # the corpus came back as <img src="imgs/img_in_chart_box_….jpg"> — the
        # layout stage labelled the chart a figure, cropped it, and the crop was
        # never returned, so that region of the page was simply lost. The model
        # supports chart parsing across 11 chart categories; we were not asking
        # for it. It fires only on regions the layout detector labels a chart,
        # so a page without one is unaffected.
        # pipeline_version="v1" pins the pipeline to the generation this MODEL_ID
        # belongs to: PaddleOCR-VL v1 = PP-DocLayoutV2 + PaddleOCR-VL-0.9B. The
        # paddleocr package now DEFAULTS to v1.6, so without this the unpinned
        # dependency silently pairs the v1 model with newer pipeline components
        # and the run stops being a reproducible "PaddleOCR-VL v1" result.
        # NOTE: paddleocr[doc-parser] is still unpinned in the image — pin it to
        # an exact version before publishing any number from this engine.
        kwargs = {
            "pipeline_version": "v1",
            "vl_rec_backend": "vllm-server",
            "vl_rec_server_url": VLLM_URL,
            "use_chart_recognition": True,
            # v1's FULL pipeline has always had an optional seal sub-pipeline
            # (use_seal_recognition / seal_res_list), independent of the seal
            # capability the 1.5 VLM later gained. This row benchmarks the
            # pipeline, so the capability is enabled globally and Paddle decides
            # whether a seal is present — enabling it only for the document we
            # know has one would give the benchmark oracle knowledge.
            "use_seal_recognition": True,
            # Benchmark policy: documented defaults, left off. Turning these on
            # is a defensible ALTERNATIVE configuration ("Paddle with full
            # preprocessing"), but it must then apply to the whole corpus, not
            # just the photographed pages.
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
        }
        last_exc = None
        for attempt in range(3):
            try:
                # Layout detection on the GPU. No CPU stage anywhere in this app.
                try:
                    self._pipeline = PaddleOCRVL(device="gpu", **kwargs)
                except TypeError:
                    self._pipeline = PaddleOCRVL(**kwargs)
                break
            except Exception as exc:
                last_exc = exc
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"PaddleOCRVL init failed: {last_exc}")

        # Only half the pipeline is pinned: MODEL_REVISION covers the 0.9B VLM
        # that vLLM serves, but the layout model (PP-DocLayoutV2 for v1) is
        # resolved independently by PaddleX's downloader and can drift. Record
        # whatever actually loaded so the run says which layout model produced
        # it rather than relying on the docs.
        self._layout_info = "unknown"
        try:
            for attr in ("layout_detection_model", "_layout_detection_model",
                         "layout_model", "doc_layout_model"):
                model = getattr(self._pipeline, attr, None)
                if model is not None:
                    self._layout_info = str(
                        getattr(model, "model_name", None)
                        or getattr(model, "model_dir", None)
                        or type(model).__name__
                    )
                    break
        except Exception:
            pass

        # Warm both stages so the first real document doesn't carry compilation.
        from PIL import Image as _PILImage
        import tempfile
        from pathlib import Path as _Path
        try:
            with tempfile.TemporaryDirectory() as td:
                p = _Path(td) / "warm.png"
                _PILImage.new("RGB", (1024, 1024), (255, 255, 255)).save(p)
                list(self._pipeline.predict(str(p)))
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
        """Run PaddleOCR-VL's documented list-input queued pipeline on one A100."""
        import tempfile
        import time
        from pathlib import Path

        started = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory() as td:
                paths = []
                for index, item in enumerate(files):
                    mime_type = item.get("mime_type", "image/png")
                    ext = ".jpg" if "jpeg" in mime_type else ".png"
                    path = Path(td) / f"{index:04d}{ext}"
                    path.write_bytes(item["bytes"])
                    paths.append(str(path))

                rendered_results = list(
                    self._pipeline.predict(paths, use_queues=True)
                )
                if len(rendered_results) != len(files):
                    raise RuntimeError(
                        f"PaddleOCR-VL returned {len(rendered_results)} results "
                        f"for {len(files)} inputs"
                    )
                results = []
                for (item, path), rendered in zip(zip(files, paths), rendered_results):
                    response = self._result_to_response(rendered, _page_size(path))
                    response["document_path"] = item.get("document_path")
                    results.append(response)
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
            startup = getattr(self, "_startup_gpu_seconds", 0.0)
            self._startup_cost_applied = True
            return startup, True
        return 0.0, False

    def _error_response(self, exc: Exception) -> dict:
        return {
            "raw_text": "",
            "structured_output": {},
            "gpu_seconds": 0.0,
            "gpu_type": GPU_TYPE,
            "model_version": f"{MODEL_ID} pipeline (PP-DocLayoutV3 + vLLM, GPU)",
            "error": f"{exc.__class__.__name__}: {str(exc) or repr(exc)}",
        }

    def _run_ocr_impl(self, image_bytes: bytes, mime_type: str = "image/png") -> dict:
        import tempfile
        import time
        from pathlib import Path

        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as td:
            ext = ".jpg" if "jpeg" in mime_type else ".png"
            p = Path(td) / f"input{ext}"
            p.write_bytes(image_bytes)

            rendered_results = list(self._pipeline.predict(str(p)))
            if len(rendered_results) != 1:
                raise RuntimeError(
                    f"PaddleOCR-VL returned {len(rendered_results)} results for one input"
                )
            response = self._result_to_response(rendered_results[0], _page_size(str(p)))

        response["gpu_seconds"] = time.perf_counter() - t0
        return response

    @staticmethod
    def _dig(res, *keypaths):
        """First of `keypaths` that resolves in res.json, else None."""
        for keypath in keypaths:
            try:
                node = res.json
                for key in keypath:
                    node = node[key]
                return node
            except Exception:
                continue
        return None

    def _result_to_response(self, res, page_size=(0, 0)) -> dict:
        """Keep everything the ONE page call already produced.

        The pipeline returns a rich result object; this used to reduce it to a
        markdown string plus a block count, so tables were re-scraped out of
        assembled markdown and the geometry the pipeline already computed was
        thrown away. Nothing here adds an inference call — it stops discarding
        the output of the call we already pay for.
        """
        md = res.markdown
        text = md.get("markdown_texts", "") if isinstance(md, dict) else str(md)

        structured: dict = {}

        # --- layout blocks: label, content, bbox, reading order ---------------
        page_w, page_h = page_size
        parsing = self._dig(res, ("res", "parsing_res_list"), ("parsing_res_list",))
        if isinstance(parsing, list):
            blocks = []
            for block in parsing:
                if not isinstance(block, dict):
                    continue
                raw_box = block.get("block_bbox")
                blocks.append({
                    "label": block.get("block_label"),
                    # Paddle reports PIXELS here; the scorer expects 0-1. These
                    # were being stored raw, so every box read as e.g.
                    # [1, 3, 766, 1026] and could never match.
                    "bbox": _normalise_box(raw_box, page_w, page_h),
                    "bbox_px": raw_box,
                    "order": block.get("block_order"),
                    "content": block.get("block_content"),
                })
            if blocks:
                structured["layout_blocks"] = blocks

        # --- tables: the pipeline's own table result, not scraped markdown ----
        # table_res_list carries pred_html per table, which is the table the
        # model actually recognised. Falling back to table-labelled parsing
        # blocks keeps older pipeline versions working.
        tables = []
        table_source = None
        table_res = self._dig(res, ("res", "table_res_list"), ("table_res_list",))
        if isinstance(table_res, list):
            for table in table_res:
                if not isinstance(table, dict):
                    continue
                html = table.get("pred_html")
                if html:
                    tables.append({"html": str(html),
                                   "cell_boxes": table.get("cell_box_list")})
        if tables:
            table_source = "table_res_list"
        # paddleocr 3.7 has a reported regression where table_res_list comes back
        # empty while parsing_res_list still carries the table HTML (seen on
        # v1.6; unverified on v1). The fallback covers it, and table_source
        # records which path actually produced the table so this can be checked
        # from the saved output instead of assumed from docs.
        if not tables and isinstance(parsing, list):
            for block in parsing:
                if isinstance(block, dict) and str(block.get("block_label")) == "table":
                    tables.append({"html": str(block.get("block_content", ""))})
            if tables:
                table_source = "parsing_res_list"
        if tables:
            structured["tables"] = tables
            structured["table_source"] = table_source

        # --- geometry: the pipeline's global OCR result -----------------------
        # rec_texts paired with rec_polys/rec_boxes is native word/line geometry.
        # This is why paddleocr_vl was wrongly marked ineligible for that leg.
        # overall_ocr_res is the pipeline's global OCR sub-result: rec_texts
        # paired with rec_boxes/rec_polys, i.e. text-LINE geometry, which is the
        # closest native match to the corpus's word/line references. Two fixed
        # key paths missed it entirely on the v1 pipeline, so the geometry leg
        # silently fell back to layout blocks. Search the result for it instead
        # of assuming where it lives, and record where it was found.
        ocr = self._dig(res, ("res", "overall_ocr_res"), ("overall_ocr_res",))
        if not isinstance(ocr, dict):
            def _find(node, depth=0):
                if depth > 4 or not isinstance(node, dict):
                    return None
                if "rec_texts" in node:
                    return node
                for value in node.values():
                    if isinstance(value, dict):
                        found = _find(value, depth + 1)
                        if found:
                            return found
                return None
            try:
                ocr = _find(res.json)
                if ocr:
                    structured["ocr_source"] = "searched"
            except Exception:
                ocr = None
        if isinstance(ocr, dict):
            rec_texts = ocr.get("rec_texts") or []
            boxes = ocr.get("rec_boxes") or ocr.get("rec_polys") or ocr.get("dt_polys") or []
            page_w, page_h = page_size
            # Paddle calls these text-recognition results and detection boxes,
            # NOT words: one rec_texts entry can be a whole detected line or
            # span ("Total amount due") under a single box. Named accordingly so
            # nothing downstream mistakes them for word-level geometry — the
            # scorer's line-merging path is the right comparison against
            # word-level GT. The other native levels are layout_blocks[*].bbox
            # and, per table, cell_boxes.
            regions = []
            for span, box in zip(rec_texts, boxes):
                flat = _normalise_box(box, page_w, page_h)
                if flat and str(span).strip():
                    regions.append({"text": str(span), "bbox": flat})
            if regions:
                structured["text_regions"] = regions
                structured["geometry_granularity"] = "text_line_or_span"
            structured["ocr_texts"] = list(rec_texts)

        # DIAGNOSTIC: the shape of res.json, so the extractors above are written
        # against what the pipeline really returns instead of what the docs
        # imply. Remove once the key paths are confirmed.
        try:
            def _shape(node, depth=0):
                if depth > 2:
                    return type(node).__name__
                if isinstance(node, dict):
                    return {k: _shape(v, depth + 1) for k, v in list(node.items())[:40]}
                if isinstance(node, list):
                    return [f"list[{len(node)}]",
                            _shape(node[0], depth + 1) if node else None]
                return type(node).__name__
            structured["_debug_result_shape"] = _shape(res.json)
        except Exception as exc:
            structured["_debug_result_shape"] = f"{type(exc).__name__}: {exc}"

        return {
            "raw_text": text,
            "structured_output": structured,
            "gpu_type": GPU_TYPE,
            "model_version": (
                f"{MODEL_ID}@{MODEL_REVISION[:12]} pipeline v1 "
                f"(layout={getattr(self, '_layout_info', 'unknown')}, "
                f"vLLM, GPU, queued batch)"
            ),
        }
