from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from modal_apps._shared import GPU_SECOND_RATES_USD, calculate_modal_cost
from runners.base import OCRResult, OCRRunner, OCRTask

# Registry-derived Modal configuration. Response GPU type overrides the registry cost fallback.
from engines import ENGINES as _ENGINE_SPECS

MODAL_ENGINES: dict[str, tuple[str, str, str, bool]] = {
    name: (spec.modal_app, spec.modal_class, spec.gpu_type, spec.supports_pdf)
    for name, spec in _ENGINE_SPECS.items()
    if spec.kind == "modal"
}

# Measure one batch on one container/GPU and amortize its time across documents.
# Batch size comes from EngineSpec; queue time and startup are recorded separately.
MODAL_BATCH_SIZES: dict[str, int] = {
    name: spec.batch_size
    for name, spec in _ENGINE_SPECS.items()
    if spec.kind == "modal" and spec.batch_size > 0
}

# Kept as a public name for compatibility with older scripts.  Benchmark execution
# no longer fans requests out from the local machine; batching happens inside the
# one measured Modal method call.
MODAL_CONCURRENCY: dict[str, int] = {}


class ModalRunner(OCRRunner):
    """Generic runner that calls a deployed Modal app and converts the response to OCRResult."""

    def __init__(self, engine_key: str) -> None:
        if engine_key not in MODAL_ENGINES:
            raise ValueError(
                f"Unknown Modal engine '{engine_key}'. "
                f"Available: {list(MODAL_ENGINES)}"
            )
        app_name, class_name, gpu_type, pdf_support = MODAL_ENGINES[engine_key]
        self.name = engine_key
        self.accepts_task = _ENGINE_SPECS[engine_key].accepts_task
        self._app_name = app_name
        self._class_name = class_name
        self._gpu_type = gpu_type
        self.supports_pdf = pdf_support
        self.batch_size = MODAL_BATCH_SIZES.get(engine_key, 0)
        self.supports_batch = engine_key in MODAL_BATCH_SIZES
        self.concurrency = MODAL_CONCURRENCY.get(engine_key, 1)
        self.supports_concurrency = self.concurrency > 1
        self._cls = None  # lazy lookup

    def _get_cls(self):
        if self._cls is None:
            import modal
            self._cls = modal.Cls.from_name(self._app_name, self._class_name)
        return self._cls

    @staticmethod
    def _mime_type(document_path: Path) -> str:
        suffix = document_path.suffix.lower()
        return (
            "application/pdf" if suffix == ".pdf"
            else "image/jpeg" if suffix in (".jpg", ".jpeg")
            else "image/png"
        )

    def _to_result(self, document_path: Path, response: dict) -> OCRResult:
        gpu_seconds = response.get("gpu_seconds", 0.0)
        # The app reports the GPU it actually ran on; trust that over the registry
        # fallback so changing GPU_TYPE in one place cannot mis-bill a whole run.
        gpu_type = response.get("gpu_type") or self._gpu_type
        cost = calculate_modal_cost(gpu_seconds, gpu_type)

        return OCRResult(
            engine_name=self.name,
            document_path=str(document_path),
            raw_text=response.get("raw_text", ""),
            structured_output=response.get("structured_output", {}),
            latency_seconds=gpu_seconds,
            cost_per_page=cost,
            page_count=1,
            model_version=response.get("model_version", self.name),
            metadata={
                "gpu_type": gpu_type,
                "gpu_seconds": gpu_seconds,
                "startup_gpu_seconds": response.get("startup_gpu_seconds", 0.0),
                "is_cold_start": response.get("is_cold_start", False),
            },
            # The app's whole reply. raw_text/structured_output are the two keys
            # we read; an app that also returns a finish reason, a per-page
            # breakdown or its own error string keeps it here rather than
            # having it dropped on the floor.
            raw_response=dict(response) if isinstance(response, dict) else {},
            error=response.get("error"),
        )

    def _task_prompt(self, task: OCRTask | None) -> str | None:
        """The prompt to send, or None to keep the app's built-in default.

        Sending None is what makes leg 1 byte-identical after prompts became
        per-request parameters: the app falls back to its own constant.
        """
        if task is None:
            return None
        if not self.accepts_task:
            raise ValueError(
                f"{self.name} is a fixed pipeline and cannot be given a task "
                f"(leg '{task.leg}'). engines.py declares accepts_task=False."
            )
        return task.prompt

    def run(self, document_path: Path, task: OCRTask | None = None) -> OCRResult:
        cls = self._get_cls()
        raw_bytes = document_path.read_bytes()
        mime_type = self._mime_type(document_path)
        prompt = self._task_prompt(task)

        response_format = self._response_format(task)
        if prompt is None:
            response = cls().run_ocr.remote(raw_bytes, mime_type)
        elif response_format is None:
            response = cls().run_ocr.remote(raw_bytes, mime_type, prompt)
        else:
            response = cls().run_ocr.remote(raw_bytes, mime_type, prompt,
                                            response_format)
        return self._to_result(document_path, response)

    @staticmethod
    def _response_format(task: OCRTask | None) -> dict | None:
        """The provider constrained-decoding directive for this task."""
        if task is None or not task.wants_json:
            return None
        if task.json_schema:
            return {"type": "json_schema",
                    "json_schema": {"name": "extraction",
                                    "schema": task.json_schema}}
        return {"type": "json_object"}

    def run_batch(
        self, document_paths: list[Path], task: OCRTask | None = None
    ) -> dict[str, OCRResult]:
        cls = self._get_cls()
        batch_id = uuid4().hex
        prompt = self._task_prompt(task)
        files = [
            {
                "document_path": str(path),
                "bytes": path.read_bytes(),
                "mime_type": self._mime_type(path),
            }
            for path in document_paths
        ]

        response_format = self._response_format(task)
        if prompt is None:
            batch_response = cls().run_ocr_batch.remote(files)
        elif response_format is None:
            batch_response = cls().run_ocr_batch.remote(files, prompt)
        else:
            batch_response = cls().run_ocr_batch.remote(files, prompt,
                                                        response_format)
        if not isinstance(batch_response, dict):
            raise TypeError(
                f"{self.name}.run_ocr_batch returned {type(batch_response).__name__}; "
                "expected the batch timing envelope"
            )

        responses = batch_response.get("results")
        if not isinstance(responses, list):
            raise TypeError(f"{self.name}.run_ocr_batch response has no results list")
        if len(responses) != len(document_paths):
            raise ValueError(
                f"{self.name}.run_ocr_batch returned {len(responses)} results "
                f"for {len(document_paths)} inputs"
            )

        batch_gpu_seconds = float(batch_response.get("batch_gpu_seconds") or 0.0)
        if batch_gpu_seconds <= 0:
            raise ValueError(
                f"{self.name}.run_ocr_batch returned invalid batch_gpu_seconds="
                f"{batch_gpu_seconds!r}"
            )
        # One GPU is reserved for the full measured method interval.  Dividing that
        # interval by all attempted documents gives both effective seconds/document
        # and billable GPU-seconds/document.  Startup/model load is outside the method
        # timer and is therefore never included in either value.
        batch_size = len(document_paths)
        gpu_seconds_per_document = batch_gpu_seconds / batch_size
        gpu_type = batch_response.get("gpu_type") or self._gpu_type
        startup_gpu_seconds = float(batch_response.get("startup_gpu_seconds") or 0.0)
        is_cold_start = bool(batch_response.get("is_cold_start"))

        results = {}
        for path, response in zip(document_paths, responses):
            response = dict(response)
            response["gpu_seconds"] = gpu_seconds_per_document
            response["gpu_type"] = gpu_type
            response["startup_gpu_seconds"] = startup_gpu_seconds
            response["is_cold_start"] = is_cold_start
            result_path = Path(response.get("document_path") or path)
            result = self._to_result(result_path, response)
            result.metadata.update({
                "measurement_scope": "modal_batch",
                "batch_id": batch_id,
                "batch_gpu_seconds": batch_gpu_seconds,
                "batch_size": batch_size,
                "effective_gpu_seconds_per_document": gpu_seconds_per_document,
                "gpu_rate_usd_per_second": GPU_SECOND_RATES_USD[gpu_type.upper()],
                "startup_excluded": True,
            })
            results[str(path)] = result
        return results

    def prepare_item(
        self, document_path: Path, task: OCRTask | None = None
    ) -> dict:
        """Build a file dict for a tagged batch, with per-item prompt and
        response_format baked in so all passes can go in one remote call."""
        item: dict = {
            "document_path": str(document_path),
            "bytes": document_path.read_bytes(),
            "mime_type": self._mime_type(document_path),
        }
        if task is not None:
            prompt = self._task_prompt(task)
            if prompt is not None:
                item["prompt"] = prompt
            fmt = self._response_format(task)
            if fmt is not None:
                item["response_format"] = fmt
        return item

    def run_tagged_batch(self, items: list[dict]) -> list[OCRResult]:
        """Run a batch where each item carries its own prompt and response_format.

        Unlike run_batch (one task for all documents), this accepts pre-built
        file dicts so the harness can merge all passes into one container call.
        Returns results as a list in the same order as the input items.
        """
        cls = self._get_cls()
        batch_id = uuid4().hex

        batch_response = cls().run_ocr_batch.remote(items)
        if not isinstance(batch_response, dict):
            raise TypeError(
                f"{self.name}.run_ocr_batch returned "
                f"{type(batch_response).__name__}; expected the batch timing "
                f"envelope"
            )

        responses = batch_response.get("results")
        if not isinstance(responses, list):
            raise TypeError(
                f"{self.name}.run_ocr_batch response has no results list"
            )
        if len(responses) != len(items):
            raise ValueError(
                f"{self.name}.run_ocr_batch returned {len(responses)} results "
                f"for {len(items)} inputs"
            )

        batch_gpu_seconds = float(
            batch_response.get("batch_gpu_seconds") or 0.0
        )
        if batch_gpu_seconds <= 0:
            raise ValueError(
                f"{self.name}.run_ocr_batch returned invalid "
                f"batch_gpu_seconds={batch_gpu_seconds!r}"
            )
        n_items = len(items)
        gpu_seconds_per_item = batch_gpu_seconds / n_items
        gpu_type = batch_response.get("gpu_type") or self._gpu_type
        startup_gpu_seconds = float(
            batch_response.get("startup_gpu_seconds") or 0.0
        )
        is_cold_start = bool(batch_response.get("is_cold_start"))

        results: list[OCRResult] = []
        for item, response in zip(items, responses):
            response = dict(response)
            response["gpu_seconds"] = gpu_seconds_per_item
            response["gpu_type"] = gpu_type
            response["startup_gpu_seconds"] = startup_gpu_seconds
            response["is_cold_start"] = is_cold_start
            result_path = Path(
                response.get("document_path") or item.get("document_path", "")
            )
            result = self._to_result(result_path, response)
            result.metadata.update({
                "measurement_scope": "modal_batch",
                "batch_id": batch_id,
                "batch_gpu_seconds": batch_gpu_seconds,
                "batch_size": n_items,
                "effective_gpu_seconds_per_document": gpu_seconds_per_item,
                "gpu_rate_usd_per_second": GPU_SECOND_RATES_USD[
                    gpu_type.upper()
                ],
                "startup_excluded": True,
            })
            results.append(result)
        return results
