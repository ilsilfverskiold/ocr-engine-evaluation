from __future__ import annotations

import base64
import os
from pathlib import Path

from runners.base import OCRResult, OCRRunner, OCRTask, provider_payload

COST_PER_PAGE = 0.004        # $4.00 / 1000 pages (OCR)
COST_PER_PAGE_ANNOTATION = 0.005  # $5.00 / 1000 pages (annotations / schema extraction)
MODEL_ID = "mistral-ocr-4-0"


class MistralOCRRunner(OCRRunner):
    """Mistral OCR API (mistral-ocr-4-0)."""

    name = "mistral_ocr_4"
    supports_pdf = True
    accepts_task = True  # schema leg only, via the OCR API's Annotations feature

    def __init__(self) -> None:
        from mistralai.client import Mistral
        self._client = Mistral(api_key=os.environ["MISTRAL_API_KEY"], retry_config=None)

    def run(self, document_path: Path, task: OCRTask | None = None) -> OCRResult:
        """No free-text prompt exists on this endpoint. The schema leg is
        answered through the OCR API's own Annotations feature
        (document_annotation_format + a JSON schema) — the vendor's documented
        structured-extraction path, not a prompt we invented. Any other task
        is ignored: markdown and tables come from the default conversion.

        UNVALIDATED: exercise on 2 documents before any full run."""
        t0 = self._timer()

        schema = task.json_schema if (task and task.leg == "schema" and task.json_schema) else None
        suffix = document_path.suffix.lower()
        if suffix == ".pdf":
            result = self._run_pdf(document_path, schema)
        else:
            result = self._run_image(document_path, schema)

        elapsed = self._elapsed(t0)
        result.latency_seconds = elapsed
        return result

    @staticmethod
    def _annotation_kwargs(schema: dict | None) -> dict:
        if not schema:
            return {}
        return {
            "document_annotation_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "document_extraction",
                    "schema": schema,
                    "strict": False,
                },
            }
        }

    def _run_pdf(self, path: Path, schema: dict | None = None) -> OCRResult:
        raw_bytes = path.read_bytes()
        b64 = base64.b64encode(raw_bytes).decode()

        response = self._client.ocr.process(
            model=MODEL_ID,
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{b64}",
            },
            **self._annotation_kwargs(schema),
        )
        return self._parse_response(response, path, annotated=bool(schema))

    def _run_image(self, path: Path, schema: dict | None = None) -> OCRResult:
        raw_bytes = path.read_bytes()
        b64 = base64.b64encode(raw_bytes).decode()
        mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"

        response = self._client.ocr.process(
            model=MODEL_ID,
            document={
                "type": "image_url",
                "image_url": f"data:{mime};base64,{b64}",
            },
            **self._annotation_kwargs(schema),
        )
        return self._parse_response(response, path, annotated=bool(schema))

    def _parse_response(self, response, path: Path, annotated: bool = False) -> OCRResult:
        pages = getattr(response, "pages", []) or []
        text_parts = []
        for page in pages:
            text_parts.append(getattr(page, "markdown", "") or "")
        raw_text = "\n\n".join(text_parts)
        page_count = len(pages) or 1

        if annotated:
            # The Annotations response carries the extraction as
            # document_annotation (a JSON string). For the schema leg that JSON
            # IS the answer, so it becomes the scored output.
            annotation = getattr(response, "document_annotation", None)
            if annotation:
                raw_text = annotation if isinstance(annotation, str) else str(annotation)

        return OCRResult(
            engine_name=self.name,
            document_path=str(path),
            raw_text=raw_text,
            structured_output=provider_payload(response),
            latency_seconds=0.0,  # filled by caller
            cost_per_page=COST_PER_PAGE_ANNOTATION if annotated else COST_PER_PAGE,
            page_count=page_count,
            model_version=MODEL_ID,
        )
