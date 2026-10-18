from __future__ import annotations

import base64
import os
from pathlib import Path

from runners.base import OCRResult, OCRRunner, OCRTask, provider_payload, sent_record

COST_PER_PAGE = 0.004        # $4.00 / 1000 pages (OCR)
COST_PER_PAGE_ANNOTATION = 0.005  # $5.00 / 1000 pages (annotations / schema extraction)
MODEL_ID = "mistral-ocr-4-1"
# KV has no supplied schema, but document_annotation_format accepts only
# json_schema (docs.mistral.ai client-python OCR docs): an open label -> value
# object carries no reference keys. The leg prompt goes in document_annotation_prompt.
KV_ANNOTATION_SCHEMA = {"type": "object", "additionalProperties": {"type": ["string", "null"]}}


class MistralOCRRunner(OCRRunner):
    """Mistral OCR 4.1: conversion, and schema/KV through document annotations."""

    name = "mistral_ocr_4"
    supports_pdf = True
    accepts_task = True  # schema and key_value, via the OCR API's Annotations feature

    def __init__(self) -> None:
        from mistralai.client import Mistral
        self._client = Mistral(api_key=os.environ["MISTRAL_API_KEY"], retry_config=None)

    def run(self, document_path: Path, task: OCRTask | None = None) -> OCRResult:
        """Markdown and tables come from the default conversion, which takes
        no prompt. Schema and key_value use the OCR API's Annotations feature:
        schema sends the page's JSON schema only; key_value sends an open
        label -> value schema plus the leg prompt as document_annotation_prompt."""
        t0 = self._timer()

        annotation, sent = self._annotation(task)
        suffix = document_path.suffix.lower()
        if suffix == ".pdf":
            result = self._run_pdf(document_path, annotation, sent)
        else:
            result = self._run_image(document_path, annotation, sent)

        elapsed = self._elapsed(t0)
        result.latency_seconds = elapsed
        return result

    @staticmethod
    def _annotation(task: OCRTask | None) -> tuple[dict, dict]:
        """(document annotation kwargs, record of what reaches the engine) for a task."""
        if task and task.leg == "schema" and task.json_schema:
            kwargs = {"document_annotation_format": {"type": "json_schema", "json_schema": {
                "name": "document_extraction", "schema": task.json_schema, "strict": False}}}
            return kwargs, sent_record(
                None, "no instruction text: the page's JSON schema only",
                "JSON schema as document_annotation_format (json_schema, strict false)",
                schema_separately=True)
        if task and task.leg == "key_value":
            kwargs = {"document_annotation_format": {"type": "json_schema", "json_schema": {
                "name": "key_values", "schema": KV_ANNOTATION_SCHEMA, "strict": False}},
                "document_annotation_prompt": task.prompt}
            return kwargs, sent_record(
                task.prompt, "task prompt as document_annotation_prompt",
                "open label -> value JSON schema as document_annotation_format (strict false)",
                schema_separately=True)
        return {}, sent_record(None, "no instruction text: the OCR endpoint takes no prompt")

    def _run_pdf(self, path: Path, annotation: dict, sent: dict) -> OCRResult:
        raw_bytes = path.read_bytes()
        b64 = base64.b64encode(raw_bytes).decode()

        response = self._client.ocr.process(
            model=MODEL_ID,
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{b64}",
            },
            **annotation,
        )
        return self._parse_response(response, path, annotated=bool(annotation), sent=sent)

    def _run_image(self, path: Path, annotation: dict, sent: dict) -> OCRResult:
        raw_bytes = path.read_bytes()
        b64 = base64.b64encode(raw_bytes).decode()
        mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"

        response = self._client.ocr.process(
            model=MODEL_ID,
            document={
                "type": "image_url",
                "image_url": f"data:{mime};base64,{b64}",
            },
            **annotation,
        )
        return self._parse_response(response, path, annotated=bool(annotation), sent=sent)

    def _parse_response(self, response, path: Path, annotated: bool = False,
                        sent: dict | None = None) -> OCRResult:
        pages = getattr(response, "pages", []) or []
        text_parts = []
        for page in pages:
            text_parts.append(getattr(page, "markdown", "") or "")
        raw_text = "\n\n".join(text_parts)
        page_count = len(pages) or 1

        if annotated:
            # The Annotations response carries the extraction as
            # document_annotation (a JSON string). For schema and key_value that
            # JSON IS the answer, so it becomes the scored output.
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
            model_version=getattr(response, "model", None) or "unreported",
            metadata={"requested_model": MODEL_ID,
                      "sent": sent or sent_record(None, "no instruction text: the OCR endpoint takes no prompt")},
        )
