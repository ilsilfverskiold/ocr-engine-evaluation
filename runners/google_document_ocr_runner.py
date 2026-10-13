"""Google Document AI — Enterprise Document OCR processor (OCR_PROCESSOR).

Online `process_document` through google-cloud-documentai, authenticated with
Application Default Credentials / a service account (a Gemini API key does not
work here). This row is the OCR processor only: Form Parser and Layout Parser
are different products and must not be substituted under this name, so the
runner checks the processor type when it can read it.

Environment:
  GOOGLE_DOCAI_PROJECT, GOOGLE_DOCAI_LOCATION ("us" / "eu" / ...),
  GOOGLE_DOCAI_PROCESSOR_ID                         — required
  GOOGLE_DOCAI_PROCESSOR_VERSION                    — optional but strongly
      preferred, e.g. "pretrained-ocr-v2.1-2024-08-07" (Stable). Without it the
      processor's default version is resolved once via get_processor and then
      requested explicitly, so every document in a run uses one version.

Request options (docs.cloud.google.com/document-ai/docs/enterprise-document-ocr,
REST ProcessOptions/OcrConfig reference, google-cloud-documentai 3.15.0
types/document_io.py; all read 2026-09-30) — every OCR option is set
explicitly so the configuration is recorded, not implied:
  enable_native_pdf_parsing=False   PDFs only; the corpus is images.
  enable_image_quality_scores=False "Adds additional latency comparable to
                                    regular OCR"; diagnostics, not text.
  enable_symbol=False               symbol boxes not needed (tokens suffice).
  disable_character_boxes_detection=False  documented default (detector on).
  hints / advanced_ocr_options      unset: automatic language, ML layout.
  premium_features: compute_style_info=False, enable_math_ocr=False,
      enable_selection_mark_detection=False — the "OCR add ons", billed
      separately ($6/1,000 pages on the pricing page), OFF by default.
  imageless_mode=True               "Option to remove images from the
                                    document" — drops the rendered page image
                                    from the reply; no effect on OCR output.

Output mapping (REST Document reference):
  raw_text                 = Document.text (the service's reading order)
  structured_output.words  = pages[].tokens — "A detected token"; word-level
                             boxes from layout.bounding_poly.normalized_vertices
                             ("relative to the original image and range from
                             0 to 1"), text via layout.text_anchor.
  structured_output.lines  = pages[].lines, kept as metadata only.
No markdown, table, key_value or schema output exists on this processor.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from runners.base import OCRResult, OCRRunner, OCRTask, provider_payload

# cloud.google.com/document-ai/pricing, read 2026-09-30: Enterprise Document OCR
# Processor "0 count to 1,000 count $0.00 (Free); 1,000 count to 5,000,000 count
# $1.50; 5,000,000 count and above $0.60" per 1,000 pages. The per-page figure
# below is the paid-tier list rate; an unused monthly free allowance would make
# the invoice lower. Add-ons ($6.00/1,000) are disabled.
COST_PER_PAGE = 0.0015


def _layout_text(layout: Any, text: str) -> str:
    """Text of a layout element from its text_anchor segments (sample code in
    the Enterprise OCR docs: layout_to_text)."""
    anchor = getattr(layout, "text_anchor", None)
    segments = getattr(anchor, "text_segments", None) or []
    return "".join(text[int(s.start_index): int(s.end_index)] for s in segments)


def _layout_box(layout: Any, page_w: float, page_h: float) -> tuple[list[float] | None, list | None]:
    """[x0,y0,x1,y1] normalised 0-1 plus the original polygon.

    normalized_vertices are used when present. Pixel `vertices` are only a
    fallback, normalised by the page dimension (unit reported alongside)."""
    poly = getattr(layout, "bounding_poly", None)
    if poly is None:
        return None, None
    nverts = list(getattr(poly, "normalized_vertices", None) or [])
    if nverts:
        pts = [[float(v.x), float(v.y)] for v in nverts]
    else:
        verts = list(getattr(poly, "vertices", None) or [])
        if not verts or not page_w or not page_h:
            return None, None
        pts = [[float(v.x) / page_w, float(v.y) / page_h] for v in verts]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)], pts


class GoogleDocumentOCRRunner(OCRRunner):
    """Document AI Enterprise Document OCR, online processing."""

    name = "google_document_ocr"
    supports_pdf = False   # PDF path not built; corpus is single-page images
    accepts_task = False   # fixed OCR processor: no prompt, no schema

    def __init__(self) -> None:
        from google.api_core.client_options import ClientOptions
        from google.cloud import documentai

        self._documentai = documentai
        self._project = os.environ["GOOGLE_DOCAI_PROJECT"]
        self._location = os.environ["GOOGLE_DOCAI_LOCATION"]
        self._processor_id = os.environ["GOOGLE_DOCAI_PROCESSOR_ID"]
        self._requested_version = os.environ.get("GOOGLE_DOCAI_PROCESSOR_VERSION") or None
        # The sample sets a regional endpoint: "You must set the api_endpoint
        # if you use a location other than 'us'."
        self._client = documentai.DocumentProcessorServiceClient(
            client_options=ClientOptions(
                api_endpoint=f"{self._location}-documentai.googleapis.com"
            )
        )
        self._resource: str | None = None
        self._processor_info: dict[str, Any] = {}
        self._process_options = documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(
                enable_native_pdf_parsing=False,
                enable_image_quality_scores=False,
                enable_symbol=False,
                disable_character_boxes_detection=False,
                premium_features=documentai.OcrConfig.PremiumFeatures(
                    compute_style_info=False,
                    enable_math_ocr=False,
                    enable_selection_mark_detection=False,
                ),
            )
        )

    # -------------------------------------------------------------- resolving
    def _resolve_resource(self) -> str:
        """Pin one processor-version resource for the whole run and check that
        the processor really is the OCR processor (not Form/Layout Parser)."""
        if self._resource:
            return self._resource
        client = self._client
        processor_name = client.processor_path(
            self._project, self._location, self._processor_id)
        info: dict[str, Any] = {"processor": processor_name}
        try:
            processor = client.get_processor(name=processor_name)
            info.update({
                "processor_type": processor.type_,
                "processor_display_name": processor.display_name,
                "default_processor_version": processor.default_processor_version,
            })
            if processor.type_ and processor.type_ != "OCR_PROCESSOR":
                raise ValueError(
                    f"{self.name}: processor {self._processor_id} is "
                    f"{processor.type_}, not OCR_PROCESSOR. Form Parser / Layout "
                    f"Parser must not be benchmarked under this engine name."
                )
        except ValueError:
            raise
        except Exception as exc:  # e.g. missing documentai.processors.get
            info["processor_lookup_error"] = f"{exc.__class__.__name__}: {exc}"

        if self._requested_version:
            resource = client.processor_version_path(
                self._project, self._location, self._processor_id,
                self._requested_version)
            info["version_source"] = "GOOGLE_DOCAI_PROCESSOR_VERSION"
        elif info.get("default_processor_version"):
            resource = info["default_processor_version"]
            info["version_source"] = "processor default, resolved once at start"
        else:
            # Unpinned: the server uses the processor's default version, which
            # can change between runs. Recorded so it is never mistaken for pinned.
            resource = processor_name
            info["version_source"] = "UNPINNED processor default (could not resolve)"
        try:
            if "/processorVersions/" in resource:
                version = client.get_processor_version(name=resource)
                info.update({
                    "processor_version_display_name": version.display_name,
                    "processor_version_state": str(version.state),
                    "processor_version_model_type": str(getattr(version, "model_type", "")),
                })
        except Exception as exc:
            info["processor_version_lookup_error"] = f"{exc.__class__.__name__}: {exc}"
        self._processor_info = info
        self._resource = resource
        return resource

    # -------------------------------------------------------------------- run
    def run(self, document_path: Path, task: OCRTask | None = None) -> OCRResult:
        if task is not None:
            raise ValueError(f"{self.name} is a fixed OCR processor and takes no task "
                             f"(got leg {task.leg!r})")
        if document_path.suffix.lower() == ".pdf":
            raise ValueError(f"{self.name}: PDF input not implemented (corpus is images)")
        documentai = self._documentai
        resource = self._resolve_resource()
        mime = ("image/jpeg" if document_path.suffix.lower() in (".jpg", ".jpeg")
                else "image/png")

        t0 = self._timer()
        request = documentai.ProcessRequest(
            name=resource,
            raw_document=documentai.RawDocument(
                content=document_path.read_bytes(), mime_type=mime),
            process_options=self._process_options,
            imageless_mode=True,
        )
        result = self._client.process_document(request=request)
        elapsed = self._elapsed(t0)
        document = result.document
        text = document.text or ""

        words: list[dict] = []
        lines: list[dict] = []
        page_meta: list[dict] = []
        for page in document.pages:
            dim = page.dimension
            pw, ph = float(dim.width or 0), float(dim.height or 0)
            for token in page.tokens:
                bbox, poly = _layout_box(token.layout, pw, ph)
                token_text = _layout_text(token.layout, text).strip()
                if bbox and token_text:
                    words.append({"text": token_text, "bbox": bbox, "poly": poly,
                                  "confidence": float(token.layout.confidence),
                                  "page": page.page_number})
            for line in page.lines:
                bbox, poly = _layout_box(line.layout, pw, ph)
                line_text = _layout_text(line.layout, text).strip()
                if bbox and line_text:
                    lines.append({"text": line_text, "bbox": bbox, "poly": poly,
                                  "confidence": float(line.layout.confidence)})
            page_meta.append({
                "page_number": page.page_number,
                "dimension": {"width": dim.width, "height": dim.height, "unit": dim.unit},
                "n_tokens": len(page.tokens), "n_lines": len(page.lines),
                "n_blocks": len(page.blocks), "n_paragraphs": len(page.paragraphs),
                # Non-empty transforms mean the service deskewed/rotated the page.
                # normalized_vertices are documented as relative to the ORIGINAL
                # image; the first run must confirm that on a skewed page.
                "n_transforms": len(page.transforms),
                "detected_languages": [
                    {"code": lang.language_code, "confidence": lang.confidence}
                    for lang in page.detected_languages
                ],
            })

        # The processor that actually produced the document, if the reply names
        # it (Document.revisions[].processor — "identify the processor by its
        # resource name"). Otherwise the resource we pinned and requested.
        returned_processors = [r.processor for r in document.revisions if r.processor]
        version_used = returned_processors[0] if returned_processors else resource
        version_id = version_used.rsplit("/processorVersions/", 1)[-1] \
            if "/processorVersions/" in version_used else version_used
        model_version = (f"documentai-enterprise-ocr {version_id}"
                         + ("" if returned_processors else " (requested)"))

        structured: dict[str, Any] = {"pages": page_meta}
        if words:
            structured["words"] = words
            structured["geometry_granularity"] = "word"
            structured["geometry_coordinates"] = "normalized_xyxy"
        if lines:
            structured["lines"] = lines

        try:
            payload = type(result).to_dict(result)
        except Exception:
            payload = provider_payload(result)

        page_count = len(document.pages) or 1
        return OCRResult(
            engine_name=self.name,
            document_path=str(document_path),
            raw_text=text,
            structured_output=structured,
            latency_seconds=elapsed,
            cost_per_page=COST_PER_PAGE * page_count,
            page_count=page_count,
            model_version=model_version,
            metadata={
                **self._processor_info,
                "processor_version_requested": resource,
                "processor_version_returned": returned_processors or None,
                "process_options": {
                    "enable_native_pdf_parsing": False,
                    "enable_image_quality_scores": False,
                    "enable_symbol": False,
                    "disable_character_boxes_detection": False,
                    "premium_features": "all off",
                    "imageless_mode": True,
                },
                "billing_basis": "pages x $1.50/1,000 list rate (free first 1,000/month ignored)",
            },
            # provider_payload applies the shared size cap to the converted dict.
            raw_response=provider_payload(payload),
        )
