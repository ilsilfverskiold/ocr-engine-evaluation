"""Mistral OCR 4.1 (`mistral-ocr-4-1`) — a NEW engine row, not a change to
`mistral_ocr_4`.

Reuses runners/mistral_runner.py (client construction and the Annotations
request shape) without modifying it, so the historical `mistral_ocr_4` row keeps
its behaviour.

Request contract (all verified 2026-09-30):
  * model id `mistral-ocr-4-1` — docs.mistral.ai/resources/changelogs
    ("We released OCR 4.1 (`mistral-ocr-4-1`). `mistral-ocr-latest` and
    `mistral-ocr-4` now point to it.") and docs.mistral.ai/models/ocr-4-1.
  * `table_format="html"` — "Returns tables separately as HTML tables"; the page
    markdown then carries placeholders such as `[tbl-3.html](tbl-3.html)`
    (docs.mistral.ai/studio/document-processing/basic_ocr).
  * `include_blocks=True` — "Return paragraph-level bounding boxes for all
    content blocks" (docs.mistral.ai/api/endpoint/ocr; mistralai 2.9.1
    models/ocrrequest.py).
  * `extract_header=False`, `extract_footer=False` — the documented default
    ("OCR treats headers and footers as part of the main content"), set
    explicitly so header/footer text stays in the scored markdown.
  * `confidence_scores_granularity="block"` — page + block confidence
    (changelog, 2026-07-16). Word confidence is NOT word geometry: the API
    returns no word boxes, so geometry here is block/paragraph granularity.
  * Schema leg: `document_annotation_format` with `type="json_schema"` (the
    request field accepts only json_schema — mistralai 2.9.1 ocrrequest.py
    docstring), same shape as the mistral_ocr_4 runner.

Response contract (mistralai 2.9.1 models/*.py, generated from the OpenAPI):
  OCRResponse{pages, model, usage_info{pages_processed, doc_size_bytes},
  document_annotation}; OCRPageObject{index, markdown, images, tables, blocks,
  confidence_scores, header, footer, hyperlinks, dimensions{dpi,height,width}};
  OCRTableObject{id, content, format}; blocks carry type, top_left_x/y,
  bottom_right_x/y (int), content, confidence_scores and, for tables, table_id.

key_value leg: NOT offered. The OCR endpoint has no label-discovery contract.
The only extraction route is document annotation, which requires a caller-
supplied json_schema (`document_annotation_prompt` is optional and "A
document_annotation_format must be provided"). A KV answer would therefore need
a schema we invent (e.g. an open string->string object) plus our own prompt —
that is our construction, not a vendor contract. Left out; see the registry
note for the user's decision.
"""

from __future__ import annotations

import base64
import html as _html
import re
from pathlib import Path
from typing import Any

from runners.base import OCRResult, OCRTask, provider_payload
from runners.mistral_runner import MistralOCRRunner

MODEL_ID = "mistral-ocr-4-1"
# Published list prices, docs.mistral.ai/models/ocr-4-1, read 2026-09-30:
# "OCR: $4/1000 Pages", "Annotations - Structured: $5/1000 Annotated Pages".
# UNVERIFIED: whether an annotated page is billed $5 in total or $5 on top of
# the $4 OCR charge. Treated as $5 total, like the mistral_ocr_4 runner; check
# the first invoice line items after the schema pass.
COST_PER_PAGE = 0.004
COST_PER_PAGE_ANNOTATION = 0.005

# The request options sent on every call (the schema pass adds the annotation
# format on top of the identical OCR configuration).
OCR_OPTIONS: dict[str, Any] = {
    "table_format": "html",
    "include_blocks": True,
    "extract_header": False,
    "extract_footer": False,
    "confidence_scores_granularity": "block",
}

# Block types that carry no readable text of their own and must never be used
# as text geometry (figure boxes are not text boxes).
_NON_TEXT_BLOCK_TYPES = {"image"}

# `[tbl-3.html](tbl-3.html)` — a markdown link, never an image (`![...]`).
_PLACEHOLDER_RE = re.compile(r"(?<!!)\[(?P<label>[^\]\n]*)\]\((?P<target>[^)\s]+)\)")
_TAG_RE = re.compile(r"<[^>]+>")


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Attribute or key lookup across SDK models and plain dicts."""
    for name in names:
        if isinstance(obj, dict):
            if name in obj and obj[name] is not None:
                return obj[name]
        else:
            value = getattr(obj, name, None)
            if value is not None:
                return value
    return default


def _html_to_text(content: str) -> str:
    """Visible text of an HTML/markdown fragment, for geometry text matching."""
    text = _TAG_RE.sub(" ", content or "")
    return re.sub(r"\s+", " ", _html.unescape(text)).strip()


def reintegrate_tables(markdown: str, tables: list[dict]) -> tuple[str, list[str], list[str]]:
    """Replace table placeholders in page markdown with the returned table HTML.

    With table_format="html" the page markdown holds only a link placeholder
    per table; without this step text and markdown legs would lose every cell.
    Returns (markdown, placed_ids, unplaced_ids). Tables the markdown never
    references are appended at the end of the page and reported as unplaced,
    so their content is scored but the position change is visible.
    """
    by_key: dict[str, str] = {}
    for table in tables:
        table_id = str(table.get("id") or "")
        if table_id:
            by_key[table_id] = table.get("content") or ""
    placed: list[str] = []

    def _swap(match: re.Match) -> str:
        target, label = match.group("target"), match.group("label")
        # UNVERIFIED whether table ids carry the ".html" suffix the documented
        # placeholder shows; accept both spellings.
        for key in (target, label, target.rsplit(".", 1)[0], label.rsplit(".", 1)[0]):
            if key in by_key:
                placed.append(key)
                return "\n\n" + by_key[key] + "\n\n"
        return match.group(0)

    out = _PLACEHOLDER_RE.sub(_swap, markdown or "")
    unplaced = [tid for tid in by_key if tid not in placed]
    if unplaced:
        out = out.rstrip() + "\n\n" + "\n\n".join(by_key[t] for t in unplaced) + "\n"
    return out, placed, unplaced


class MistralOCR41Runner(MistralOCRRunner):
    """Mistral OCR API, model `mistral-ocr-4-1` (default conversion + blocks)."""

    name = "mistral_ocr_4_1"
    supports_pdf = False  # corpus is single-page images; PDF path not built
    accepts_task = True   # schema leg only, via Annotations

    # __init__ is inherited: Mistral(api_key=os.environ["MISTRAL_API_KEY"]).

    def run(self, document_path: Path, task: OCRTask | None = None) -> OCRResult:
        if task is not None and task.leg != "schema":
            # A silently ignored task would be a benchmark lie: the registry
            # only routes the schema leg here.
            raise ValueError(
                f"{self.name} accepts only the schema task; got leg {task.leg!r}"
            )
        if document_path.suffix.lower() == ".pdf":
            raise ValueError(f"{self.name}: PDF input not implemented (corpus is images)")
        schema = task.json_schema if task is not None else None
        if task is not None and not schema:
            raise ValueError(f"{self.name}: schema task arrived without a json_schema")

        t0 = self._timer()
        raw_bytes = document_path.read_bytes()
        mime = ("image/jpeg" if document_path.suffix.lower() in (".jpg", ".jpeg")
                else "image/png")
        b64 = base64.b64encode(raw_bytes).decode()
        response = self._client.ocr.process(
            model=MODEL_ID,
            document={"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
            **OCR_OPTIONS,
            **self._annotation_kwargs(schema),
        )
        elapsed = self._elapsed(t0)
        result = self._parse_41(response, document_path, annotated=bool(schema))
        result.latency_seconds = elapsed
        return result

    # ------------------------------------------------------------------ parse
    def _parse_41(self, response: Any, path: Path, annotated: bool) -> OCRResult:
        pages = _get(response, "pages", default=[]) or []
        page_markdown: list[str] = []
        raw_markdown: list[str] = []
        tables_out: list[dict] = []
        regions: list[dict] = []
        layout_blocks: list[dict] = []
        page_meta: list[dict] = []
        unplaced_all: list[str] = []

        for page in pages:
            md = _get(page, "markdown", default="") or ""
            raw_markdown.append(md)
            tables = []
            for table in _get(page, "tables", default=[]) or []:
                tables.append({
                    "id": _get(table, "id"),
                    "content": _get(table, "content", default=""),
                    "format": _get(table, "format_", "format"),
                })
            merged, placed, unplaced = reintegrate_tables(md, tables)
            page_markdown.append(merged)
            unplaced_all.extend(unplaced)
            for table in tables:
                if table["content"] and str(table["format"]).lower() == "html":
                    tables_out.append({"html": table["content"], "id": table["id"]})

            dims = _get(page, "dimensions")
            width = _get(dims, "width") if dims is not None else None
            height = _get(dims, "height") if dims is not None else None
            dpi = _get(dims, "dpi") if dims is not None else None

            for order, block in enumerate(_get(page, "blocks", default=[]) or []):
                btype = str(_get(block, "type", default="") or "")
                coords = [_get(block, k) for k in
                          ("top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y")]
                bbox = None
                if width and height and all(c is not None for c in coords):
                    x0, y0, x1, y1 = (float(c) for c in coords)
                    bbox = [x0 / width, y0 / height, x1 / width, y1 / height]
                content = _get(block, "content", default="") or ""
                layout_blocks.append({
                    "label": btype,
                    "bbox": bbox,
                    "bbox_px": coords,
                    "order": order,
                    "page": _get(page, "index"),
                    "content": content,
                    "table_id": _get(block, "table_id"),
                    "image_id": _get(block, "image_id"),
                })
                text = _html_to_text(content)
                if bbox and text and btype not in _NON_TEXT_BLOCK_TYPES:
                    regions.append({"text": text, "bbox": bbox, "label": btype})

            page_meta.append({
                "index": _get(page, "index"),
                "dimensions": {"width": width, "height": height, "dpi": dpi},
                "header": _get(page, "header"),
                "footer": _get(page, "footer"),
                "placed_table_ids": placed,
                "unplaced_table_ids": unplaced,
            })

        markdown = "\n\n".join(page_markdown)
        structured: dict[str, Any] = {
            "raw_model_output": "\n\n".join(raw_markdown),  # placeholders intact
            "pages": page_meta,
        }
        if tables_out:
            structured["tables"] = tables_out
        if layout_blocks:
            structured["layout_blocks"] = layout_blocks
        if regions:
            # Paragraph/block boxes — never reported as words.
            structured["text_regions"] = regions
            structured["geometry_granularity"] = "block"
            structured["geometry_coordinates"] = "normalized_xyxy"
        if unplaced_all:
            structured["tables_appended_unplaced"] = unplaced_all

        # Local image size, so the first run can confirm that block pixel
        # coordinates share the frame of pages[].dimensions (see probe list).
        try:
            from PIL import Image, ImageOps
            with Image.open(path) as img:
                structured["image_size_local"] = list(ImageOps.exif_transpose(img).size)
        except Exception:
            pass

        raw_text = markdown
        if annotated:
            annotation = _get(response, "document_annotation")
            structured["markdown"] = markdown  # the conversion rides along
            raw_text = annotation if isinstance(annotation, str) else (
                "" if annotation is None else str(annotation))

        usage = _get(response, "usage_info")
        pages_processed = _get(usage, "pages_processed") if usage is not None else None
        billed_pages = int(pages_processed) if pages_processed else max(len(pages), 1)
        rate = COST_PER_PAGE_ANNOTATION if annotated else COST_PER_PAGE
        returned_model = _get(response, "model")

        return OCRResult(
            engine_name=self.name,
            document_path=str(path),
            raw_text=raw_text,
            structured_output=structured,
            latency_seconds=0.0,  # set by run()
            cost_per_page=rate * billed_pages,
            page_count=len(pages) or 1,
            model_version=str(returned_model) if returned_model else f"{MODEL_ID} (requested)",
            metadata={
                "requested_model": MODEL_ID,
                "returned_model": returned_model,
                "request_options": {**OCR_OPTIONS,
                                    "document_annotation": bool(annotated)},
                "usage_info": {
                    "pages_processed": pages_processed,
                    "doc_size_bytes": _get(usage, "doc_size_bytes") if usage is not None else None,
                },
                "billing_basis": ("usage_info.pages_processed x list rate"
                                  if pages_processed else "page count x list rate"),
                "rate_usd_per_page": rate,
            },
            raw_response=provider_payload(response),
        )
