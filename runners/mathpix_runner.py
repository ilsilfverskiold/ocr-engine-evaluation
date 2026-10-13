"""Mathpix OCR — POST https://api.mathpix.com/v3/text (image OCR).

Source for every option below: docs.mathpix.com/reference/post-v3-text and
docs.mathpix.com/guides/image-ocr, read 2026-09-30.

Two configurations, because the API documents a conflict between them:
"Combining include_word_data: true with enable_document_layout: true is not
supported."

  default pass (task=None) — the conversion: text/markdown/tables legs.
      enable_document_layout=True  "For a full page image, send
                                   enable_document_layout: true" (guide);
                                   the default is snippet-oriented.
      formats=["text","html","data"], data_options.include_table_html=True
                                   text = Mathpix Markdown (raw_text);
                                   html/data carry table HTML.
      include_line_data=True       line contours, kept as metadata.
      include_page_info=True       documented default, explicit: headers,
                                   footers and page numbers stay in the text.
      math delimiters              documented defaults \\( \\) and \\[ \\],
                                   explicit; formula content is not deleted.
      enable_tables_fallback       NOT sent: documented as deprecated, "has no
                                   effect".
  geometry pass (task.leg == "geometry") — word boxes.
      include_word_data=True, enable_document_layout=False,
      include_line_data=True (row count for billing, line contours).

The geometry task's prompt is a CONFIGURATION SELECTOR, not text sent to the
vendor: /v3/text takes no prompt. The registry maps the geometry leg to the
sentinel "__mathpix_word_data__" (same pattern as marker's converter
sentinels) so the harness plans a separate pass; the string itself is never
transmitted.

Upload: multipart form-data, file field "file" and all options as stringified
JSON in "options_json" (guide example). "EXIF data is ignored for all images,
including EXIF orientation" — so EXIF orientation is applied locally before
upload when present, and recorded. Multipart body limit: "the body limit
applies to the entire multipart payload including the file" = 5 MB.
"""

from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path
from typing import Any

from runners.base import OCRResult, OCRRunner, OCRTask, provider_payload

API_URL = "https://api.mathpix.com/v3/text"
GEOMETRY_SENTINEL = "__mathpix_word_data__"
# Limits & Quotas (docs.mathpix.com/reference/limits-and-quotas): v3/text
# "5 MB JSON body"; multipart "body limit applies to the entire multipart
# payload including the file". Keep margin for the form fields.
MAX_UPLOAD_BYTES = 5 * 1024 * 1024 - 64 * 1024
REQUEST_TIMEOUT_S = 120

# mathpix.com/pricing/api, read 2026-09-30: v3/text "$0.002 / image" (0-1M);
# "Images containing more than 12 rows of text may be billed at the v3/pdf
# per-page rate" = $0.005/page. The row count is estimated from line_data.
COST_PER_IMAGE = 0.002
COST_PER_IMAGE_OVER_12_ROWS = 0.005

_COMMON = {
    "include_page_info": True,
    "math_inline_delimiters": ["\\(", "\\)"],
    "math_display_delimiters": ["\\[", "\\]"],
    "rm_spaces": True,
}
DEFAULT_OPTIONS: dict[str, Any] = {
    **_COMMON,
    "formats": ["text", "html", "data"],
    "data_options": {"include_table_html": True, "include_tsv": True},
    "enable_document_layout": True,
    "include_line_data": True,
}
GEOMETRY_OPTIONS: dict[str, Any] = {
    **_COMMON,
    "formats": ["text"],
    "enable_document_layout": False,
    "include_word_data": True,
    "include_line_data": True,
}

_TABLE_RE = re.compile(r"<table\b.*?</table>", re.S | re.I)


def _tables_from_html(html: str) -> list[str]:
    return _TABLE_RE.findall(html or "")


def _cnt_box(cnt: Any, width: float, height: float) -> tuple[list[float] | None, list | None]:
    """Mathpix `cnt` pixel contour -> normalised [x0,y0,x1,y1] + normalised poly."""
    try:
        pts = [[float(p[0]), float(p[1])] for p in cnt]
    except (TypeError, ValueError, IndexError):
        return None, None
    if not pts or not width or not height:
        return None, None
    norm = [[x / width, y / height] for x, y in pts]
    xs = [p[0] for p in norm]
    ys = [p[1] for p in norm]
    return [min(xs), min(ys), max(xs), max(ys)], norm


def _estimated_rows(line_data: list[dict]) -> int | None:
    """Text-bearing leaf lines — an estimate of Mathpix's billing 'rows'."""
    if not line_data:
        return None
    return sum(1 for line in line_data
               if (line.get("text") or "").strip() and not line.get("children_ids"))


class MathpixRunner(OCRRunner):
    """Mathpix v3/text; default conversion + separately configured word pass."""

    name = "mathpix"
    supports_pdf = False   # /v3/pdf (async) not built; corpus is images
    accepts_task = True    # geometry only, as a configuration selector

    def __init__(self) -> None:
        import requests  # noqa: F401  (fail early if missing)
        self._app_id = os.environ["MATHPIX_APP_ID"]
        self._app_key = os.environ["MATHPIX_APP_KEY"]

    # ----------------------------------------------------------------- upload
    @staticmethod
    def _prepare_upload(path: Path) -> tuple[bytes, str, str, dict]:
        """Bytes to upload + (filename, mime, preprocessing record).

        Pixels are never resampled. EXIF orientation is applied (Mathpix ignores
        it) and, only if the file is over the multipart limit, the SAME pixels
        are re-encoded as a maximally compressed PNG (lossless)."""
        from PIL import Image, ImageOps

        raw = path.read_bytes()
        mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        record: dict[str, Any] = {"original_bytes": len(raw), "resampled": False}
        with Image.open(io.BytesIO(raw)) as img:
            orientation = img.getexif().get(0x0112, 1)
            record["exif_orientation"] = orientation
            needs_transpose = orientation not in (None, 1)
            if not needs_transpose and len(raw) <= MAX_UPLOAD_BYTES:
                record["image_size"] = list(img.size)
                record["upload"] = "original bytes"
                return raw, path.name, mime, record
            fixed = ImageOps.exif_transpose(img)
            record["image_size"] = list(fixed.size)
            buf = io.BytesIO()
            fixed.save(buf, format="PNG", optimize=True, compress_level=9)
            data = buf.getvalue()
        record["upload"] = ("exif-transposed lossless PNG" if needs_transpose
                            else "lossless PNG re-encode (size limit)")
        record["uploaded_bytes"] = len(data)
        if len(data) > MAX_UPLOAD_BYTES:
            # No downscaling: that would change what is being measured.
            # UNVERIFIED alternative: async /v3/pdf takes 1 GB uploads, but its
            # page lists PDF/DOCX/PPTX — image input there is not confirmed.
            raise ValueError(
                f"{path.name}: {len(data)} bytes after lossless re-encode exceeds "
                f"Mathpix v3/text 5 MB multipart limit; not downscaled"
            )
        return data, path.stem + ".png", "image/png", record

    def _post(self, data: bytes, filename: str, mime: str, options: dict) -> dict:
        import requests

        resp = requests.post(
            API_URL,
            headers={"app_id": self._app_id, "app_key": self._app_key},
            files={"file": (filename, data, mime)},
            data={"options_json": json.dumps(options)},
            timeout=REQUEST_TIMEOUT_S,
        )
        try:
            body = resp.json()
        except ValueError:
            resp.raise_for_status()
            raise RuntimeError(f"Mathpix returned non-JSON (HTTP {resp.status_code})")
        if resp.status_code >= 400 or body.get("error"):
            raise RuntimeError(
                f"Mathpix HTTP {resp.status_code}: {body.get('error')} "
                f"{json.dumps(body.get('error_info'))} request_id={body.get('request_id')}"
            )
        return body

    # -------------------------------------------------------------------- run
    def run(self, document_path: Path, task: OCRTask | None = None) -> OCRResult:
        if document_path.suffix.lower() == ".pdf":
            raise ValueError(f"{self.name}: PDF input needs /v3/pdf (not implemented)")
        geometry = False
        if task is not None:
            if task.leg != "geometry" or task.prompt != GEOMETRY_SENTINEL:
                raise ValueError(
                    f"{self.name} accepts only the geometry configuration task "
                    f"({GEOMETRY_SENTINEL!r}); got leg {task.leg!r}"
                )
            geometry = True
        options = GEOMETRY_OPTIONS if geometry else DEFAULT_OPTIONS

        data, filename, mime, prep = self._prepare_upload(document_path)
        t0 = self._timer()
        body = self._post(data, filename, mime, options)
        elapsed = self._elapsed(t0)

        width = float(body.get("image_width") or 0)
        height = float(body.get("image_height") or 0)
        rotation = body.get("auto_rotate_degrees") or 0
        line_data = body.get("line_data") or []

        lines = []
        for line in line_data:
            bbox, poly = _cnt_box(line.get("cnt"), width, height)
            lines.append({
                "id": line.get("id"), "parent_id": line.get("parent_id"),
                "type": line.get("type"), "subtype": line.get("subtype"),
                "text": line.get("text"), "bbox": bbox, "poly": poly,
                "conversion_output": line.get("conversion_output"),
                "confidence": line.get("confidence"),
            })

        structured: dict[str, Any] = {
            "preprocessing": prep,
            "lines": lines,
            "processed_image_size": [body.get("image_width"), body.get("image_height")],
            "auto_rotate_degrees": rotation,
            "auto_rotate_confidence": body.get("auto_rotate_confidence"),
        }

        if geometry:
            words, other = [], []
            for word in body.get("word_data") or []:
                bbox, poly = _cnt_box(word.get("cnt"), width, height)
                text = (word.get("text") or "").strip()
                entry = {"text": text, "bbox": bbox, "poly": poly,
                         "type": word.get("type"),
                         "confidence": word.get("confidence")}
                # Only word elements of type "text" are words. math/table/
                # diagram/equation_number elements can span several tokens.
                if word.get("type") == "text" and text and bbox:
                    words.append(entry)
                else:
                    other.append(entry)
            if rotation:
                # UNVERIFIED which frame cnt uses after auto-rotation; boxes in
                # a rotated frame would be silently mis-scored, so they are
                # withheld from the scorer rather than guessed back.
                structured["words_unmapped_rotated"] = words
                structured["geometry_error"] = (
                    f"Mathpix returned rotated geometry ({rotation} degrees); "
                    "coordinate frame is not verified")
                structured["geometry_note"] = (
                    f"auto_rotate_degrees={rotation}: word boxes withheld")
            elif words:
                structured["words"] = words
                structured["geometry_granularity"] = "word"
                structured["geometry_coordinates"] = "normalized_xyxy"
            if other:
                structured["word_data_other"] = other
        else:
            tables: list[dict] = []
            for item in body.get("data") or []:
                value = item.get("value") or ""
                if "<table" in value.lower():
                    tables.append({"html": value, "source": f"data[{item.get('type')}]"})
            if not tables:
                tables = [{"html": t, "source": "html"} for t in
                          _tables_from_html(body.get("html") or "")]
            if tables:
                structured["tables"] = tables
            if body.get("html"):
                structured["html"] = body["html"]

        rows = _estimated_rows(line_data)
        over_12 = rows is None or rows > 12
        cost = COST_PER_IMAGE_OVER_12_ROWS if over_12 else COST_PER_IMAGE
        version = body.get("version")

        return OCRResult(
            engine_name=self.name,
            document_path=str(document_path),
            raw_text=body.get("text") or "",   # Mathpix Markdown
            structured_output=structured,
            latency_seconds=elapsed,
            cost_per_page=cost,
            page_count=1,
            model_version=(f"mathpix-v3/text {version}" if version
                           else "mathpix-v3/text (version not returned)"),
            metadata={
                "request_id": body.get("request_id"),
                "version": version,
                "configuration": "geometry(word_data)" if geometry else "default(document_layout)",
                "request_options": options,
                "estimated_text_rows": rows,
                "billing_basis": (
                    "estimated: >12 text rows (or unknown) -> $0.005, else $0.002; "
                    "the invoice decides"),
                "is_printed": body.get("is_printed"),
                "is_handwritten": body.get("is_handwritten"),
                "confidence": body.get("confidence"),
                "confidence_rate": body.get("confidence_rate"),
            },
            raw_response=provider_payload(body),
        )
