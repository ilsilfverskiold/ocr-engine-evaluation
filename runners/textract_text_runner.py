from __future__ import annotations

import io
from pathlib import Path

from runners.base import OCRResult, OCRRunner, OCRTask, provider_payload

COST_PER_PAGE = 0.0015  # $1.50 / 1000 pages — DetectDocumentText tier
_MAX_BYTES = 9_000_000  # Textract sync limit is 10MB; stay under with margin


def _prepare_bytes(path: Path) -> bytes:
    """Return image bytes, resizing if over Textract's 10MB sync limit."""
    raw = path.read_bytes()
    if len(raw) <= _MAX_BYTES or path.suffix.lower() == ".pdf":
        return raw
    from PIL import Image
    img = Image.open(path).convert("RGB")
    quality = 85
    while quality >= 40:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        if buf.tell() <= _MAX_BYTES:
            return buf.getvalue()
        quality -= 15
    # Last resort: halve dimensions
    buf = io.BytesIO()
    img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


class TextractTextRunner(OCRRunner):
    """AWS Textract text-only detection (DetectDocumentText)."""

    name = "textract_text"
    supports_pdf = True
    supports_structured = False

    def __init__(self) -> None:
        import boto3
        self._client = boto3.client("textract")

    def run(self, document_path: Path, task: OCRTask | None = None) -> OCRResult:
        """DetectDocumentText takes no prompt: `task` is accepted for the
        uniform runner contract and ignored."""
        t0 = self._timer()
        raw_bytes = _prepare_bytes(document_path)

        response = self._client.detect_document_text(
            Document={"Bytes": raw_bytes}
        )
        elapsed = self._elapsed(t0)

        blocks = response.get("Blocks", [])
        lines = [b["Text"] for b in blocks if b["BlockType"] == "LINE"]
        text = "\n".join(lines)
        words = _word_boxes(blocks)

        return OCRResult(
            engine_name=self.name,
            document_path=str(document_path),
            raw_text=text,
            structured_output={"words": words} if words else {},
            latency_seconds=elapsed,
            cost_per_page=COST_PER_PAGE,
            page_count=1,
            model_version="aws-textract-detect-document-text",
            raw_response=provider_payload(response),
        )


def _word_boxes(blocks: list[dict]) -> list[dict]:
    """WORD blocks -> geometry-leg words. Textract's BoundingBox is already
    normalised 0-1 (Left/Top/Width/Height), converted here to [x0,y0,x1,y1]."""
    words = []
    for block in blocks:
        if block.get("BlockType") != "WORD":
            continue
        box = (block.get("Geometry") or {}).get("BoundingBox") or {}
        left, top = box.get("Left", 0.0), box.get("Top", 0.0)
        words.append({
            "text": block.get("Text", ""),
            "bbox": [left, top, left + box.get("Width", 0.0), top + box.get("Height", 0.0)],
            "confidence": (block.get("Confidence") or 0.0) / 100.0,
        })
    return words
