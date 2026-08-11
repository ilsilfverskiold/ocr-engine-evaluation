from __future__ import annotations

from pathlib import Path

from runners.base import OCRResult, OCRRunner, OCRTask


class TesseractRunner(OCRRunner):
    name = "tesseract"
    supports_pdf = False  # needs image input; orchestrator converts PDFs

    def __init__(self, lang: str = "eng") -> None:
        self.lang = lang

    def run(self, document_path: Path, task: OCRTask | None = None) -> OCRResult:
        """pytesseract exposes no prompt: `task` is accepted for the
        uniform runner contract and ignored."""
        import pytesseract
        from PIL import Image

        t0 = self._timer()
        img = Image.open(document_path)
        text = pytesseract.image_to_string(img, lang=self.lang)
        # Word boxes for the geometry leg, normalised 0-1 against the image.
        # image_to_data is the documented word-level interface; conf == -1 rows
        # are layout blocks, not words.
        data = pytesseract.image_to_data(
            img, lang=self.lang, output_type=pytesseract.Output.DICT
        )
        width, height = img.size
        words = []
        for i, word in enumerate(data["text"]):
            if not word.strip() or int(data["conf"][i]) < 0:
                continue
            x, y = data["left"][i], data["top"][i]
            w, h = data["width"][i], data["height"][i]
            words.append({
                "text": word,
                "bbox": [x / width, y / height, (x + w) / width, (y + h) / height],
                "confidence": int(data["conf"][i]) / 100.0,
            })
        elapsed = self._elapsed(t0)

        version = pytesseract.get_tesseract_version()

        return OCRResult(
            engine_name=self.name,
            document_path=str(document_path),
            raw_text=text,
            structured_output={"words": words} if words else {},
            latency_seconds=elapsed,
            cost_per_page=0.0,
            page_count=1,
            model_version=str(version),
        )
