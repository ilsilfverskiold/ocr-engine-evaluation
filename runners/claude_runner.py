from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from runners.base import OCRResult, OCRRunner, OCRTask, provider_payload

_MAX_BYTES = 4_900_000  # Claude API hard limit is 5MB; stay under with margin


def _geometry_input(path: Path):
    """Known standard-tier frame; no implicit provider coordinate resizing.

    https://platform.claude.com/docs/en/build-with-claude/vision-coordinates
    """
    import math
    from PIL import Image, ImageOps
    image = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    original = image.size
    scale = min(1.0, 1568 / max(original))
    while True:
        width, height = [max(1, int(v * scale)) for v in original]
        if math.ceil(width / 28) * math.ceil(height / 28) <= 1568:
            break
        scale *= .995
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    prompt = (
        f"Locate every visible word in this {width} by {height} pixel image. "
        'Return only a JSON array of {"text":"printed word","bbox":[left,top,right,bottom]}. '
        'Use absolute pixel coordinates in this supplied image, with (0,0) at the top left, '
        'x increasing rightward and y downward. Give one tight rectangle per word at its actual '
        'printed location. Keep punctuation attached. Do not combine words into lines, reconstruct '
        'a cleaner layout, or infer positions from reading order. Include headers, footers and table text.'
    )
    return buf.getvalue(), prompt, {'source_size': list(original), 'sent_size': [width, height],
                                    'recipe': 'sonnet-geometry-pixels-v2', 'coordinates': 'absolute pixels in sent image'}


def _prepare_image(path: Path) -> tuple[bytes, str]:
    """Return (bytes, media_type), compressing if the image exceeds Claude's 5MB limit."""
    raw = path.read_bytes()
    media_type = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    if len(raw) <= _MAX_BYTES:
        return raw, media_type
    from PIL import Image
    img = Image.open(path).convert("RGB")
    for quality in [85, 70, 55, 40]:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        if buf.tell() <= _MAX_BYTES:
            return buf.getvalue(), "image/jpeg"
    # Last resort: halve dimensions
    buf = io.BytesIO()
    img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), "image/jpeg"

# Claude Sonnet 4.6 pricing (approximate per page = 1 image ~1500 in + ~800 out tokens)
INPUT_PRICE_PER_TOKEN = 3.0 / 1_000_000   # $3/M input
OUTPUT_PRICE_PER_TOKEN = 15.0 / 1_000_000  # $15/M output


class ClaudeRunner(OCRRunner):
    """Claude Sonnet 4.6 — image-based OCR via Anthropic API."""

    name = "claude_sonnet_4_6"
    supports_pdf = False  # orchestrator converts PDF pages to images
    accepts_task = True

    _PROMPT = (
        "Transcribe ALL text visible in this document image exactly as it appears. "
        "Preserve structure, tables (as markdown), reading order, and special characters. "
        "Do not add commentary, summaries, or corrections. Output only the transcribed text."
    )

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=0)
        self._model = model

    def run(self, document_path: Path, task: OCRTask | None = None) -> OCRResult:
        t0 = self._timer()
        raw_bytes, media_type = _prepare_image(document_path)

        # task=None reproduces leg 1 byte for byte.
        prompt = task.prompt if task else self._PROMPT
        output_cap = 4096
        geometry_frame = None
        if task and task.leg == 'geometry':
            raw_bytes, prompt, geometry_frame = _geometry_input(document_path)
            media_type = 'image/png'
            output_cap = 32768
        b64 = base64.standard_b64encode(raw_bytes).decode()
        kwargs: dict = {}
        if task and task.wants_json:
            # Preserve the benchmark's prompt-based JSON route. No assistant
            # prefill: this model rejected it in the historical control.
            kwargs["temperature"] = 0

        content: list[dict] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            },
            {"type": "text", "text": prompt},
        ]
        messages: list[dict] = [{"role": "user", "content": content}]

        message = self._client.messages.create(
            model=self._model,
            max_tokens=output_cap,
            messages=messages,
            **kwargs,
        )
        elapsed = self._elapsed(t0)

        text = message.content[0].text if message.content else ""
        structured = {}
        if geometry_frame:
            from evaluation.geometry_leg import _box_items, _extract_box
            width, height = geometry_frame['sent_size']
            items = _box_items(text) or []
            words = []
            for item in items:
                box = _extract_box(item) if isinstance(item, dict) else None
                words.append({**item, 'pixel_bbox': list(box),
                              'bbox': [box[0]/width, box[1]/height, box[2]/width, box[3]/height]}
                             if box else item)
            structured = {'words': words, 'geometry_coordinates': 'normalized_from_recorded_pixel_frame',
                          'geometry_frame': geometry_frame}
        usage = message.usage
        cost = (
            usage.input_tokens * INPUT_PRICE_PER_TOKEN
            + usage.output_tokens * OUTPUT_PRICE_PER_TOKEN
        )

        return OCRResult(
            engine_name=self.name,
            document_path=str(document_path),
            raw_text=text,
            structured_output=structured,
            latency_seconds=elapsed,
            cost_per_page=cost,
            page_count=1,
            model_version=self._model,
            metadata={
                "prompt_used": prompt,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "leg": task.leg if task else "default",
                "max_output_tokens": output_cap,
                "temperature": kwargs.get("temperature", "provider_default"),
                "sdk_max_retries": 0,
                "geometry_frame": geometry_frame,
                # A reply cut off at max_tokens is invalid JSON that looks like
                # the model failing the leg. stop_reason is the difference.
                "stop_reason": getattr(message, "stop_reason", None),
            },
            # raw_text above is content[0] only; this is every block that came
            # back, with the stop_reason and usage that explain a short reply.
            raw_response=provider_payload(message),
        )
