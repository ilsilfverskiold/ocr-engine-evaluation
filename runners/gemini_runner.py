from __future__ import annotations

import os
from pathlib import Path

from runners.base import OCRResult, OCRRunner, OCRTask, provider_payload

INPUT_PRICE_PER_TOKEN = 0.25 / 1_000_000   # gemini-3.1-flash-lite
OUTPUT_PRICE_PER_TOKEN = 1.50 / 1_000_000  # gemini-3.1-flash-lite
ESTIMATED_COST_PER_PAGE = 0.0007


class GeminiRunner(OCRRunner):
    """Gemini 3.1 Flash-Lite — image-based OCR via google-genai SDK."""

    name = "gemini_flash"
    supports_pdf = False
    accepts_task = True

    _PROMPT = (
        "You are a precise document transcription engine. "
        "Transcribe ALL text visible in this document image exactly as it appears, "
        "preserving structure, tables (as markdown), and reading order. "
        "Do not add commentary, summaries, or corrections. Output only the transcribed text."
    )

    def __init__(self, model: str = "gemini-3.1-flash-lite") -> None:
        from google import genai
        self._client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self._model_name = model

    def run(self, document_path: Path, task: OCRTask | None = None) -> OCRResult:
        from google.genai import types

        t0 = self._timer()
        raw_bytes = document_path.read_bytes()
        mime = "image/jpeg" if document_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"

        # task=None reproduces leg 1 byte for byte: same prompt, same default
        # generation config, no response_mime_type.
        prompt = task.prompt if task else self._PROMPT
        config = None
        if task and task.wants_json:
            config = types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            )

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=[
                types.Part.from_bytes(data=raw_bytes, mime_type=mime),
                prompt,
            ],
            **({"config": config} if config is not None else {}),
        )
        elapsed = self._elapsed(t0)
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        cost = (
            input_tokens * INPUT_PRICE_PER_TOKEN
            + output_tokens * OUTPUT_PRICE_PER_TOKEN
            if usage else ESTIMATED_COST_PER_PAGE
        )

        return OCRResult(
            engine_name=self.name,
            document_path=str(document_path),
            raw_text=response.text or "",
            structured_output={},
            latency_seconds=elapsed,
            cost_per_page=cost,
            page_count=1,
            model_version=self._model_name,
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "leg": task.leg if task else "default",
            },
            # `response.text` is only the first text part; a truncated or
            # safety-blocked reply looks identical to an empty one without the
            # finish_reason that lives here.
            raw_response=provider_payload(response),
        )
