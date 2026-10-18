from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OCRTask:
    """What a leg asks the engine for."""

    leg: str
    prompt: str
    # Injected into the prompt AND passed to the provider's JSON mode where one
    # exists. None for legs that ask for text rather than conforming JSON.
    json_schema: dict[str, Any] | None = None
    # "json" asks the runner to enable the provider's structured-output mode.
    response_format: str | None = None

    @property
    def wants_json(self) -> bool:
        return self.response_format == "json" or self.json_schema is not None


@dataclass
class OCRResult:
    engine_name: str
    document_path: str
    raw_text: str
    structured_output: dict[str, Any]  # tables as HTML/JSON, forms as KV; {} if N/A
    latency_seconds: float
    cost_per_page: float               # USD; 0.0 for local engines
    page_count: int
    model_version: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # Persist the complete provider reply so parsing/scoring can be revised without another paid call.
    raw_response: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class OCRRunner(ABC):
    name: str = "base"
    supports_pdf: bool = True
    supports_structured: bool = False
    # Mirrors EngineSpec.accepts_task. A runner that leaves this False ignores
    # any task it is handed; the orchestrator never sends one.
    accepts_task: bool = False

    @abstractmethod
    def run(self, document_path: Path, task: OCRTask | None = None) -> OCRResult: ...

    def run_safe(self, document_path: Path, task: OCRTask | None = None) -> OCRResult:
        try:
            result = self.run(document_path, task)
            # A runner that was handed no task and reports nothing sent no instruction.
            if task is None and "sent" not in result.metadata:
                result.metadata["sent"] = sent_record(None, "none: no task passed, default conversion")
            return result
        except Exception as exc:
            message = str(exc) or repr(exc)
            return OCRResult(
                engine_name=self.name,
                document_path=str(document_path),
                raw_text="",
                structured_output={},
                latency_seconds=0.0,
                cost_per_page=0.0,
                page_count=0,
                model_version="unknown",
                error=f"{exc.__class__.__name__}: {message}",
            )

    @staticmethod
    def _timer() -> float:
        """Returns a perf_counter start time. Use with _elapsed()."""
        return time.perf_counter()

    @staticmethod
    def _elapsed(start: float) -> float:
        return time.perf_counter() - start


def sent_record(text: str | None, source: str, constraint: str = "none (native output)",
                schema_separately: bool = False) -> dict[str, Any]:
    """What actually reached the engine on one call, stored as OCRResult.metadata["sent"].

    `text` is the instruction text the engine received (None when it received
    none); `source` says where it came from. The call log records this, never
    the task prompt a runner may have ignored.
    """
    return {"text": text, "source": source, "constraint": constraint,
            "schema_separately": schema_separately}


def provider_payload(response: Any, max_bytes: int = 4_000_000) -> dict[str, Any]:
    """An SDK response object as plain JSON-serialisable data.

    Every engine's full reply is persisted beside the text we scored. A reply
    we did not keep cannot be re-checked by hand or re-scored after a scorer
    fix without paying for the call again — and "trust me, the number is
    right" is not something a benchmark gets to say. Falls back to a
    truncation note rather than failing a run over a large or exotic payload.
    """
    def plain(obj: Any, depth: int = 0) -> Any:
        if depth > 12:
            return str(obj)[:500]
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [plain(v, depth + 1) for v in obj]
        if isinstance(obj, dict):
            return {str(k): plain(v, depth + 1) for k, v in obj.items()}
        for attr in ("model_dump", "dict", "to_dict"):
            method = getattr(obj, attr, None)
            if callable(method):
                try:
                    return plain(method(), depth + 1)
                except Exception:
                    pass
        if hasattr(obj, "__dict__"):
            return {k: plain(v, depth + 1)
                    for k, v in vars(obj).items() if not k.startswith("_")}
        return str(obj)[:500]

    try:
        data = plain(response)
        if not isinstance(data, dict):
            data = {"response": data}
        if len(json.dumps(data, ensure_ascii=False, default=str)) > max_bytes:
            return {"_truncated": "provider payload exceeded size limit",
                    "_keys": sorted(data)[:40]}
        return data
    except Exception as exc:
        return {"_unserialisable": f"{exc.__class__.__name__}: {exc}"}
