from __future__ import annotations

import os
from pathlib import Path

from runners.base import OCRResult, OCRRunner, OCRTask, provider_payload

COST_PER_PAGE = 0.00375  # cost_effective tier: 3 credits/page × $1.25/1K credits


def _sanitize_schema_for_llama(schema: dict) -> dict:
    """Rewrite a JSON Schema into the subset LlamaExtract accepts.

    LlamaExtract requires: root ``{"type": "object", "properties": {...}}``,
    and ``additionalProperties`` must be a boolean, not a schema object.
    """
    import copy
    s = copy.deepcopy(schema)

    def _walk(node: dict) -> None:
        if not isinstance(node, dict):
            return
        if "additionalProperties" in node and not isinstance(node["additionalProperties"], bool):
            node["additionalProperties"] = True
        for v in node.values():
            if isinstance(v, dict):
                _walk(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        _walk(item)

    _walk(s)

    if s.get("type") == "object" and "properties" not in s:
        s["properties"] = {}

    return s


class LlamaParseRunner(OCRRunner):
    """LlamaParse document parsing API (llama-cloud v2.5+)."""

    name = "llamaparse"
    supports_pdf = True
    accepts_task = True  # schema leg only, via LlamaExtract (llama-cloud-services)

    def __init__(self) -> None:
        # The vendor's wrapper handles the upload -> poll -> result job flow;
        # the raw llama_cloud client exposes only the job primitives.
        import nest_asyncio
        nest_asyncio.apply()  # the wrapper's sync parse() re-enters its loop on repeat calls
        from llama_cloud_services import LlamaParse
        self._parser = LlamaParse(
            api_key=os.environ["LLAMA_CLOUD_API_KEY"],
            result_type="markdown",
            verbose=False,
            show_progress=False,
        )

    def run(self, document_path: Path, task: OCRTask | None = None) -> OCRResult:
        """The parse endpoint takes no prompt. The schema leg is answered by
        LlamaExtract — LlamaIndex's own schema-extraction product beside
        LlamaParse — so the engine competes with the vendor's documented path,
        not a prompt we invented. Other tasks are ignored: markdown and tables
        come from the default conversion.

        UNVALIDATED: exercise on 2 documents before any full run. Needs the
        llama-cloud-services package (in requirements.txt)."""
        if task is not None and task.leg == "schema" and task.json_schema:
            return self._run_extract(document_path, task.json_schema)
        t0 = self._timer()
        result = self._parser.parse(str(document_path))
        elapsed = self._elapsed(t0)

        pages = getattr(result, "pages", None) or []
        page_count = max(1, len(pages))
        raw_text = "\n".join(
            (getattr(p, "md", None) or getattr(p, "text", None) or "") for p in pages
        ).strip()

        return OCRResult(
            engine_name=self.name,
            document_path=str(document_path),
            raw_text=raw_text,
            # The parse result carries per-page items with types and boxes;
            # keeping only the concatenated markdown threw that away.
            structured_output=provider_payload(result),
            latency_seconds=elapsed,
            cost_per_page=COST_PER_PAGE,
            page_count=page_count,
            model_version="llamaparse-latest",
        )

    def _run_extract(self, document_path: Path, schema: dict) -> OCRResult:
        import json as _json

        t0 = self._timer()
        try:
            from llama_cloud_services import LlamaExtract
        except ImportError as exc:
            raise RuntimeError(
                "LlamaExtract requires the llama-cloud-services package "
                "(pip install llama-cloud-services)"
            ) from exc

        extractor = LlamaExtract(api_key=os.environ["LLAMA_CLOUD_API_KEY"])
        # Stateless per-call extraction: one agent per schema shape, reused by
        # name so repeated documents with the same doc_type do not recreate it.
        agent_name = f"ocr-benchmark-{abs(hash(_json.dumps(schema, sort_keys=True))) % 10**8}"
        try:
            agent = extractor.get_agent(name=agent_name)
        except Exception:
            agent = None
        if agent is None:
            agent = extractor.create_agent(name=agent_name,
                                           data_schema=_sanitize_schema_for_llama(schema))
        result = agent.extract(str(document_path))
        data = getattr(result, "data", None) or {}
        elapsed = self._elapsed(t0)

        return OCRResult(
            engine_name=self.name,
            document_path=str(document_path),
            raw_text=_json.dumps(data, ensure_ascii=False),
            structured_output=provider_payload(result),
            latency_seconds=elapsed,
            cost_per_page=COST_PER_PAGE,  # extraction pricing differs; verify on the 2-doc run
            page_count=1,
            model_version="llama-extract",
        )
