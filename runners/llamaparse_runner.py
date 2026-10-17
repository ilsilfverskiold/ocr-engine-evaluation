from __future__ import annotations

import os
import html
import re
from pathlib import Path

from runners.base import OCRResult, OCRRunner, OCRTask, provider_payload

COST_PER_PAGE = 0.00375  # cost_effective tier: 3 credits/page × $1.25/1K credits
EXTRACT_COST_PER_PAGE = 0.025  # v1 MULTIMODAL: 20 credits × $1.25/1000


def _audit_text_view(page: dict) -> tuple[str, list[dict]]:
    """Score the native text unchanged and expose differences from native items."""
    native = str(page.get("text") or "").strip()
    additions = []

    def words(value: str) -> list[str]:
        return re.findall(r"\w+", html.unescape(re.sub(r"<[^>]+>", " ", value)).casefold())

    seen = words(native)

    def add(value, source):
        for line in str(value or "").splitlines():
            # Normalize item table Markdown only for the audit. The native
            # page text remains the scored text, unchanged.
            if re.fullmatch(r"[\s|:-]+", line):
                continue
            plain = html.unescape(re.sub(r"<[^>]+>", " ", line)).strip().strip("|").strip()
            if "|" in plain:
                plain = "\t".join(cell.strip() for cell in plain.split("|"))
            tokens = words(plain)
            if tokens and not any(seen[i:i + len(tokens)] == tokens
                                  for i in range(len(seen) - len(tokens) + 1)):
                additions.append({"source": source, "text": plain})

    for index, item in enumerate(page.get("items") or []):
        add(item.get("value") or item.get("md") or item.get("html"), f"items[{index}]")
    for field in ("pageHeaderMarkdown", "pageFooterMarkdown", "printedPageNumber"):
        add(page.get(field), field)
    return native, additions


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

        Schema extraction submits one stateless v1 Multimodal job per page."""
        if task is not None and task.leg == "schema" and task.json_schema:
            return self._run_extract(document_path, task.json_schema)
        t0 = self._timer()
        result = self._parser.parse(str(document_path))
        elapsed = self._elapsed(t0)

        pages = getattr(result, "pages", None) or []
        page_count = max(1, len(pages))
        if any((getattr(p, "md", None) or "").strip() and
               not (getattr(p, "text", None) or "").strip() for p in pages):
            raise ValueError("LlamaParse returned Markdown without native text for a page")
        page_payloads = [provider_payload(p, max_bytes=32_000_000) for p in pages]
        if any("_truncated" in payload for payload in page_payloads):
            raise ValueError("LlamaParse page payload exceeded the retention limit")
        text_views = [_audit_text_view(payload) for payload in page_payloads]
        raw_text = "\n".join(view[0] for view in text_views).strip()
        markdown = "\n".join((getattr(p, "md", None) or "") for p in pages).strip()
        structured = provider_payload(result, max_bytes=32_000_000)
        if "_truncated" in structured:
            raise ValueError("LlamaParse result exceeded the retention limit")
        structured["markdown"] = markdown
        structured["native_text"] = "\n".join((getattr(p, "text", None) or "") for p in pages).strip()
        structured["text_item_differences"] = [view[1] for view in text_views]

        return OCRResult(
            engine_name=self.name,
            document_path=str(document_path),
            raw_text=raw_text,
            # The parse result carries per-page items with types and boxes;
            # keeping only the concatenated markdown threw that away.
            structured_output=structured,
            latency_seconds=elapsed,
            cost_per_page=COST_PER_PAGE,
            page_count=page_count,
            model_version="llamaparse-latest",
        )

    def _run_extract(self, document_path: Path, schema: dict) -> OCRResult:
        """One stateless vendor extraction job, using the historical Multimodal mode.

        The installed llama-cloud SDK sends one HTTP request per method (no
        automatic retries). Polling reads the same job; it never resubmits it.
        No persistent per-document agents or randomized agent names are needed.
        """
        import base64
        import json
        import time
        from llama_cloud import ExtractConfig, ExtractMode, ExtractTarget, FileData, StatusEnum
        from llama_cloud.client import LlamaCloud

        t0 = self._timer()
        client = LlamaCloud(token=os.environ["LLAMA_CLOUD_API_KEY"])
        config = ExtractConfig(extraction_mode=ExtractMode.MULTIMODAL,
                               extraction_target=ExtractTarget.PER_DOC)
        effective_schema = _sanitize_schema_for_llama(schema)
        if effective_schema != schema:
            raise ValueError("Vendor schema conversion would change the shared schema")
        suffix = document_path.suffix.lower()
        mime = "application/pdf" if suffix == ".pdf" else "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
        job = client.llama_extract.extract_stateless(
            config=config, data_schema=effective_schema,
            file=FileData(data=base64.b64encode(document_path.read_bytes()).decode(), mime_type=mime))
        reply = None
        error = None
        try:
            while True:
                status = client.llama_extract.get_job(job.id)
                if status.status in (StatusEnum.SUCCESS, StatusEnum.PARTIAL_SUCCESS):
                    reply = client.llama_extract.get_run_by_job_id(job.id)
                    break
                if status.status != StatusEnum.PENDING:
                    raise RuntimeError(f"Job {job.id} ended with {status.status}")
                if self._elapsed(t0) > 600:
                    raise TimeoutError(f"Job {job.id} still pending; retrieve it, do not resubmit")
                time.sleep(2)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}; existing extraction job {job.id}, do not resubmit"
        data = getattr(reply, "data", None) or {}
        return OCRResult(
            engine_name=self.name, document_path=str(document_path),
            raw_text=json.dumps(data, ensure_ascii=False),
            structured_output=provider_payload(reply) if reply else {},
            latency_seconds=self._elapsed(t0), cost_per_page=EXTRACT_COST_PER_PAGE,
            page_count=1, model_version="llama-extract",
            metadata={"extraction_mode": "MULTIMODAL", "extraction_target": "PER_DOC",
                      "interface": "v1 stateless extraction", "job_id": job.id,
                      "submission_attempts": 1, "cost_is_estimate": True},
            raw_response={"job": provider_payload(job), "result": provider_payload(reply) if reply else {}},
            error=error,
        )
