"""Datalab hosted API: /api/v1/convert (default conversion) and /api/v1/extract
(schema leg). Sources read 2026-09-30:

- https://documentation.datalab.to/api-reference/convert-document
- https://documentation.datalab.to/api-reference/convert-result-check
- https://documentation.datalab.to/docs/recipes/conversion/conversion-api-overview
- https://documentation.datalab.to/api-reference/extract-structured-data
- https://documentation.datalab.to/docs/recipes/structured-extraction/api-overview
- https://documentation.datalab.to/docs/recipes/structured-extraction/balanced-mode
- https://documentation.datalab.to/platform/billing
- https://documentation.datalab.to/docs/welcome/sdk/conversion (cost_breakdown keys)
- https://www.datalab.to/pricing

Flow (per the conversion overview's REST example): multipart POST with
X-API-Key -> persist request_id / request_check_url -> poll the check URL until
status == "complete" (or "failed") under a deadline with backoff -> if the
reply carries a signed result_url, download it WITHOUT the API key and merge
the non-null polling fields over it -> check `success` (a completed job can be
success=false) -> parse.

The hosted service's model versions are not pinned or pinnable from the
client; the `versions` dict the service returns is recorded as model_version.
Hosted models are not assumed identical to the public Surya/Chandra weights.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

from runners.base import OCRResult, OCRRunner, OCRTask, provider_payload

API_BASE = "https://www.datalab.to/api/v1"

# Explicit conversion mode. "accurate" would be a separate engine id
# (different price and pipeline), never a silent change to this one.
CONVERT_MODE = "balanced"
# markdown -> raw_text (the service's own Markdown); json -> block tree with
# bbox/polygon per block (conversion overview: "json: includes bounding boxes
# and block types"). Comma-separated multi-format per the API reference.
OUTPUT_FORMAT = "markdown,json"
# Keep headers/footers: the documented additional_config keys.
ADDITIONAL_CONFIG = {"keep_pageheader_in_output": True,
                     "keep_pagefooter_in_output": True}
# Word boxes are a paid add-on (+$3/1K pages) and only appear in HTML output.
# OFF: geometry for this engine comes from JSON block boxes (granularity
# "block"). Word geometry would be a separately identified configuration.
WORD_BBOXES = False
EXTRAS = ""                         # none requested
# Image extraction returns every figure as base64 in the reply; nothing in the
# benchmark scores it and it inflates the persisted payload. Synthetic image
# captions (disable_image_captions) stay at the service default (on).
DISABLE_IMAGE_EXTRACTION = True
# A cached result would make latency meaningless for a benchmark run.
SKIP_CACHE = True

# Schema leg: the vendor's Extract API. balanced is the documented default
# extraction_mode; stated explicitly because teams with recent extraction use
# default to "fast" (structured-extraction overview, note dated June 4 2026).
# `mode` (parsing) is set to the engine's conversion mode; when omitted, the
# service parses balanced extraction in "accurate" mode.
EXTRACTION_MODE = "balanced"

# List rates (www.datalab.to/pricing, read 2026-09-30), used ONLY when the
# reply carries no cost_breakdown. Billing rounds each request up to the cent.
CONVERT_BALANCED_USD_PER_PAGE = 4.00 / 1000
EXTRACT_BALANCED_USD_PER_PAGE = 15.00 / 1000   # plus parsing; "+ fees" possible

POLL_DEADLINE_SECONDS = 900
POLL_INITIAL_DELAY = 2.0
POLL_MAX_DELAY = 15.0
HTTP_TIMEOUT = 120

IMAGE_BLOCK_TYPES = frozenset({"Picture", "Figure", "Diagram", "PictureGroup",
                               "FigureGroup"})
_META_SUFFIXES = ("_citations", "_meta", "_score")
_TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)


def _mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def _html_text(html_text: str) -> str:
    from bs4 import BeautifulSoup

    return BeautifulSoup(html_text or "", "html.parser").get_text(" ", strip=True)


def _strip_extraction_meta(node):
    """Drop the documented per-field annotations (`<field>_citations`,
    `<field>_meta`, `<field>_score`) that sit beside each extracted value, so
    the scored JSON is the schema instance. Only removed when the base field
    exists in the same object; the unmodified reply is kept separately."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            base = next((key[: -len(s)] for s in _META_SUFFIXES if key.endswith(s)), None)
            if base is not None and base in node:
                continue
            out[key] = _strip_extraction_meta(value)
        return out
    if isinstance(node, list):
        return [_strip_extraction_meta(v) for v in node]
    return node


def _json_blocks(doc) -> tuple[list[dict], list[dict], list[dict]]:
    """Walk the convert JSON tree -> (layout_blocks, text_regions, tables).

    UNVERIFIED shape: the API reference types `json` only as an object. The
    walker assumes the open-source Marker JSONOutput layout (Document ->
    Page children -> blocks with block_type/html/polygon/bbox/children;
    github.com/datalab-to/marker marker/renderers/json.py) and normalises by
    each Page block's own bbox. Anything else yields no geometry rather than
    guessed geometry. Probe: inspect raw_response.result.json on 2 pages.
    """
    layout, regions, tables = [], [], []
    if isinstance(doc, str):
        try:
            doc = json.loads(doc)
        except json.JSONDecodeError:
            return layout, regions, tables
    if not isinstance(doc, dict):
        return layout, regions, tables

    def norm(bbox, page_box):
        try:
            px0, py0, px1, py1 = (float(v) for v in page_box)
            x0, y0, x1, y1 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            return None
        w, h = px1 - px0, py1 - py0
        if w <= 0 or h <= 0:
            return None
        clamp = lambda v: min(1.0, max(0.0, v))  # noqa: E731
        out = [clamp((x0 - px0) / w), clamp((y0 - py0) / h),
               clamp((x1 - px0) / w), clamp((y1 - py0) / h)]
        return out if out[2] > out[0] and out[3] > out[1] else None

    def walk(block, page_index, page_box):
        children = block.get("children") or []
        btype = str(block.get("block_type") or "")
        if children:
            for child in children:
                if isinstance(child, dict):
                    walk(child, page_index, page_box)
            return
        html = block.get("html") or ""
        bbox = norm(block.get("bbox"), page_box) if block.get("bbox") else None
        order = len(layout)
        layout.append({"label": btype, "id": block.get("id"), "page": page_index,
                       "order": order, "bbox": bbox, "content": html})
        for match in _TABLE_RE.finditer(html):
            tables.append({"html": match.group(0), "block_type": btype,
                           "page": page_index})
        text = _html_text(html)
        if text and bbox and btype not in IMAGE_BLOCK_TYPES:
            regions.append({"text": text, "bbox": bbox, "label": btype,
                            "page": page_index})

    pages = [c for c in (doc.get("children") or []) if isinstance(c, dict)]
    for page_index, page in enumerate(pages):
        page_box = page.get("bbox")
        if not page_box:
            continue
        for child in page.get("children") or []:
            if isinstance(child, dict):
                walk(child, page_index, page_box)
    return layout, regions, tables


class DatalabRunner(OCRRunner):
    """Datalab hosted Convert (mode=balanced) + Extract (schema leg)."""

    name = "datalab_ocr"
    supports_pdf = True
    accepts_task = True   # schema leg only, via the vendor Extract API

    def __init__(self) -> None:
        import requests

        self._requests = requests
        self._api_key = os.environ["DATALAB_API_KEY"]
        self._session = requests.Session()

    # ---------------------------------------------------------------- entry
    def run(self, document_path: Path, task: OCRTask | None = None) -> OCRResult:
        if task is None:
            return self._run_convert(document_path)
        if task.leg == "schema" and task.json_schema:
            return self._run_extract(document_path, task.json_schema)
        # Convert takes no prompt; key_value has no schema-free Extract
        # contract. Never silently answer a task with the default conversion.
        raise ValueError(
            f"datalab_ocr cannot answer task leg {task.leg!r}: Convert has no "
            "prompt, and Extract requires a JSON schema (schema leg only)")

    # ---------------------------------------------------------------- HTTP
    def _submit(self, endpoint: str, path: Path, data: dict) -> dict:
        url = f"{API_BASE}/{endpoint}"
        for attempt in range(4):
            with path.open("rb") as fh:
                resp = self._session.post(
                    url, headers={"X-API-Key": self._api_key},
                    files={"file": (path.name, fh, _mime(path))},
                    data=data, timeout=HTTP_TIMEOUT)
            # 429 = not accepted; safe to resubmit. Other errors are not
            # retried: a 5xx after upload may still have created a billed job.
            if resp.status_code == 429 and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            body = resp.json()
            if body.get("success") is False:
                raise RuntimeError(f"Datalab {endpoint} rejected: {body.get('error')}")
            if not body.get("request_check_url"):
                raise RuntimeError(f"Datalab {endpoint} reply has no request_check_url")
            return body
        raise RuntimeError(f"Datalab {endpoint}: rate limited after retries")

    def _poll(self, check_url: str) -> tuple[dict, bool]:
        host = urlparse(check_url).hostname or ""
        if not (host == "datalab.to" or host.endswith(".datalab.to")):
            # Never send the key to a host the API did not document.
            raise RuntimeError(f"unexpected request_check_url host {host!r}")
        deadline = time.monotonic() + POLL_DEADLINE_SECONDS
        delay = POLL_INITIAL_DELAY
        while True:
            try:
                resp = self._session.get(check_url, headers={"X-API-Key": self._api_key},
                                         timeout=HTTP_TIMEOUT)
                if resp.status_code == 429 or resp.status_code >= 500:
                    result = None
                else:
                    resp.raise_for_status()
                    result = resp.json()
            except self._requests.ConnectionError:
                result = None
            if result is not None:
                status = result.get("status")
                if status == "failed":
                    return result, False
                if status == "complete":
                    downloaded = False
                    if result.get("result_url"):
                        # Signed URL authorises itself: no API key header.
                        dl = self._requests.get(result["result_url"], timeout=HTTP_TIMEOUT)
                        dl.raise_for_status()
                        result = {**dl.json(),
                                  **{k: v for k, v in result.items() if v is not None}}
                        downloaded = True
                    return result, downloaded
            if time.monotonic() + delay > deadline:
                raise TimeoutError(f"Datalab result not ready after "
                                   f"{POLL_DEADLINE_SECONDS}s: {check_url}")
            time.sleep(delay)
            delay = min(POLL_MAX_DELAY, delay * 1.5)

    @staticmethod
    def _cost(result: dict, page_count: int, list_rate: float) -> tuple[float, str]:
        breakdown = result.get("cost_breakdown") or {}
        cents = breakdown.get("final_cost_cents") if isinstance(breakdown, dict) else None
        if isinstance(cents, (int, float)):
            return (cents / 100.0) / max(1, page_count), "cost_breakdown.final_cost_cents"
        return list_rate, "list rate constant (no cost_breakdown returned)"

    @staticmethod
    def _model_version(result: dict, submit: dict, requested: str) -> str:
        versions = result.get("versions") or submit.get("versions")
        if versions:
            if not isinstance(versions, str):
                versions = json.dumps(versions, sort_keys=True)
            return f"{requested} versions={versions}"
        return f"{requested} (requested)"

    def _metadata(self, submit: dict, result: dict, downloaded: bool, config: dict,
                  cost_usd: float, cost_source: str) -> dict:
        md = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        return {
            "request_id": submit.get("request_id"),
            "request_check_url": submit.get("request_check_url"),
            "status": result.get("status"),
            "success": result.get("success"),
            "result_url_downloaded": downloaded,
            "request_config": config,
            "cost_breakdown": result.get("cost_breakdown"),
            "cost_usd_per_page": cost_usd,
            "cost_source": cost_source,
            "page_count": result.get("page_count"),
            "failed_pages": md.get("failed_pages"),
            "parse_quality_score": result.get("parse_quality_score"),
            "runtime": result.get("runtime"),
            "versions": result.get("versions") or submit.get("versions"),
        }

    # ------------------------------------------------------------ convert
    def _run_convert(self, path: Path) -> OCRResult:
        t0 = self._timer()
        config = {
            "mode": CONVERT_MODE,
            "output_format": OUTPUT_FORMAT,
            "additional_config": json.dumps(ADDITIONAL_CONFIG),
            "word_bboxes": str(WORD_BBOXES).lower(),
            "disable_image_extraction": str(DISABLE_IMAGE_EXTRACTION).lower(),
            "skip_cache": str(SKIP_CACHE).lower(),
        }
        if EXTRAS:
            config["extras"] = EXTRAS
        submit = self._submit("convert", path, config)
        result, downloaded = self._poll(submit["request_check_url"])
        latency = self._elapsed(t0)

        page_count = int(result.get("page_count") or 1)
        cost, cost_source = self._cost(result, page_count, CONVERT_BALANCED_USD_PER_PAGE)
        requested = f"datalab-convert mode={CONVERT_MODE}"
        error = None
        if result.get("success") is False or result.get("status") == "failed":
            error = f"DatalabError: {result.get('error') or 'conversion failed'}"

        layout, regions, tables = _json_blocks(result.get("json"))
        structured: dict = {}
        if tables:
            structured["tables"] = tables
        if regions:
            structured["text_regions"] = regions
            structured["geometry_granularity"] = "block"
        if layout:
            structured["layout_blocks"] = layout

        return OCRResult(
            engine_name=self.name,
            document_path=str(path),
            raw_text=(result.get("markdown") or "") if not error else "",
            structured_output=structured,
            latency_seconds=latency,
            cost_per_page=cost,
            page_count=page_count,
            model_version=self._model_version(result, submit, requested),
            metadata=self._metadata(submit, result, downloaded, config, cost, cost_source),
            raw_response={"submit": provider_payload(submit),
                          "result": provider_payload(result)},
            error=error,
        )

    # ------------------------------------------------------------ extract
    def _run_extract(self, path: Path, schema: dict) -> OCRResult:
        if not isinstance(schema, dict) or "properties" not in schema:
            # Extract requires a 'properties' key (API reference, page_schema).
            raise ValueError("Datalab Extract needs a JSON schema with 'properties'")
        t0 = self._timer()
        config = {
            "page_schema": json.dumps(schema),
            "extraction_mode": EXTRACTION_MODE,
            "mode": CONVERT_MODE,
            "skip_cache": str(SKIP_CACHE).lower(),
        }
        submit = self._submit("extract", path, config)
        result, downloaded = self._poll(submit["request_check_url"])
        latency = self._elapsed(t0)

        page_count = int(result.get("page_count") or 1)
        cost, cost_source = self._cost(
            result, page_count, EXTRACT_BALANCED_USD_PER_PAGE + CONVERT_BALANCED_USD_PER_PAGE)
        requested = f"datalab-extract extraction_mode={EXTRACTION_MODE} mode={CONVERT_MODE}"
        error = None
        if result.get("success") is False or result.get("status") == "failed":
            error = f"DatalabError: {result.get('error') or 'extraction failed'}"

        raw = result.get("extraction_schema_json")
        parsed = raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
        answer = (json.dumps(_strip_extraction_meta(parsed), ensure_ascii=False)
                  if parsed is not None else (raw if isinstance(raw, str) else ""))
        meta_config = {k: v for k, v in config.items() if k != "page_schema"}
        return OCRResult(
            engine_name=self.name,
            document_path=str(path),
            raw_text=answer if not error else "",
            structured_output={
                "raw_model_output": raw if isinstance(raw, str) else json.dumps(raw),
                "extraction_mode": result.get("extraction_mode"),
            },
            latency_seconds=latency,
            cost_per_page=cost,
            page_count=page_count,
            model_version=self._model_version(result, submit, requested),
            metadata=self._metadata(submit, result, downloaded, meta_config,
                                    cost, cost_source),
            raw_response={"submit": provider_payload(submit),
                          "result": provider_payload(result)},
            error=error,
        )
