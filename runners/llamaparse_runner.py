from __future__ import annotations

import os
import html
import re
from pathlib import Path

from runners.base import OCRResult, OCRRunner, OCRTask, provider_payload, sent_record

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


GRANULAR_TIER = "cost_effective"
GRANULAR_LEVELS = ("word", "line")


def image_frame(document_path) -> tuple[int, int] | None:
    """Pixel size of an image input; None for a PDF (its page frame applies)."""
    path = Path(document_path)
    if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp") \
            or not path.is_file():
        return None
    from PIL import Image
    with Image.open(path) as image:
        return image.size


def layout_blocks_from_pages(pages: list[dict], frame: tuple[float, float] | None) -> list[dict]:
    """Item boxes (heading, text, table) from a parse reply as normalized xyxy.

    ``items[].bBox`` is ``x, y, w, h`` with a top-left origin in the frame of
    the page image LlamaParse rendered. For an image input that is the source
    image's pixel size (checked on all 17 geometry pages: item envelopes match
    the references at scale 1.00), NOT the ``width``/``height`` the page
    reports, which are 72-dpi points (754 px -> 271.44 pt). ``frame`` is that
    pixel size; for a PDF input (None) the page's own width/height apply.
    One box per layout item, never per word: scored as ``layout_block``.
    """
    blocks: list[dict] = []
    for page in pages:
        width, height = frame or (page.get("width"), page.get("height"))
        if not (width and height):
            continue
        for item in page.get("items") or []:
            box = item.get("bBox") or {}
            if not all(k in box for k in ("x", "y", "w", "h")):
                continue
            text = item.get("value") or item.get("md") or ""
            if item.get("type") == "table" and item.get("rows"):
                text = " ".join(" ".join(str(cell) for cell in row) for row in item["rows"])
            if not str(text).strip():
                continue
            blocks.append({"text": str(text), "label": item.get("type"),
                           "bbox": [box["x"] / width, box["y"] / height,
                                    (box["x"] + box["w"]) / width, (box["y"] + box["h"]) / height]})
    return blocks


def regions_from_grounded_items(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Word and line boxes from the ``grounded_items`` sidecar.

    One row per page with ``page_width``/``page_height`` (page points,
    top-left origin), so the frame is declared per row. A word or line
    carries a ``bbox`` (x, y, w, h) and a ``span``, a [start, end) UTF-8 byte
    range into the text it belongs to: for a text item the item's ``md`` (or
    ``caption``, per ``grounding.source``); for a table item each cell's own
    text, which the item carries as ``rows[r][c]`` (checked on the funsd_003
    probe: cell span [0, 15] is "NAME OF ACCOUNT", not the Markdown's
    "| NAME OF ACCOU"). Returns (words, lines), both normalized xyxy.
    """
    words: list[dict] = []
    lines: list[dict] = []
    for row in rows:
        if not row.get("success", True):
            continue
        page_w, page_h = row.get("page_width"), row.get("page_height")
        if not (page_w and page_h):
            continue

        def norm(box):
            return [box["x"] / page_w, box["y"] / page_h,
                    (box["x"] + box["w"]) / page_w, (box["y"] + box["h"]) / page_h]

        def collect(lines_node, text: str) -> None:
            text_bytes = str(text or "").encode("utf-8")

            def sliced(span):
                try:
                    return text_bytes[int(span[0]):int(span[1])].decode("utf-8", errors="replace")
                except (TypeError, ValueError, IndexError):
                    return ""

            for line in lines_node or []:
                if not isinstance(line, dict):
                    continue
                if isinstance(line.get("bbox"), dict) and line.get("span"):
                    lines.append({"text": sliced(line["span"]), "bbox": norm(line["bbox"])})
                for word in line.get("words") or []:
                    if isinstance(word, dict) and isinstance(word.get("bbox"), dict) and word.get("span"):
                        words.append({"text": sliced(word["span"]), "bbox": norm(word["bbox"])})

        for item in row.get("items") or []:
            grounding = item.get("grounding") or {}
            if isinstance(grounding.get("rows"), list):
                cell_texts = item.get("rows") or []
                for r, cells in enumerate(grounding["rows"]):
                    for c, cell in enumerate(cells or []):
                        if not isinstance(cell, dict):
                            continue
                        try:
                            text = cell_texts[r][c]
                        except (IndexError, TypeError):
                            text = ""
                        collect(cell.get("lines"), text)
            else:
                source = grounding.get("source") or "md"
                collect(grounding.get("lines"), item.get(source) or item.get("md") or "")
    return ([w for w in words if w["text"].strip()],
            [l for l in lines if l["text"].strip()])


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
        frame = image_frame(document_path)
        # An image whose pixel frame cannot be read gets no declared boxes:
        # the page's point frame would misplace them. The stored reply still
        # carries the items, so scoring can recover them with the image.
        blocks = (layout_blocks_from_pages(page_payloads, frame)
                  if frame is not None or Path(document_path).suffix.lower() == ".pdf" else [])
        if blocks:
            # Declared geometry contract: item boxes in the image pixel frame, normalized.
            structured["text_regions"] = blocks
            structured["geometry_granularity"] = "layout_block"
            structured["geometry_coordinates"] = "normalized_xyxy"

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
            metadata={"sent": sent_record(None, "no instruction text: the parse endpoint takes no prompt")},
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
        # cite_sources: the vendor's documented grounding output, one citation
        # (page, verbatim text, box) per leaf field, no listed extra credits.
        # Saved with the call; nothing scores it.
        config = ExtractConfig(extraction_mode=ExtractMode.MULTIMODAL,
                               extraction_target=ExtractTarget.PER_DOC,
                               cite_sources=True)
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
                      "cite_sources": True,
                      "interface": "v1 stateless extraction", "job_id": job.id,
                      "submission_attempts": 1, "cost_is_estimate": True,
                      "sent": sent_record(
                          None, "no instruction text: LlamaExtract receives only the schema",
                          "JSON schema as LlamaExtract data_schema (MULTIMODAL)",
                          schema_separately=True)},
            raw_response={"job": provider_payload(job), "result": provider_payload(reply) if reply else {}},
            error=error,
        )



class LlamaParseGranularRunner(OCRRunner):
    """LlamaParse with ``output_options.granular_bboxes``: one box per word and
    per line, the parse endpoint's documented geometry output.

    A separately measured configuration beside ``llamaparse`` (same tier and
    price; the pattern of textract_text / textract_structured), so the saved
    conversion run is untouched. The installed SDK predates the option, so
    the request carries it as plain JSON through the SDK's HTTP client and the
    finished job is read back as raw JSON: the SDK's typed result model would
    drop ``result_content_metadata.grounded_items``. The sidecar rows are
    stored whole so the adapter can be corrected offline without re-paying.
    """

    name = "llamaparse_granular"
    supports_pdf = True
    accepts_task = False

    def __init__(self) -> None:
        from llama_cloud.client import LlamaCloud
        self._client = LlamaCloud(token=os.environ["LLAMA_CLOUD_API_KEY"])

    def _get_json(self, path: str) -> dict:
        import urllib.parse
        wrapper = self._client._client_wrapper
        response = wrapper.httpx_client.request(
            "GET", urllib.parse.urljoin(f"{wrapper.get_base_url()}/", path),
            headers=wrapper.get_headers(), timeout=60)
        response.raise_for_status()
        return response.json()

    def run(self, document_path: Path, task: OCRTask | None = None) -> OCRResult:
        import json
        import time
        import httpx

        t0 = self._timer()
        with open(document_path, "rb") as handle:
            uploaded = self._client.files.upload_file(upload_file=handle)
        options = {"granular_bboxes": list(GRANULAR_LEVELS)}
        job = self._client.v_2.parse_file(file_id=uploaded.id, tier=GRANULAR_TIER,
                                          version="latest", output_options=options)
        while True:
            result = self._get_json(f"api/v2/parse/{job.id}")
            status = str((result.get("job") or {}).get("status") or result.get("status") or "").upper()
            if status == "COMPLETED":
                break
            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"parse job {job.id} ended {status}: "
                                   f"{(result.get('job') or {}).get('error_message')}")
            if self._elapsed(t0) > 600:
                raise TimeoutError(f"parse job {job.id} still {status}; retrieve it, do not resubmit")
            time.sleep(2)
        sidecar = (result.get("result_content_metadata") or {}).get("grounded_items") or {}
        url = sidecar.get("presigned_url")
        if not url:
            raise RuntimeError(f"parse job {job.id} completed without a grounded_items sidecar; "
                               "granular_bboxes may be unsupported for this input")
        download = httpx.get(url, timeout=120)
        download.raise_for_status()
        rows = [json.loads(line) for line in download.text.splitlines() if line.strip()]
        elapsed = self._elapsed(t0)
        words, lines = regions_from_grounded_items(rows)
        raw_text = "\n".join(line["text"] for line in lines) or " ".join(word["text"] for word in words)
        structured = {
            "grounded_items": rows,
            "text_regions": words,
            "line_regions": lines,
            "geometry_granularity": "word",
            "geometry_coordinates": "normalized_xyxy",
            "granular_bboxes": list(GRANULAR_LEVELS),
            "parse_job": {k: v for k, v in result.items() if k not in ("items", "markdown", "text")},
        }
        return OCRResult(
            engine_name=self.name,
            document_path=str(document_path),
            raw_text=raw_text,
            structured_output=structured,
            latency_seconds=elapsed,
            cost_per_page=COST_PER_PAGE,
            page_count=max(1, len(rows)),
            model_version="llamaparse-latest granular_bboxes",
            metadata={"sent": sent_record(None, "no instruction text: the parse endpoint takes no prompt; "
                                                "output_options.granular_bboxes=[word, line]"),
                      "tier": GRANULAR_TIER, "job_id": job.id, "cost_is_estimate": True,
                      "sidecar_bytes": sidecar.get("size_bytes")},
        )
