from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from runners.base import OCRResult, OCRRunner, OCRTask


def _complete_export(rendered: str, native_document: dict) -> tuple[str, list[dict]]:
    """Retain recognized text items the Docling serializer did not traverse."""
    import html
    import re

    def words(value):
        clean = html.unescape(re.sub(r"<[^>]+>", " ", value))
        clean = re.sub(r"\\([\\`*_{}\[\]()#+.!<>|])", r"\1", clean)
        return re.findall(r"\w+", clean.casefold())

    output = rendered.rstrip()
    seen = words(output)
    additions = []

    def insertion_after_parent_table(item):
        parent = (item.get("parent") or {}).get("$ref", "")
        if not parent.startswith("#/tables/"):
            return None
        try:
            table = (native_document.get("tables") or [])[int(parent.rsplit("/", 1)[1])]
        except (IndexError, ValueError):
            return None
        cells = [str(cell.get("text") or "").strip() for cell in
                 (table.get("data") or {}).get("table_cells") or []]
        cells = [cell for cell in cells if len(cell) >= 4]
        if not cells:
            return None
        start = output.find(cells[0])
        end = output.find(cells[-1], max(0, start)) if start >= 0 else -1
        if end < 0:
            return None
        line_end = output.find("\n", end + len(cells[-1]))
        return len(output) if line_end < 0 else line_end + 1

    entries = [(f"texts[{index}]", item) for index, item in
               enumerate(native_document.get("texts") or [])]
    for table_index, table in enumerate(native_document.get("tables") or []):
        for cell_index, cell in enumerate((table.get("data") or {}).get("table_cells") or []):
            entries.append((f"tables[{table_index}].cells[{cell_index}]", cell))
    for source, item in entries:
        value = str(item.get("text") or "").strip()
        tokens = words(value)
        if tokens and not any(seen[i:i + len(tokens)] == tokens
                              for i in range(len(seen) - len(tokens) + 1)):
            position = insertion_after_parent_table(item)
            additions.append({"source": source, "label": item.get("label"),
                              "content_layer": item.get("content_layer"),
                              "parent": item.get("parent"), "text": value,
                              "insert_after": position})
            if position is None or position == len(output):
                output += "\n\n" + value
            else:
                output = output[:position] + value + "\n\n" + output[position:]
            seen = words(output)
    return output, additions

def _quiet_torch_stack() -> None:
    """Silence docling's model stack so the benchmark's own output survives.

    Loading docling pulls in torch, transformers and rapidocr, which between
    them emit graph-break traces, deprecation notices and autotune advice —
    hundreds of lines per document, straight to stderr, burying the per-document
    results this tool exists to print. None of it is actionable here: the models
    run correctly, and the warnings are addressed to people editing the models.
    Set before the import, because some of it is configured at import time.
    """
    import logging
    import os
    import warnings

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCHDYNAMO_VERBOSE", "0")
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    os.environ.setdefault("TORCH_LOGS", "-all")
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Seed RapidOCR's logger to prevent its import-time console handler from polluting reports.
    for name in ("RapidOCR", "rapidocr"):
        seeded = logging.getLogger(name)
        if not seeded.handlers:
            seeded.addHandler(logging.NullHandler())
        seeded.propagate = False

    _silence_loggers()
    try:
        import torch._dynamo as dynamo
        dynamo.config.verbose = False
        dynamo.config.suppress_errors = True
    except Exception:
        pass


_NOISY_PREFIXES = ("torch", "transformers", "docling", "rapidocr", "RapidOCR",
                   "PIL", "matplotlib", "timm", "huggingface_hub", "filelock")


def _silence_loggers() -> None:
    """Mute the noisy libraries, including loggers created during import.

    Setting a parent's level is not enough: these libraries create child
    loggers with their own explicit levels and their own handlers, so the walk
    over the live registry has to happen after the import as well as before.
    """
    import logging

    for name in list(logging.root.manager.loggerDict):
        if name.split(".")[0] in _NOISY_PREFIXES or name.startswith(_NOISY_PREFIXES):
            logger = logging.getLogger(name)
            logger.setLevel(logging.ERROR)
            logger.propagate = False


class DoclingRunner(OCRRunner):
    """
    Docling (IBM Research) — local CPU document understanding library.
    Handles both PDF and image inputs natively.
    pip install docling
    """

    name = "docling"
    supports_pdf = True
    supports_structured = True

    def __init__(self) -> None:
        self._converter = None  # lazy init

    def _get_converter(self):
        if self._converter is None:
            _quiet_torch_stack()
            from docling.document_converter import DocumentConverter
            _silence_loggers()          # again: the import created more of them
            self._converter = DocumentConverter()
            _silence_loggers()          # and so did building the pipeline
        return self._converter

    def run(self, document_path: Path, task: OCRTask | None = None) -> OCRResult:
        """Docling is a fixed pipeline: `task` is accepted for the uniform
        runner contract and ignored."""
        t0 = self._timer()
        converter = self._get_converter()
        result = converter.convert(str(document_path))
        elapsed = self._elapsed(t0)

        doc = result.document
        from docling_core.types.doc import ContentLayer
        layers = set(ContentLayer)
        raw_text = doc.export_to_text(included_content_layers=layers, traverse_pictures=True)
        markdown = doc.export_to_markdown(included_content_layers=layers, traverse_pictures=True)
        native_document = doc.export_to_dict()
        if any(str(getattr(item, "text", "")).strip() for item in getattr(doc, "texts", [])) \
                and (not raw_text.strip() or not markdown.strip()):
            raise ValueError("Docling has recognized text items absent from an export")
        provider_text, provider_markdown = raw_text, markdown
        raw_text, text_supplements = _complete_export(raw_text, native_document)
        markdown, markdown_supplements = _complete_export(markdown, native_document)

        # Collect table HTML for structured output
        tables: list[dict] = []
        for table in doc.tables:
            try:
                # `doc` is required as of docling 2.x — without it the call is
                # deprecated, prints a warning per table, and returns an empty
                # string, so the table leg silently fell back to parsing the
                # markdown text and we persisted empty tables in the outputs.
                try:
                    html = table.export_to_html(doc=doc)
                except TypeError:
                    html = table.export_to_html()
                if html:
                    tables.append({"html": html})
            except Exception:
                pass

        structured_output = {
            "markdown": markdown,
            "provider_text": provider_text,
            "provider_markdown": provider_markdown,
            "text_supplements": text_supplements,
            "markdown_supplements": markdown_supplements,
            "content_layers": sorted(layer.value for layer in layers),
            "native_document": native_document,
        }
        if tables:
            structured_output["tables"] = tables

        # Page count from document metadata
        page_count = len(doc.pages) if hasattr(doc, "pages") else 1

        return OCRResult(
            engine_name=self.name,
            document_path=str(document_path),
            raw_text=raw_text,
            structured_output=structured_output,
            latency_seconds=elapsed,
            cost_per_page=0.0,
            page_count=page_count,
            model_version=f"docling=={version('docling')}",
        )
