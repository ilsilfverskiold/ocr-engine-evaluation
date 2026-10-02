from __future__ import annotations

from pathlib import Path
from typing import Any

from runners.base import OCRResult, OCRRunner, OCRTask, provider_payload

COST_PER_PAGE = 0.065  # AnalyzeDocument TABLES + FORMS in US West/Oregon: $0.015 + $0.050/page


class TextractStructuredRunner(OCRRunner):
    """AWS Textract structured extraction (AnalyzeDocument with TABLES + FORMS)."""

    name = "textract_structured"
    supports_pdf = True
    supports_structured = True

    def __init__(self) -> None:
        import boto3
        self._client = boto3.client("textract")

    def run(self, document_path: Path, task: OCRTask | None = None) -> OCRResult:
        """AnalyzeDocument takes no prompt: `task` is accepted for the
        uniform runner contract and ignored. The schema and key_value legs
        are judged by the LLM judge against the FORMS output."""
        t0 = self._timer()
        from runners.textract_text_runner import _prepare_bytes
        raw_bytes = _prepare_bytes(document_path)

        response = self._client.analyze_document(
            Document={"Bytes": raw_bytes},
            FeatureTypes=["TABLES", "FORMS"],
        )
        elapsed = self._elapsed(t0)

        blocks = response.get("Blocks", [])
        text_lines = [
            b["Text"] for b in blocks if b["BlockType"] == "LINE"
        ]
        raw_text = "\n".join(text_lines)

        from runners.textract_text_runner import _word_boxes
        structured_output = {
            "tables": self._extract_tables(blocks),
            "forms": self._extract_key_values(blocks),
            "words": _word_boxes(blocks),
        }

        return OCRResult(
            engine_name=self.name,
            document_path=str(document_path),
            raw_text=raw_text,
            structured_output=structured_output,
            latency_seconds=elapsed,
            cost_per_page=COST_PER_PAGE,
            page_count=1,
            model_version="aws-textract-analyze-document",
            # structured_output above is OUR walk of the block graph. At
            # $0.065/page, a bug in _extract_tables must be re-derivable from
            # the blocks rather than re-bought.
            raw_response=provider_payload(response),
        )

    def _extract_tables(self, blocks: list[dict]) -> list[dict[str, Any]]:
        """Walk the block graph into HTML tables.

        Textract returns cells, not records: each CELL carries RowIndex,
        ColumnIndex and optional RowSpan/ColumnSpan, and its text lives in
        child WORD/SELECTION_ELEMENT blocks. Three things a naive walk gets
        wrong and this one handles:
          * merged cells — spans are emitted as rowspan/colspan and the covered
            positions are not re-emitted as empty cells,
          * missing positions — a row that skips a column index gets an empty
            <td> so column alignment survives,
          * checkbox cells — SELECTION_ELEMENT children contribute their
            selection status, which is otherwise silently dropped.
        """
        block_map = {b["Id"]: b for b in blocks}
        tables = []
        for block in blocks:
            if block["BlockType"] != "TABLE":
                continue

            cells: dict[tuple[int, int], dict] = {}
            for rel in block.get("Relationships", []):
                if rel["Type"] != "CHILD":
                    continue
                for cell_id in rel["Ids"]:
                    cell = block_map.get(cell_id, {})
                    if cell.get("BlockType") != "CELL":
                        continue
                    cells[(cell["RowIndex"], cell["ColumnIndex"])] = {
                        "text": self._cell_text(cell, block_map),
                        "rowspan": int(cell.get("RowSpan", 1) or 1),
                        "colspan": int(cell.get("ColumnSpan", 1) or 1),
                    }

            if not cells:
                continue

            max_row = max(r for r, _ in cells)
            max_col = max(c for _, c in cells)

            # Positions covered by a merged cell must not be emitted again.
            covered: set[tuple[int, int]] = set()
            for (r, c), cell in cells.items():
                for dr in range(cell["rowspan"]):
                    for dc in range(cell["colspan"]):
                        if dr or dc:
                            covered.add((r + dr, c + dc))

            html_rows = []
            for r in range(1, max_row + 1):
                cells_html = []
                for c in range(1, max_col + 1):
                    if (r, c) in covered:
                        continue
                    cell = cells.get((r, c))
                    if cell is None:
                        cells_html.append("<td></td>")
                        continue
                    attrs = ""
                    if cell["rowspan"] > 1:
                        attrs += f' rowspan="{cell["rowspan"]}"'
                    if cell["colspan"] > 1:
                        attrs += f' colspan="{cell["colspan"]}"'
                    cells_html.append(f"<td{attrs}>{cell['text']}</td>")
                if cells_html:
                    html_rows.append(f"<tr>{''.join(cells_html)}</tr>")
            tables.append({"html": f"<table>{''.join(html_rows)}</table>"})
        return tables

    def _cell_text(self, cell: dict, block_map: dict) -> str:
        """Join a cell's child WORDs; include checkbox state for SELECTION_ELEMENTs."""
        parts: list[str] = []
        for rel in cell.get("Relationships", []):
            if rel["Type"] != "CHILD":
                continue
            for child_id in rel["Ids"]:
                child = block_map.get(child_id, {})
                if child.get("BlockType") == "WORD":
                    parts.append(child.get("Text", ""))
                elif child.get("BlockType") == "SELECTION_ELEMENT":
                    if child.get("SelectionStatus") == "SELECTED":
                        parts.append("[X]")
        return " ".join(p for p in parts if p).strip()

    def _extract_key_values(self, blocks: list[dict]) -> list[dict[str, str]]:
        # One implementation, shared with the scorer: a checkbox value is a
        # SELECTION_ELEMENT (SELECTED / NOT_SELECTED), not a WORD, and used
        # to be dropped here as "".
        from evaluation.kv_schema_code import textract_forms
        return textract_forms(blocks)
