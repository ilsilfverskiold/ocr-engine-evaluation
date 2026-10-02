from __future__ import annotations

from pathlib import Path
from typing import Any
from html import escape

from runners.base import OCRResult, OCRRunner, OCRTask, provider_payload

COST_PER_PAGE = 0.065  # AnalyzeDocument TABLES + FORMS in US West/Oregon: $0.015 + $0.050/page
TABLE_ADAPTER_VERSION = "textract-merged-cells-v1 (2026-10-02)"


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
        uniform runner contract and ignored. Key-value code checks the native
        FORMS pairs and table cells; this engine has no schema interface."""
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
            "table_adapter_version": TABLE_ADAPTER_VERSION,
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

    @classmethod
    def _extract_tables(cls, blocks: list[dict], *, _include_merged: bool = True) -> list[dict[str, Any]]:
        """Walk the block graph into HTML tables.

        Textract returns cells, not records: each CELL carries RowIndex,
        ColumnIndex; merged spans are separate MERGED_CELL relationships.
        https://docs.aws.amazon.com/textract/latest/dg/how-it-works-tables.html
        Cell text lives in
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
                        "text": cls._cell_text(cell, block_map),
                        "rowspan": int(cell.get("RowSpan", 1) or 1),
                        "colspan": int(cell.get("ColumnSpan", 1) or 1),
                    }

            # MERGED_CELL points to child CELL blocks, which in turn point to
            # WORDs. Looking only at TABLE.CHILD loses the provider's joins.
            for rel in block.get("Relationships", []):
                if not _include_merged or rel["Type"] != "MERGED_CELL":
                    continue
                for merged_id in rel["Ids"]:
                    merged = block_map.get(merged_id, {})
                    if merged.get("BlockType") != "MERGED_CELL":
                        continue
                    children = [block_map[cid] for child_rel in merged.get("Relationships", [])
                                if child_rel["Type"] == "CHILD" for cid in child_rel["Ids"]
                                if cid in block_map and block_map[cid].get("BlockType") == "CELL"]
                    children.sort(key=lambda c: (c["RowIndex"], c["ColumnIndex"]))
                    text = " ".join(filter(None, (cls._cell_text(c, block_map) for c in children)))
                    cells[merged["RowIndex"], merged["ColumnIndex"]] = {
                        "text": text,
                        "rowspan": int(merged.get("RowSpan", 1) or 1),
                        "colspan": int(merged.get("ColumnSpan", 1) or 1),
                    }

            if not cells:
                continue

            max_row = max(r + cell["rowspan"] - 1 for (r, c), cell in cells.items())
            max_col = max(c + cell["colspan"] - 1 for (r, c), cell in cells.items())

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
                    cells_html.append(f"<td{attrs}>{escape(cell['text'])}</td>")
                # Even an entirely covered row occupies a rowspan position.
                html_rows.append(f"<tr>{''.join(cells_html)}</tr>")
            tables.append({"html": f"<table>{''.join(html_rows)}</table>"})
        if _include_merged:
            # The provider supplies both primitive cells and merged groups.
            # Preserve both; joining a multiline address must not discard its
            # original constituent cells or existing key/value evidence.
            elementary = cls._extract_tables(blocks, _include_merged=False)
            for table, source in zip(tables, elementary):
                table["cell_html"] = source["html"]
        return tables

    @staticmethod
    def _cell_text(cell: dict, block_map: dict) -> str:
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
