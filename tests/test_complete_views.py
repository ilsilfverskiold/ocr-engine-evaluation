import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_benchmark as rb
from evaluation.modal_metrics import audit_modal_records
from modal_apps.mineru_pipeline_app import _complete_views
from modal_apps.paddleocr_vl_app import _complete_markdown, _plain_text_from_blocks
from runners.docling_runner import DoclingRunner, _complete_export
from runners.llamaparse_runner import LlamaParseRunner


OUTPUTS = (Path(__file__).resolve().parents[1]
           / "results/ocr_outputs/reconciled_outputs")


def saved_call(suffix):
    matches = list(OUTPUTS.glob(f"**/{suffix}"))
    if len(matches) != 1:
        raise AssertionError((suffix, matches))
    return json.loads(matches[0].read_text())


class CompleteViewsTest(unittest.TestCase):
    def test_replayed_view_is_not_a_second_modal_measurement(self):
        rate = 0.000694
        measured = {
            "engine_name": "paddleocr_vl", "document_path": "invoice.png",
            "pass_name": "default", "timestamp": "physical",
            "measurement_scope": "modal_batch", "batch_id": "batch-one",
            "batch_gpu_seconds": 2.0, "batch_size": 1,
            "effective_gpu_seconds_per_document": 2.0,
            "latency_seconds": 2.0, "steady_latency_seconds": 2.0,
            "cost_per_page": 2 * rate, "steady_cost_per_page": 2 * rate,
            "startup_excluded": True, "gpu_type": "A100-80GB",
            "gpu_rate_usd_per_second": rate,
        }
        replay = {**measured, "timestamp": "replay", "measurement_scope": "replayed_saved_call",
                  "batch_id": None, "batch_size": None, "batch_gpu_seconds": None}
        summaries, errors, used_batches = audit_modal_records([measured, replay])
        self.assertTrue(used_batches)
        self.assertEqual(errors, [])
        self.assertEqual(summaries[0]["documents"], 1)
        self.assertEqual(rb.engine_operational([measured, replay])["paddleocr_vl"]["documents"], 1)

    def test_llamaparse_scores_native_text_and_markdown_separately(self):
        runner = LlamaParseRunner.__new__(LlamaParseRunner)
        runner._parser = SimpleNamespace(parse=lambda _: SimpleNamespace(
            pages=[SimpleNamespace(text="plain Invoice no: 123", md="# Invoice no: 123",
                                   items=[], pageHeaderMarkdown="", pageFooterMarkdown="")]))
        result = runner.run(Path("invoice.png"))
        self.assertEqual(result.raw_text, "plain Invoice no: 123")
        self.assertEqual(result.structured_output["markdown"], "# Invoice no: 123")

    def test_docling_exports_both_views_with_furniture(self):
        from docling_core.types.doc import ContentLayer, DocItemLabel, DoclingDocument

        sample = DoclingDocument(name="layer-check")
        sample.add_text(label=DocItemLabel.TEXT, text="FURNITURE",
                        content_layer=ContentLayer.FURNITURE)
        sample.add_text(label=DocItemLabel.TEXT, text="BODY",
                        content_layer=ContentLayer.BODY)
        self.assertNotIn("FURNITURE", sample.export_to_text())
        self.assertIn("FURNITURE", sample.export_to_text(
            included_content_layers=set(ContentLayer)))

        seen = []

        class Document:
            tables = []
            pages = [1]

            def export_to_text(self, **kwargs):
                assert kwargs["traverse_pictures"] is True
                seen.append(("text", kwargs["included_content_layers"]))
                return "FURNITURE\nBODY"

            def export_to_markdown(self, **kwargs):
                assert kwargs["traverse_pictures"] is True
                seen.append(("markdown", kwargs["included_content_layers"]))
                return "# FURNITURE\nBODY"

            def export_to_dict(self):
                return {"texts": []}

        runner = DoclingRunner()
        with patch.object(runner, "_get_converter", return_value=SimpleNamespace(
                convert=lambda _: SimpleNamespace(document=Document()))):
            result = runner.run(Path("invoice.png"))
        self.assertEqual(result.raw_text, "FURNITURE\nBODY")
        self.assertEqual(result.structured_output["markdown"], "# FURNITURE\nBODY")
        self.assertEqual(len(seen), 2)
        for _, layers in seen:
            self.assertEqual(layers, set(ContentLayer))

        native = {"texts": [{"label": "footnote", "content_layer": "body",
                             "parent": {"$ref": "#/tables/0"},
                             "text": "1 Net Change in Portfolio is the difference."}]}
        completed, additions = _complete_export("Account statement", native)
        self.assertIn("Net Change in Portfolio", completed)
        self.assertEqual(additions[0]["label"], "footnote")
        native["tables"] = [{"data": {"table_cells": [
            {"text": "Current Period"}, {"text": "END OF PERIOD"}]}}]
        native["texts"][0]["parent"] = {"$ref": "#/tables/0"}
        completed, additions = _complete_export(
            "| Current Period |\n| END OF PERIOD |\n\nNEXT SECTION", native)
        self.assertLess(completed.index("Net Change in Portfolio"),
                        completed.index("NEXT SECTION"))
        self.assertIsNotNone(additions[0]["insert_after"])
        escaped, additions = _complete_export("SPAR\\_Saved", {"texts": [
            {"label": "text", "text": "SPAR_Saved"}]})
        self.assertEqual(escaped, "SPAR\\_Saved")
        self.assertEqual(additions, [])

    def test_paddle_saved_header_reaches_both_views(self):
        call = saved_call("easy__invoice__invoice_003/paddleocr_vl/default.call.json")
        blocks = call["structured_output"]["layout_blocks"]
        text = _plain_text_from_blocks(blocks)
        markdown, supplements = _complete_markdown(
            call["raw_text"], blocks, historical_filtered=True)
        self.assertIn("Invoice no: 26343874", text)
        self.assertIn("Invoice no: 26343874", markdown)
        self.assertTrue(any(item["label"] == "header" for item in supplements))

    def test_mineru_saved_footer_and_table_plain_text(self):
        call = saved_call("easy__receipt__sroie_001/mineru_pipeline/default.call.json")
        so = call["structured_output"]
        text, markdown, supplements = _complete_views(so["middle_json_page"], so["markdown"])
        footer = "GOODS SOLD ARE NOT RETURNABLE"
        self.assertIn(footer, text)
        self.assertIn(footer, markdown)
        self.assertNotIn("<table", text.lower())
        self.assertTrue(any(item["source"] == "discarded_blocks" and footer in item["text"]
                            for item in supplements))


if __name__ == "__main__":
    unittest.main()
