"""Complete Docling exports from a saved native document, without another OCR pass.

Usage: python scripts/backfill_docling_views.py DOCLING_RUN PATCH_RUN
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_benchmark as rb
from evaluation.ground_truth import GroundTruthLoader
from runners.docling_runner import _complete_export


def main(source_run: str, patch_run: str) -> None:
    records = rb.read_jsonl(rb.RUNS_DIR / f"{source_run}_inference.jsonl")
    if len(records) != 93 or any(rec["engine_name"] != "docling" for rec in records):
        raise ValueError("Expected the complete 93-page Docling conversion run")
    inference_file = rb.RUNS_DIR / f"{patch_run}_inference.jsonl"
    if inference_file.exists():
        raise FileExistsError(inference_file)
    loader = GroundTruthLoader()
    rows = []
    text_added = markdown_added = 0
    for rec in records:
        if rec.get("error"):
            raise ValueError(f"Docling conversion failed for {rec['document_path']}: {rec['error']}")
        doc = Path(rec["document_path"])
        gt = loader.load(doc)
        original = rb._read_stored_call(source_run, "docling", doc, gt, "default")
        if original is None:
            raise FileNotFoundError((source_run, doc))
        fixed = copy.deepcopy(original)
        so = fixed["structured_output"]
        native = so.get("native_document")
        if not native:
            raise ValueError(f"Docling native document missing: {doc}")
        provider_text = fixed["raw_text"]
        provider_markdown = so["markdown"]
        fixed["raw_text"], text_supplements = _complete_export(provider_text, native)
        so["markdown"], markdown_supplements = _complete_export(provider_markdown, native)
        so["provider_text"] = provider_text
        so["provider_markdown"] = provider_markdown
        so["text_supplements"] = text_supplements
        so["markdown_supplements"] = markdown_supplements
        fixed["legs_fed"] = ["text", "markdown", "tables"]
        if _complete_export(fixed["raw_text"], native)[1] or \
                _complete_export(so["markdown"], native)[1]:
            raise ValueError(f"Docling recognized text still absent from a view: {doc}")
        text_added += len(text_supplements)
        markdown_added += len(markdown_supplements)
        digest = hashlib.sha256(json.dumps(original, sort_keys=True, ensure_ascii=False)
                                .encode("utf-8")).hexdigest()
        fixed.setdefault("metadata", {})["view_backfill"] = {
            "source_timestamp": source_run,
            "source_call_json_sha256": digest,
            "method": "complete both exports from retained native Docling document",
        }
        folder = f"{gt.tier}__{gt.doc_type}__{doc.stem}"
        target = rb.OCR_OUTPUTS_DIR / patch_run / folder / "docling" / "default.call.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(fixed, ensure_ascii=False), encoding="utf-8")
        row = {**rec, "timestamp": patch_run, "legs": ["text", "markdown", "tables"],
               "measurement_scope": "local_conversion_reexport"}
        row.pop("source_timestamp", None)
        rows.append(row)
    inference_file.parent.mkdir(parents=True, exist_ok=True)
    inference_file.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                              encoding="utf-8")
    manifest = {"timestamp": patch_run, "engines": ["docling"],
                "legs": ["text", "markdown", "tables"], "source_run": source_run,
                "kind": "native-document-export-completion", "scored": False,
                "coverage": 93, "text_supplements": text_added,
                "markdown_supplements": markdown_added}
    (rb.RUNS_DIR / f"{patch_run}_run.json").write_text(json.dumps(manifest, indent=2),
                                                        encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
