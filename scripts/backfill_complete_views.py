"""Create replayable text/Markdown calls from retained native provider outputs.

Usage: python scripts/backfill_complete_views.py SOURCE_RUN PATCH_RUN
The source archive is read only. PATCH_RUN is a local source for --rescore.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_benchmark as rb
from evaluation.ground_truth import GroundTruthLoader
from modal_apps.mineru_pipeline_app import _block_lines, _complete_views, _plain
from modal_apps.paddleocr_vl_app import _complete_markdown, _plain_text_from_blocks, _block_text
from runners.llamaparse_runner import _audit_text_view


def _words(value: str) -> list[str]:
    return re.findall(r"\w+", _plain(value).casefold())


def _contains(haystack: str, needle: str) -> bool:
    big, small = _words(haystack), _words(needle)
    return not small or any(big[i:i + len(small)] == small
                            for i in range(len(big) - len(small) + 1))


def _repair(engine: str, call: dict) -> tuple[dict, dict]:
    fixed = copy.deepcopy(call)
    so = fixed["structured_output"]
    report = {}
    if engine == "llamaparse":
        pages = so["pages"]
        views = [_audit_text_view(page) for page in pages]
        fixed["raw_text"] = "\n".join(view[0] for view in views).strip()
        so["native_text"] = fixed["raw_text"]
        so["markdown"] = "\n".join(str(page.get("md") or "") for page in pages).strip()
        so["text_item_differences"] = [view[1] for view in views]
        assert fixed["raw_text"] == "\n".join(str(page.get("text") or "") for page in pages).strip()
        report = {"pages": len(pages), "item_differences": sum(len(v[1]) for v in views)}
    elif engine == "paddleocr_vl":
        blocks = so.get("layout_blocks") or []
        provider_md = fixed["raw_text"]
        so["provider_markdown"] = provider_md
        so["markdown"], so["markdown_supplements"] = _complete_markdown(
            provider_md, blocks, historical_filtered=True)
        fixed["raw_text"] = _plain_text_from_blocks(blocks)
        missing_text = [i for i, block in enumerate(blocks) if block.get("content") and
                        not _contains(fixed["raw_text"], _block_text(str(block["content"]))) ]
        missing_md = [i for i, block in enumerate(blocks) if block.get("content") and
                      not _contains(so["markdown"], _block_text(str(block["content"]))) ]
        if missing_text or missing_md:
            raise ValueError(f"Paddle coverage failure: text={missing_text} markdown={missing_md}")
        report = {"blocks": len(blocks), "markdown_supplements": len(so["markdown_supplements"])}
    elif engine == "mineru_pipeline":
        page = so["middle_json_page"]
        provider_md = so["markdown"]
        fixed["raw_text"], so["markdown"], additions = _complete_views(page, provider_md)
        so["provider_markdown"] = provider_md
        so["markdown_supplements"] = additions
        missing = []
        count = 0
        for group in ("para_blocks", "discarded_blocks"):
            for block in page.get(group) or []:
                for line in _block_lines(block):
                    count += 1
                    if not _contains(fixed["raw_text"], line) or not _contains(so["markdown"], line):
                        missing.append((group, line[:80]))
        if missing:
            raise ValueError(f"MinerU coverage failure: {missing[:10]}")
        if "<table" in fixed["raw_text"].lower():
            raise ValueError("MinerU text still contains HTML table markup")
        report = {"lines": count, "markdown_supplements": len(additions),
                  "discarded_blocks": len(page.get("discarded_blocks") or [])}
    else:
        raise ValueError(engine)
    source = json.dumps(call, ensure_ascii=False, sort_keys=True).encode("utf-8")
    fixed.setdefault("metadata", {})["view_backfill"] = {
        "source_call_json_sha256": hashlib.sha256(source).hexdigest(),
        "method": "replayed retained native text, Markdown and recognized blocks",
    }
    return fixed, report


def main(source_run: str, patch_run: str) -> None:
    if source_run == patch_run:
        raise ValueError("patch run must differ from source")
    records = rb.read_jsonl(rb.RUNS_DIR / f"{source_run}_inference.jsonl")
    targets = {"llamaparse": 93, "paddleocr_vl": 93, "mineru_pipeline": 17}
    chosen = [rec for rec in records if rec["engine_name"] in targets and
              (rec.get("pass_name") or "default") == "default" and not rec.get("error")]
    counts = {engine: sum(rec["engine_name"] == engine for rec in chosen) for engine in targets}
    if counts != targets:
        raise ValueError(f"Unexpected saved coverage: {counts}, expected {targets}")
    destination = rb.RUNS_DIR / f"{patch_run}_inference.jsonl"
    if destination.exists():
        raise FileExistsError(destination)
    rows = []
    summary = {engine: [] for engine in targets}
    loader = GroundTruthLoader()
    for rec in chosen:
        engine = rec["engine_name"]
        doc = Path(rec["document_path"])
        ground_truth = loader.load(doc)
        original_run = rec.get("source_timestamp") or source_run
        call = rb._read_stored_call(original_run, engine, doc, ground_truth, "default")
        if call is None:
            raise FileNotFoundError((original_run, engine, doc))
        fixed, report = _repair(engine, call)
        fixed["metadata"]["view_backfill"]["source_timestamp"] = original_run
        folder = f"{ground_truth.tier}__{ground_truth.doc_type}__{doc.stem}"
        target = rb.OCR_OUTPUTS_DIR / patch_run / folder / engine / "default.call.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(fixed, ensure_ascii=False), encoding="utf-8")
        row = {**rec, "timestamp": patch_run, "legs": ["text", "markdown"]}
        row.pop("source_timestamp", None)
        # This is a second view of an existing physical call, not another
        # member of its Modal batch. The original call still carries the batch
        # measurement on its tables/geometry row in the combined rescore.
        for key in ("batch_id", "batch_gpu_seconds", "batch_size"):
            row[key] = None
        row["measurement_scope"] = "replayed_saved_call"
        rows.append(row)
        summary[engine].append(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                           encoding="utf-8")
    manifest = {"timestamp": patch_run, "engines": list(targets), "legs": ["text", "markdown"],
                "source_run": source_run, "kind": "saved-call-view-backfill", "scored": False,
                "coverage": {engine: len(items) for engine, items in summary.items()},
                "limitation": "Docling native document/text was not retained in historical calls."}
    (rb.RUNS_DIR / f"{patch_run}_run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({engine: {"pages": len(items),
                               "markdown_supplements": sum(item.get("markdown_supplements", 0) for item in items),
                               "item_differences": sum(item.get("item_differences", 0) for item in items)}
                      for engine, items in summary.items()}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
