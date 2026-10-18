"""Audit per-leg source identity against the pre-reconciliation Git archive.

Read-only audit of the original published run. The one-off source repair has
been retired; this command cannot rewrite source calls, ledgers or manifests.
No OCR or judge requests are made.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUN = "20261017T065717Z"
ARCHIVE = "b80c0c82:results/current.zip"
ARCHIVE_SHA256 = "6c2f1c65b9295e1689c473039b14369bf6899dec969cafda7756d537255bda13"


def relative(value):
    parts = Path(value).parts
    return Path(*parts[parts.index("documents"):]).as_posix()


def payload(call):
    # These are the fields consumed by the scorers. Prompt/transport metadata
    # can have separately documented corrections without changing the answer.
    return {k: call.get(k) for k in
            ("raw_text", "structured_output", "raw_response", "json_schema")}


def main():
    archive_bytes = subprocess.check_output(["git", "show", ARCHIVE], cwd=ROOT)
    if hashlib.sha256(archive_bytes).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("Unexpected source archive checksum")
    archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    members = set(archive.namelist())
    ledger_path = ROOT / f"results/runs/{RUN}_inference.jsonl"
    rows = [json.loads(line) for line in archive.read(
        f"results/runs/{RUN}_inference.jsonl").splitlines()]
    restored = {}
    changes = Counter()
    missing = []
    for row in rows:
        row["document_path"] = relative(row["document_path"])
        if row.get("source_document_path"):
            row["source_document_path"] = relative(row["source_document_path"])
        source = row.get("source_timestamp") or row["timestamp"]
        doc = Path(row["document_path"])
        suffix = (f"{row['tier']}__{row['doc_type']}__{doc.stem}/"
                  f"{row['engine_name']}/{row.get('pass_name') or 'default'}.call.json")
        original = f"results/ocr_outputs/{source}/{suffix}"
        reconciled = ROOT / f"results/ocr_outputs/reconciled_outputs/{suffix}"
        row["original_source_timestamp"] = source
        if original not in members:
            if not row.get("error"):
                missing.append(original)
            row["source_timestamp"] = f"reconciled_sources/{source}"
            row["output_retained"] = False
            continue
        call = json.loads(archive.read(original))
        existing = json.loads(reconciled.read_text()) if reconciled.exists() else None
        if existing is not None and payload(call) == payload(existing):
            row["source_timestamp"] = "reconciled_outputs"
        else:
            row["source_timestamp"] = f"reconciled_sources/{source}"
            target = ROOT / f"results/ocr_outputs/{row['source_timestamp']}/{suffix}"
            call["document"] = row["document_path"]
            call.setdefault("metadata", {}).update(
                original_source_timestamp=source, reconciled_run=RUN)
            restored[target] = call
            for leg in row.get("legs", []):
                changes[f"{row['engine_name']}/{leg}"] += 1
        row["output_retained"] = True
    if missing:
        raise RuntimeError(f"Source archive lacks successful calls: {missing}")
    baseline = [json.loads(line) for line in archive.read(
        f"results/runs/{RUN}_results.jsonl").splitlines()]
    published = [json.loads(line) for line in (ROOT / f"results/runs/{RUN}_results.jsonl").read_text().splitlines()]
    for records in (baseline, published):
        for row in records:
            for key in ("source_document_path", "source_timestamp", "original_source_timestamp",
                        "original_result_source_timestamp"):
                row.pop(key, None)
            for key in ("document_path", "source_document_path"):
                if row.get(key):
                    row[key] = relative(row[key])
    if baseline != published:
        raise RuntimeError("Published scores differ from the source archive beyond path/provenance metadata")
    report = {"archive": ARCHIVE, "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
              "published_run": RUN, "score_rows_unchanged": len(baseline),
              "inference_rows": len(rows), "restored_source_calls": len(restored),
              "source_changes_by_engine_leg": dict(sorted(changes.items()))}
    report['published_output_calls'] = len({
        (r['source_timestamp'], r['document_path'], r['engine_name'], r.get('pass_name') or 'default')
        for r in rows if r.get('output_retained')})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
