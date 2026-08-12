"""
OCR Benchmark Orchestrator
==========================
Usage:
  python run_benchmark.py                          # full run, all engines, tiers, legs
  python run_benchmark.py --estimate               # print cost estimate, no API calls
  python run_benchmark.py --check                  # verify API keys + Modal deployments
  python run_benchmark.py --dry-run                # discover docs + validate, no inference
  python run_benchmark.py --legs text,tables       # run a subset of the six legs
  python run_benchmark.py --engines tesseract,docling --tiers easy --skip-judge
  python run_benchmark.py --resume                 # continue the latest interrupted run
  python run_benchmark.py --report                 # re-print the report from stored results

Engines are declared in engines.py — the single registry. Legs are declared in
evaluation/legs.py and specified in docs/scoring-design.md.

Two files are written per run:
  results/runs/{ts}_inference.jsonl   one record per (engine, document, pass)
  results/runs/{ts}_results.jsonl     one record per (engine, document, leg)

They are separate because they count different things. The Modal cost audit
re-derives every GPU figure from the inference records and aborts the report on
any inconsistency; folding several leg rows into that stream would break the
identity `batch_gpu_seconds / batch_size` it checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from modal_apps._shared import calculate_modal_cost

load_dotenv()

ROOT = Path(__file__).parent
DOCS_ROOT = ROOT / "documents"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
SUMMARIES_DIR = RESULTS_DIR / "summaries"
LOGS_DIR = RESULTS_DIR / "logs"
OCR_OUTPUTS_DIR = RESULTS_DIR / "ocr_outputs"
JUDGE_CACHE_DIR = RESULTS_DIR / "judge_checklists"

from engines import (  # noqa: E402
    DEFAULT_ENGINES,
    ENGINES,
    LOCAL_CPU_ENGINES,
    MODAL_ENGINE_NAMES as MODAL_ENGINES,
    build_runner,
)
from runners.base import OCRTask
from evaluation.legs import (  # noqa: E402
    ALL_LEGS,
    DEFAULT_PASS,
    LEGS,
    parse_legs,
    plan_passes,
    skipped_engines,
)

# One stage-2 judge call per (engine, document); the stage-1 checklist is built
# once per document and cached, so it is amortised across every engine.
# Measured 2026-08-09: a 29-item checklist cost $0.0032 in verdicts. Checklists
# are capped at evaluation.judge.MAX_CHECKLIST_ITEMS, so the worst case is a
# full-size list rather than the 568-item page that cost $0.24 before the cap.
JUDGE_COST_PER_ENGINE_DOC = 0.0050
JUDGE_CHECKLIST_COST_PER_DOC = 0.0060

# Table judge: image-based cell checklist (stage 1) and per-engine verdicts
# (stage 2).  Checklists are capped at table_judge.MAX_CHECKLIST_ITEMS = 60.
TABLE_JUDGE_CHECKLIST_COST_PER_DOC = 0.0080   # image input is larger than text
TABLE_JUDGE_COST_PER_ENGINE_DOC = 0.0040

# Below this a leg's result is a failure worth naming in the report.
FAIL_THRESHOLD = 0.50
POOR_THRESHOLD = 0.70

# A leg whose engines all land within this range is not separating them.
SPREAD_WARN_RANGE = 0.02

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
TIER_ORDER = ["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# Logging — console + results/logs/{timestamp}.log
# ---------------------------------------------------------------------------

_LIBRARY_NOISE = ("torch", "transformers", "docling", "rapidocr", "RapidOCR",
                  "PIL", "timm", "huggingface_hub", "filelock", "matplotlib",
                  "urllib3", "httpx", "modal", "botocore", "boto3")


class _DropLibraryNoise(logging.Filter):
    """Keep third-party logging off the terminal report.

    Local model stacks are loud — docling pulls in torch, transformers and
    rapidocr, which emit graph-break traces, deprecation notices and per-region
    "text detection result is empty" warnings. Left alone they bury the
    per-document results this tool exists to print, and because those libraries
    create their loggers lazily during inference, muting the loggers that exist
    at startup does not hold. Filtering at our own handler does, whenever the
    logger appears. Warnings and errors from the benchmark itself pass through;
    anything at ERROR from a library still gets shown, since that may explain a
    failed document.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            return True
        return not record.name.startswith(_LIBRARY_NOISE)


def setup_logging(timestamp: str) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{timestamp}.log"

    # Libraries log through the root logger; ours is "benchmark". Attach the
    # filter to root's handlers so their records are dropped before printing.
    root = logging.getLogger()
    if not any(isinstance(f, _DropLibraryNoise) for f in root.filters):
        root.addFilter(_DropLibraryNoise())
    for handler in root.handlers:
        if not any(isinstance(f, _DropLibraryNoise) for f in handler.filters):
            handler.addFilter(_DropLibraryNoise())

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Log file: {log_path}")
    return logger


# ---------------------------------------------------------------------------
# Document discovery
# ---------------------------------------------------------------------------

def discover_documents(tiers: list[str]) -> list[Path]:
    docs = []
    for tier in tiers:
        tier_dir = DOCS_ROOT / tier
        if not tier_dir.exists():
            continue
        for p in sorted(tier_dir.rglob("*")):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            if "_gt" in p.stem or "_page" in p.stem:
                continue
            docs.append(p)
    return docs


# ---------------------------------------------------------------------------
# --estimate
# ---------------------------------------------------------------------------

# Provider free allowances, used by --estimate to show what a run could cost
# with them applied. Conditions are stated in the output because they are not
# universal: the Textract allowance exists only for the first 3 months of an
# AWS account, and the Modal credit only for new accounts.
FREE_TIER = {
    # engine: (free_calls, condition)
    "textract_text": (1000, "AWS free tier: 1,000 pages/month, first 3 months"),
    "textract_structured": (100, "AWS free tier: 100 pages/month, first 3 months"),
    "llamaparse": (1000, "LlamaParse free tier: 1,000 pages/day, ongoing"),
}
MODAL_FREE_CREDIT_USD = 30.0  # one-time credit for new Modal accounts


def _measured_modal_costs() -> dict[tuple[str, str], float]:
    """Mean measured $/page per (engine, pass) from stored runs, newest winning.

    Modal cost is GPU-seconds x rate, so it depends on how long the model
    actually runs — and a schema-extraction call has no reason to take the same
    GPU time as a conversion call. Static registry figures were measured from
    the reference run's CONVERSION pass only; wherever a real measured figure
    exists for a pass, the estimate uses it instead.
    """
    sums: dict[tuple[str, str], list[float]] = {}
    for path in sorted(RUNS_DIR.glob("*_inference.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("measurement_scope") != "modal_batch":
                continue
            cost = r.get("cost_per_page")
            if cost is None:
                continue
            # Records key the engine as "engine_name"; reading "engine" here
            # crashed --estimate the moment any Modal run existed to learn from.
            engine = r.get("engine_name") or r.get("engine")
            if not engine:
                continue
            sums.setdefault((engine, r.get("pass_name") or "default"), []).append(float(cost))
    return {k: sum(v) / len(v) for k, v in sums.items() if v}


def print_estimate(
    engines: list[str], legs: list[str], documents: list[Path], skip_judge: bool,
    judged_doc_count: int,
) -> None:
    col_e, col_p = 26, 11
    doc_count = len(documents)
    measured = _measured_modal_costs()

    # A prompted pass only calls documents that carry that leg's ground truth —
    # the run skips the rest — so the estimate must count the same way or it
    # overstates: a geometry pass touches 17 documents, not 93.
    from evaluation.ground_truth import GroundTruthLoader
    loader = GroundTruthLoader()
    gt_by_doc = {}
    for d in documents:
        try:
            gt_by_doc[d] = loader.load(d)
        except Exception:
            continue

    def eligible_docs(plan) -> int:
        if plan.leg_for_task is None:
            return doc_count  # default conversion runs on every document
        return sum(
            1 for g in gt_by_doc.values()
            if any(g.has_gt_for(leg) for leg in plan.legs)
        )
    header = (
        f"{'Engine':<{col_e}} | {'passes':>6} | {'$/page':>{col_p}} | "
        f"{'calls':>7} | {'total':>10} | notes"
    )
    sep = "-" * len(header)
    print(f"\nCost estimate — {doc_count} documents x {len(engines)} engines "
          f"x legs [{', '.join(legs)}]")
    print(sep)
    print(header)
    print(sep)

    subtotal = 0.0
    free_saving = 0.0
    modal_cost = 0.0
    free_notes: list[str] = []
    for eng in engines:
        spec = ENGINES.get(eng)
        if spec is None:
            continue
        plans = plan_passes(eng, legs)
        per_pass = [(plan, eligible_docs(plan)) for plan in plans]
        calls = sum(n for _, n in per_pass)
        if spec.kind == "modal":
            # Per-pass: measured GPU time where a stored run has it, the static
            # conversion-pass figure otherwise (flagged below).
            total = sum(
                measured.get((eng, plan.name), spec.cost_per_page) * n
                for plan, n in per_pass
            )
            unmeasured = [plan.name for plan, n in per_pass
                          if n and (eng, plan.name) not in measured]
        else:
            total = spec.cost_per_page * calls
            unmeasured = []
        subtotal += total
        if spec.kind == "modal":
            modal_cost += total
        if eng in FREE_TIER and calls:
            free_calls, condition = FREE_TIER[eng]
            covered = min(calls, free_calls)
            saving = covered * spec.cost_per_page
            free_saving += saving
            free_notes.append(
                f"  {eng}: {covered} of {calls} calls free (-${saving:.2f}) — {condition}"
            )
        prefix = "~" if ("variable" in spec.cost_notes or "Modal" in spec.cost_notes) else " "
        note = spec.cost_notes
        if not plans:
            note = (note + "; " if note else "") + "no selected leg is answerable"
        breakdown = " + ".join(f"{plan.name}:{n}" for plan, n in per_pass) if len(per_pass) > 1 else ""
        if breakdown:
            note = (note + "; " if note else "") + f"calls = {breakdown}"
        if unmeasured and spec.kind == "modal":
            note = (note + "; " if note else "") + (
                f"no measured GPU time yet for {','.join(unmeasured)} — using the "
                f"conversion-pass figure")
        print(
            f"{eng:<{col_e}} | {len(plans):>6} | {prefix}${spec.cost_per_page:.5f}  | "
            f"{calls:>7} | {prefix}${total:>8.4f} | {note}"
        )

    print(sep)
    print(f"{'Subtotal OCR':<{col_e}} | {'':>6} | {'':>{col_p}} | {'':>7} | ~${subtotal:>8.2f}")

    judge_grand = 0.0
    if not skip_judge and judged_doc_count:
        eligible = [e for e in engines if plan_passes(e, legs)]
        checklists = JUDGE_CHECKLIST_COST_PER_DOC * judged_doc_count
        verdicts = JUDGE_COST_PER_ENGINE_DOC * judged_doc_count * len(eligible)
        judge_total = checklists + verdicts
        judge_grand += judge_total
        print(
            f"{'LLM judge (silver-GT)':<{col_e}} | {'':>6} | {'':>{col_p}} | "
            f"{judged_doc_count * len(eligible):>7} | ~${judge_total:>8.2f} | "
            f"{judged_doc_count} silver-GT docs; checklists built once and reused"
        )

    if "tables" in legs and os.environ.get("GOOGLE_API_KEY"):
        table_doc_count = sum(
            1 for g in gt_by_doc.values() if g.has_gt_for("tables")
        )
        if table_doc_count:
            eligible = [e for e in engines if plan_passes(e, legs)]
            t_checklists = TABLE_JUDGE_CHECKLIST_COST_PER_DOC * table_doc_count
            t_verdicts = TABLE_JUDGE_COST_PER_ENGINE_DOC * table_doc_count * len(eligible)
            t_total = t_checklists + t_verdicts
            judge_grand += t_total
            print(
                f"{'LLM judge (tables)':<{col_e}} | {'':>6} | {'':>{col_p}} | "
                f"{table_doc_count * len(eligible):>7} | ~${t_total:>8.2f} | "
                f"{table_doc_count} table docs; Gemini builds cell reference from image"
            )

    if judge_grand > 0:
        print(f"{'Grand total':<{col_e}} | {'':>6} | {'':>{col_p}} | {'':>7} | "
              f"~${subtotal + judge_grand:>8.2f}")
    else:
        print(f"{'Grand total (no judge)':<{col_e}} | {'':>6} | {'':>{col_p}} | {'':>7} | "
              f"~${subtotal:>8.2f}")

    if free_notes or modal_cost:
        print("\nWith provider free tiers (conditions apply — check your accounts):")
        for line in free_notes:
            print(line)
        if modal_cost:
            covered = min(modal_cost, MODAL_FREE_CREDIT_USD)
            print(f"  Modal engines: ${modal_cost:.2f} of GPU time — fully inside the "
                  f"one-time ${MODAL_FREE_CREDIT_USD:.0f} new-account credit"
                  if modal_cost <= MODAL_FREE_CREDIT_USD else
                  f"  Modal engines: ${modal_cost:.2f} of GPU time — first "
                  f"${covered:.2f} covered by the one-time ${MODAL_FREE_CREDIT_USD:.0f} credit")
        best_case = max(0.0, subtotal - free_saving - min(modal_cost, MODAL_FREE_CREDIT_USD))
        print(f"  Best case with all allowances: ~${best_case:.2f} + judge")

    print(f"\nA pass is one inference call per document. Legs an engine answers from its")
    print(f"default conversion share one pass; only a leg that needs its own prompt adds one.")
    print()


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------

def check_setup(engines: list[str], log: logging.Logger) -> bool:
    import subprocess

    ok = True
    log.info("--- API key check ---")
    if "llamaparse" in engines:
        # The schema leg uses LlamaExtract, a separate package from the parse SDK.
        try:
            import llama_cloud_services  # noqa: F401
            log.info("  [OK] llama-cloud-services (LlamaExtract, schema leg)")
        except ImportError:
            log.info("  [MISSING] llama-cloud-services — llamaparse cannot run the "
                     "schema leg (pip install llama-cloud-services)")
    for eng in engines:
        for key in (ENGINES[eng].required_env if eng in ENGINES else ()):
            if os.environ.get(key):
                log.info(f"  [OK] {key}")
            else:
                log.info(f"  [MISSING] {key}  (needed for {eng})")
                ok = False
    if os.environ.get("GOOGLE_API_KEY"):
        log.info("  [OK] GOOGLE_API_KEY (LLM judge)")
    else:
        log.info("  [MISSING] GOOGLE_API_KEY (needed for the judge — or use --skip-judge)")

    modal_engines = [e for e in engines if e in MODAL_ENGINES]
    if modal_engines:
        log.info("\n--- Modal check ---")
        try:
            result = subprocess.run(["modal", "profile", "list"], capture_output=True, text=True)
        except FileNotFoundError:
            log.info("  [MISSING] modal CLI not found — pip install -r requirements.txt "
                     "inside your venv, then: modal setup")
            return False
        auth_output = f"{result.stdout}\n{result.stderr}".lower()
        if result.returncode == 0 and "no token" not in auth_output:
            log.info("  [OK] Modal authenticated")
        else:
            log.info("  [MISSING] Modal not authenticated — run: modal setup")
            return False

        import io
        from PIL import Image as PILImage
        from runners.modal_runner import MODAL_BATCH_SIZES
        from runners.modal_runner import MODAL_ENGINES as ENGINE_MAP

        buf = io.BytesIO()
        PILImage.new("RGB", (32, 32), (255, 255, 255)).save(buf, "PNG")
        ping_bytes = buf.getvalue()

        for eng in modal_engines:
            try:
                import modal
                app_name, class_name, _, _ = ENGINE_MAP[eng]
                cls = modal.Cls.from_name(app_name, class_name)
                if eng in MODAL_BATCH_SIZES:
                    cls().run_ocr_batch.remote([
                        {"document_path": "check.png", "bytes": ping_bytes,
                         "mime_type": "image/png"},
                    ])
                    log.info(f"  [OK] {eng} ({app_name}) — live, batch API available")
                else:
                    cls().run_ocr.remote(ping_bytes, "image/png")
                    log.info(f"  [OK] {eng} ({app_name}) — live")
            except Exception as exc:
                log.info(f"  [DOWN] {eng} — {exc}")
                log.info(f"         Fix: redeploy {ENGINES[eng].modal_app_file}")
                ok = False

    return ok


# ---------------------------------------------------------------------------
# Record IO
# ---------------------------------------------------------------------------

_write_lock = threading.Lock()


def log_record(record: dict, path: Path) -> None:
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _write_ocr_output(
    engine: str, doc_path: Path, gt, text: str, timestamp: str,
    pass_name: str, structured_output: dict | None = None,
    result=None, task=None, legs: tuple[str, ...] = (),
) -> None:
    """Two files per (document, engine, pass), in a folder per engine:

        <tier>__<type>__<stem>/<engine>/<pass>.txt        what we scored
        <tier>__<type>__<stem>/<engine>/<pass>.call.json  the whole call

    `.call.json` holds the prompt sent, the provider's complete reply before
    any parsing, the structured output and the cost/latency — so a scorer fix
    can be re-applied from disk instead of re-paying for the call.

    Deliberately NOT written: a separate `_structured.json` and `_tables.html`
    (both are already inside `.call.json`) and a copy of the ground truth
    (it lives beside the document in documents/). Those made one document
    folder run to ~100 near-duplicate files.
    """
    folder = f"{gt.tier}__{gt.doc_type}__{doc_path.stem}"
    out_dir = OCR_OUTPUTS_DIR / timestamp / folder / engine
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{pass_name}.txt").write_text(text, encoding="utf-8")

    if result is not None:
        envelope = {
            "engine": engine,
            "document": str(doc_path),
            "pass_name": pass_name,
            "legs_fed": list(legs),
            # The exact ask. An app that falls back to its own default reports
            # the prompt it actually used, because a stored `null` becomes
            # unreadable as soon as that default changes.
            "prompt_sent": (
                getattr(task, "prompt", None)
                or ((getattr(result, "raw_response", None) or {}).get("prompt_used"))
                or ((getattr(result, "metadata", None) or {}).get("prompt_used"))
            ),
            "response_format": getattr(task, "response_format", None),
            "json_schema_sent": getattr(task, "json_schema", None) is not None,
            "model_version": getattr(result, "model_version", None),
            "latency_seconds": getattr(result, "latency_seconds", None),
            "cost_per_page": getattr(result, "cost_per_page", None),
            "page_count": getattr(result, "page_count", None),
            "error": getattr(result, "error", None),
            # What we scored, beside what actually arrived.
            "raw_text": text,
            "structured_output": getattr(result, "structured_output", None) or {},
            "metadata": getattr(result, "metadata", None) or {},
            "raw_response": getattr(result, "raw_response", None) or {},
        }
        try:
            payload = json.dumps(envelope, ensure_ascii=False, indent=1, default=str)
        except Exception as exc:  # never lose a run over an exotic payload
            payload = json.dumps(
                {"_unserialisable": f"{exc.__class__.__name__}: {exc}",
                 "raw_text": text}, ensure_ascii=False, indent=1)
        (out_dir / f"{pass_name}.call.json").write_text(payload, encoding="utf-8")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_leg(leg: str, engine: str, text: str, gt, structured_output: dict | None,
              use_judge: bool | None = None):
    """Dispatch to the leg's scorer. Every scorer takes (output, ground truth).

    ``use_judge`` controls LLM-judge scoring for legs that support it:
      None  = auto (use judge when GOOGLE_API_KEY is set)
      True  = force judge on
      False = force judge off, use deterministic scoring
    """
    if leg == "text":
        from evaluation.text_leg import score_text
        return score_text(text, gt)
    if leg == "markdown":
        from evaluation.markdown_leg import score_markdown
        from evaluation.normalizers import markdown_text
        spec = ENGINES[engine]
        rendered = (structured_output or {}).get("markdown")
        if rendered:
            emitted = rendered
        else:
            emitted = (text if spec.answers(leg) == "prompted"
                       else markdown_text(text, engine))
        return score_markdown(emitted, gt, engine, use_judge=use_judge)
    if leg == "tables":
        from evaluation.table_leg import score_tables
        return score_tables(text, gt, engine, structured_output,
                            use_judge=use_judge)
    if leg == "key_value":
        from evaluation.keyvalue_leg import score_key_value
        return score_key_value(text, gt, engine, structured_output)
    if leg == "schema":
        from evaluation.schema_leg import score_schema
        return score_schema(text, gt, engine, structured_output)
    if leg == "geometry":
        from evaluation.geometry_leg import score_geometry
        # Native engines (tesseract, textract) put word boxes in
        # structured_output["words"]; prompted VLMs return the leg's JSON in
        # their text output. parse_word_boxes accepts either shape.
        # "words" is word-level (tesseract, textract). "text_regions" is
        # line/span-level — paddle's rec_boxes, and any engine whose native
        # geometry is coarser. Kept as distinct keys so a coarse source is never
        # silently reported as word boxes; the scorer merges to lines when the
        # reference says so.
        so = structured_output or {}
        predicted = so.get("words") or so.get("text_regions") or text
        return score_geometry(predicted, gt)
    raise ValueError(f"No scorer for leg {leg!r}")


def _base_record(engine: str, doc_path: Path, gt, timestamp: str) -> dict:
    return {
        "engine_name": engine,
        "tier": gt.tier,
        "doc_type": gt.doc_type,
        "document_path": str(doc_path),
        "gt_source": gt.gt_source,
        "timestamp": timestamp,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_legs(records: list[dict]) -> dict:
    """Mean score per (engine, tier, doc_type, leg).

    Never pooled across document types: a 43-field tax form and a 4-field
    receipt are not the same difficulty, so they are never averaged into one
    number without the report saying so.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("error"):
            continue
        groups[(r["engine_name"], r["tier"], r["doc_type"], r["leg"])].append(r)

    agg = {}
    for key, rows in groups.items():
        scored = [r for r in rows if r.get("applicable") and r.get("score") is not None]
        scores = [r["score"] for r in scored]
        agg[key] = {
            "engine": key[0], "tier": key[1], "doc_type": key[2], "leg": key[3],
            "score": round(sum(scores) / len(scores), 4) if scores else None,
            "n": len(scored),
            "n_documents": len(rows),
            "n_not_applicable": len(rows) - len(scored),
            "n_fail": sum(1 for s in scores if s < FAIL_THRESHOLD),
            "n_poor": sum(1 for s in scores if s < POOR_THRESHOLD),
        }
    return agg


def engine_leg_means(agg: dict) -> dict[tuple[str, str], float]:
    """Mean of doc-type means per (engine, leg) — not a pooled document mean."""
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for v in agg.values():
        if v["score"] is not None:
            buckets[(v["engine"], v["leg"])].append(v["score"])
    return {k: sum(v) / len(v) for k, v in buckets.items()}


def engine_operational(inference: list[dict]) -> dict[str, dict]:
    """Per-engine cost and latency, document-weighted, cold starts excluded.

    Returns both the default-pass cost (one call — what production deployment
    costs) and the all-passes total (the benchmark's full spend).  The scale
    projection and per-engine summary use the default-pass figure so engines
    are compared on the same basis regardless of how many legs the benchmark
    asked each one.
    """
    per_doc_cost: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    per_doc_latency: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    # Single-call cost: the pass that serves the text leg (every engine has
    # one).  This is "one call per page" — what production deployment costs,
    # comparable across engines regardless of how many extra legs the benchmark
    # asked each one.  For most engines this is the "default" pass; for engines
    # that always need a prompt it is named "text" or "text+markdown", etc.
    per_pass_cost: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_pass_latency: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    passes: dict[str, set[str]] = defaultdict(set)

    for r in inference:
        if r.get("error"):
            continue
        engine, doc = r["engine_name"], r["document_path"]
        cost = r.get("steady_cost_per_page")
        if cost is None:
            cost = r.get("cost_per_page")
        latency = r.get("steady_latency_seconds")
        if latency is None:
            latency = r.get("latency_seconds")
        if cost is not None:
            per_doc_cost[engine][doc] += float(cost)
        if latency is not None:
            per_doc_latency[engine][doc] += float(latency)
        pass_name = r.get("pass_name") or DEFAULT_PASS
        passes[engine].add(pass_name)
        if cost is not None:
            per_pass_cost[engine][pass_name].append(float(cost))
        if latency is not None:
            per_pass_latency[engine][pass_name].append(float(latency))

    def _pick_primary_pass(engine: str) -> str:
        """The pass that represents 'one call per page' for this engine.

        Priority: 'default' (most engines) > any pass containing 'text'
        (engines that always prompt) > the pass with the most records.
        """
        engine_passes = passes.get(engine, set())
        if DEFAULT_PASS in engine_passes:
            return DEFAULT_PASS
        text_passes = [p for p in engine_passes if "text" in p]
        if text_passes:
            return min(text_passes, key=len)
        all_counts = {
            p: len(per_pass_cost.get(engine, {}).get(p, []))
            for p in engine_passes
        }
        return max(all_counts, key=lambda p: all_counts[p]) if all_counts else DEFAULT_PASS

    stats = {}
    for engine in set(per_doc_cost) | set(per_doc_latency):
        costs = list(per_doc_cost[engine].values())
        latencies = list(per_doc_latency[engine].values())
        latency_all = sum(latencies) / len(latencies) if latencies else None
        primary = _pick_primary_pass(engine)
        pc = per_pass_cost.get(engine, {}).get(primary, [])
        pl = per_pass_latency.get(engine, {}).get(primary, [])
        single_cost = sum(pc) / len(pc) if pc else None
        single_lat = sum(pl) / len(pl) if pl else None
        pass_stats = {}
        for pn in passes[engine]:
            ppc = per_pass_cost.get(engine, {}).get(pn, [])
            ppl = per_pass_latency.get(engine, {}).get(pn, [])
            avg_c = sum(ppc) / len(ppc) if ppc else None
            avg_l = sum(ppl) / len(ppl) if ppl else None
            pass_stats[pn] = {
                "cost_per_page": avg_c,
                "latency_s": avg_l,
                "pages_per_hour": (3600 / avg_l) if avg_l else None,
                "documents": len(ppc) or len(ppl),
            }
        stats[engine] = {
            "cost_per_page": single_cost,
            "latency_s": single_lat,
            "pages_per_hour": (3600 / single_lat) if single_lat else None,
            "cost_all_passes": sum(costs) / len(costs) if costs else None,
            "latency_all_passes": latency_all,
            "passes": len(passes[engine]),
            "documents": len(per_doc_cost[engine]) or len(per_doc_latency[engine]),
            "per_pass": pass_stats,
        }
    return stats


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_gt_provenance(records: list[dict], log: logging.Logger) -> None:
    """What kind of ground truth the scores rest on. Printed before any score.

    Not every document's GT came from the same place, and that changes how much
    weight a number deserves — so it frames the tables rather than footnoting
    them.
    """
    docs_by_source: dict[str, set[str]] = defaultdict(set)
    for r in records:
        docs_by_source[r.get("gt_source") or "human"].add(r.get("document_path", ""))
    total = sum(len(v) for v in docs_by_source.values())
    if not total:
        return

    labels = {
        "human": "human (source dataset annotation)",
        "model_verified": "model-transcribed, engine-consensus verified",
        "silver_tesseract": "machine-generated (Tesseract) — judge only",
    }
    log.info("")
    log.info("=" * 100)
    log.info("Ground truth provenance")
    log.info("=" * 100)
    for source in ("human", "model_verified", "silver_tesseract"):
        n = len(docs_by_source.get(source, ()))
        if n:
            log.info(f"  {source:<18} {n:>3} documents   {labels[source]}")

    n_model = len(docs_by_source.get("model_verified", ()))
    if n_model:
        try:
            from evaluation.ground_truth import MODEL_VERIFIED_GT
            annotators = sorted({v.get("annotator", "?") for v in MODEL_VERIFIED_GT.values()})
        except Exception:
            annotators = []
        log.info("")
        log.info(f"  NOTE: {n_model} document(s) carry ARTIFICIAL ground truth produced by "
                 f"{', '.join(annotators) or 'a model'} and")
        log.info("        cross-checked against OCR engine consensus, not human annotation.")
        log.info("        Their scores are real but rest on a model's reading of the page —")
        log.info("        see documents/gt_provenance.json for the per-document audit trail.")
    log.info("")


def _print_gt_limitations(log: logging.Logger) -> None:
    """Known defects in the reference data, ranked by severity.

    Recorded in documents/gt_limitations.json rather than left implicit: a
    benchmark carrying defective reference data quietly produces numbers that
    describe the annotation instead of the engines.
    """
    path = DOCS_ROOT / "gt_limitations.json"
    if not path.exists():
        return
    try:
        entries = json.loads(path.read_text(encoding="utf-8")).get("limitations", [])
    except Exception:
        return
    if not entries:
        return

    order = {"blocks": 0, "distorts": 1, "weakens": 2, "noted": 3}
    entries.sort(key=lambda e: order.get(e.get("severity"), 9))

    log.info("=" * 100)
    log.info("Known ground-truth limitations")
    log.info("=" * 100)
    for e in entries:
        docs = e.get("documents", [])
        scope = "all documents" if docs == ["*"] else f"{len(docs)} doc(s)"
        legs = ", ".join(e.get("legs", [])) or "-"
        log.info(f"  [{e.get('severity','?'):<8}] {e.get('id','?'):<38} {scope:<16} legs: {legs}")
        log.info(f"             {e.get('summary','')}")
    log.info("")
    log.info("  Full detail incl. measured effect: documents/gt_limitations.json")
    log.info("")


def _fmt(value, width: int = 6, places: int = 3) -> str:
    return f"{value:>{width}.{places}f}" if value is not None else f"{'-':>{width}}"


def _reference_counts(records: list[dict], legs: list[str]) -> dict[str, int]:
    """How many of the run's documents have a reference for each leg.

    Read through GroundTruthLoader, not by probing sidecar paths, so the count
    printed is the count scoring saw.
    """
    from evaluation.ground_truth import GroundTruthLoader

    loader = GroundTruthLoader()
    counts = {leg: 0 for leg in legs}
    for path in sorted({r["document_path"] for r in records}):
        try:
            gt = loader.load(Path(path))
        except Exception:
            continue
        for leg in legs:
            if gt.has_gt_for(leg):
                counts[leg] += 1
    return counts


def _print_leg_results(
    agg: dict, records: list[dict], legs: list[str], engines: list[str],
    operational: dict, log: logging.Logger,
) -> None:
    """Per-leg results by (engine, tier, doc_type), with cost and latency beside.

    Accuracy per dollar is the whole question, so cost and latency sit in the
    same table as the score rather than in a separate one.
    """
    skips = skipped_engines(legs, engines)
    references = _reference_counts(records, legs)

    for leg in legs:
        spec = LEGS[leg]
        leg_rows = [v for v in agg.values() if v["leg"] == leg]
        leg_records = [r for r in records if r["leg"] == leg]
        scored_docs = {
            r["document_path"] for r in leg_records
            if r.get("applicable") and r.get("score") is not None
        }
        n_reference = references.get(leg, 0)

        div = "=" * 100
        log.info(div)
        log.info(f"Leg: {spec.title}  —  {spec.question}")
        log.info(f"  Scored on: {spec.scored_on}")
        method = spec.method
        if "LLM judge" in method:
            judge_model_keys = ("judge_model", "judge_verdict_model",
                                "judge_checklist_model")
            judge_models: set[str] = set()
            for r in leg_records:
                m = r.get("metrics") or {}
                for k in judge_model_keys:
                    v = m.get(k)
                    if v:
                        judge_models.add(v)
            if not judge_models:
                judge_models.add("gemini-2.5-pro")
            method = f"LLM judge ({', '.join(sorted(judge_models))})"
        log.info(f"  Method: {method}")
        log.info(div)

        if not scored_docs:
            # Say which of the two causes it is: nothing to compare against, or
            # nobody able to answer. They call for different work.
            if n_reference == 0:
                log.info("  UNAVAILABLE — no document in this run has a reference for "
                         "this leg,")
                log.info("  so nothing can be measured. This is a missing-ground-truth "
                         "result, not an engine result.")
            else:
                log.info(f"  UNAVAILABLE — {n_reference} document(s) have a reference, "
                         f"but no engine produced a score.")
                reason = next(
                    (r.get("reason") for r in leg_records if r.get("reason")), None
                )
                if reason:
                    log.info(f"  Reason: {reason}")
            for engine, note in skips.get(leg, []):
                log.info(f"    not competing: {engine:<22} {note}")
            log.info("")
            continue

        eligible = sorted({v["engine"] for v in leg_rows if v["score"] is not None})
        log.info(f"  {len(scored_docs)} of {n_reference} documents with a reference "
                 f"produced a score; {len(eligible)} of {len(engines)} engines competed.")

        for engine, note in skips.get(leg, []):
            log.info(f"    not competing: {engine:<22} {note}")
        log.info("")

        means = engine_leg_means(agg)
        ordered = sorted(eligible, key=lambda e: means.get((e, leg), -1.0), reverse=True)

        for tier in TIER_ORDER:
            doc_types = sorted({
                v["doc_type"] for v in leg_rows
                if v["tier"] == tier and v["score"] is not None
            })
            if not doc_types:
                continue
            lookup = {
                (v["engine"], v["doc_type"]): v
                for v in leg_rows if v["tier"] == tier
            }
            width = 11
            cols = [f"{dt[:width]:^{width}}" for dt in doc_types]
            sub_cols = [f"{'score  (n)':^{width}}" for _ in doc_types]
            if len(doc_types) > 1:
                cols.append(f"{'avg':^{width}}")
                sub_cols.append(f"{'':^{width}}")
            header = f"  {'Engine':<24} | " + " | ".join(cols)
            log.info(f"  --- {tier} tier " + "-" * max(0, len(header) - len(tier) - 12))
            log.info(header)
            log.info(f"  {'':<24} | " + " | ".join(sub_cols))
            log.info("  " + "-" * (len(header) - 2))

            def _tier_avg(engine: str) -> float:
                scores = [
                    lookup[(engine, dt)]["score"]
                    for dt in doc_types
                    if (engine, dt) in lookup and lookup[(engine, dt)]["score"] is not None
                ]
                return sum(scores) / len(scores) if scores else -1.0

            tier_ordered = sorted(
                [e for e in ordered if any((e, dt) in lookup for dt in doc_types)],
                key=_tier_avg, reverse=True,
            )
            for engine in tier_ordered:
                cells = []
                tier_scores = []
                for dt in doc_types:
                    row = lookup.get((engine, dt))
                    if row is None or row["score"] is None:
                        cells.append(f"{'-':^{width}}")
                    else:
                        cells.append(f"{row['score']:.3f} ({row['n']:>2}){'':<{max(0, width - 10)}}")
                        tier_scores.append(row["score"])
                if len(doc_types) > 1:
                    if tier_scores:
                        tier_avg = sum(tier_scores) / len(tier_scores)
                        cells.append(f"{tier_avg:.3f}{'':>{max(0, width - 5)}}")
                    else:
                        cells.append(f"{'-':^{width}}")
                log.info(f"  {engine:<24} | " + " | ".join(cells))
            log.info("")

        # Accuracy beside what it costs.
        log.info(f"  {'Engine':<24} | {'score':>7} | {'$/page':>9} | {'lat(s)':>8} | "
                 f"{'docs':>5} | {'fails':>7}")
        log.info("  " + "-" * 78)
        for engine in ordered:
            mean = means.get((engine, leg))
            ops = operational.get(engine) or {}
            n = sum(v["n"] for v in leg_rows if v["engine"] == engine)
            fails = sum(v["n_fail"] for v in leg_rows if v["engine"] == engine)
            cost = ops.get("cost_per_page")
            latency = ops.get("latency_s")
            log.info(
                f"  {engine:<24} | {_fmt(mean, 7)} | "
                f"{('$' + format(cost, '.5f')) if cost is not None else '-':>9} | "
                f"{_fmt(latency, 8, 2)} | {n:>5} | {fails:>3}/{n:<3}"
            )
        log.info("  score = mean of per-document-type means, never pooled across document types.")

        # For tables leg: show score with vs without sampled documents
        if leg == "tables":
            from collections import defaultdict as _dd
            from evaluation.table_judge import MAX_CHECKLIST_ITEMS as _MAX_CL
            eng_all: dict[str, list[float]] = _dd(list)
            eng_uns: dict[str, list[float]] = _dd(list)
            for r in leg_records:
                if r["leg"] != leg or not r.get("applicable") or r.get("score") is None:
                    continue
                m = r.get("metrics") or {}
                was_sampled = m.get("judge_sampled",
                                    m.get("judge_checklist_size") == _MAX_CL)
                eng_all[r["engine_name"]].append(r["score"])
                if not was_sampled:
                    eng_uns[r["engine_name"]].append(r["score"])
            if any(len(eng_all[e]) != len(eng_uns.get(e, [])) for e in eng_all):
                log.info("")
                log.info(f"  {'Engine':<24}   {'all':>7}  {'no-sample':>9}  "
                         f"{'diff':>6}  {'sampled':>7}")
                log.info("  " + "-" * 62)
                for engine in ordered:
                    a = eng_all.get(engine, [])
                    u = eng_uns.get(engine, [])
                    if not a:
                        continue
                    ma = sum(a) / len(a)
                    mu = sum(u) / len(u) if u else None
                    diff = f"{mu - ma:+.3f}" if mu is not None else "  n/a"
                    ns = len(a) - len(u)
                    log.info(
                        f"  {engine:<24}   {ma:>7.3f}  "
                        f"{'n/a' if mu is None else f'{mu:>9.3f}'}  "
                        f"{diff:>6}  {ns:>3}/{len(a)}"
                    )

            # TEDS vs Judge comparison
            eng_judge: dict[str, list[float]] = _dd(list)
            eng_teds: dict[str, list[float]] = _dd(list)
            for r in leg_records:
                if r["leg"] != leg or not r.get("applicable") or r.get("score") is None:
                    continue
                m = r.get("metrics") or {}
                ca = m.get("cell_accuracy")
                td = m.get("teds")
                if ca is not None:
                    eng_judge[r["engine_name"]].append(ca)
                if td is not None:
                    eng_teds[r["engine_name"]].append(td)
            if eng_judge and eng_teds:
                log.info("")
                log.info(f"  {'Engine':<24}   {'judge':>7}  {'TEDS':>7}  {'diff':>6}  {'docs':>5}")
                log.info("  " + "-" * 58)
                for engine in ordered:
                    j = eng_judge.get(engine, [])
                    t = eng_teds.get(engine, [])
                    if not j and not t:
                        continue
                    mj = sum(j) / len(j) if j else None
                    mt = sum(t) / len(t) if t else None
                    d = (f"{mj - mt:+.3f}" if mj is not None and mt is not None
                         else "  n/a")
                    n = max(len(j), len(t))
                    log.info(
                        f"  {engine:<24}   "
                        f"{'n/a' if mj is None else f'{mj:>7.3f}'}  "
                        f"{'n/a' if mt is None else f'{mt:>7.3f}'}  "
                        f"{d:>6}  {n:>5}"
                    )

        log.info("")


def _print_spread_warnings(agg: dict, legs: list[str], log: logging.Logger) -> None:
    """A leg that does not spread engines is not measuring anything.

    If every engine scores the same the test is too easy or the reference is
    too thin — that is how schemas covering four fields of a twenty-five-field
    page were caught. Surfaced as a warning rather than left for a reader to
    notice.
    """
    means = engine_leg_means(agg)
    warnings = []
    for leg in legs:
        scores = [v for (e, l), v in means.items() if l == leg]
        if len(scores) < 2:
            continue
        spread = max(scores) - min(scores)
        if spread <= SPREAD_WARN_RANGE:
            warnings.append((leg, spread, min(scores), max(scores), len(scores)))

    if not warnings:
        return
    div = "=" * 100
    log.info(div)
    log.info("WARNING — legs that did not separate the engines")
    log.info(div)
    for leg, spread, low, high, n in warnings:
        log.info(f"  {leg:<12} {n} engines scored between {low:.3f} and {high:.3f} "
                 f"(spread {spread:.3f})")
    log.info("  A leg that does not spread engines is not measuring them. Either the")
    log.info("  documents are too easy for it, or the reference is too thin to discriminate.")
    log.info("")


def _print_failures(records: list[dict], log: logging.Logger) -> None:
    """Documents where an engine scored badly, with the reason.

    Every number in the tables above can be traced to a document from here.
    """
    failures = [
        r for r in records
        if r.get("applicable") and r.get("score") is not None
        and r["score"] < POOR_THRESHOLD and not r.get("error")
    ]
    if not failures:
        return

    div = "=" * 100
    log.info(div)
    log.info(f"Failing (engine, document, leg) results  "
             f"(hard fail: score<{FAIL_THRESHOLD}  |  soft: score<{POOR_THRESHOLD})")
    log.info(div)

    by_engine: dict[str, list[dict]] = defaultdict(list)
    for r in failures:
        by_engine[r["engine_name"]].append(r)

    for engine in sorted(by_engine):
        rows = sorted(by_engine[engine], key=lambda r: r["score"])
        hard = sum(1 for r in rows if r["score"] < FAIL_THRESHOLD)
        log.info(f"\n  {engine}  —  {hard} hard, {len(rows) - hard} soft")
        log.info(f"  {'Doc':<28} {'Leg':<10} {'Score':>6}  Likely issue")
        log.info("  " + "-" * 92)
        for r in rows[:25]:
            flag = "FAIL" if r["score"] < FAIL_THRESHOLD else "warn"
            log.info(
                f"  [{flag}] {Path(r['document_path']).name:<22} {r['leg']:<10} "
                f"{r['score']:>6.3f}  {_diagnose(r)}"
            )
        if len(rows) > 25:
            log.info(f"  ... and {len(rows) - 25} more (full detail in the per-doc CSV)")
    log.info("")


def _diagnose(record: dict) -> str:
    """Name the failure from the metrics the scorer already recorded.

    Only facts the scorer measured — no invented correlate.
    """
    leg = record["leg"]
    m = record.get("metrics") or {}
    reason = record.get("reason")

    def metric(key: str, default: float) -> float:
        # `m.get(key) or default` would turn a measured 0.0 — the worst
        # possible value — into the default and skip the diagnosis entirely.
        value = m.get(key)
        return default if value is None else value

    if leg == "text":
        if m.get("repetition_suspected"):
            return f"output {m.get('length_ratio')}x reference length — repetition loop"
        if metric("word_precision", 1) < 0.65 and metric("word_recall", 0) >= 0.80:
            return "reads the page but adds labels/structure not in the reference"
        if metric("gt_coverage", 0) < 0.4:
            return "most of the page is missing from the output"
        return "missing content"
    if leg == "markdown":
        if metric("element_recall", 0) < 0.5:
            return "most structural blocks absent"
        if metric("order_score", 1) < 0.6:
            return "blocks present but out of reading order"
        return "partial structure"
    if leg == "tables":
        if reason:
            return reason
        if m.get("output_rows", 0) == 0:
            return "no table recovered"
        if m.get("output_rows") != m.get("reference_rows"):
            return (f"grid mismatch: {m.get('output_rows')}x{m.get('output_cols')} vs "
                    f"reference {m.get('reference_rows')}x{m.get('reference_cols')}")
        return "cell values wrong"
    if leg == "key_value":
        invented = m.get("kv_pairs_invented_value") or 0
        if invented:
            return f"{invented} blank field(s) filled with invented values"
        return "fields missing or wrong"
    if leg == "schema":
        invented = m.get("schema_fields_invented_value") or 0
        if invented:
            return f"{invented} blank field(s) filled with invented values"
        return "fields missing or wrong"
    if leg == "geometry":
        return reason or "boxes missing or misplaced"
    return reason or "below threshold"


def _print_inference_errors(inference: list[dict], log: logging.Logger) -> None:
    succeeded: set[tuple[str, str, str]] = set()
    for r in inference:
        if not r.get("error"):
            key = (r["engine_name"], r["document_path"], r.get("pass_name") or "")
            succeeded.add(key)
    errors = [
        r for r in inference
        if r.get("error")
        and (r["engine_name"], r["document_path"], r.get("pass_name") or "") not in succeeded
    ]
    if not errors:
        return
    seen: set[tuple[str, str, str]] = set()
    deduped = []
    for r in errors:
        key = (r["engine_name"], r["document_path"], r.get("pass_name") or "")
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    errors = deduped
    div = "=" * 100
    log.info(div)
    log.info("Inference errors — these (engine, document, pass) calls failed and should be re-run")
    log.info("  Run:  python run_benchmark.py --rerun-errors")
    log.info(div)
    log.info(f"  {'Engine':<24} {'Pass':<10} {'Doc':<28} Error")
    log.info("  " + "-" * 92)
    for r in sorted(errors, key=lambda x: (x["engine_name"], x["document_path"])):
        log.info(
            f"  {r['engine_name']:<24} {(r.get('pass_name') or '-'):<10} "
            f"{Path(r['document_path']).name:<28} {str(r['error'])[:40]}"
        )
    log.info("")


def _print_engine_summary(
    agg: dict, legs: list[str], operational: dict, log: logging.Logger
) -> None:
    """One row per engine: every leg it answered, beside cost and throughput."""
    means = engine_leg_means(agg)
    engines = sorted({v["engine"] for v in agg.values()})
    if not engines:
        return

    active = [leg for leg in legs if any((e, leg) in means for e in engines)]
    if not active:
        return

    div = "=" * 100
    log.info(div)
    log.info("Per-engine summary — every leg, beside what it costs")
    log.info(div)
    header = f"  {'Engine':<24} | " + " | ".join(f"{leg[:9]:>9}" for leg in active)
    header += f" | {'$/page':>9} | {'pages/hr':>9}"
    log.info(header)
    log.info("  " + "-" * (len(header) - 2))

    def sort_key(engine: str) -> float:
        vals = [means[(engine, leg)] for leg in active if (engine, leg) in means]
        return sum(vals) / len(vals) if vals else -1.0

    for engine in sorted(engines, key=sort_key, reverse=True):
        cells = []
        for leg in active:
            value = means.get((engine, leg))
            cells.append(f"{value:>9.3f}" if value is not None else f"{'-':>9}")
        ops = operational.get(engine) or {}
        cost = ops.get("cost_per_page")
        pph = ops.get("pages_per_hour")
        log.info(
            f"  {engine:<24} | " + " | ".join(cells)
            + f" | {('$' + format(cost, '.5f')) if cost is not None else '-':>9}"
            + f" | {format(pph, ',.0f') if pph else '-':>9}"
        )
    log.info("  Blank = the engine does not answer that leg; see the per-leg tables for why.")
    log.info("  $/page = one call (default conversion pass). Comparable across engines.")
    log.info("")


SCALE_VOLUMES = [1_000, 10_000, 100_000, 1_000_000]


def _print_scale_projection(operational: dict, log: logging.Logger) -> None:
    """Cost and time at volume — what matters when choosing an engine for production."""
    engines_with_cost = {
        e: ops for e, ops in operational.items()
        if ops.get("cost_per_page") is not None or ops.get("pages_per_hour") is not None
    }
    if not engines_with_cost:
        return

    div = "=" * 100
    log.info(div)
    log.info("Scale projection  (steady-state; startup/model-load excluded)")
    log.info(div)

    vol_labels = ["1K", "10K", "100K", "1M"]
    header = (
        f"  {'Engine':<24} | {'pages/hr':>9} | {'$/page':>8} | "
        + " | ".join(f"{'cost@' + l:>10}" for l in vol_labels)
        + " | "
        + " | ".join(f"{'hrs@' + l:>8}" for l in vol_labels)
    )
    log.info(header)
    log.info("  " + "-" * (len(header) - 2))

    sorted_engines = sorted(
        engines_with_cost,
        key=lambda e: (-(engines_with_cost[e].get("cost_per_page") or 0), e),
    )

    for eng in sorted_engines:
        ops = engines_with_cost[eng]
        cpp = ops.get("cost_per_page") or 0.0
        pph = ops.get("pages_per_hour")
        pph_str = f"{pph:>9,.0f}" if pph else f"{'-':>9}"

        cost_cols = " | ".join(
            f"${cpp * v:>9,.2f}" for v in SCALE_VOLUMES
        )
        hrs_cols = " | ".join(
            f"{v / pph:>7.1f}h" if pph and pph > 0 else f"{'-':>7} "
            for v in SCALE_VOLUMES
        )
        log.info(
            f"  {eng:<24} | {pph_str} | ${cpp:>7.5f} | {cost_cols} | {hrs_cols}"
        )

    log.info("")
    log.info("  Hours assume single-stream processing. Parallelising across N workers")
    log.info("  divides wall-clock time by N; cost stays the same.")
    log.info("")


def _pass_to_legs(engines: list[str]) -> dict[str, dict[str, list[str]]]:
    """Build {engine: {pass_name: [leg, ...]}} from the authoritative plan."""
    mapping: dict[str, dict[str, list[str]]] = {}
    for eng in engines:
        if eng not in ENGINES:
            continue
        plans = plan_passes(eng, list(ALL_LEGS))
        mapping[eng] = {p.name: list(p.legs) for p in plans}
    return mapping


def _print_per_leg_costs(
    operational: dict, engines: list[str], log: logging.Logger
) -> None:
    """Per-leg cost and latency — what each use case costs per page."""
    pass_legs = _pass_to_legs(engines)
    has_data = {e for e in engines if e in operational
                and operational[e].get("per_pass")}
    if not has_data:
        return

    legs_seen: list[str] = []
    for leg in ALL_LEGS:
        for eng in has_data:
            for pn, legs in pass_legs.get(eng, {}).items():
                if leg in legs and leg not in legs_seen:
                    legs_seen.append(leg)
    if not legs_seen:
        return

    div = "=" * 100
    log.info(div)
    log.info("Per-leg cost and latency  (what each use case costs per page)")
    log.info(div)

    col_w = 12
    header = f"  {'Engine':<24}"
    for leg in legs_seen:
        header += f" | {'$/pg':>{col_w//2}}  {'lat':>{col_w//2-1}}"
    log.info(f"  {'':24}   " + "   ".join(
        f"{leg:^{col_w+1}}" for leg in legs_seen
    ))
    hdr2 = f"  {'Engine':<24}"
    for _ in legs_seen:
        hdr2 += f" | {'$/page':>6} {'sec':>5}"
    log.info(hdr2)
    log.info("  " + "-" * (25 + len(legs_seen) * 15))

    sorted_engines = sorted(
        has_data,
        key=lambda e: -(operational[e].get("cost_per_page") or 0),
    )

    for eng in sorted_engines:
        per_pass = operational[eng].get("per_pass", {})
        eng_pass_legs = pass_legs.get(eng, {})
        leg_to_pass: dict[str, str] = {}
        for pn, legs in eng_pass_legs.items():
            for leg in legs:
                leg_to_pass[leg] = pn

        row = f"  {eng:<24}"
        for leg in legs_seen:
            pn = leg_to_pass.get(leg)
            if pn and pn in per_pass:
                ps = per_pass[pn]
                c = ps.get("cost_per_page")
                l = ps.get("latency_s")
                c_s = f"${c:.4f}" if c is not None else "-"
                l_s = f"{l:.1f}" if l is not None else "-"
                row += f" | {c_s:>6} {l_s:>5}"
            else:
                row += f" | {'—':>6} {'—':>5}"
        log.info(row)

    log.info("")
    log.info("  Legs sharing a pass (e.g. text+markdown+tables) show the same")
    log.info("  cost — it is one API call. Combining legs across passes sums their costs.")
    log.info("")


def _print_judged_documents(records: list[dict], log: logging.Logger) -> None:
    judged = [r for r in records if r["leg"] == "judge" and r.get("score") is not None]
    if not judged:
        return
    div = "=" * 100
    log.info(div)
    log.info("LLM-judged documents — the silver-GT pages, scored against the image")
    log.info("  Item accuracy = checklist items the engine got right / items on the page.")
    log.info("  DIRECTIONAL: the checklist is model-generated and has not been validated")
    log.info("  against human judgement. Not comparable to the GT-scored legs above.")
    log.info(div)
    by_engine: dict[str, list[float]] = defaultdict(list)
    checklist_sizes: list[int] = []
    for r in judged:
        by_engine[r["engine_name"]].append(r["score"])
        size = (r.get("metrics") or {}).get("checklist_size")
        if size:
            checklist_sizes.append(int(size))

    engines = sorted(by_engine, key=lambda e: -sum(by_engine[e]) / len(by_engine[e]))

    by_tier_type: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in judged:
        by_tier_type[(r.get("tier", "?"), r.get("doc_type", "?"))][r["engine_name"]].append(
            r["score"]
        )

    width = 11
    for tier in TIER_ORDER:
        doc_types = sorted({dt for (t, dt) in by_tier_type if t == tier})
        if not doc_types:
            continue
        cols = [f"{dt[:width]:^{width}}" for dt in doc_types]
        sub_cols = [f"{'score  (n)':^{width}}" for _ in doc_types]
        if len(doc_types) > 1:
            cols.append(f"{'avg':^{width}}")
            sub_cols.append(f"{'':^{width}}")
        header = f"  {'Engine':<24} | " + " | ".join(cols)
        log.info(f"  --- {tier} tier " + "-" * max(0, len(header) - len(tier) - 12))
        log.info(header)
        log.info(f"  {'':<24} | " + " | ".join(sub_cols))
        log.info("  " + "-" * (len(header) - 2))

        def _judge_tier_avg(engine: str) -> float:
            scores = []
            for dt in doc_types:
                dt_scores = by_tier_type[(tier, dt)].get(engine, [])
                if dt_scores:
                    scores.append(sum(dt_scores) / len(dt_scores))
            return sum(scores) / len(scores) if scores else -1.0

        tier_ordered = sorted(
            [e for e in engines if any(by_tier_type[(tier, dt)].get(e) for dt in doc_types)],
            key=_judge_tier_avg, reverse=True,
        )
        for engine in tier_ordered:
            cells = []
            tier_scores = []
            for dt in doc_types:
                dt_scores = by_tier_type[(tier, dt)].get(engine, [])
                if dt_scores:
                    avg = sum(dt_scores) / len(dt_scores)
                    cells.append(f"{avg:.3f} ({len(dt_scores):>2}){'':<{max(0, width - 10)}}")
                    tier_scores.append(avg)
                else:
                    cells.append(f"{'-':^{width}}")
            if len(doc_types) > 1:
                if tier_scores:
                    cells.append(f"{sum(tier_scores) / len(tier_scores):.3f}{'':>{max(0, width - 5)}}")
                else:
                    cells.append(f"{'-':^{width}}")
            log.info(f"  {engine:<24} | " + " | ".join(cells))
        log.info("")

    log.info(f"  {'Engine':<24} | {'item acc':>9} | {'docs':>5}")
    log.info("  " + "-" * 46)
    for engine in engines:
        scores = by_engine[engine]
        log.info(f"  {engine:<24} | {sum(scores) / len(scores):>9.3f} | {len(scores):>5}")
    if checklist_sizes:
        log.info(f"\n  Checklist size: {min(checklist_sizes)}-{max(checklist_sizes)} items "
                 f"per document (mean {sum(checklist_sizes) / len(checklist_sizes):.1f}).")
        log.info("  Anything the checklist missed is untested for every engine equally.")
    log.info("")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def _write_per_doc_csv(records: list[dict], path: Path) -> None:
    metric_keys: list[str] = []
    for r in records:
        for k in (r.get("metrics") or {}):
            if k not in metric_keys:
                metric_keys.append(k)
    fields = [
        "engine_name", "tier", "doc_type", "document_path", "gt_source",
        "leg", "pass_name", "applicable", "score", "reason",
        "latency_seconds", "cost_per_page", "model_version", "timestamp", "error",
    ] + metric_keys
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = {k: v for k, v in r.items() if k != "metrics" and k != "detail"}
            row.update(r.get("metrics") or {})
            w.writerow(row)


def _write_summary_csv(agg: dict, path: Path) -> None:
    fields = ["engine", "tier", "doc_type", "leg", "score", "n", "n_documents",
              "n_not_applicable", "n_fail", "n_poor"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in sorted(
            agg.values(), key=lambda r: (r["leg"], r["tier"], r["doc_type"], r["engine"])
        ):
            w.writerow(row)


def _write_engine_csv(
    agg: dict, legs: list[str], operational: dict, path: Path
) -> None:
    means = engine_leg_means(agg)
    engines = sorted({v["engine"] for v in agg.values()})
    fields = ["engine"] + [f"{leg}_score" for leg in legs] + [
        "cost_per_page_usd", "latency_s_per_page", "pages_per_hour", "passes"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for engine in engines:
            ops = operational.get(engine) or {}
            row = {"engine": engine}
            for leg in legs:
                value = means.get((engine, leg))
                row[f"{leg}_score"] = round(value, 4) if value is not None else None
            row["cost_per_page_usd"] = (
                round(ops["cost_per_page"], 6) if ops.get("cost_per_page") is not None else None
            )
            row["latency_s_per_page"] = (
                round(ops["latency_s"], 4) if ops.get("latency_s") is not None else None
            )
            row["pages_per_hour"] = (
                round(ops["pages_per_hour"], 1) if ops.get("pages_per_hour") else None
            )
            row["passes"] = ops.get("passes")
            w.writerow(row)


# ---------------------------------------------------------------------------
# Report entry point
# ---------------------------------------------------------------------------

def write_run_manifest(
    timestamp: str, engines: list[str], tiers: list[str], legs: list[str]
) -> None:
    """What the run was asked to do.

    Without it, re-reporting could only see the legs that produced records —
    and a leg with no ground truth produces none, so the one result worth
    printing (that it could not be measured) would silently vanish.
    """
    (RUNS_DIR / f"{timestamp}_run.json").write_text(
        json.dumps(
            {"timestamp": timestamp, "engines": engines, "tiers": tiers, "legs": legs},
            indent=2,
        ),
        encoding="utf-8",
    )


def read_run_manifest(timestamp: str) -> dict:
    path = RUNS_DIR / f"{timestamp}_run.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def report(timestamp: str, log: logging.Logger, legs: list[str] | None = None) -> None:
    leg_records = read_jsonl(RUNS_DIR / f"{timestamp}_results.jsonl")
    inference = read_jsonl(RUNS_DIR / f"{timestamp}_inference.jsonl")

    if not leg_records:
        log.info("No results to report.")
        return

    # Modal cost audit gate — the report is not printed if a GPU figure cannot
    # be re-derived from the stored batch fields.
    from evaluation.modal_metrics import audit_modal_records

    summaries, errors, used_batches = audit_modal_records(inference)
    if errors:
        preview = "\n".join(f"  - {e}" for e in errors[:20])
        more = f"\n  - ... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError(
            f"Modal timing/cost audit failed; report was not generated:\n{preview}{more}"
        )
    if used_batches:
        log.info("Modal timing/cost audit: PASS")
        log.info("  Engine                     GPU          Docs  Batches  "
                 "GPU s/doc   Docs/s      $/doc")
        for row in summaries:
            log.info(
                f"  {row['engine']:<26} {row['gpu_type']:<11} {row['documents']:>5} "
                f"{row['batches']:>8} {row['gpu_seconds_per_document']:>10.6f} "
                f"{row['documents_per_second']:>8.6f} ${row['cost_per_document']:.7f}"
            )

    manifest = read_run_manifest(timestamp)
    active_legs = legs or manifest.get("legs") or [
        leg for leg in ALL_LEGS if any(r["leg"] == leg for r in leg_records)
    ]
    engines = manifest.get("engines") or sorted({r["engine_name"] for r in leg_records})
    engines = [e for e in engines if e in ENGINES] or sorted(
        {r["engine_name"] for r in leg_records}
    )
    # Re-check GT applicability so report reflects current GT state
    # (e.g. charts excluded from tables) without requiring a rescore.
    from evaluation.ground_truth import GroundTruthLoader
    _gt_loader = GroundTruthLoader()
    _applicable_records = []
    for r in leg_records:
        if r["leg"] == "judge":
            continue
        if r.get("applicable") and r.get("score") is not None:
            try:
                gt = _gt_loader.load(Path(r["document_path"]))
                if not gt.has_gt_for(r["leg"]):
                    r = dict(r, applicable=False, score=None)
            except (FileNotFoundError, ValueError):
                pass
        _applicable_records.append(r)
    agg = aggregate_legs(_applicable_records)
    operational = engine_operational(inference)

    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    per_doc = SUMMARIES_DIR / f"{timestamp}_per_doc.csv"
    summary = SUMMARIES_DIR / f"{timestamp}_summary.csv"
    engine_csv = SUMMARIES_DIR / f"{timestamp}_engines.csv"
    _write_per_doc_csv(leg_records, per_doc)
    _write_summary_csv(agg, summary)
    _write_engine_csv(agg, active_legs, operational, engine_csv)

    # Order fixed by docs/architecture.md: provenance and limitations frame the
    # scores, then the scores, then cost, then what failed.
    _print_gt_provenance(_applicable_records, log)
    _print_gt_limitations(log)
    _print_leg_results(agg, _applicable_records, active_legs, engines, operational, log)
    _print_judged_documents(_applicable_records, log)
    _print_engine_summary(agg, active_legs, operational, log)
    _print_scale_projection(operational, log)
    _print_per_leg_costs(operational, engines, log)
    _print_spread_warnings(agg, active_legs, log)
    _print_failures(_applicable_records, log)
    _print_inference_errors(inference, log)

    log.info("Results written to:")
    log.info(f"  {per_doc}")
    log.info(f"  {summary}")
    log.info(f"  {engine_csv}")
    log.info(f"  {LOGS_DIR / (timestamp + '.log')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="OCR Benchmark Runner")
    p.add_argument("--engines", default=",".join(DEFAULT_ENGINES),
                   help="Comma-separated, any subset of: " + ", ".join(ENGINES)
                        + " (default: all)")
    p.add_argument("--tiers", default="easy,medium,hard",
                   help="Comma-separated, any subset of: easy, medium, hard (default: all)")
    p.add_argument("--legs", default="all",
                   help=f"Comma-separated legs to run (default: all). "
                        f"Known: {', '.join(ALL_LEGS)}")
    p.add_argument("--skip-judge", action="store_true",
                   help="Skip the LLM judge on silver-GT documents")
    p.add_argument("--judge-model", default=None, metavar="MODEL",
                   help="Override the Gemini model used for judge verdicts "
                        "(all judges). Supported: gemini-2.5-flash, "
                        "gemini-2.5-pro, gemini-3.5-flash, "
                        "gemini-3.5-flash-lite, gemini-3.6-flash")
    p.add_argument("--estimate", action="store_true",
                   help="Print cost estimate and exit — no API calls")
    p.add_argument("--check", action="store_true",
                   help="Verify API keys and Modal deployments, then exit")
    p.add_argument("--dry-run", action="store_true",
                   help="Discover documents, plan passes and validate — no inference")
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel engine threads (default: number of engines)")
    p.add_argument("--max-docs", type=int, default=0,
                   help="Limit to the first N discovered documents")
    p.add_argument("--resume", nargs="?", const="latest", metavar="TIMESTAMP",
                   help="Resume an interrupted run; completed (engine, doc, pass) "
                        "calls are skipped and results appended")
    p.add_argument("--report", nargs="?", const="latest", metavar="TIMESTAMP",
                   help="Regenerate the report from stored results — no OCR re-run")
    p.add_argument("--rerun-errors", nargs="?", const="latest", metavar="TIMESTAMP",
                   help="Re-run only the calls that errored in a previous run")
    return p.parse_args()


def _resolve_timestamp(value: str) -> str:
    if value != "latest":
        return value
    candidates = sorted(RUNS_DIR.glob("*_results.jsonl"))
    if not candidates:
        raise FileNotFoundError("No results JSONL found in results/runs/")
    return candidates[-1].name.replace("_results.jsonl", "")


def validate_keys(engines: list[str], skip_judge: bool) -> list[str]:
    missing = []
    for eng in engines:
        for key in (ENGINES[eng].required_env if eng in ENGINES else ()):
            if not os.environ.get(key):
                missing.append(f"{key}  (needed for {eng})")
    if not skip_judge and not os.environ.get("GOOGLE_API_KEY"):
        missing.append("GOOGLE_API_KEY  (needed for the judge — or use --skip-judge)")
    return missing


def _print_dry_run(
    engines: list[str], legs: list[str], documents: list[Path], log: logging.Logger
) -> None:
    from evaluation.ground_truth import GroundTruthLoader

    loader = GroundTruthLoader()
    log.info("")
    log.info("=" * 100)
    log.info("Dry run — what a real run would do")
    log.info("=" * 100)

    log.info(f"\nDocuments: {len(documents)}")
    gts = []
    failed = 0
    for doc in documents:
        try:
            gts.append(loader.load(doc))
        except Exception as exc:
            failed += 1
            log.warning(f"  [GT ERROR] {doc.relative_to(ROOT)}: {exc}")
    log.info(f"  loaded through GroundTruthLoader: {len(gts)}  |  failed: {failed}")

    provenance: dict[str, int] = defaultdict(int)
    for gt in gts:
        provenance[gt.gt_source] += 1
    for source, n in sorted(provenance.items()):
        log.info(f"    {source:<18} {n:>3}")

    log.info("\nGround truth available per leg:")
    for leg in legs:
        n = sum(1 for gt in gts if gt.has_gt_for(leg))
        status = "" if n else "   <- UNAVAILABLE, no document has a reference"
        log.info(f"  {leg:<12} {n:>3} / {len(gts)} documents{status}")

    log.info("\nInference passes per engine:")
    total_calls = 0
    for engine in engines:
        plans = plan_passes(engine, legs)
        calls = len(plans) * len(documents)
        total_calls += calls
        detail = ", ".join(
            f"{p.name}[{'+'.join(p.legs)}]" for p in plans
        ) or "none — answers no selected leg"
        log.info(f"  {engine:<24} {len(plans)} pass(es) x {len(documents)} docs "
                 f"= {calls:>4} calls   {detail}")
    log.info(f"\n  Total inference calls: {total_calls}")

    skips = skipped_engines(legs, engines)
    if skips:
        log.info("\nEngines not competing, and why:")
        for leg, rows in skips.items():
            log.info(f"  {leg}:")
            for engine, note in rows:
                log.info(f"    {engine:<24} {note}")
    log.info("")


def main() -> None:
    args = parse_args()
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    try:
        legs = parse_legs(args.legs)
    except ValueError as exc:
        print(exc)
        sys.exit(1)

    unknown = [e for e in engines if e not in ENGINES]
    if unknown:
        print(f"Unknown engine(s): {', '.join(unknown)}. Known: {', '.join(ENGINES)}")
        sys.exit(1)

    if args.judge_model:
        from evaluation.judge import PRICING as _judge_pricing
        if args.judge_model not in _judge_pricing:
            print(f"Unknown judge model: {args.judge_model}. "
                  f"Supported: {', '.join(sorted(_judge_pricing))}")
            sys.exit(1)
        # The judges read this at call time — one flag reaches all of them.
        os.environ["OCR_JUDGE_MODEL"] = args.judge_model

    # key_value and schema have no deterministic scorer — they are judged or
    # they are nothing. Asking for them together with --skip-judge is a
    # contradiction: refuse an explicit ask, silently drop them from the
    # default all-legs selection.
    judge_only = [l for l in legs if l in ("key_value", "schema")]
    if args.skip_judge and judge_only:
        if args.legs.strip().lower() != "all":
            print(f"--skip-judge cannot run judge-only leg(s): "
                  f"{', '.join(judge_only)}. These legs have no deterministic "
                  f"scorer — drop them from --legs, or drop --skip-judge.")
            sys.exit(1)
        legs = [l for l in legs if l not in ("key_value", "schema")]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = setup_logging(ts)

    if args.report:
        run_ts = _resolve_timestamp(args.report)
        log = setup_logging(run_ts)
        log.info(f"Regenerating report from run {run_ts}")
        report(run_ts, log)
        return

    if args.rerun_errors:
        run_ts = _resolve_timestamp(args.rerun_errors)
        for name in ("_inference.jsonl", "_results.jsonl"):
            path = RUNS_DIR / f"{run_ts}{name}"
            records = read_jsonl(path)
            kept = [r for r in records if not r.get("error")]
            if len(kept) != len(records):
                with open(path, "w", encoding="utf-8") as f:
                    for r in kept:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                log.info(f"--rerun-errors: cleared {len(records) - len(kept)} "
                         f"errored record(s) from {path.name}")
        args.resume = run_ts

    documents = discover_documents(tiers)
    if legs != list(ALL_LEGS):
        # An explicit --legs selection means the user wants THOSE legs
        # exercised: keep only documents with ground truth for at least one of
        # them, so `--legs geometry --max-docs 2` validates on 2 geometry
        # documents instead of truncating to 2 ineligible ones and doing
        # nothing.
        from evaluation.ground_truth import GroundTruthLoader
        _loader = GroundTruthLoader()
        eligible = [d for d in documents
                    if any(_loader.load(d).has_gt_for(leg) for leg in legs)]
        if eligible:
            documents = eligible
    if args.max_docs:
        documents = documents[:max(0, args.max_docs)]

    if args.estimate:
        if not documents:
            print("No documents found under documents/ — fetch the corpus first:")
            print("  python scripts/download_data.py")
            return
        from evaluation.ground_truth import GroundTruthLoader
        loader = GroundTruthLoader()
        judged = sum(
            1 for d in documents
            if loader.load(d).gt_source == "silver_tesseract"
        )
        print_estimate(engines, legs, documents, args.skip_judge, judged)
        return

    if args.check:
        sys.exit(0 if check_setup(engines, log) else 1)

    if not documents:
        log.error(f"No documents found in {DOCS_ROOT}")
        log.error("Fetch the corpus first:  python scripts/download_data.py")
        sys.exit(1)

    log.info(f"Found {len(documents)} documents  |  tiers: {tiers}  |  "
             f"engines: {len(engines)}  |  legs: {legs}")

    if args.dry_run:
        _print_dry_run(engines, legs, documents, log)
        return

    missing = validate_keys(engines, args.skip_judge)
    if missing:
        log.error("Missing environment variables:")
        for m in missing:
            log.error(f"  {m}")
        log.error("\nSet them in .env or export them. Exiting.")
        sys.exit(1)

    from evaluation.ground_truth import GroundTruthLoader
    gt_loader = GroundTruthLoader()

    judge_client = None
    judge_document = None
    if not args.skip_judge:
        try:
            from evaluation.judge import judge_document  # noqa: F811
            judge_client = None
        except ImportError as exc:
            log.warning(f"LLM judge unavailable ({exc}); silver-GT documents will "
                        f"carry no judged score. Use --skip-judge to silence this.")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    if args.resume:
        ts = _resolve_timestamp(args.resume)
        log = setup_logging(ts)
    inference_jsonl = RUNS_DIR / f"{ts}_inference.jsonl"
    results_jsonl = RUNS_DIR / f"{ts}_results.jsonl"
    write_run_manifest(ts, engines, tiers, legs)

    done: set[tuple[str, str, str]] = set()
    if args.resume:
        for r in read_jsonl(inference_jsonl):
            if not r.get("error"):
                done.add((r["engine_name"], r["document_path"],
                          r.get("pass_name") or DEFAULT_PASS))
        log.info(f"Resuming run {ts} — {len(done)} completed call(s) will be skipped")

    total_calls = sum(len(plan_passes(e, legs)) * len(documents) for e in engines)
    completed = 0
    completed_lock = threading.Lock()
    total_spend = 0.0
    spend_lock = threading.Lock()

    log.info(f"\nRun ID: {ts}")
    log.info(f"Engines: {len(engines)}  |  Documents: {len(documents)}  |  "
             f"Inference calls: {total_calls}")
    local_engines = [e for e in engines if e in LOCAL_CPU_ENGINES]
    cloud_engines = [e for e in engines if e not in LOCAL_CPU_ENGINES]
    log.info(f"Local CPU engines (sequential): {local_engines or 'none'}")
    log.info(f"Cloud/Modal engines (parallel): {len(cloud_engines)}")
    log.info("")

    def record_pass(
        engine: str, doc_path: Path, gt, plan, result, error: str | None,
        task=None,
    ) -> None:
        """Write the inference record, then one leg record per leg it feeds."""
        nonlocal total_spend

        base = _base_record(engine, doc_path, gt, ts)
        if error or result is None:
            # A failed call is the one you most want to read afterwards: the
            # provider's error body, or a reply that arrived but could not be
            # used. Persist whatever came back before recording the failure.
            if result is not None:
                _write_ocr_output(engine, doc_path, gt, result.raw_text or "", ts,
                                  plan.name, result.structured_output,
                                  result=result, task=task, legs=plan.legs)
            log_record({
                **base, "pass_name": plan.name, "legs": list(plan.legs),
                "latency_seconds": None, "cost_per_page": None,
                "model_version": result.model_version if result else "unknown",
                "error": error or "no result returned",
            }, inference_jsonl)
            for leg in plan.legs:
                log_record({
                    **base, "leg": leg, "pass_name": plan.name,
                    "applicable": False, "score": None,
                    "reason": "inference failed", "metrics": {}, "detail": {},
                    "error": error or "no result returned",
                }, results_jsonl)
            log.warning(f"  [{engine}] ERROR on {doc_path.name}: {error}")
            return

        metadata = result.metadata or {}
        is_batch = metadata.get("measurement_scope") == "modal_batch"
        startup = float(metadata.get("startup_gpu_seconds") or 0.0)
        latency = float(result.latency_seconds)
        cost = float(result.cost_per_page)
        if is_batch:
            # The batch timer starts after model load, so the measured interval
            # is already steady state.
            steady_latency, steady_cost = latency, cost
        else:
            steady_latency = max(latency - startup, 0.0)
            steady_cost = (
                calculate_modal_cost(steady_latency, metadata.get("gpu_type", "T4"))
                if startup and engine in MODAL_ENGINES else cost
            )

        with spend_lock:
            total_spend += cost

        _write_ocr_output(engine, doc_path, gt, result.raw_text, ts, plan.name,
                          result.structured_output, result=result, task=task,
                          legs=plan.legs)

        log_record({
            **base,
            "pass_name": plan.name,
            "legs": list(plan.legs),
            # docs/scoring-design.md rule 2: a result whose prompt cannot be
            # reproduced is not a result. None = the engine's own default ask.
            "prompt_used": task.prompt if task is not None else None,
            "latency_seconds": latency if is_batch else round(latency, 3),
            "cost_per_page": cost if is_batch else round(cost, 6),
            "startup_seconds": round(startup, 3),
            "is_cold_start": bool(metadata.get("is_cold_start")),
            "steady_latency_seconds": (
                steady_latency if is_batch else round(steady_latency, 3)
            ),
            "steady_cost_per_page": steady_cost if is_batch else round(steady_cost, 6),
            "measurement_scope": metadata.get("measurement_scope"),
            "batch_id": metadata.get("batch_id"),
            "batch_gpu_seconds": (
                float(metadata["batch_gpu_seconds"])
                if metadata.get("batch_gpu_seconds") is not None else None
            ),
            "batch_size": (
                int(metadata["batch_size"])
                if metadata.get("batch_size") is not None else None
            ),
            "effective_gpu_seconds_per_document": (
                float(metadata["effective_gpu_seconds_per_document"])
                if metadata.get("effective_gpu_seconds_per_document") is not None else None
            ),
            "startup_excluded": metadata.get("startup_excluded"),
            "gpu_type": metadata.get("gpu_type"),
            "gpu_rate_usd_per_second": metadata.get("gpu_rate_usd_per_second"),
            "model_version": result.model_version,
            "error": None,
        }, inference_jsonl)

        parts = []
        for leg in plan.legs:
            score = score_leg(leg, engine, result.raw_text, gt, result.structured_output,
                              use_judge=False if args.skip_judge else None)
            log_record({
                **base,
                "leg": leg,
                "pass_name": plan.name,
                "applicable": score.applicable,
                "score": score.score,
                "reason": score.reason,
                "metrics": score.as_metrics(),
                "detail": score.as_detail(),
                "latency_seconds": round(steady_latency, 3),
                "cost_per_page": round(steady_cost, 6),
                "model_version": result.model_version,
                "error": None,
            }, results_jsonl)
            if score.applicable and score.score is not None:
                parts.append(f"{leg}={score.score:.3f}")
            else:
                parts.append(f"{leg}=n/a")

        log.info(f"  [{engine}/{plan.name}] {doc_path.name}  " + "  ".join(parts)
                 + f"  {steady_latency:.1f}s  ${cost:.5f}")

        # The judge covers the documents whose ground truth cannot be trusted.
        if (
            judge_document is not None
            and plan.name == DEFAULT_PASS
            and gt.gt_source == "silver_tesseract"
        ):
            _run_judge(engine, doc_path, gt, result.raw_text, base)

    def _run_judge(engine: str, doc_path: Path, gt, text: str, base: dict) -> None:
        nonlocal total_spend
        try:
            JUDGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            verdict = judge_document(doc_path, text, engine, client=judge_client)
        except Exception as exc:
            log.warning(f"  [{engine}] judge error on {doc_path.name}: {exc}")
            return
        cost = float(getattr(verdict, "cost_usd", 0.0) or 0.0)
        with spend_lock:
            total_spend += cost
        items = list(getattr(verdict, "items", []) or [])
        log_record({
            **base,
            "leg": "judge",
            "pass_name": DEFAULT_PASS,
            "applicable": getattr(verdict, "item_accuracy", None) is not None,
            "score": getattr(verdict, "item_accuracy", None),
            "reason": None,
            "metrics": {
                "item_accuracy": getattr(verdict, "item_accuracy", None),
                "checklist_size": getattr(verdict, "checklist_size", len(items)),
                "items_checked": len(items),
            },
            # Per-item verdicts are kept, not just the total, so any judged
            # number can be traced back to the items behind it.
            "detail": {"items": _jsonable(items)},
            "cost_per_page": round(cost, 6),
            "error": None,
        }, results_jsonl)

        # Say so on the terminal. Without this the silver documents print
        # "text=n/a geometry=n/a" and look skipped, when in fact the judge ran
        # on them — it is the only score they get, and it is what the pause
        # after that line was spent on.
        accuracy = getattr(verdict, "item_accuracy", None)
        if accuracy is None:
            log.info(f"  [{engine}/judge] {doc_path.name}  judge=n/a  "
                     f"({getattr(verdict, 'error', None) or 'no verdict'})")
        else:
            log.info(f"  [{engine}/judge] {doc_path.name}  judge={accuracy:.3f}  "
                     f"({len(items)} checklist items)  ${cost:.5f}")

    def run_engine(engine: str) -> None:
        nonlocal completed
        plans = plan_passes(engine, legs)
        if not plans:
            log.info(f"[SKIP] {engine} answers none of the selected legs "
                     f"({ENGINES[engine].leg_notes or 'not eligible'})")
            return
        try:
            runner = build_runner(engine)
        except Exception as exc:
            log.warning(f"[SKIP] Cannot instantiate {engine}: {exc}")
            return

        if engine in MODAL_ENGINES and getattr(runner, "supports_batch", False):
            _run_modal_all_passes(engine, runner, plans)
            return

        for plan in plans:
            pending: list[tuple[Path, object]] = []
            for doc_path in documents:
                if (engine, str(doc_path), plan.name) in done:
                    with completed_lock:
                        completed += 1
                    continue
                try:
                    gt = gt_loader.load(doc_path)
                except Exception as exc:
                    log.warning(f"{engine} <- {doc_path.name}  GT ERROR: {exc}")
                    continue
                if plan.leg_for_task and not any(gt.has_gt_for(l) for l in plan.legs):
                    with completed_lock:
                        completed += 1
                    continue
                pending.append((doc_path, gt))

            if not pending:
                continue

            task = None
            if plan.leg_for_task:
                task = LEGS[plan.leg_for_task].task_for(
                    pending[0][1].json_schema if LEGS[plan.leg_for_task].needs_schema else None,
                    engine=plan.engine,
                )
            else:
                text_prompt = ENGINES[plan.engine].leg_prompts.get("text")
                if text_prompt and ENGINES[plan.engine].accepts_task:
                    task = OCRTask(leg="text", prompt=text_prompt,
                                   json_schema=None, response_format=None)

            with completed_lock:
                completed += len(pending)
                seq = completed
            log.info(f"[{seq}/{total_calls}] {engine} pass '{plan.name}' "
                     f"-> legs {'+'.join(plan.legs)} ({len(pending)} docs)")

            for doc_path, gt in pending:
                call_task = task
                if plan.leg_for_task and LEGS[plan.leg_for_task].needs_schema:
                    call_task = LEGS[plan.leg_for_task].task_for(gt.json_schema, engine=plan.engine)
                result = runner.run_safe(doc_path, call_task)
                record_pass(engine, doc_path, gt, plan, result, result.error, task=call_task)

    def _run_modal_all_passes(engine, runner, plans) -> None:
        """Merge every pass into one remote call so the container stays warm.

        Each item carries its own prompt and response_format via
        runner.prepare_item(), so the vLLM server processes all legs without
        an intervening cold start between passes.
        """
        nonlocal completed

        all_items: list[tuple[dict, Path, object, object, object]] = []

        for plan in plans:
            for doc_path in documents:
                if (engine, str(doc_path), plan.name) in done:
                    with completed_lock:
                        completed += 1
                    continue
                try:
                    gt = gt_loader.load(doc_path)
                except Exception as exc:
                    log.warning(f"{engine} <- {doc_path.name}  GT ERROR: {exc}")
                    continue
                if plan.leg_for_task and not any(gt.has_gt_for(l) for l in plan.legs):
                    with completed_lock:
                        completed += 1
                    continue

                task = None
                if plan.leg_for_task:
                    spec = LEGS[plan.leg_for_task]
                    task = spec.task_for(
                        gt.json_schema if spec.needs_schema else None,
                        engine=plan.engine,
                    )
                else:
                    text_prompt = ENGINES[plan.engine].leg_prompts.get("text")
                    if text_prompt and ENGINES[plan.engine].accepts_task:
                        task = OCRTask(leg="text", prompt=text_prompt,
                                       json_schema=None, response_format=None)

                file_dict = runner.prepare_item(doc_path, task)
                all_items.append((file_dict, doc_path, gt, plan, task))

        if not all_items:
            return

        with completed_lock:
            completed += len(all_items)
            seq = completed
        pass_names = sorted(set(p.name for _, _, _, p, _ in all_items))
        log.info(f"[{seq}/{total_calls}] {engine} ALL passes "
                 f"({', '.join(pass_names)}) -> {len(all_items)} items, "
                 f"one container session")

        batch_size = getattr(runner, "batch_size", 0) or len(all_items)
        for start in range(0, len(all_items), batch_size):
            chunk = all_items[start:start + batch_size]
            file_dicts = [fd for fd, _, _, _, _ in chunk]
            try:
                results = runner.run_tagged_batch(file_dicts)
            except Exception as exc:
                message = f"{exc.__class__.__name__}: {str(exc) or repr(exc)}"
                for _, doc_path, gt, plan, task in chunk:
                    record_pass(engine, doc_path, gt, plan, None, message,
                                task=task)
                continue
            for (_, doc_path, gt, plan, task), result in zip(chunk, results):
                record_pass(engine, doc_path, gt, plan, result,
                            result.error if result else
                            "Modal batch returned no result",
                            task=task)

    # Phase 1: local CPU engines, one at a time so the machine is not saturated.
    if local_engines:
        log.info("Phase 1: local CPU engines (sequential)")
        for engine in local_engines:
            run_engine(engine)

    # Phase 2: cloud + Modal engines, in parallel — they are I/O bound.
    if cloud_engines:
        workers = args.workers if args.workers > 0 else len(cloud_engines)
        log.info(f"Phase 2: cloud/Modal engines ({workers} parallel workers)")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_engine, e): e for e in cloud_engines}
            for future in as_completed(futures):
                if future.exception():
                    log.error(f"[{futures[future]}] thread failed: {future.exception()}")

    log.info(f"\nTotal spend this run: ${total_spend:.4f}")
    log.info("Aggregating results...\n")
    report(ts, log, legs)


def _jsonable(items) -> list:
    """Per-item judge verdicts, whatever shape the judge returns them in."""
    out = []
    for item in items:
        if isinstance(item, dict):
            out.append(item)
        elif hasattr(item, "__dict__"):
            out.append({k: v for k, v in vars(item).items()})
        else:
            out.append({"value": str(item)})
    return out


if __name__ == "__main__":
    main()
