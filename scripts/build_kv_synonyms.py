"""Prepare or execute the explicit key-only synonym plan. Preparation is offline.

python scripts/build_kv_synonyms.py prepare --source RUN --output /tmp/kv-plan.json
python scripts/build_kv_synonyms.py execute /tmp/kv-plan.json --budget-usd 3
The execute command costs money; obtain approval for the saved plan first.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import kv_synonyms as ks
from evaluation.ground_truth import load_corpus
from evaluation.kv_schema_code import engine_pairs
from evaluation.shared_fields import CACHE_ROOT, reference_id


def prepare(sources: list[str]) -> dict:
    import run_benchmark as rb
    corpus = {Path(g.document_path).stem: g for g in load_corpus()}
    keys = defaultdict(set)
    calls = []
    for source in sources:
        records = rb.read_jsonl(rb.RUNS_DIR / f"{source}_inference.jsonl")
        if not records:
            raise ValueError(f"No inference records for {source}")
        for rec in records:
            if rec.get("error") or "key_value" not in rec.get("legs", []):
                continue
            gt = corpus[Path(rec["document_path"]).stem]
            if not gt.metadata.get("structured_gold_reference", {}).get("shared_field_selection"):
                continue
            timestamp = rec.get("source_timestamp") or source
            name = rec.get("pass_name") or "default"
            call = rb._read_stored_call(timestamp, rec["engine_name"], Path(gt.document_path), gt, name)
            if call is None:
                raise ValueError(f"Missing source call: {timestamp}/{rec['engine_name']}/{name}/{Path(gt.document_path).stem}")
            pairs, _ = engine_pairs(rec["engine_name"], call.get("raw_text") or "",
                                    call.get("structured_output") or {})
            stem = Path(gt.document_path).stem
            keys[stem].update(str(k) for k, _ in pairs)
            calls.append({"source": timestamp, "engine": rec["engine_name"], "page": stem,
                          "pass": name, "call_sha256": ks.digest(call)})
    pages = [ks.prepare_page(corpus[stem], sorted(values)) for stem, values in sorted(keys.items())]
    plan = {"version": ks.VERSION, "model": ks.MODEL, "pricing": ks.PRICES,
            "pages": pages, "source_calls": calls,
            "request_count": sum(len(p["requests"]) for p in pages),
            "planning_reserve_usd": sum(r["planning_reserve_usd"] for p in pages for r in p["requests"])}
    plan["id"] = ks.digest(plan)
    return plan


def execute(plan: dict, budget: float) -> None:
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("Budget must be a positive finite USD amount")
    if plan["id"] != ks.digest({k: v for k, v in plan.items() if k != "id"}):
        raise ValueError("Plan changed; prepare and review it again")
    if plan["version"] != ks.VERSION or plan["model"] != ks.MODEL:
        raise ValueError("Plan/model version differs from this implementation")
    if budget < plan["planning_reserve_usd"]:
        raise ValueError(f"Budget below plan reserve ${plan['planning_reserve_usd']:.4f}")
    corpus = {Path(g.document_path).stem: g for g in load_corpus()}
    for page in plan["pages"]:
        if reference_id(corpus[page["page"]]) != page["reference_id"]:
            raise ValueError(f"Reference changed: {page['page']}; rebuild the plan")
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    load_dotenv(ROOT / ".env")
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"],
                         http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)))
    run_dir = CACHE_ROOT / "calls" / plan["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    spent, replies = 0.0, {}
    # Reserve before submission, including interrupted/failed calls. Rerunning
    # cannot silently resubmit a paid call whose outcome was not saved.
    for page in plan["pages"]:
        for request in page["requests"]:
            path = run_dir / (request["id"] + ".json")
            if path.exists():
                record = json.loads(path.read_text())
                spent += max(float(record.get("reserved_usd", 0)), float(record.get("cost_usd", 0)))
                if "reply" not in record:
                    raise ValueError(f"Unfinished paid request needs review before retry: {path}")
            else:
                reserve = request["planning_reserve_usd"]
                if spent + reserve > budget:
                    raise ValueError("Budget exhausted before next request")
                record = {"request_id": request["id"], "reserved_usd": reserve}
                path.write_text(json.dumps(record))
                spent += reserve
                response = client.models.generate_content(
                    model=request["model"], contents=request["prompt"],
                    config=types.GenerateContentConfig(
                        temperature=request["config"]["temperature"],
                        max_output_tokens=request["config"]["max_output_tokens"],
                        thinking_config=types.ThinkingConfig(thinking_budget=request["config"]["thinking_budget"]),
                        response_mime_type="application/json", response_schema=request["response_schema"]))
                usage = response.usage_metadata
                record.update(raw_response=response.model_dump(mode="json"), raw_text=response.text,
                              input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                              output_tokens=(getattr(usage, "candidates_token_count", 0) or 0) +
                                            (getattr(usage, "thoughts_token_count", 0) or 0))
                record["cost_usd"] = (record["input_tokens"] * plan["pricing"]["input_per_million"] +
                                      record["output_tokens"] * plan["pricing"]["output_per_million"]) / 1_000_000
                spent += max(0, record["cost_usd"] - reserve)
                path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
                record["reply"] = json.loads(response.text)
                ks.validate_reply(request, record["reply"])
                path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
            ks.validate_reply(request, record["reply"])
            replies[request["id"]] = record
            print(f"{page['page']}: key batch saved; reserved ${spent:.4f}", flush=True)
    # Publish no partial union: all engines must see the same completed cache.
    for page in plan["pages"]:
        ks.freeze_page(page, replies, CACHE_ROOT)
    print(f"Frozen {len(plan['pages'])} page synonym caches; actual saved cost "
          f"${sum(r.get('cost_usd', 0) for r in replies.values()):.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source", required=True, help="Comma-separated inference run IDs")
    prep.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("execute")
    run.add_argument("plan", type=Path)
    run.add_argument("--budget-usd", type=float, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        plan = prepare(args.source.split(","))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
        print(f"{len(plan['pages'])} pages; {plan['request_count']} calls; "
              f"planning reserve ${plan['planning_reserve_usd']:.4f}; no API calls made.\n{args.output.resolve()}")
    else:
        execute(json.loads(args.plan.read_text()), args.budget_usd)


if __name__ == "__main__":
    main()
