"""Stage 1 of the key_value/schema scoring: printed labels per reference field.

One call per document, engine-blind: gemini-3.1-pro-preview is shown the page
image and our reference field list and returns, for each field, the label(s)
printed on the page for it and whether it is printed as a label+value pair.
It is NOT asked for the value (an earlier draft was; the model then echoed our
value instead of reading the page). Output is cached under
``results/kv_alias_cache/<stem>__<model>.json`` with the prompt, the raw
reply, token counts, cost and the per-path mapping stage 2 reads.

Run:  python -m evaluation.alias_pass --estimate        (no calls)
      python -m evaluation.alias_pass --run             (only documents without a cache)
      python -m evaluation.alias_pass --run --docs sroie_000,invoice_000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from evaluation.kv_schema_code import ALIAS_CACHE_DIR, ALIAS_MODEL

PROMPT = """You are looking at a document image. Below is a list of reference fields with the values our annotators recorded for this page.

For EACH numbered field, report:
- "printed_labels": every label printed on the page for that value, exactly as printed (e.g. "ROC NO:", "Round Amt:", "Total GST @6%:"). Empty list if the value is printed with no label (e.g. a company name in a header).
- "is_printed_pair": true if the field appears on the page as a label followed by its value.

Do not invent labels. Transcribe exactly as printed.

FIELDS:
{fields}

Output JSON only: {{"fields": [{{"n": 1, "printed_labels": ["..."], "is_printed_pair": true}}]}}"""

SCHEMA = {"type": "OBJECT", "required": ["fields"], "properties": {"fields": {"type": "ARRAY", "items": {"type": "OBJECT",
          "required": ["n", "printed_labels", "is_printed_pair"], "properties": {"n": {"type": "INTEGER"},
          "printed_labels": {"type": "ARRAY", "items": {"type": "STRING"}}, "is_printed_pair": {"type": "BOOLEAN"}}}}}}

# list prices per token, ai.google.dev/gemini-api/docs/pricing, read 2026-10-01
PRICE = {"gemini-3.1-pro-preview": (2.0 / 1e6, 12.0 / 1e6)}
IMAGE_TOKENS = 1120   # Gemini 3 default media resolution for one image


def _items_for(gt):
    from evaluation.schema_judge import build_schema_checklist
    return build_schema_checklist(gt.true_json, gt.json_schema, Path(gt.document_path).stem, use_cache=True, scalar_only=False)


def _fields_text(items) -> str:
    return "\n".join(f"{i}. {it.path}: {it.value if not it.is_blank else '(blank)'}" for i, it in enumerate(items, 1))


def cache_path(stem: str, model: str = ALIAS_MODEL) -> Path:
    return ALIAS_CACHE_DIR / f"{stem}__{model}.json"


def estimate(gts) -> tuple[int, float]:
    """(documents without a cache, estimated USD) — 2.58 chars/token for text, 38.9 output tokens per field, calibrated on sroie_000."""
    n, usd = 0, 0.0
    pi, po = PRICE[ALIAS_MODEL]
    for gt in gts:
        stem = Path(gt.document_path).stem
        if cache_path(stem).exists():
            continue
        items = _items_for(gt)
        prompt = PROMPT.format(fields=_fields_text(items))
        usd += (IMAGE_TOKENS + len(prompt) / 2.58) * pi + len(items) * 30 * po
        n += 1
    return n, usd


def run_one(gt, model: str = ALIAS_MODEL) -> dict:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    items = _items_for(gt)
    prompt = PROMPT.format(fields=_fields_text(items))
    img = Path(gt.document_path)
    mime = "image/jpeg" if img.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    t0 = time.time()
    resp = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=img.read_bytes(), mime_type=mime), prompt],
        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json", response_schema=SCHEMA),
    )
    u = resp.usage_metadata
    fields = json.loads(resp.text).get("fields", [])
    amap = {f.get("n"): f for f in fields}
    by_path = {it.path: {"printed_labels": list(amap.get(i, {}).get("printed_labels") or []),
                         "is_printed_pair": bool(amap.get(i, {}).get("is_printed_pair"))}
               for i, it in enumerate(items, 1)}
    pi, po = PRICE[model]
    out = {
        "document": str(gt.document_path), "model": model, "prompt": prompt, "raw": resp.text,
        "tokens": {"input": u.prompt_token_count, "output": u.candidates_token_count,
                   "thinking": getattr(u, "thoughts_token_count", 0) or 0},
        "cost_usd": (u.prompt_token_count or 0) * pi + ((u.candidates_token_count or 0) + (getattr(u, "thoughts_token_count", 0) or 0)) * po,
        "seconds": round(time.time() - t0, 2), "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_version": "labels-only v1 (2026-10-01)", "by_path": by_path,
    }
    ALIAS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path(Path(gt.document_path).stem, model).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    from dotenv import load_dotenv
    root = Path(__file__).parent.parent
    load_dotenv(dotenv_path=str(root / ".env"))
    load_dotenv(dotenv_path=str(root.parent / ".env"))
    from evaluation.ground_truth import load_corpus
    p = argparse.ArgumentParser()
    p.add_argument("--estimate", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--docs", default=None)
    a = p.parse_args()
    gts = [gt for gt in load_corpus() if gt.has_gt_for("schema")]
    if a.docs:
        want = set(a.docs.split(","))
        gts = [gt for gt in gts if Path(gt.document_path).stem in want]
    n, usd = estimate(gts)
    print(f"{len(gts)} documents with structured GT; {n} without a stage-1 cache; estimated ${usd:.2f}")
    if not a.run:
        return
    spent = 0.0
    for gt in gts:
        stem = Path(gt.document_path).stem
        if cache_path(stem).exists():
            continue
        out = run_one(gt)
        spent += out["cost_usd"]
        print(f"  {stem}: {len(out['by_path'])} fields, tokens {out['tokens']}, ${out['cost_usd']:.4f}, {out['seconds']}s")
    print(f"done; spent ${spent:.2f}")


if __name__ == "__main__":
    main()
