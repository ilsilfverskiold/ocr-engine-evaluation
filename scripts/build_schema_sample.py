"""Freeze one shared schema field sample using local tokenizer sizing, no inference.

Tokenizer JSON files are supplied explicitly; no weights or API calls are used.
Runtime inference/scoring loads the resulting manifest without these dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.ground_truth import load_corpus
from evaluation.legs import LEGS
from evaluation.schema_contract import extraction_schema, restrict_schema, sample_reference_id

VERSION = "shared-schema-sample-v1"
# Entire image allowances plus 512 tokens for chat framing/safety. DeepSeek's
# pinned vLLM v0.21 processor: 1024 base, 640 crops, max six crops => <=933
# visual tokens, rounded to 1024 here. Qwen/GLM use their processor maxima.
LIMITS = {
    "deepseek_ocr": {"context": 8192, "image": 1024, "output": 2048, "tokenizer": "deepseek"},
    "qwen3_vl": {"context": 24576, "image": 16384, "output": 4096, "tokenizer": "qwen"},
    "glm_ocr": {"context": 20480, "image": 12288, "output": 4096, "tokenizer": "glm"},
}


def build(tokenizer_dir: Path) -> dict:
    from tokenizers import Tokenizer
    paths = {v["tokenizer"]: tokenizer_dir / (v["tokenizer"] + "-tokenizer.json")
             for v in LIMITS.values()}
    tokenizers = {name: Tokenizer.from_file(str(path)) for name, path in paths.items()}
    documents = {}
    for gt in load_corpus():
        if gt.json_schema is None:
            continue
        stem = Path(gt.document_path).stem
        schema = extraction_schema(gt.json_schema, stem)
        excluded = set((gt.metadata.get("structured_gold_reference") or {}).get("excluded_fields", []))
        candidates = [key for key in schema["properties"] if key not in excluded]
        if not isinstance(gt.true_json, dict) or any(key not in gt.true_json for key in candidates):
            raise ValueError(f"{stem}: sample builder requires reviewed flat fields")
        # Hash ordering avoids selecting only the first section or sorting by
        # short/easy values. Greedy filling is maximal, not a knapsack optimum.
        order = sorted(candidates, key=lambda key: hashlib.sha256(
            f"{VERSION}\0{stem}\0{key}".encode()).hexdigest())

        def measure(keys):
            chosen = restrict_schema(schema, keys)
            answer = json.dumps({key: gt.true_json[key] for key in keys}, ensure_ascii=False, indent=2)
            sizes = {}
            for engine, limits in LIMITS.items():
                tokenizer = tokenizers[limits["tokenizer"]]
                prompt = LEGS["schema"].task_for(chosen, engine=engine).prompt
                p = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
                a = len(tokenizer.encode(answer, add_special_tokens=False).ids)
                sizes[engine] = {"prompt_tokens": p, "reference_answer_tokens": a,
                    "context_reserved": p + limits["image"] + limits["output"] + 512,
                    "fits": a * 1.25 <= limits["output"] and
                            p + limits["image"] + limits["output"] + 512 <= limits["context"]}
            return sizes

        keys = []
        for key in order:
            proposed = keys + [key]
            if all(v["fits"] for v in measure(proposed).values()):
                keys = proposed
        if not keys:
            raise ValueError(f"{stem}: no field fits; do not submit an empty schema")
        # Restore reference order after selection and recheck the exact request.
        keys = [key for key in candidates if key in keys]
        sizes = measure(keys)
        if not all(v["fits"] for v in sizes.values()):
            raise ValueError(f"{stem}: serialized sample exceeds budget")
        documents[stem] = {"reference_id": sample_reference_id(gt),
            "image_sha256": hashlib.sha256(Path(gt.document_path).read_bytes()).hexdigest(),
            "available": len(candidates), "keys": keys, "sizing": sizes}
    return {"version": VERSION, "policy": "hash-order greedy fill; identical keys across engines; 25% answer headroom; 512 context headroom",
        "limits": LIMITS, "tokenizers_sha256": {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in paths.items()},
        "deepseek_processor_source": "https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/transformers_utils/processors/deepseek_ocr.py",
        "documents": documents}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "documents/schema_sample.json")
    args = parser.parse_args()
    manifest = build(args.tokenizer_dir)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    pages = list(manifest["documents"].values())
    print(f"{len(pages)} pages; {sum(len(p['keys']) for p in pages)} / "
          f"{sum(p['available'] for p in pages)} fields; "
          f"{sum(len(p['keys']) == p['available'] for p in pages)} pages retain all fields")
