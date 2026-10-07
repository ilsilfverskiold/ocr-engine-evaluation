"""Shared runtime configuration, cache identity and strict judge JSON decoding."""
import hashlib
import json

MODEL = "gemini-2.5-flash"

PRICES = {"input_per_million": .30, "output_per_million": 2.50,
          "source": "https://ai.google.dev/gemini-api/docs/pricing", "checked": "2026-10-06"}

CONTROL_CONFIG = {"temperature": 0, "max_output_tokens": 4096, "thinking_budget": 1024}

def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()

def decode_reply(text: str) -> dict:
    """JSON's usual last-key-wins behavior must not conceal duplicate decisions."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON property: {key}")
            result[key] = value
        return result
    reply = json.loads(text, object_pairs_hook=unique)
    if not isinstance(reply, dict):
        raise ValueError("Judge response must be an object")
    return reply
