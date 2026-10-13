"""Numeric/label normalization and schema-guided field traversal."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

# Value normalisation

_NUMBER_RE = re.compile(r"^[+-]?[\d.,\s]*\d[\d.,\s]*$")
_CURRENCY_CHARS = "$£€¥₹"


def normalize_string(value: Any) -> str:
    """Trim, collapse whitespace, case-fold. Unicode forms unified first."""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip().casefold()


def as_number(value: Any) -> float | None:
    """The numeric value of `value`, or None if it is not a number.

    Handles what document text does to numbers: currency symbols, thousands
    separators in either convention, percentages, and accounting negatives in
    parentheses. `1.249,50` and `1,249.50` are both 1249.5.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text:
        return None

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative, text = True, text[1:-1].strip()
    text = text.rstrip("%").strip()
    for symbol in _CURRENCY_CHARS:
        text = text.replace(symbol, "")
    text = text.replace(" ", " ").strip()
    if text.startswith("-"):
        negative, text = not negative, text[1:].strip()
    if text.startswith("+"):
        text = text[1:].strip()
    if not text or not _NUMBER_RE.match(text):
        return None

    text = text.replace(" ", "")
    if "," in text and "." in text:
        # Whichever separator comes last is the decimal point.
        decimal_sep = "," if text.rfind(",") > text.rfind(".") else "."
        thousands_sep = "." if decimal_sep == "," else ","
        text = text.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif "," in text:
        parts = text.split(",")
        # "1,50" is a decimal comma; "1,500" and "1,234,567" are separators.
        if len(parts) == 2 and len(parts[1]) != 3:
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")
    elif text.count(".") > 1:
        text = text.replace(".", "")

    try:
        number = float(text)
    except ValueError:
        return None
    return -number if negative else number


def is_blank(value: Any) -> bool:
    """True for the several ways a document says "this field has no value".

    A missing key and an explicit null both mean "not present" and must be
    treated alike, or an engine is penalised for how it spells absence.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return normalize_string(value) in ("", "null", "none", "n/a", "na", "-", "--")
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


# Label normalisation

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def normalize_label(label: str) -> str:
    """`Invoice No.:` and `invoice_no` and `invoiceNo` all become `invoice no`.

    Case- and punctuation-insensitive with the trailing colon stripped, and
    snake_case/camelCase split into words so a schema key and the label printed
    on the page can meet in the middle.
    """
    text = unicodedata.normalize("NFKC", str(label or "")).strip()
    text = _CAMEL_BOUNDARY.sub(" ", text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


# Field walking

@dataclass(frozen=True)
class Field:
    """One comparable value in the reference, with the path that located it."""

    path: str
    value: Any
    in_array: bool = False


def walk_fields(
    obj: Any, schema: dict | None = None, prefix: str = "", in_array: bool = False
) -> list[Field]:
    """Flatten a reference to its leaf values, walked over the schema's keys.

    When a schema is supplied it drives the walk, so a key the schema declares
    but the page does not carry still counts as a field the engine had to get
    right (by returning null). Arrays are aligned in order and each cell is a
    field, exactly as docs/scoring-design.md specifies.
    """
    fields: list[Field] = []
    schema_type = (schema or {}).get("type") if isinstance(schema, dict) else None
    # Human review may make an object nullable without changing its fields.
    # Keep walking those declared child keys when the reference object is null.
    if isinstance(schema_type, list):
        schema_type = next((t for t in schema_type if t != "null"), None)

    if isinstance(obj, dict) or schema_type == "object":
        properties = (schema or {}).get("properties") if isinstance(schema, dict) else None
        keys = list(properties) if properties else []
        if isinstance(obj, dict):
            keys += [k for k in obj if k not in keys]
        for key in keys:
            child_schema = (properties or {}).get(key) if properties else None
            value = obj.get(key) if isinstance(obj, dict) else None
            path = f"{prefix}.{key}" if prefix else str(key)
            fields.extend(walk_fields(value, child_schema, path, in_array))
        return fields

    if isinstance(obj, list):
        item_schema = (schema or {}).get("items") if isinstance(schema, dict) else None
        for index, value in enumerate(obj):
            fields.extend(
                walk_fields(value, item_schema, f"{prefix}[{index}]", True)
            )
        return fields

    fields.append(Field(path=prefix, value=obj, in_array=in_array))
    return fields


# Scoring
