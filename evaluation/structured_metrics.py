"""Field accuracy: the comparison shared by the key-value and schema legs.

Two ideas do all the work here.

**Normalise at comparison time, never in the stored ground truth.** `$1,249.50`
and `1249.5` are the same amount and must score as a match, but the reference
keeps the string the page prints and the engine's reply stays exactly as it
came back. Both remain inspectable when a number looks wrong.

**No partial credit on a value.** One wrong digit in an identifier or an amount
means the field is wrong. That is where OCR actually fails, and a similarity
score would quietly convert a wrong invoice number into 0.95 of a right one.

What is deliberately tolerant: which key a value arrived under (case,
punctuation, snake_case vs the printed label), whether "not present" was
expressed as an explicit null or an absent key, and any extra keys the engine
volunteered.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Any

# --------------------------------------------------------------------------
# Value normalisation
# --------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"^[+-]?[\d.,\s]*\d[\d.,\s]*$")
_CURRENCY_CHARS = "$£€¥₹"

_DATE_PATTERNS: tuple[tuple[re.Pattern[str], tuple[str, str, str]], ...] = (
    (re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$"), ("y", "m", "d")),
    (re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})$"), ("a", "b", "y")),
    (re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{2})$"), ("a", "b", "yy")),
)

_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"]
    )
}
_TEXT_DATE_RE = re.compile(
    r"^(\d{1,2})\s+([a-z]{3,9})\s+(\d{4})$|^([a-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})$"
)


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


def as_iso_date(value: Any) -> str | None:
    """ISO-8601 form of a date string, or None if it is not a date.

    Ambiguous day/month order (`01/02/2019`) is reported as the pair of
    readings it could be, so `15/01/2019` == `2019-01-15` matches while
    `01/02/2019` matches both `2019-02-01` and `2019-01-02`. Being explicit
    beats silently picking a locale.
    """
    candidates = _date_candidates(value)
    return sorted(candidates)[0] if candidates else None


def _date_candidates(value: Any) -> set[str]:
    if isinstance(value, (int, float, bool)) or value is None:
        return set()
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    if not text:
        return set()

    for pattern, order in _DATE_PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        groups = match.groups()
        try:
            if order[0] == "y":
                return {date(int(groups[0]), int(groups[1]), int(groups[2])).isoformat()}
            year = int(groups[2])
            if order[2] == "yy":
                year += 2000 if year < 70 else 1900
            a, b = int(groups[0]), int(groups[1])
            out = set()
            for day, month in ((a, b), (b, a)):
                try:
                    out.add(date(year, month, day).isoformat())
                except ValueError:
                    continue
            return out
        except ValueError:
            return set()

    match = _TEXT_DATE_RE.match(text)
    if match:
        if match.group(1):
            day, month_name, year = match.group(1), match.group(2), match.group(3)
        else:
            month_name, day, year = match.group(4), match.group(5), match.group(6)
        month = _MONTHS.get(month_name[:3])
        if month:
            try:
                return {date(int(year), month, int(day)).isoformat()}
            except ValueError:
                return set()
    return set()


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


def values_match(reference: Any, predicted: Any) -> bool:
    """Do these two values say the same thing?

    Order of attempts: blankness, then number by value, then date to ISO, then
    the normalised string. No fuzzy fallback — a near miss on a value is a miss.
    """
    ref_blank, pred_blank = is_blank(reference), is_blank(predicted)
    if ref_blank or pred_blank:
        return ref_blank and pred_blank

    ref_number, pred_number = as_number(reference), as_number(predicted)
    if ref_number is not None and pred_number is not None:
        return abs(ref_number - pred_number) < 1e-9

    ref_dates, pred_dates = _date_candidates(reference), _date_candidates(predicted)
    if ref_dates and pred_dates:
        return bool(ref_dates & pred_dates)

    return normalize_string(reference) == normalize_string(predicted)


# --------------------------------------------------------------------------
# Label normalisation
# --------------------------------------------------------------------------

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


def label_variants(path: str) -> set[str]:
    """Every label a dotted path could reasonably be reported under.

    `summary.total_gross_worth` matches an engine that returned it nested, flat
    as `total_gross_worth`, or under the printed label `Total Gross Worth`.
    """
    variants = {normalize_label(path)}
    stripped = re.sub(r"\[\d+\]", "", path)
    variants.add(normalize_label(stripped))
    segments = [s for s in stripped.split(".") if s]
    if segments:
        variants.add(normalize_label(segments[-1]))
        if len(segments) > 1:
            variants.add(normalize_label(" ".join(segments[-2:])))
    return {v for v in variants if v}


# --------------------------------------------------------------------------
# Field walking
# --------------------------------------------------------------------------

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


def lookup_path(obj: Any, path: str) -> tuple[bool, Any]:
    """Read a dotted/indexed path out of parsed engine output.

    Returns (found, value). `found=False` and an explicit null are different
    facts to the caller even though `values_match` treats them alike.

    Keys may themselves contain dots. A financial statement's columns are
    dates — `values` holds `"31.1.04"` — so splitting the path on every dot
    turned one real key into three tokens and found nothing. Feeding those
    documents their own reference scored 0.41: every engine was capped there
    no matter what it read. Keys are therefore matched greedily, longest
    first, against the keys that actually exist at each level.
    """
    return _resolve(obj, path)


def _resolve(current: Any, rest: str) -> tuple[bool, Any]:
    if not rest:
        return True, current
    if rest.startswith("["):
        match = re.match(r"\[(\d+)\]", rest)
        if not match or not isinstance(current, list):
            return False, None
        index = int(match.group(1))
        if index >= len(current):
            return False, None
        return _resolve(current[index], rest[match.end():].lstrip("."))
    if not isinstance(current, dict):
        return False, None
    for key in sorted(current, key=len, reverse=True):
        key = str(key)
        if rest == key:
            return True, current[key]
        if rest.startswith(key + "."):
            found, value = _resolve(current[key], rest[len(key) + 1:])
            if found:
                return True, value
        if rest.startswith(key + "["):
            found, value = _resolve(current[key], rest[len(key):])
            if found:
                return True, value
    return False, None


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

@dataclass
class FieldVerdict:
    path: str
    expected: Any
    predicted: Any
    found: bool
    correct: bool
    # "match" | "wrong_value" | "missing" | "invented_value"
    #
    # invented_value is called out separately because it is the failure that
    # matters most in production: the page shows a blank field and the engine
    # filled it in.
    kind: str


@dataclass
class FieldAccuracy:
    score: float | None
    total: int
    correct: int
    missing: int
    wrong_value: int
    invented_value: int
    blank_total: int
    blank_correct: int
    verdicts: list[FieldVerdict] = field(default_factory=list)

    def as_metrics(self) -> dict[str, float | int | None]:
        return {
            "field_accuracy": self.score,
            "fields_total": self.total,
            "fields_correct": self.correct,
            "fields_missing": self.missing,
            "fields_wrong_value": self.wrong_value,
            "fields_invented_value": self.invented_value,
            "blank_fields_total": self.blank_total,
            "blank_fields_correct": self.blank_correct,
        }


def score_fields(reference_fields: list[Field], predicted: Any) -> FieldAccuracy:
    """Field accuracy = matched fields / reference fields.

    A field is matched when the value found at its path says the same thing as
    the reference. Keys the engine added that the reference does not mention
    are ignored, not penalised — the leg measures whether the asked-for values
    came back, not whether anything else did.
    """
    verdicts: list[FieldVerdict] = []
    for ref in reference_fields:
        found, value = lookup_path(predicted, ref.path)
        correct = values_match(ref.value, value if found else None)
        if correct:
            kind = "match"
        elif is_blank(ref.value):
            kind = "invented_value"
        elif not found or is_blank(value):
            kind = "missing"
        else:
            kind = "wrong_value"
        verdicts.append(
            FieldVerdict(ref.path, ref.value, value if found else None, found, correct, kind)
        )

    total = len(verdicts)
    correct_count = sum(1 for v in verdicts if v.correct)
    blank = [v for v in verdicts if is_blank(v.expected)]
    return FieldAccuracy(
        score=round(correct_count / total, 4) if total else None,
        total=total,
        correct=correct_count,
        missing=sum(1 for v in verdicts if v.kind == "missing"),
        wrong_value=sum(1 for v in verdicts if v.kind == "wrong_value"),
        invented_value=sum(1 for v in verdicts if v.kind == "invented_value"),
        blank_total=len(blank),
        blank_correct=sum(1 for v in blank if v.correct),
        verdicts=verdicts,
    )
