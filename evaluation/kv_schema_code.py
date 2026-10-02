"""Code verdicts for the key_value and schema legs.

Stage 1 (``evaluation/alias_pass.py``, once per document, engine-blind):
gemini-3.1-pro-preview sees the page image and our reference field list and
returns, per field, the label(s) printed on the page for it and whether it is
printed as a label+value pair. It is not asked for values. Cached under
``results/kv_alias_cache/<stem>__<model>.json``.

Stage 2 (this module, per engine, no model):
  1. parse the answer into pairs — JSON leaves (fenced or embedded JSON;
     every block of an answer with several fenced blocks; truncated JSON
     salvaged as far as it parses; duplicate keys are all kept, where a
     dict would keep only the last), or for Textract its FORMS
     pairs plus 2-column table rows / header->cell pairs;
  2. cleanup: an empty-valued key containing ":" is split at the first colon;
     a multi-line value whose later lines are "Label: value" lines (an engine
     that transcribed a whole column under one label) yields those lines as
     pairs of their own and the remaining lines as a second value for the
     key — the original pair is kept too;
  3. key match: the field path, its last segment, or (key_value only) a printed
     label — normalised (camelCase split), one-typo tolerant on tokens longer
     than 3 characters; a label of >= 2 tokens may sit inside a longer engine
     key ("Tax Id (Seller)" ~ "Tax Id:"); a line number ("5b") matches a key
     that starts with it; a key with the same words as the name in any
     order, stop words aside ("name of lessee" ~ lessee.name), unless the key
     ends in a stop word ("[name of lessee], of" is the fragment printed
     before the NEXT blank);
     For blank scalar fields, a key naming the semantic field takes
     precedence over repeated printed-label candidates. Otherwise a matching
     blank may satisfy an ambiguous repeated label; another field's explicit
     value is not attributed solely by label containment.
  4. value match: EXACT after normalisation — numbers as numbers, dates as
     dates, strings lowercased with punctuation collapsed. No containment:
     "193.00 SR" is not 193.00; a 10-star card mask is not the printed 12-x
     one. Two exceptions, both narrow:
       * checkboxes (reference true/false): true ~ true/yes/✓/☑/[x]/checked/
         selected, or the item's own label as the value ("Insulin": "Insulin"),
         or a radio group whose value is this option ("Filing Status":
         "Married filing separately"); false ~ false/no/☐/unchecked/
         NOT_SELECTED/empty, or a radio group that selected another option;
       * a printed label shared by several reference fields ("Your first name
         and middle initial" -> firstName + middleInitial; "Date" -> date +
         datetime), or one that itself names several things (" and ") when
         the reference holds only one of them: the reference value may be a
         whole-token part of the engine's value, because the page prints
         one value for both;
  5. key_value only, when no matched key holds the value:
       * a block printed under one label may come back as several pairs
         whose keys contain the label ("Seller Name" + "Seller Address" for
         "Seller:"): their ordered concatenation is the value;
       * when no key matched, or the field's printed label is a sentence
         fragment ("located at", "A security deposit of" — a lease prints
         no labels, so every engine keys those blanks differently), the value
         (4+ characters) may sit under a key that belongs to no other field;
       * a date the page prints as several blanks ("31" day of "December
         2025") is composed from the values returned for the field's printed
         labels;
  6. arrays (schema): a reference row is aligned to the engine row that
     shares a value in another column (the same name or id), else to the row
     with the same index when the engine returned as many rows. A cell is
     judged in that row only: a blank cell is `blank_ok` unless THAT row has
     a value for it — another row's value is not an invention. Without an
     aligned row a value is accepted from any row of its column and a blank
     is not held against the engine.

key_value denominator = the fields the page prints as label+value pairs (from
stage 1). A document without a stage-1 file (``alias_missing``), or none of
whose fields is a printed pair (``fields_total`` 0), is not applicable on
key_value — never "0 for every engine". schema = all fields.
"""

from __future__ import annotations

import difflib
import itertools
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
import unicodedata

SCORER_VERSION = "kv-schema-code-v10 (2026-10-02; checkbox group membership)"
ALIAS_MODEL = "gemini-3.1-pro-preview"
ALIAS_CACHE_DIR = Path(__file__).parent.parent / "results" / "kv_alias_cache"


# ---------------------------------------------------------------- stage-1 cache
def load_aliases(document_stem: str, model: str = ALIAS_MODEL) -> dict | None:
    """{field path: {"printed_labels": [...], "is_printed_pair": bool}} or None."""
    path = ALIAS_CACHE_DIR / f"{document_stem}__{model}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        aliases = data.get("by_path") or None
        # Source-image review of the five California leases: a blank is
        # labeled both by its preceding clause and by its trailing unit.
        # Keep the original cache and both conventions used by native APIs.
        if aliases and document_stem in {f"lease_{i:03d}" for i in range(5)}:
            for path, unit in (("lease.years", "year(s)"), ("lease.months", "month(s)")):
                if path in aliases:
                    labels = aliases[path].setdefault("printed_labels", [])
                    if unit not in labels:
                        labels.append(unit)
                    aliases[path]["review"] = "lease-unit-labels-v1 (2026-10-02)"
        return aliases
    except Exception:
        return None


# ---------------------------------------------------------------- parsing
class _Obj(list):
    """A JSON object as its ordered (key, value) pairs. Duplicate keys are kept:
    Gemini wrote "[street address], State of" three times on a lease, and
    ``dict`` would keep only the last."""


def _loads(s: str):
    return json.loads(s, object_pairs_hook=_Obj)


def flatten(obj: Any, prefix: str = "") -> list[list[str]]:
    out: list[list[str]] = []
    if isinstance(obj, _Obj):
        for k, v in obj:
            out += flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out += flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += flatten(v, f"{prefix}[{i}]")
    else:
        out.append([prefix, "" if obj is None else str(obj)])
    return out


def parse_json_answer(text: str) -> tuple[Any, str]:
    t = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", (text or "").strip())
    try:
        return _loads(t), "parsed as JSON"
    except Exception:
        pass
    # An answer with several fenced JSON blocks (Claude wrote a first object,
    # a note, then a corrected one on invoice_002): every block is read; a
    # salvage of the first alone would drop the correction.
    blocks = []
    for b in re.findall(r"```(?:json)?\s*\n(.*?)```", text or "", re.S):
        try:
            o = _loads(b.strip())
        except Exception:
            continue
        if isinstance(o, _Obj):
            blocks.append(o)
    if len(blocks) >= 2:
        merged = _Obj()
        for o in blocks:
            merged.extend(o)
        return merged, f"parsed ({len(blocks)} fenced JSON blocks, all read)"
    a, b = t.find("{"), t.rfind("}")
    if a >= 0 and b > a:
        try:
            return _loads(t[a:b + 1]), "parsed (embedded JSON)"
        except Exception:
            pass
    if a >= 0:
        body = t[a:].rstrip()
        cuts = [m.start() for m in re.finditer(r"[,}\]]", body)]
        for cut in reversed(cuts[-400:]):
            prefix = body[:cut].rstrip().rstrip(",")
            stack: list[str] = []
            ins = esc = False
            for ch in prefix:
                if ins:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        ins = False
                elif ch == '"':
                    ins = True
                elif ch in "{[":
                    stack.append("}" if ch == "{" else "]")
                elif ch in "}]" and stack:
                    stack.pop()
            if ins:
                continue
            try:
                return _loads(prefix + "".join(reversed(stack))), f"parsed (truncated JSON salvaged: {len(body) - cut} chars dropped)"
            except Exception:
                continue
    return None, "not JSON"


def textract_forms(blocks: list[dict]) -> list[dict[str, str]]:
    """Textract FORMS pairs from the raw block graph. A checkbox value is a
    SELECTION_ELEMENT child, not a WORD: it is read as SELECTED /
    NOT_SELECTED (joined with any words beside it) instead of being dropped.
    Used by the runner at capture time and by the scorer when it re-reads a
    stored raw_response."""
    block_map = {b["Id"]: b for b in blocks}

    def text_of(block: dict) -> str:
        parts = []
        for rel in block.get("Relationships", []):
            if rel["Type"] != "CHILD":
                continue
            for cid in rel["Ids"]:
                child = block_map.get(cid, {})
                if child.get("BlockType") == "WORD":
                    parts.append(child.get("Text", ""))
                elif child.get("BlockType") == "SELECTION_ELEMENT":
                    parts.append(child.get("SelectionStatus", "") or "")
        return " ".join(p for p in parts if p).strip()

    pairs = []
    for block in blocks:
        if block.get("BlockType") != "KEY_VALUE_SET" or "KEY" not in block.get("EntityTypes", []):
            continue
        value_text = ""
        for rel in block.get("Relationships", []):
            if rel["Type"] == "VALUE":
                for vid in rel["Ids"]:
                    value_text = text_of(block_map.get(vid, {}))
        pairs.append({"key": text_of(block), "value": value_text})
    return pairs


def engine_pairs(engine: str, text: str, structured_output: dict | None) -> tuple[list[list[str]], str]:
    """(label, value) pairs from what the engine returned, and how."""
    if engine == "textract_structured":
        so = structured_output or {}
        pairs = [[str(p.get("key", "")), str(p.get("value", ""))] for p in so.get("forms", [])]
        from evaluation.code_judges import _html_grid
        for t in so.get("tables", []):
            if not isinstance(t, dict):
                continue
            for html in dict.fromkeys(t.get(key) for key in ("cell_html", "html")):
                if not html:
                    continue
                grid = _html_grid(html)
                if grid and all(len(r) == 2 for r in grid):
                    pairs += [[r[0], r[1]] for r in grid]
                elif len(grid) > 1:
                    hdr = grid[0]
                    pairs += [[hdr[j] if j < len(hdr) else f"col{j+1}", cell] for r in grid[1:] for j, cell in enumerate(r)]
        return pairs, "Textract FORMS pairs + table cells (no JSON interface)"
    obj, how = parse_json_answer(text or "")
    return (flatten(obj) if obj is not None else []), how


# one "Label: value" line inside a multi-line value
_EMBEDDED = re.compile(r"^([A-Za-z][^:]{0,50}?)\s*:\s*(\S.*?)\s*$")


def cleanup(pairs: list[list[str]]) -> tuple[list[list[str]], list[str]]:
    out, log = [], []
    for k, v in pairs:
        if (v is None or str(v).strip() == "") and ":" in str(k):
            a, b = str(k).split(":", 1)
            if a.strip() and b.strip():
                out.append([a.strip(), b.strip()])
                log.append(f"{k!r} -> {a.strip()!r} : {b.strip()!r}")
                continue
        out.append([k, v])
        lines = [l.strip() for l in str(v).split("\n") if l.strip()] if isinstance(v, str) and "\n" in v else []
        if len(lines) >= 2:
            rest, embedded = [], []
            for line in lines:
                m = _EMBEDDED.match(line)
                if m:
                    embedded.append([m.group(1).strip(), m.group(2).strip()])
                else:
                    rest.append(line)
            if embedded:
                out += embedded
                if rest:
                    out.append([k, " ".join(rest)])
                log.append(f"{k!r}: {len(embedded)} embedded 'label: value' line(s) read as pairs of their own; "
                           f"the other lines kept as a second value")
    return out, log


# ---------------------------------------------------------------- matching
def norm_label(s: Any) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(s))   # digitalAssets -> digital Assets
    s = s.lower()
    s = re.sub(r"\[\d+\]", " ", s)
    s = s.replace("_", " ").replace(".", " ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def _tok_eq(x: str, y: str) -> bool:
    # Label typo tolerance must not merge numbered fields (line2/line6).
    # A letter typo with the same digits may still be ordinary OCR noise.
    return x == y or (len(x) > 3 and re.findall(r"\d+", x) == re.findall(r"\d+", y)
                     and difflib.SequenceMatcher(None, x, y).ratio() >= 0.8)


_STOP = {"of", "the", "and", "a", "an", "to", "in", "on", "at", "for", "by", "with", "from", "or", "as", "is", "be"}


def _same_terminal_index(engine_key: Any, ref: Any) -> bool:
    """Primitive-array positions are columns/categories, not reorderable rows.

    Record rows (items[0].price) are aligned separately. Their row indices
    must not constrain label matching, but values[0] must not borrow values[1].
    """
    a = re.search(r"\[(\d+)\]$", str(engine_key))
    b = re.search(r"\[(\d+)\]$", str(ref))
    return not b or bool(a and a.group(1) == b.group(1))


def label_eq(engine_key: Any, ref: Any) -> bool:
    """Engine key ~ reference name/label: same tokens (one-typo tolerant, with unchanged numeric identifiers), or
    the reference (at least 2 tokens) appears contiguously inside a longer
    engine key — "Tax Id (Seller)" for the printed label "Tax Id:", or the
    whole printed line of a tax form for its label. Never the other way
    round: a short engine key ("prescription drugs") does not match every
    reference path that contains it. A one-token reference that is a line
    number ("5b", "7") matches a key that starts with it ("5b Taxable
    amount") or a short key holding it ("Line 7")."""
    if not _same_terminal_index(engine_key, ref):
        return False
    ta, tb = norm_label(engine_key).split(), norm_label(ref).split()
    if not ta or not tb:
        return False
    if len(ta) == len(tb):
        return all(_tok_eq(x, y) for x, y in zip(ta, tb))
    if len(ta) < len(tb):
        return False
    if len(tb) < 2:
        tok = tb[0]
        if not any(ch.isdigit() for ch in tok):
            # a one-word name inside a longer key is not a match: "name" is
            # in "APP NAME", "returns" in "puerReturns", "Total" in "Total
            # Item(s)" (tried 2026-10-01: one gain, a dozen wrong columns)
            return False
        return ta[0] == tok or (len(ta) <= 3 and tok in ta)
    return any(all(_tok_eq(x, y) for x, y in zip(tb, ta[i:i + len(tb)]))
               for i in range(len(ta) - len(tb) + 1))


def name_eq(engine_key: Any, ref: Any) -> bool:
    """Same words in any order, stop words aside: "name of lessee" ~
    lessee.name, "street address (lessor)" ~ lessor.streetAddress, "Issue
    date" ~ "Date of issue". Not for a key that ends in a stop word
    ("[name of lessee], of" is the fragment printed before the NEXT blank),
    not for one-word names, not for keys longer than 8 tokens."""
    if not _same_terminal_index(engine_key, ref):
        return False
    ta, tb = norm_label(engine_key).split(), norm_label(ref).split()
    if not ta or not tb or len(ta) > 8 or ta[-1] in _STOP:
        return False
    sa, sb = {t for t in ta if t not in _STOP}, {t for t in tb if t not in _STOP}
    return len(sb) >= 2 and sa == sb


def is_fragment(label: Any) -> bool:
    """A printed "label" that is a sentence fragment leading into a blank
    ("located at", "A security deposit of", "and"): it ends in a stop word
    or is nothing but stop words. Every engine keys such a blank differently."""
    toks = norm_label(label).split()
    return bool(toks) and ((toks[-1] in _STOP and len(toks[-1]) >= 2) or all(t in _STOP for t in toks))


def _contains(ta: list[str], tb: list[str]) -> bool:
    """tb appears contiguously in ta."""
    return bool(tb) and len(ta) > len(tb) and any(ta[i:i + len(tb)] == tb for i in range(len(ta) - len(tb) + 1))


_TRUE = {"true", "yes", "y", "✓", "✔", "☑", "☒", "x", "[x]", "[✓]", "[✔]", "checked", "selected", "on", "ticked", "t", "☑️"}
_FALSE = {"false", "no", "n", "☐", "□", "[ ]", "[]", "unchecked", "not selected", "not_selected", "notselected", "off", "none", "null", "", "-", "unticked", "f"}


# Source-reviewed 1040 group: the independent nonresident-spouse election
# is stored under filingStatus too, but is not one of its five status options.
_GROUP_OPTIONS = {"filingStatus": {"single", "HOH", "MFJ", "MFS", "QSS"}}


def as_bool(s: Any):
    """True / False for a checkbox-style value, else None."""
    if isinstance(s, bool):
        return s
    t = " ".join(str(s).strip().lower().split())
    if t in _TRUE:
        return True
    if t in _FALSE:
        return False
    return None


def is_bool_expected(expected: Any) -> bool:
    return isinstance(expected, bool) or str(expected).strip() in ("True", "False", "true", "false")


def as_number(s: Any):
    """Parse a complete numeric value, keeping signs and every digit.

    Punctuation group separators must form groups of three; whitespace is
    formatting, as in the existing field policy. A single comma with one or
    two trailing digits is decimal; a single dot is decimal. Accounting
    parentheses mean negative here; decorative printed lease amounts are
    handled explicitly by field name in value_equal, not guessed globally.
    """
    t = unicodedata.normalize("NFKC", str(s)).strip().replace("−", "-")
    negative = t.startswith("(") and t.endswith(")")
    if negative:
        t = t[1:-1].strip()
    match = re.fullmatch(r"([+-])?\s*(?:RM|[$€£¥₹])?\s*([+-])?\s*(\d[\d.,\s]*)\s*%?", t, re.I)
    if not match or (match[1] and match[2]) or (negative and (match[1] or match[2])):
        return None
    digits = re.sub(r"\s+", "", match[3])
    decimal = None
    if "." in digits and "," in digits:
        decimal = "." if digits.rfind(".") > digits.rfind(",") else ","
    elif digits.count(".") == 1:
        decimal = "."
    elif digits.count(",") == 1 and re.search(r",\d{1,2}$", digits):
        decimal = ","
    integer, fraction = digits, ""
    if decimal:
        if digits.count(decimal) != 1:
            return None
        integer, fraction = digits.rsplit(decimal, 1)
        if not fraction.isdigit():
            return None
    # Permit exactly one punctuation grouping convention.
    separators = {c for c in integer if not c.isdigit()}
    if separators:
        if len(separators) != 1:
            return None
        sep = next(iter(separators))
        if sep not in (",", ".", " ") or not re.fullmatch(r"\d{1,3}(?:" + re.escape(sep) + r"\d{3})+", integer):
            return None
        integer = integer.replace(sep, "")
    if not integer.isdigit():
        return None
    try:
        number = Decimal(integer + ("." + fraction if fraction else ""))
        return number.copy_negate() if negative or (match[1] or match[2]) == "-" else number
    except InvalidOperation:
        return None


def as_date(s: Any):
    t = " ".join(str(s).strip().split())
    for f in ("%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%d/%m/%y"):
        try:
            return datetime.strptime(t, f).date()
        except Exception:
            pass
    return None


def norm_str(s: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(s).lower()).split())


_BLANK_WORDS = {"", "n a", "na", "none", "null", "nil", "nan"}
_DASHES = re.compile(r"^[-\u2010-\u2015]+$")


def is_blank_value(v: Any) -> bool:
    """An engine value that says "nothing here": empty, null, n/a, or the
    dash(es) a table prints in an empty cell."""
    if v is None:
        return True
    t = str(v).strip()
    return bool(_DASHES.match(t)) or norm_str(t) in _BLANK_WORDS


def _symbols(s: Any) -> str:
    """A value with no letters or digits ("£", "---", "✓") compared as itself:
    dashes of any length or kind are one dash, spaces dropped."""
    t = re.sub(r"\s+", "", str(s).strip())
    return "-" if _DASHES.match(t) else t


def value_equal(expected: Any, got: Any, names: list[str] | None = None, shared_label: bool = False) -> bool:
    """`names`: the field's own names/labels (a checked box may be written as
    its label); `shared_label`: the page prints one value for several
    reference fields, so a whole-token part of the engine value counts."""
    if expected is None or str(expected).strip() == "":
        return False
    if is_bool_expected(expected):
        want = str(expected).strip().lower() == "true"
        have = as_bool(got)
        if have is None and want and names and any(label_eq(got, n) for n in names):
            have = True          # "Insulin": "Insulin" — the checked item named as its own value
        return have is not None and have == want
    # Identifiers are strings, not quantities. Keep their leading zeros and
    # all digits while accepting printed separators (SSNs, phone numbers).
    name_tokens = {token for n in names or [] for token in norm_label(n).split()}
    identifier = bool(name_tokens & {"id", "identifier", "ssn", "tel", "phone", "telephone",
                                     "iban", "postcode", "postal", "zip", "number", "no"})
    count_field = bool(name_tokens & {"quantity", "qty", "count"}) or any(
        re.search(r"\bnumber of\b", norm_label(n)) for n in names or [])
    if identifier and not count_field:
        e, g = norm_str(expected), norm_str(got)
        if shared_label and as_number(expected) is None:
            te, tg = e.split(), g.split()
            if te and len(tg) > len(te) and any(tg[i:i + len(te)] == te for i in range(len(tg) - len(te) + 1)):
                return True
        return bool(e) and e.replace(" ", "") == g.replace(" ", "")
    # These source-reviewed lease fields transcribe a second positive amount
    # printed in parentheses in prose, not an accounting negative.
    printed_amount = any(str(n).split(".")[-1] in ("monthlyAmountPrinted", "securityDepositPrinted") for n in names or [])
    if printed_amount and re.fullmatch(r"\(\s*[$€£]\s*\d[\d., ]*\)", str(got).strip()):
        got = str(got).strip()[1:-1]
    # A tax-summary rate may be printed @6; @ is a rate prefix, not a digit.
    if "rate" in name_tokens:
        expected, got = re.sub(r"^@\s*", "", str(expected)), re.sub(r"^@\s*", "", str(got))
    a, b = as_number(expected), as_number(got)
    if shared_label and a is None:
        te, tg = norm_str(expected).split(), norm_str(got).split()
        if te and len(tg) > len(te) and any(tg[i:i + len(te)] == te for i in range(len(tg) - len(te) + 1)):
            return True
    # A printed count may include its unit. Restrict this to count fields;
    # neither amounts nor arbitrary prose can borrow a matching number.
    count_names = {"quantity", "qty", "total items", "item count", "number of items",
                   "total quantity", "item qty", "item quantity", "count"}
    if a is not None and b is None and any(norm_label(n) in count_names for n in names or []):
        match = re.fullmatch(r"\s*([-+]?\d[\d., ]*)\s+units?\s*", str(got), re.I)
        if match:
            b = as_number(match.group(1))
    if a is not None and b is not None:
        return a == b
    if a is not None or b is not None:
        # A malformed amount cannot recover by stripping punctuation below.
        return False
    da, db = as_date(expected), as_date(got)
    if da and db:
        return da == db
    e, g = norm_str(expected), norm_str(got)
    if not e:
        # a reference made only of symbols ("£", "---"): the normalised
        # form is empty for any engine value, so compare the symbols
        return bool(_symbols(expected)) and _symbols(expected) == _symbols(got)
    # whitespace is formatting: the invoice_002 reference holds "WV79662"
    # where the page prints "WV 79662" and every engine reads it so
    return e == g or e.replace(" ", "") == g.replace(" ", "")


@dataclass
class FieldVerdict:
    path: str
    expected: Any
    is_blank: bool
    verdict: str          # correct | blank_ok | wrong | invented | missing
    key: str = ""
    value: str = ""
    how: str = ""
    printed_pair: bool | None = None


@dataclass
class KvSchemaCodeResult:
    field_accuracy: float
    fields_total: int             # the denominator used
    fields_all: int               # every reference field
    correct: int
    missing: int
    wrong: int
    blank_total: int
    blank_correct: int
    parsed_how: str
    pairs: int
    alias_missing: bool
    cleanup_log: list[str] = field(default_factory=list)
    verdicts: list[FieldVerdict] = field(default_factory=list)


# ---------------------------------------------------------------- arrays
_ROW = re.compile(r"^(.*\[\d+\])\.(.+)$")


def _row_split(path: Any) -> tuple[str | None, str | None]:
    """"consultants[0].sales" -> ("consultants[0]", "sales"); not an array
    cell -> (None, None)."""
    m = _ROW.match(str(path))
    return (m.group(1), m.group(2)) if m else (None, None)


def _row_index(row_key: str) -> int:
    m = re.search(r"\[(\d+)\]$", row_key)
    return int(m.group(1)) if m else -1


def align_rows(items, pairs: list[list[str]]) -> dict[str, str | None]:
    """reference row key -> engine row key, one engine row per reference row.
    Rows of the same array are paired greedily by how many non-blank values
    they share in other columns — a name, description or id counts 2, a
    quantity, price or flag 1 (a receipt has many rows with quantity 1) —
    ties going to the same index. A reference row that shares nothing is
    paired with the row of the same index when the engine returned as many
    rows of that array as the reference holds and that row is still free;
    else None."""
    gt_rows: dict[str, list] = {}
    for it in items:
        rk, _ = _row_split(it.path)
        if rk:
            gt_rows.setdefault(rk, []).append(it)
    eng_rows: dict[str, list[int]] = {}
    for p, (k, _) in enumerate(pairs):
        rk, _ = _row_split(k)
        if rk:
            eng_rows.setdefault(rk, []).append(p)
    out: dict[str, str | None] = {rk: None for rk in gt_rows}
    scored: list[tuple[int, int, str, str]] = []     # (score, same-index, gt row, engine row)
    for rk, its in gt_rows.items():
        pref = norm_label(rk)
        for erk in eng_rows:
            if norm_label(erk) != pref:
                continue
            n = 0
            for it in its:
                if it.is_blank:
                    continue
                fld = _row_split(it.path)[1]
                if any(label_eq(_row_split(pairs[p][0])[1], fld) and value_equal(it.value, pairs[p][1], [it.path, fld]) for p in eng_rows[erk]):
                    n += 1 if (as_number(it.value) is not None or is_bool_expected(it.value) or len(norm_str(it.value)) < 3) else 2
            if n > 0:
                scored.append((n, 1 if _row_index(erk) == _row_index(rk) else 0, rk, erk))
    taken: set[str] = set()
    for n, same, rk, erk in sorted(scored, key=lambda t: (-t[0], -t[1])):
        if out[rk] is None and erk not in taken:
            out[rk], taken = erk, taken | {erk}
    for rk in gt_rows:
        if out[rk] is not None:
            continue
        pref = norm_label(rk)
        cands = [erk for erk in eng_rows if norm_label(erk) == pref]
        same_array = sum(1 for g in gt_rows if norm_label(g) == pref)
        if cands and len(cands) == same_array:
            erk = next((e for e in cands if _row_index(e) == _row_index(rk)), None)
            if erk is not None and erk not in taken:
                out[rk], taken = erk, taken | {erk}
    return out


# ---------------------------------------------------------------- verdicts
def score_fields(items, pairs: list[list[str]], leg: str, aliases: dict | None) -> KvSchemaCodeResult:
    """`items` are SchemaChecklistItem-like (path, value, is_blank)."""
    pairs, clog = cleanup(pairs)
    # a stage-1 file whose field paths do not match this reference (another
    # GT version) is as good as none
    if aliases and not any(it.path in aliases for it in items):
        aliases = None
    verdicts: list[FieldVerdict] = []
    # An answer with no pairs at all (not JSON, empty) earns nothing — not
    # even the blank fields, or a non-answer would score the blanks for free.
    answered = len(pairs) > 0
    reference_paths = {it.path for it in items}
    # printed labels shared by several reference fields (one printed value)
    label_count: dict[str, int] = {}
    for v in (aliases or {}).values():
        for lab in v.get("printed_labels") or []:
            label_count[norm_label(lab)] = label_count.get(norm_label(lab), 0) + 1

    def names_of(it) -> list[str]:
        ns = [it.path, it.path.split(".")[-1]]
        if leg == "key_value":
            ns += list((aliases or {}).get(it.path, {}).get("printed_labels") or [])
        return ns

    def key_matches(k: Any, ns: list[str]) -> bool:
        # A supplied structured path already identifies its field. It must
        # not satisfy a different missing field through a shared leaf name
        # or fuzzy alias. Bare printed labels remain ambiguous in the KV leg.
        if k in reference_paths and (leg == "schema" or "." in k or "[" in k):
            return k == ns[0]
        explicit = semantic_owners.get(k, set())
        if len(explicit) == 1 and ns[0] not in explicit:
            return False
        return any(label_eq(k, n) or name_eq(k, n) for n in ns)

    def key_names(k: Any, ns: list[str]) -> bool:
        """The key IS the field's name or label (same tokens, or the same
        words in another order) — not a longer key that merely contains the
        label. A 1040 prints "Last name" three times; a dependent's last
        name under "Dependent 1 Last name" is not an invented spouse."""
        tk = norm_label(k).split()
        return any((len(tk) == len(norm_label(n).split()) and label_eq(k, n)) or name_eq(k, n) for n in ns)

    all_names = [names_of(it) for it in items]
    # A qualified printed key such as 'State of (lessor)' identifies the
    # lessor, even though other fields share the bare 'State of' alias.
    # Bare labels and combined first-name/initial labels keep their shared
    # interpretation; only an unambiguous full semantic name takes priority.
    semantic_owners = {k: {it.path for it in items if "." in it.path
                           and key_names(k, [it.path])}
                       for k, _ in pairs}
    # which reference fields each engine key belongs to (by name/label)
    owners = [{j for j, ns in enumerate(all_names) if key_matches(k, ns)} for k, _ in pairs]
    aligned = align_rows(items, pairs)
    claimed = {erk for erk in aligned.values() if erk is not None}   # engine rows that belong to a reference row
    eng_row_pairs: dict[str, list[int]] = {}
    for p, (k, _) in enumerate(pairs):
        rk, _ = _row_split(k)
        if rk:
            eng_row_pairs.setdefault(rk, []).append(p)

    for i, it in enumerate(items):
        names = all_names[i]
        a = (aliases or {}).get(it.path, {})
        labels = list(a.get("printed_labels") or []) if leg == "key_value" else []
        pp = a.get("is_printed_pair") if aliases else None
        # one printed value for several reference fields: a label listed for
        # two fields, or a label that itself names several things ("Your
        # first name and middle initial" holds "Ruth J"; the reference may
        # have only firstName = "Ruth")
        shared = any(label_count.get(norm_label(lab), 0) > 1 for lab in labels) or (
            not is_bool_expected(it.value) and as_number(it.value) is None
            and any(" and " in norm_label(lab) for lab in labels))

        # -- arrays: judge the cell in the aligned engine row only
        rk, fld = _row_split(it.path)
        erk = aligned.get(rk) if rk else None
        if erk is not None:
            row_c = [(pairs[p][0], pairs[p][1]) for p in eng_row_pairs.get(erk, [])
                     if label_eq(_row_split(pairs[p][0])[1], fld) or name_eq(_row_split(pairs[p][0])[1], fld)]
            where = f"row {erk!r} aligned to {rk!r}"
            if it.is_blank:
                filled = [(k, v) for k, v in row_c if not is_blank_value(v)]
                if filled:
                    k, v = filled[0]
                    verdicts.append(FieldVerdict(it.path, it.value, True, "invented", str(k), str(v), f"{where}; the row holds a value for this blank cell", pp))
                else:
                    verdicts.append(FieldVerdict(it.path, it.value, True, "blank_ok", how=f"{where}; no value in that row for this blank cell", printed_pair=pp))
                continue
            hit = [(k, v) for k, v in row_c if value_equal(it.value, v, names, shared)]
            if hit:
                k, v = hit[0]
                verdicts.append(FieldVerdict(it.path, it.value, False, "correct", str(k), str(v), f"{where}; value equal after normalisation", pp))
            elif row_c:
                k, v = row_c[0]
                verdicts.append(FieldVerdict(it.path, it.value, False, "wrong", str(k), str(v), f"{where}; value differs", pp))
            else:
                verdicts.append(FieldVerdict(it.path, it.value, False, "missing", how=f"{where}; that row has no {fld!r}", printed_pair=pp))
            continue
        if rk and it.is_blank:
            # no engine row could be aligned to this reference row: another
            # row's value in this column is not an invention here
            if answered:
                verdicts.append(FieldVerdict(it.path, it.value, True, "blank_ok", how="no engine row aligned to this row; nothing attributable to this blank cell", printed_pair=pp))
            else:
                verdicts.append(FieldVerdict(it.path, it.value, True, "missing", how="no pairs at all", printed_pair=pp))
            continue

        # a reference row no engine row could be aligned to takes its value
        # from any row of the column, except rows that belong to another
        # reference row
        cands = [(k, v) for k, v in pairs if key_matches(k, names)
                 and (not rk or _row_split(k)[0] is not None)
                 and not (rk and _row_split(k)[0] in claimed)]
        # An explicitly named path outranks generic printed-label matches
        # in either field leg. A wrong total cannot borrow another amount.
        exact = [(k, v) for k, v in cands if k == it.path]
        if exact:
            cands = exact
        if it.is_blank:
            # A semantic field name beats an ambiguous printed label. The
            # spouse's explicitly filled "Last name (spouse)" cannot borrow
            # a blank from "Last name (dependent 1)" merely because both
            # contain "Last name". Repeated labels without a field-specific
            # key still use the candidates below (e.g. a lease's "year(s)").
            field_names = [it.path, it.path.split(".")[-1]]
            specific = [(k, v) for k, v in cands if key_names(k, field_names)]
            if specific:
                cands = specific
            elif leg == "key_value":
                # A generic repeated unit label may describe the filled
                # initial term, not the separate blank renewal term. Only
                # discard it when another field explains the exact value;
                # an explicit renewal field remains attributable and fails.
                kept = []
                for k, v in cands:
                    explained = [other.path for j, other in enumerate(items)
                                 if j != i and not other.is_blank and
                                 any(label_eq(k, label) for label in
                                     (aliases or {}).get(other.path, {}).get("printed_labels", [])) and
                                 value_equal(other.value, v, all_names[j])]
                    if not is_blank_value(v) and explained:
                        clog.append(f"{it.path}: ambiguous label {k!r} with value {v!r} "
                                    f"is explained by filled field(s) {explained}; not attributed to this blank")
                    else:
                        kept.append((k, v))
                cands = kept
        if it.is_blank and cands and not any(is_blank_value(v) for _, v in cands):
            # every matching pair holds a value: an invention only when one of
            # them NAMES the field. A key that merely contains the label
            # belongs to another line of the page ("Dependent 1 Last name"
            # for a blank spouse; "NOT_SELECTED The Lessee may have the
            # right…" for a blank renewal period).
            named = [(k, v) for k, v in cands if key_names(k, names)]
            if not named:
                verdicts.append(FieldVerdict(it.path, it.value, True, "blank_ok", str(cands[0][0]), str(cands[0][1]),
                                             "only keys that contain the label hold a value (another line of the page); nothing names this field", pp))
                continue
            cands = named
        if not cands and is_bool_expected(it.value) and not it.is_blank:
            # a radio group / list of checked items: the key is the group
            # ("Filing Status", "prescriptionDrugs[0]") and the value names
            # the option. This option is true when a group value names it,
            # false when the group chose something else.
            parts = it.path.split(".")
            group = parts[-2] if len(parts) >= 2 else None
            siblings = [x for x in items if x.path != it.path and x.path.rsplit(".", 1)[0] == it.path.rsplit(".", 1)[0]]

            def is_group_key(k: str) -> bool:
                # the group's own key — not a sibling option's key ("Filing
                # Status - Single" is the key of `single`, not of the group)
                return bool(group) and label_eq(k, group) and not any(
                    label_eq(k, s.path) or label_eq(k, s.path.split(".")[-1]) for s in siblings)

            gp = [(k, v) for k, v in pairs if is_group_key(k) and str(v).strip()
                  and as_number(v) is None and as_date(v) is None and as_bool(v) is None]
            if gp:
                named = [(k, v) for k, v in gp if any(label_eq(v, n) for n in names)]
                # A random/unknown group answer must not clear every false
                # checkbox. Require a recognized selection, and do not infer
                # independent flags merely from their shared JSON parent.
                allowed = _GROUP_OPTIONS.get(group)
                options = [x for x in [it, *siblings] if is_bool_expected(x.value)
                           and (allowed is None or x.path.split(".")[-1] in allowed)]
                recognized = [(k, v) for k, v in gp if any(
                    label_eq(v, n) for option in options for n in names_of(option))]
                if not named and (not recognized or
                                  (allowed is not None and parts[-1] not in allowed)):
                    gp = []
                else:
                    gp = named or recognized
            if gp:
                named = [(k, v) for k, v in gp if any(label_eq(v, n) for n in names)]
                want = str(it.value).strip().lower() == "true"
                k, v = (named or gp)[0]
                ok = bool(named) == want
                verdicts.append(FieldVerdict(it.path, it.value, False, "correct" if ok else "wrong", str(k), str(v),
                                             "group key matched; its value " + ("names this option" if named else "names another option"), pp))
                continue
        hit = [(k, v) for k, v in cands if (is_blank_value(v) if it.is_blank else value_equal(it.value, v, names, shared))]
        if hit:
            k, v = hit[0]
            how = "key matched; value equal after normalisation"
            if shared and not it.is_blank and not value_equal(it.value, v):
                how = "key matched; reference value is part of the engine value (label shared by several fields)"
            verdicts.append(FieldVerdict(it.path, it.value, it.is_blank, "blank_ok" if it.is_blank else "correct", str(k), str(v), how, pp))
            continue

        # -- key_value only: the value is there, keyed another way
        if leg == "key_value" and not it.is_blank:
            found = None
            # (a) a block split over several keys that contain the label
            toks = [norm_label(n).split() for n in names]
            toks = [t for t in toks if t and not (len(t) == 1 and (len(t[0]) < 3 or t[0] in _STOP))]
            parts = [(k, v) for p, (k, v) in enumerate(pairs) if str(v).strip() and owners[p] <= {i}
                     and any(_contains(norm_label(k).split(), t) for t in toks)]
            if len(parts) >= 2 and value_equal(it.value, " ".join(str(v) for _, v in parts)):
                found = (" + ".join(repr(str(k)) for k, _ in parts), " ".join(str(v) for _, v in parts),
                         f"value is the ordered concatenation of {len(parts)} pairs whose keys contain the label")
            # (b) no key matched, or the label is a sentence fragment: the
            # value under a key that belongs to no other field
            distinctive = (len(norm_str(it.value)) >= 4 and not is_bool_expected(it.value)
                           and (as_number(it.value) is None or sum(ch.isdigit() for ch in str(it.value)) >= 4))
            name_toks = {t for n in names for t in norm_label(n).split() if t not in _STOP}
            if found is None and (not cands or any(is_fragment(lab) for lab in labels)):
                for p, (k, v) in enumerate(pairs):
                    if owners[p] - {i} or is_blank_value(v) or not value_equal(it.value, v, names):
                        continue
                    # a true/false or a short number is not distinctive on its
                    # own: the key must share words with the field's label
                    # ("c If you elect to use the lump-sum election method" ~
                    # the printed line); a name, address, id or amount is
                    if not distinctive and len(name_toks & set(norm_label(k).split())) < 3:
                        continue
                    found = (str(k), str(v), ("no key matched" if not cands else "printed label is a sentence fragment")
                             + f"; value found under {str(k)!r}, a key that belongs to no other field")
                    break
            # (c) a date printed as several blanks
            if found is None and len(labels) >= 2 and as_date(it.value):
                per = [[str(v) for k, v in pairs if str(v).strip() and label_eq(k, lab)] for lab in labels]
                per = [vals for vals in per if vals]
                if len(per) >= 2:
                    for combo in itertools.product(*per):
                        joined = " ".join(combo)
                        if as_date(joined) == as_date(it.value):
                            found = (" + ".join(repr(lab) for lab in labels), joined, f"date composed from the values of {len(per)} printed blanks")
                            break
            if found is not None:
                k, v, how = found
                verdicts.append(FieldVerdict(it.path, it.value, False, "correct", k, v, how, pp))
                continue

        if not cands:
            if it.is_blank and answered:
                # the page has no value for this field; omitting it is as right as returning null
                verdicts.append(FieldVerdict(it.path, it.value, True, "blank_ok", how="blank field not returned (nothing invented)", printed_pair=pp))
            else:
                verdicts.append(FieldVerdict(it.path, it.value, it.is_blank, "missing", how="no pair whose key matches " + ", ".join(repr(n) for n in dict.fromkeys(names)), printed_pair=pp))
            continue
        k, v = cands[0]
        verdicts.append(FieldVerdict(it.path, it.value, it.is_blank, "invented" if it.is_blank else "wrong", str(k), str(v), "key matched; value differs", pp))

    if leg == "key_value" and aliases:
        denom = [v for v in verdicts if v.printed_pair]
    elif leg == "key_value":
        denom = []            # no stage 1: the denominator is undefined -> not applicable
    else:
        denom = list(verdicts)
    correct = sum(1 for v in denom if v.verdict in ("correct", "blank_ok"))
    return KvSchemaCodeResult(
        field_accuracy=round(correct / len(denom), 4) if denom else 0.0,
        fields_total=len(denom), fields_all=len(verdicts), correct=correct,
        missing=sum(1 for v in denom if v.verdict == "missing"),
        wrong=sum(1 for v in denom if v.verdict in ("wrong", "invented")),
        blank_total=sum(1 for v in denom if v.is_blank),
        blank_correct=sum(1 for v in denom if v.verdict == "blank_ok"),
        parsed_how="", pairs=len(pairs), alias_missing=(leg == "key_value" and not aliases),
        cleanup_log=clog, verdicts=verdicts,
    )
