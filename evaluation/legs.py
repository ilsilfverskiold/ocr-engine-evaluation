"""The six legs: what each one asks the engine for, and how a run is grouped
into inference passes.

Authoritative spec: docs/scoring-design.md. This module is that document turned
into data — the prompts below are the asks it fixes, verbatim, so every engine
is asked for the same thing and the benchmark holds one variable still.

A leg needs an inference *pass* only when an engine cannot answer it from the
default conversion. Legs 1-3 come out of one default call for almost every
engine; legs 4-5 are one prompted call each. `plan_passes()` works out the
minimum set for a given engine and leg selection, so selecting more legs never
costs more calls than the legs actually require.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from engines import ENGINES
from runners.base import OCRTask

def instance_template(schema: dict[str, Any]) -> Any:
    """JSON Schema -> an empty example instance of it.

    Some cards demonstrate the extraction ask with a filled-in example object
    rather than a JSON Schema document. GLM-OCR is one: handed
    `{"type": "object", "properties": {...}}` it echoes that meta-document back
    verbatim and scores zero; handed `{"invoice_no": "", ...}` it fills the
    values in. Same schema, same fields, same scoring — only the representation
    the model was trained to expect differs.
    """
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((value for value in kind if value != "null"), None)
    if kind == "object":
        return {key: instance_template(value)
                for key, value in (schema.get("properties") or {}).items()}
    if kind == "array":
        return [instance_template(schema.get("items") or {})]
    return ""


TEXT = "text"
MARKDOWN = "markdown"
TABLES = "tables"
KEY_VALUE = "key_value"
SCHEMA = "schema"
GEOMETRY = "geometry"

ALL_LEGS: tuple[str, ...] = (TEXT, MARKDOWN, TABLES, KEY_VALUE, SCHEMA, GEOMETRY)

# The default conversion pass. `task=None` — no prompt is sent, so an engine
# behaves exactly as it does out of the box.
DEFAULT_PASS = "default"


@dataclass(frozen=True)
class LegSpec:
    name: str
    title: str
    question: str          # the user question this leg answers
    prompt: str | None     # the ask; None = the engine's default conversion
    scored_on: str         # headline metric, for the report header
    method: str = "deterministic"
    needs_schema: bool = False
    # Name of the inference pass this leg's prompt belongs to. Legs sharing a
    # pass name share one call. Legs answered natively never trigger a pass.
    pass_name: str = DEFAULT_PASS

    def task_for(self, json_schema: dict[str, Any] | None = None,
                 engine: str | None = None) -> OCRTask | None:
        """Build this leg's task, honouring any model-card prompt the engine
        declares. A specialised model degrades under a generic instruction, so
        engines.py wins over the generic prompt here — see
        docs/scoring-design.md, "Prompting is part of the measurement"."""
        prompt = self.prompt
        schema_render = "json_schema"
        if engine is not None:
            from engines import ENGINES
            prompt = ENGINES[engine].prompt_for(self.name, prompt)
            schema_render = ENGINES[engine].schema_render
        if prompt is None:
            return None
        if self.needs_schema:
            if json_schema is None:
                raise ValueError(
                    f"Leg {self.name!r} needs the document's json_schema and got None"
                )
            rendered = (instance_template(json_schema)
                        if schema_render == "instance_template" else json_schema)
            prompt = prompt.replace(
                "{schema}", json.dumps(rendered, indent=2, ensure_ascii=False)
            )
        if engine is not None:
            from engines import ENGINES
            response_format = ENGINES[engine].response_format_for(
                self.name, self.needs_schema)
        else:
            response_format = ("json" if self.needs_schema
                               or self.name in (KEY_VALUE, GEOMETRY) else None)
        return OCRTask(
            leg=self.name,
            prompt=prompt,
            json_schema=json_schema if self.needs_schema else None,
            response_format=response_format,
        )


LEGS: dict[str, LegSpec] = {
    TEXT: LegSpec(
        name=TEXT,
        title="Text accuracy",
        question="Did the engine read the words on the page?",
        prompt=None,
        scored_on="GT coverage (CER/WER/word-F1 supporting)",
    ),
    MARKDOWN: LegSpec(
        name=MARKDOWN,
        title="Markdown / reading order",
        question="Did the engine preserve the document's structure and order?",
        prompt=(
            "Convert this document to Markdown. Preserve reading order, "
            "headings, lists and tables. Output only the Markdown."
        ),
        scored_on="structural element accuracy; CER/WER and block F1 separately",
        method="code verdicts against cached image checklist",
        pass_name="markdown",
    ),
    TABLES: LegSpec(
        name=TABLES,
        title="Table extraction",
        question="Did the engine recover the table's items and their associated values?",
        prompt=(
            "Extract every printed table from this image. Output only the tables, "
            "preserving cell text, row/column associations, headers, empty cells and "
            "merged cells. Use valid Markdown for simple tables and HTML with "
            "rowspan/colspan when needed. Do not invent headers or values, split "
            "multiline cell text into unrelated rows, or turn unrelated prose/form "
            "fields into table rows."
        ),
        scored_on="reference value credit; extras, strict cell F1 and structure reported separately",
        method="label-associated logical items and non-overlapping value spans against reviewed table references",
        pass_name="tables",
    ),
    KEY_VALUE: LegSpec(
        name=KEY_VALUE,
        title="Key-value pairs",
        question="Did the engine pull the printed fields off the page?",
        prompt=(
            "Extract every printed label and its value from this document as a "
            "flat JSON object of label -> value. Use the label exactly as "
            "printed. Use null where a printed field is blank. Output only JSON."
        ),
        scored_on="shared-reference field accuracy",
        method="known keys first; judge confirms unmatched value candidates and caches variations; code checks values",
        pass_name="key_value",
    ),
    SCHEMA: LegSpec(
        name=SCHEMA,
        title="Schema JSON",
        question="Can the engine fill a schema you hand it?",
        prompt=(
            "Extract the values from this document as JSON conforming exactly "
            "to this schema. Use null for fields not present on the page. "
            "Output only JSON.\n\nSchema:\n{schema}"
        ),
        scored_on="field accuracy with extra-key penalty; JSON/schema validity separately",
        method="code verdicts against schema ground truth",
        needs_schema=True,
        pass_name="schema",
    ),
    GEOMETRY: LegSpec(
        name=GEOMETRY,
        title="Geometry",
        question="Where on the page is each word?",
        prompt=(
            'For every word, return its text and bounding box as '
            '[{"text": ..., "bbox": [x0, y0, x1, y1]}], normalised to 0-1. '
            "Output only JSON."
        ),
        scored_on="located-unit recall; precision/F1 and box-text CER/WER separately",
        pass_name="geometry",
    ),
}


def parse_legs(value: str | None) -> list[str]:
    """Parse a --legs value into leg names, preserving the canonical order."""
    if not value or value.strip().lower() == "all":
        return list(ALL_LEGS)
    requested = [v.strip() for v in value.split(",") if v.strip()]
    unknown = [v for v in requested if v not in LEGS]
    if unknown:
        raise ValueError(
            f"Unknown leg(s): {', '.join(unknown)}. Known: {', '.join(ALL_LEGS)}"
        )
    return [leg for leg in ALL_LEGS if leg in requested]


@dataclass(frozen=True)
class PassPlan:
    """One inference call per document for one engine, and the legs it feeds."""

    name: str
    engine: str
    legs: tuple[str, ...]
    leg_for_task: str | None    # which leg's prompt this pass sends; None = default


def plan_passes(engine: str, legs: list[str]) -> list[PassPlan]:
    """The minimum set of inference passes for one engine over `legs`.

    Legs the engine answers natively all fold into the default pass. Legs it
    answers only when prompted each get their own pass. Legs it cannot answer
    at all produce no pass — the orchestrator records the skip with the reason
    from the registry.
    """
    spec = ENGINES[engine]
    native: list[str] = []
    # Prompted legs are grouped by the ASK actually sent — the engine-resolved
    # prompt plus response format — not by leg name. Two legs whose resolved
    # ask is identical are one call: deepseek_ocr's markdown and tables legs
    # both send the model card's grounding prompt, and paying for that call
    # twice would double the engine's cost for no extra information.
    prompted: dict[tuple, list[str]] = {}
    for leg in legs:
        how = spec.answers(leg)
        if how == "native":
            native.append(leg)
        elif how == "prompted":
            leg_spec = LEGS[leg]
            resolved = spec.prompt_for(leg, leg_spec.prompt)
            fmt = spec.response_format_for(leg, leg_spec.needs_schema)
            # needs_schema prompts embed the per-document schema and can never
            # legitimately merge with a non-schema leg.
            prompted.setdefault((resolved, fmt, leg_spec.needs_schema), []).append(leg)

    plans: list[PassPlan] = []
    if native:
        # An OCR fine-tune whose card prescribes a task prefix must be sent it
        # explicitly, or the app's own default governs and the registry prompt
        # is a fiction: deepseek_ocr declares "Free OCR." while its app
        # defaults to the grounding markdown prompt, so its text leg was being
        # scored from a markdown conversion. Engines that declare no text
        # prompt keep None — their runner's conversion default is the ask that
        # serves text, markdown and tables together, and forcing the generic
        # text prompt on them would suppress the structure the other legs need.
        text_override = spec.leg_prompts.get(TEXT) if spec.accepts_task else None
        plans.append(PassPlan(
            DEFAULT_PASS, engine, tuple(native), TEXT if text_override else None
        ))
    for (_, _, _), pass_legs in prompted.items():
        name = pass_legs[0] if len(pass_legs) == 1 else "+".join(pass_legs)
        plans.append(PassPlan(name, engine, tuple(pass_legs), pass_legs[0]))
    return plans


def skipped_engines(legs: list[str], engines: list[str]) -> dict[str, list[tuple[str, str]]]:
    """{leg: [(engine, reason), ...]} for engines that cannot answer a leg.

    A blank cell in the report is never left unexplained: the reason comes from
    the registry's leg_notes, which is the same place the eligibility is
    declared.
    """
    out: dict[str, list[tuple[str, str]]] = {}
    for leg in legs:
        rows = []
        for engine in engines:
            spec = ENGINES.get(engine)
            if spec is None or spec.answers(leg):
                continue
            rows.append((engine, spec.leg_notes or "not eligible for this leg"))
        if rows:
            out[leg] = rows
    return out
