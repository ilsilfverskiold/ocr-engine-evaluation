"""Every reference, scored against itself, must be perfect.

This is the guard for the worst class of defect this benchmark has had: a
reference that no engine can match. It does not look like a bug — nothing
errors, every engine returns a number — it looks like a result. The tell is
that the numbers cluster, because a flaw in the shared reference costs every
engine the same.

Two instances were found on 2026-08-09 by running exactly this check:

  * the invoice table references had been generated from a markdown GT that
    used the dataset's schema field names and six of the page's eight columns,
    so six unrelated engines all scored 0.585-0.587 on invoice_000 and the leg
    ranked nothing. Against the rebuilt reference they score 0.92-1.00.
  * `lookup_path` split field paths on every dot, so a financial statement
    whose columns are dates (`values."31.1.04"`) could not be read back at
    all. Those documents capped every engine near 0.41.

Feeding a reference back as though an engine had produced it is the cheapest
way to catch both: if the ground truth cannot score 1.0 against itself, the
number the leg publishes is about us, not the engines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.geometry_leg import score_geometry
from evaluation.ground_truth import GroundTruthLoader
from evaluation.keyvalue_leg import score_key_value
from evaluation.markdown_leg import score_markdown
from evaluation.schema_leg import score_schema
from evaluation.table_leg import score_tables
from evaluation.text_leg import score_text

DOCUMENTS = Path(__file__).parent.parent / "documents"


def _corpus():
    return sorted(DOCUMENTS.glob("*/*/*.png")) + sorted(DOCUMENTS.glob("*/*/*.jpg"))


def _self_scores(leg: str):
    """[(document stem, score)] for every document carrying this leg's GT."""
    loader = GroundTruthLoader()
    out = []
    for path in _corpus():
        gt = loader.load(path)
        if not gt.has_gt_for(leg):
            continue
        if leg == "text":
            result = score_text(gt.plain_text, gt)
        elif leg == "markdown":
            # Deterministic only. Left at use_judge=None this called Gemini
            # for every markdown reference once a key was in the environment
            # (2026-09-30: 57 verdict calls from one pytest run). The
            # headline (`.score`) is now the code verdict against Gemini's
            # image checklist, which the sidecar need not satisfy; this guard
            # is about the deterministic block scorer, so read that.
            result = score_markdown(gt.markdown, gt, use_judge=False)
            if result.applicable:
                out.append((path.stem, result.markdown_score))
            continue
        elif leg == "tables":
            # paddleocr_vl is the HTML-format engine, so the reference HTML
            # travels the same converter an engine's answer would. As for
            # markdown, the guard is the deterministic TEDS, not the
            # checklist headline.
            result = score_tables(gt.table_html, gt, "paddleocr_vl", use_judge=False)
            if result.applicable:
                out.append((path.stem, result.teds))
            continue
        elif leg == "key_value":
            result = score_key_value(json.dumps(gt.true_json), gt, "gemini_flash")
        elif leg == "schema":
            result = score_schema(json.dumps(gt.true_json), gt, "gemini_flash")
        else:
            result = score_geometry(gt.word_boxes, gt)
        # A reference with nothing scorable in it (a chart whose data is an
        # array, not label->value pairs) is a correct "not applicable".
        if result.applicable:
            out.append((path.stem, result.score))
    return out


@pytest.mark.parametrize(
    "leg", [
        "text",
        # Deterministic block F1 scores every markdown reference 1.0 against
        # itself, which is what this guard checks. The JUDGED path is a
        # different metric by design — its reference is Gemini's catalogue of
        # the page image, not this sidecar — so it need not score the sidecar
        # 1.0 (run once by accident with a key present, 2026-09-30: 42 of 57
        # below 1.0). Not a defect; see docs/benchmark-audit-2026-09-30.md §5.14.
        "markdown",
        "tables",
        # key_value and schema are scored by code against the reference's
        # own fields (no verdict cache since 2026-10-01), so the reference
        # must score 1.0 against itself; chart_003 is a correct "not
        # applicable" on key_value.
        "key_value",
        "schema",
        "geometry",
    ]
)
def test_reference_scores_one_against_itself(leg):
    imperfect = [(stem, score) for stem, score in _self_scores(leg)
                 if score is None or score < 0.999]
    assert not imperfect, (
        f"{leg}: {len(imperfect)} reference(s) cannot score 1.0 against "
        f"themselves, so no engine can either — {imperfect[:5]}"
    )


def test_the_corpus_is_actually_being_checked():
    # A loader change that quietly stopped finding ground truth would make
    # every assertion above vacuous.
    assert len(_self_scores("text")) >= 80
    assert len(_self_scores("schema")) >= 60
