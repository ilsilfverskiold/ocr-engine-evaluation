"""Judge invariants that do not need a model call.

The checklist cap exists because report_001 — a page of dense numeric tables —
produced 568 checkable items, which cost $0.24 and 214 seconds to judge for a
single engine. Fourteen engines would have paid that fourteen times, and the
document would have dominated the leg for being dense rather than for being
hard. Sampling to 40 items measured the same thing (0.100 against 0.114 on the
full list) for 1/36th the cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.judge import MAX_CHECKLIST_ITEMS, ChecklistItem, _sample_checklist


def _items(n: int) -> list[ChecklistItem]:
    return [ChecklistItem(label=f"l{i}", value=f"v{i}", confident=True) for i in range(n)]


def test_short_checklists_are_untouched():
    items = _items(29)
    assert _sample_checklist(items) == items


def test_long_checklists_are_capped():
    assert len(_sample_checklist(_items(568))) == MAX_CHECKLIST_ITEMS


def test_sampling_is_deterministic_so_every_engine_faces_one_list():
    # If two engines were judged against different samples, their scores would
    # not be comparable — the whole point of caching the checklist.
    assert _sample_checklist(_items(568)) == _sample_checklist(_items(568))


def test_sampling_spans_the_whole_document_not_just_the_top():
    sampled = _sample_checklist(_items(568))
    positions = [int(item.label[1:]) for item in sampled]
    assert positions[0] == 0
    assert positions[-1] > 500          # the tail of the page is represented
    assert positions == sorted(positions)


def test_sampling_preserves_item_identity():
    sampled = _sample_checklist(_items(100))
    assert all(isinstance(item, ChecklistItem) for item in sampled)
    assert len({item.label for item in sampled}) == len(sampled)
