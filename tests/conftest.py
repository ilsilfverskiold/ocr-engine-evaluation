"""Test-session guard: no test may reach a paid judge or engine.

`run_benchmark` calls `load_dotenv()` at import, so any test importing it
puts the real GOOGLE_API_KEY into the environment — and every judge-backed
scorer treats a present key as permission to call Gemini when `use_judge` is
left at None. On 2026-09-30 the markdown self-consistency test did exactly
that: 57 verdict calls, cached under engine name "". Stripping the keys here
makes the judge client fail to construct, so a test that would have paid
now fails loudly instead.
"""

from __future__ import annotations

import os

import pytest

_PAID_KEYS = (
    "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "MISTRAL_API_KEY",
    "LLAMA_CLOUD_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
    "MATHPIX_APP_ID", "MATHPIX_APP_KEY", "DATALAB_API_KEY",
    "GOOGLE_DOCAI_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS",
)


@pytest.fixture(autouse=True)
def _no_paid_credentials(monkeypatch):
    # Imports may already have run load_dotenv(); scrub before each test too.
    for key in _PAID_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield


def pytest_sessionstart(session):
    for key in _PAID_KEYS:
        os.environ.pop(key, None)
