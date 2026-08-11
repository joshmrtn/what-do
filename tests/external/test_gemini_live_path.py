"""A real call to the real Gemini API returns something our provider can read.

Deliberately the cheapest claim available — one short disambiguation, about two
seconds. It is not here to judge Gemini's answers; `tests/model/` does that.
It is here so that a change to `GeminiClient` can be checked against the live
API without inventing a new harness, and it stays small so that checking is
never expensive enough to skip.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
import pytest

from src.ingestion.disambiguation import OllamaDisambiguationProvider
from src.utils.gemini_client import GeminiClient


def _require_gemini() -> tuple[str, str]:
    """Load the key/model or skip the test if no key is configured."""
    load_dotenv()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY not set")
    return key, os.environ.get("GEMINI_MODEL", "gemini-flash-latest")


@pytest.mark.external
def test_real_gemini_disambiguation():
    """Real Gemini classifies an obvious venue handle as 'venue'."""

    key, model = _require_gemini()
    provider = OllamaDisambiguationProvider(client=GeminiClient(api_key=key), model=model)

    result = provider.classify(
        handle="@thevaultlounge",
        context="Come enjoy live jazz at @thevaultlounge this Saturday — doors open at 7pm!",
    )
    assert result == "venue"
