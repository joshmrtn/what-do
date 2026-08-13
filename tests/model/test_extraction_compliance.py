"""Does a real model comply with the extraction prompt?

Grouped by prompt rather than by model, so the same caption goes to every
candidate. That is the shape the bench in #2 needs: one input, many models,
compared side by side.

The event and logger factories are local copies rather than imports from the
unit suite. These tests leave with #2 and should not drag a unit test module
with them.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
import io
import os

from dotenv import load_dotenv
import pytest

from src.models.event import Event
from src.processing.extraction import OllamaExtractionProvider
from src.processing.extraction_stage import ExtractionStage
from src.utils.gemini_client import GeminiClient
from src.utils.logging import get_logger
from src.utils.ollama_client import OllamaClient

SAMPLE_CAPTION = (
    "🎵 Live jazz with the Salem Jazz Collective this Saturday at The Vault Lounge! "
    "Doors open at 7pm, music starts at 8pm. $15 cover. "
    "Great cocktails, cozy atmosphere, perfect for date night. "
    "Follow @salemsjazcollective for updates!"
)


def _make_event(**kwargs: Any) -> Event:
    now = datetime(2026, 6, 22, 12, 0, 0, tzinfo=timezone.utc)
    defaults: dict[str, Any] = dict(
        event_id="evt-1",
        source_event_candidates=["cand-1"],
        source_type="apify",
        created_at=now,
        updated_at=now,
        title="Live Jazz Night",
        description="Come enjoy live jazz at the waterfront venue this Saturday.",
    )
    defaults.update(kwargs)
    return Event(**defaults)


def _make_logger():
    return get_logger("model_compliance", stream=io.StringIO())


def _require_gemini() -> tuple[str, str]:
    """Load the key/model or skip the test if no key is configured."""
    load_dotenv()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY not set")
    return key, os.environ.get("GEMINI_MODEL", "gemini-flash-latest")


@pytest.mark.model
def test_real_extraction_produces_valid_result():
    """Confirm real Ollama extraction works end-to-end with gemma4:e4b."""

    client = OllamaClient(host="http://localhost:11434", timeout=3600)
    provider = OllamaExtractionProvider(client=client, model="gemma4:e4b", min_tags=5)
    stage = ExtractionStage(provider=provider, image_fetcher=None, logger=_make_logger())

    event = _make_event(title=None, description=SAMPLE_CAPTION)
    results = stage.process([event])

    assert len(results) == 1
    result = results[0]
    assert result.extraction_degradation is None, (
        f"Extraction fell short: {result.extraction_degradation}"
    )
    assert len(result.tags) >= 5
    assert result.summary is not None and len(result.summary) > 0


@pytest.mark.model
@pytest.mark.external
def test_real_gemini_extraction():
    """Real Gemini extraction produces a structurally valid result."""

    key, model = _require_gemini()
    provider = OllamaExtractionProvider(client=GeminiClient(api_key=key), model=model, min_tags=5)

    result = provider.extract(SAMPLE_CAPTION)

    assert len(result.tags) >= 5
    assert result.summary is not None and len(result.summary) > 0


@pytest.mark.model
@pytest.mark.external
def test_real_gemini_resolves_relative_date():
    """Gemini resolves 'this Saturday' against an injected reference date."""

    key, model = _require_gemini()
    provider = OllamaExtractionProvider(client=GeminiClient(api_key=key), model=model, min_tags=5)

    result = provider.extract(
        "Live music this Saturday at 8pm at The Vault Lounge in Salem.",
        reference_date=datetime(2026, 8, 3),  # a Monday
    )

    assert result.start_time is not None
    assert result.start_time.date() == date(2026, 8, 8)
