"""Does a real model comply with the disambiguation prompt?"""

from __future__ import annotations

import pytest

from src.ingestion.disambiguation import OllamaDisambiguationProvider
from src.utils.ollama_client import OllamaClient


@pytest.mark.model
def test_real_ollama_classifies_venue_handle():
    """Confirm real Ollama can classify an obvious venue handle."""

    client = OllamaClient(host="http://localhost:11434", timeout=3600)
    provider = OllamaDisambiguationProvider(client=client, model="gemma4:e2b")

    result = provider.classify(
        handle="@thevaultlounge",
        context="Come enjoy live jazz at @thevaultlounge this Saturday — doors open at 7pm!",
    )
    assert result == "venue"
