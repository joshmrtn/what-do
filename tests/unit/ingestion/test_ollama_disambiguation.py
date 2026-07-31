"""Unit tests for OllamaDisambiguationProvider."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _make_client(response_text: str):
    client = MagicMock()
    client.chat.return_value = response_text
    return client


# ---------------------------------------------------------------------------
# OllamaDisambiguationProvider
# ---------------------------------------------------------------------------


def test_classifies_venue():
    from src.ingestion.disambiguation import OllamaDisambiguationProvider

    client = _make_client('{"classification": "venue"}')
    provider = OllamaDisambiguationProvider(client=client, model="gemma4:e2b")

    result = provider.classify(handle="@thevaultlounge", context="Great night at @thevaultlounge!")
    assert result == "venue"


def test_classifies_person():
    from src.ingestion.disambiguation import OllamaDisambiguationProvider

    client = _make_client('{"classification": "person"}')
    provider = OllamaDisambiguationProvider(client=client, model="gemma4:e2b")

    result = provider.classify(handle="@johndoe", context="Thanks to @johndoe for performing!")
    assert result == "person"


def test_classify_called_with_handle_and_context():
    from src.ingestion.disambiguation import OllamaDisambiguationProvider

    client = _make_client('{"classification": "venue"}')
    provider = OllamaDisambiguationProvider(client=client, model="gemma4:e2b")

    provider.classify(handle="@someplace", context="visit @someplace tonight")

    client.chat.assert_called_once()
    call_kwargs = client.chat.call_args
    messages = call_kwargs[1]["messages"] if "messages" in call_kwargs[1] else call_kwargs[0][1]
    combined = " ".join(str(m) for m in messages)
    assert "@someplace" in combined
    assert "visit @someplace tonight" in combined


def test_retry_on_malformed_json_succeeds_second_try():
    from src.ingestion.disambiguation import OllamaDisambiguationProvider

    client = MagicMock()
    client.chat.side_effect = [
        "here is my answer: venue",  # first call: prose, not JSON
        '{"classification": "venue"}',  # second call: valid
    ]
    provider = OllamaDisambiguationProvider(client=client, model="gemma4:e2b")

    result = provider.classify(handle="@thespot", context="live music at @thespot")
    assert result == "venue"
    assert client.chat.call_count == 2


def test_raises_after_two_failures():
    from src.ingestion.disambiguation import DisambiguationError, OllamaDisambiguationProvider

    client = _make_client("I cannot determine this from the context provided.")
    provider = OllamaDisambiguationProvider(client=client, model="gemma4:e2b")

    with pytest.raises(DisambiguationError):
        provider.classify(handle="@mystery", context="some text mentioning @mystery")

    assert client.chat.call_count == 2


def test_raises_on_unknown_classification_value():
    from src.ingestion.disambiguation import DisambiguationError, OllamaDisambiguationProvider

    client = MagicMock()
    client.chat.side_effect = [
        '{"classification": "unknown"}',
        '{"classification": "maybe"}',
    ]
    provider = OllamaDisambiguationProvider(client=client, model="gemma4:e2b")

    with pytest.raises(DisambiguationError):
        provider.classify(handle="@x", context="context")


@pytest.mark.slow
def test_real_ollama_classifies_venue_handle():
    """Confirm real Ollama can classify an obvious venue handle."""
    from src.utils.ollama_client import OllamaClient
    from src.ingestion.disambiguation import OllamaDisambiguationProvider

    client = OllamaClient(host="http://localhost:11434", timeout=3600)
    provider = OllamaDisambiguationProvider(client=client, model="gemma4:e2b")

    result = provider.classify(
        handle="@thevaultlounge",
        context="Come enjoy live jazz at @thevaultlounge this Saturday — doors open at 7pm!",
    )
    assert result == "venue"
