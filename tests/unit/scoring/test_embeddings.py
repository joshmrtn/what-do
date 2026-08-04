"""Unit tests for the embedding provider abstraction."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.scoring.embeddings import EmbeddingError, EmbeddingProvider, OllamaEmbeddingProvider
from src.utils.chat_client import LLMError


class _FakeClient:
    """Stands in for OllamaClient — records calls, returns canned vectors."""

    def __init__(self, vector=None, error=None):
        self._vector = vector if vector is not None else [0.1, 0.2, 0.3]
        self._error = error
        self.calls: list[tuple[str, str]] = []

    def embed(self, model: str, text: str) -> list[float]:
        self.calls.append((model, text))
        if self._error is not None:
            raise self._error
        return list(self._vector)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_ollama_provider_satisfies_embedding_provider():

    provider = OllamaEmbeddingProvider(client=_FakeClient())

    assert isinstance(provider, EmbeddingProvider)


def test_any_object_with_embed_satisfies_protocol():
    """The protocol is structural, so a fake substitutes with no inheritance."""

    class Stub:
        def embed(self, text: str) -> list[float]:
            return [1.0, 2.0]

    assert isinstance(Stub(), EmbeddingProvider)


# ---------------------------------------------------------------------------
# OllamaEmbeddingProvider
# ---------------------------------------------------------------------------


def test_embed_returns_vector():

    provider = OllamaEmbeddingProvider(client=_FakeClient(vector=[0.5, 0.6]))

    assert provider.embed("karaoke") == [0.5, 0.6]


def test_embed_passes_configured_model_through():

    client = _FakeClient()
    provider = OllamaEmbeddingProvider(client=client, model="custom-embed-model")

    provider.embed("karaoke")

    assert client.calls == [("custom-embed-model", "karaoke")]


def test_default_model_is_nomic_embed_text():

    client = _FakeClient()
    OllamaEmbeddingProvider(client=client).embed("karaoke")

    assert client.calls[0][0] == "nomic-embed-text"


def test_client_failure_raises_embedding_error():

    provider = OllamaEmbeddingProvider(client=_FakeClient(error=LLMError("refused")))

    with pytest.raises(EmbeddingError, match="refused"):
        provider.embed("karaoke")


def test_embedding_error_is_an_llm_error():
    """Callers already handle LLMError; embedding failures must fit that net."""

    assert issubclass(EmbeddingError, LLMError)


def test_empty_text_raises_without_calling_client():

    client = _FakeClient()
    provider = OllamaEmbeddingProvider(client=client)

    with pytest.raises(EmbeddingError, match="empty"):
        provider.embed("   ")

    assert client.calls == []


def test_empty_vector_response_raises():

    provider = OllamaEmbeddingProvider(client=_FakeClient(vector=[]))

    with pytest.raises(EmbeddingError, match="empty"):
        provider.embed("karaoke")


def test_no_network_call_made_by_provider_itself():
    """The provider must delegate transport, never reach the network directly."""

    client = MagicMock()
    client.embed.return_value = [0.1]
    provider = OllamaEmbeddingProvider(client=client)

    provider.embed("karaoke")

    client.embed.assert_called_once()


# ---------------------------------------------------------------------------
# Case folding — nomic-embed-text has an uncased vocabulary
# ---------------------------------------------------------------------------


def test_text_is_lowercased_before_embedding():
    """Capitalised words hit [UNK]: 'Karaoke', 'Trivia' and 'Death' all embed identically."""

    client = _FakeClient()
    OllamaEmbeddingProvider(client=client).embed("Karaoke Night at Koto")

    assert client.calls == [("nomic-embed-text", "karaoke night at koto")]


def test_case_differences_produce_one_cache_hit():
    """Both sides of a comparison must be folded identically."""

    client = _FakeClient()
    provider = OllamaEmbeddingProvider(client=client)

    assert provider.embed("KARAOKE") == provider.embed("karaoke")


def test_blank_after_folding_still_raises():

    with pytest.raises(EmbeddingError, match="empty"):
        OllamaEmbeddingProvider(client=_FakeClient()).embed("   ")
