"""Unit tests for the ChatClient structural protocol."""

from __future__ import annotations

from datetime import datetime, timezone

import requests

from src.ingestion.disambiguation import OllamaDisambiguationProvider
from src.models.tag import Tag
from src.processing.extraction import OllamaExtractionProvider
from src.utils.chat_client import ChatClient
from src.utils.ollama_client import OllamaClient

from tests.support.network import fetcher_policy

_HOST = "http://localhost:11434"
_NOW = datetime(2026, 8, 20, 2, 0, tzinfo=timezone.utc)


def test_ollama_client_satisfies_chat_client_protocol():

    client = OllamaClient(
        _HOST,
        session=requests.Session(),
        policy=fetcher_policy(urls=_HOST, now=_NOW),
        get_now=lambda: _NOW,
    )
    assert isinstance(client, ChatClient)


def test_object_with_chat_method_satisfies_protocol():

    class Stub:
        def chat(self, model, messages, images=None):
            return "ok"

    assert isinstance(Stub(), ChatClient)


def test_object_without_chat_is_not_a_chat_client():

    class NotAClient:
        pass

    assert not isinstance(NotAClient(), ChatClient)


def test_extraction_provider_accepts_any_chat_client():

    class Stub:
        def chat(self, model, messages, images=None):
            return '{"tags": ["a", "b", "c", "d", "e"], "summary": "a summary"}'

    provider = OllamaExtractionProvider(client=Stub(), min_tags=5)
    result = provider.extract("some event text")
    assert result.tags == [Tag(text=c) for c in "abcde"]


def test_disambiguation_provider_accepts_any_chat_client():

    class Stub:
        def chat(self, model, messages, images=None):
            return '{"classification": "venue"}'

    provider = OllamaDisambiguationProvider(client=Stub())
    assert provider.classify(handle="@place", context="live music at @place") == "venue"
