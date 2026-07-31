"""Unit tests for GeminiClient (hermetic — injected fake genai client).

The ``slow`` tests at the bottom hit the real Gemini API and are skipped unless
``GEMINI_API_KEY`` is available (from the environment or a local ``.env``).
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv


class FakeResponse:
    """Stand-in for a google.genai response object."""

    def __init__(self, text: str | None) -> None:
        self.text = text


def _fake_genai_client(response_text: str | None = "ok", raise_exc: Exception | None = None):
    client = MagicMock()
    if raise_exc is not None:
        client.models.generate_content.side_effect = raise_exc
    else:
        client.models.generate_content.return_value = FakeResponse(response_text)
    return client


def _contents(fake) -> list:
    return fake.models.generate_content.call_args.kwargs["contents"]


def test_chat_returns_response_text():
    from src.utils.gemini_client import GeminiClient

    fake = _fake_genai_client("hello from gemini")
    gc = GeminiClient(api_key="x", client=fake)
    out = gc.chat(model="gemini-2.0-flash", messages=[{"role": "user", "content": "hi"}])
    assert out == "hello from gemini"


def test_chat_passes_model_through():
    from src.utils.gemini_client import GeminiClient

    fake = _fake_genai_client()
    gc = GeminiClient(api_key="x", client=fake)
    gc.chat(model="gemini-2.0-flash", messages=[{"role": "user", "content": "hi"}])
    assert fake.models.generate_content.call_args.kwargs["model"] == "gemini-2.0-flash"


def test_chat_translates_roles():
    from src.utils.gemini_client import GeminiClient

    fake = _fake_genai_client()
    gc = GeminiClient(api_key="x", client=fake)
    gc.chat(
        model="m",
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ],
    )
    contents = _contents(fake)
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[0]["parts"][0]["text"] == "first"


def test_chat_attaches_images_to_last_user_message():
    from src.utils.gemini_client import GeminiClient

    fake = _fake_genai_client()
    gc = GeminiClient(api_key="x", client=fake)
    gc.chat(
        model="m",
        messages=[{"role": "user", "content": "describe"}],
        images=[b"\x89PNGdata"],
    )
    parts = _contents(fake)[-1]["parts"]
    assert any("text" in p for p in parts)
    inline = [p for p in parts if "inline_data" in p]
    assert len(inline) == 1
    assert inline[0]["inline_data"]["data"] == b"\x89PNGdata"


def test_chat_without_images_sends_no_inline_data():
    from src.utils.gemini_client import GeminiClient

    fake = _fake_genai_client()
    gc = GeminiClient(api_key="x", client=fake)
    gc.chat(model="m", messages=[{"role": "user", "content": "hi"}])
    parts = _contents(fake)[-1]["parts"]
    assert all("inline_data" not in p for p in parts)


def test_chat_wraps_sdk_error_as_gemini_error():
    from src.utils.gemini_client import GeminiClient, GeminiError

    fake = _fake_genai_client(raise_exc=RuntimeError("boom"))
    gc = GeminiClient(api_key="x", client=fake)
    with pytest.raises(GeminiError):
        gc.chat(model="m", messages=[{"role": "user", "content": "hi"}])


def test_none_text_response_raises_gemini_error():
    from src.utils.gemini_client import GeminiClient, GeminiError

    fake = _fake_genai_client(response_text=None)
    gc = GeminiClient(api_key="x", client=fake)
    with pytest.raises(GeminiError):
        gc.chat(model="m", messages=[{"role": "user", "content": "hi"}])


def test_gemini_error_is_llm_error():
    from src.utils.chat_client import LLMError
    from src.utils.gemini_client import GeminiError

    assert issubclass(GeminiError, LLMError)


def test_ollama_error_is_llm_error():
    from src.utils.chat_client import LLMError
    from src.utils.ollama_client import OllamaError

    assert issubclass(OllamaError, LLMError)


def test_gemini_client_satisfies_chat_client_protocol():
    from src.utils.chat_client import ChatClient
    from src.utils.gemini_client import GeminiClient

    gc = GeminiClient(api_key="x", client=MagicMock())
    assert isinstance(gc, ChatClient)


# --- live tests (real Gemini API; require GEMINI_API_KEY) ---

_SAMPLE_CAPTION = (
    "🎵 Live jazz with the Salem Jazz Collective this Saturday at The Vault Lounge! "
    "Doors open at 7pm, music starts at 8pm. $15 cover. "
    "Great cocktails, cozy atmosphere, perfect for date night. "
    "Follow @salemsjazcollective for updates!"
)


def _require_gemini() -> tuple[str, str]:
    """Load the key/model or skip the test if no key is configured."""
    load_dotenv()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY not set")
    return key, os.environ.get("GEMINI_MODEL", "gemini-flash-latest")


@pytest.mark.slow
def test_real_gemini_extraction():
    """Real Gemini extraction produces a structurally valid result."""
    from src.processing.extraction import OllamaExtractionProvider
    from src.utils.gemini_client import GeminiClient

    key, model = _require_gemini()
    provider = OllamaExtractionProvider(client=GeminiClient(api_key=key), model=model, min_tags=5)

    result = provider.extract(_SAMPLE_CAPTION)

    assert len(result.tags) >= 5
    assert result.summary is not None and len(result.summary) > 0


@pytest.mark.slow
def test_real_gemini_disambiguation():
    """Real Gemini classifies an obvious venue handle as 'venue'."""
    from src.ingestion.disambiguation import OllamaDisambiguationProvider
    from src.utils.gemini_client import GeminiClient

    key, model = _require_gemini()
    provider = OllamaDisambiguationProvider(client=GeminiClient(api_key=key), model=model)

    result = provider.classify(
        handle="@thevaultlounge",
        context="Come enjoy live jazz at @thevaultlounge this Saturday — doors open at 7pm!",
    )
    assert result == "venue"
