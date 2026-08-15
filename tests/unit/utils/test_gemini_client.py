"""Unit tests for GeminiClient — hermetic, against an injected fake genai client.

These cover the translation layer: role mapping, image attachment, error
wrapping, protocol conformance. Whether Gemini itself answers well is a
question for ``tests/model/``; whether the live API is reachable is one for
``tests/external/``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.utils.chat_client import ChatClient, LLMError
from src.utils.gemini_client import GeminiClient, GeminiError
from src.utils.ollama_client import OllamaError


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

    fake = _fake_genai_client("hello from gemini")
    gc = GeminiClient(api_key="x", client=fake)
    out = gc.chat(model="gemini-2.0-flash", messages=[{"role": "user", "content": "hi"}])
    assert out == "hello from gemini"


def test_chat_passes_model_through():

    fake = _fake_genai_client()
    gc = GeminiClient(api_key="x", client=fake)
    gc.chat(model="gemini-2.0-flash", messages=[{"role": "user", "content": "hi"}])
    assert fake.models.generate_content.call_args.kwargs["model"] == "gemini-2.0-flash"


def test_chat_translates_roles():

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

    fake = _fake_genai_client()
    gc = GeminiClient(api_key="x", client=fake)
    gc.chat(model="m", messages=[{"role": "user", "content": "hi"}])
    parts = _contents(fake)[-1]["parts"]
    assert all("inline_data" not in p for p in parts)


def test_chat_wraps_sdk_error_as_gemini_error():

    fake = _fake_genai_client(raise_exc=RuntimeError("boom"))
    gc = GeminiClient(api_key="x", client=fake)
    with pytest.raises(GeminiError):
        gc.chat(model="m", messages=[{"role": "user", "content": "hi"}])


def test_none_text_response_raises_gemini_error():

    fake = _fake_genai_client(response_text=None)
    gc = GeminiClient(api_key="x", client=fake)
    with pytest.raises(GeminiError):
        gc.chat(model="m", messages=[{"role": "user", "content": "hi"}])


def test_gemini_error_is_llm_error():

    assert issubclass(GeminiError, LLMError)


def test_ollama_error_is_llm_error():

    assert issubclass(OllamaError, LLMError)


def test_gemini_client_satisfies_chat_client_protocol():

    gc = GeminiClient(api_key="x", client=MagicMock())
    assert isinstance(gc, ChatClient)


class TestImageMimeType:
    """Every image was labelled `image/jpeg` regardless of what it actually was.

    `HttpImageFetcher` fetches whatever the listing links to, so a PNG or WebP
    was being handed to the model under a JPEG label — which Gemini may reject
    or misread. The path is live: `ExtractionStage` fetches `image_url` bytes and
    forwards them here (#7).
    """

    def _parts(self, client, image: bytes):
        contents = client._build_contents([{"role": "user", "content": "hi"}], [image])
        return [p for c in contents for p in c["parts"] if "inline_data" in p]

    def test_a_png_is_labelled_png(self):
        client = GeminiClient(api_key="k", client=object())

        parts = self._parts(client, b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

        assert parts[0]["inline_data"]["mime_type"] == "image/png"

    def test_a_jpeg_is_still_labelled_jpeg(self):
        client = GeminiClient(api_key="k", client=object())

        parts = self._parts(client, b"\xff\xd8\xff\xe0" + b"\x00" * 8)

        assert parts[0]["inline_data"]["mime_type"] == "image/jpeg"

    def test_a_webp_is_labelled_webp(self):
        client = GeminiClient(api_key="k", client=object())

        parts = self._parts(client, b"RIFF\x24\x00\x00\x00WEBP" + b"\x00" * 8)

        assert parts[0]["inline_data"]["mime_type"] == "image/webp"

    def test_each_image_is_labelled_for_itself(self):
        """A mixed batch is the case a single hardcoded label cannot serve."""
        client = GeminiClient(api_key="k", client=object())
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
        jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 8

        contents = client._build_contents([{"role": "user", "content": "hi"}], [png, jpeg])
        mimes = [
            p["inline_data"]["mime_type"]
            for c in contents
            for p in c["parts"]
            if "inline_data" in p
        ]

        assert mimes == ["image/png", "image/jpeg"]

    def test_the_bytes_are_forwarded_unchanged(self):
        client = GeminiClient(api_key="k", client=object())
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8

        parts = self._parts(client, png)

        assert parts[0]["inline_data"]["data"] == png
