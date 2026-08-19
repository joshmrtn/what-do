"""Unit tests for GeminiClient — hermetic, against an injected fake genai client.

These cover the translation layer and the politeness around it: role mapping,
image attachment, error wrapping, which SDK failures are worth another attempt,
protocol conformance. Whether Gemini itself answers well is the bench's question
(#2); whether the live API is reachable is ``tests/external/``'s.

Only the **SDK client** is a double. The policy, the throttle and the transient
check are the real objects, so these exercise the real retry and the real
backoff — with the clock and the sleep injected, so no test waits.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import httpx
import pytest
from google.genai.errors import APIError

from src.utils.chat_client import ChatClient, LLMError
from src.utils.gemini_client import (
    GEMINI_HOST,
    GeminiClient,
    GeminiError,
    gemini_transient_check,
)
from src.utils.ollama_client import OllamaError
from src.utils.secret import Secret
from tests.support.network import fetcher_policy

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)
MESSAGES = [{"role": "user", "content": "hi"}]


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


def _client(
    fake,
    *,
    max_attempts: int = 3,
    sleeps: list[float] | None = None,
    timeout_seconds: float = 30.0,
) -> GeminiClient:
    """A real policy around a faked SDK client.

    The host is assigned from `GEMINI_HOST` itself rather than spelled again
    here: a client that asked for any other host would find no policy and be
    refused, which is the assignment being tested rather than described.
    """
    policy = fetcher_policy(
        urls=f"https://{GEMINI_HOST}",
        now=NOW,
        max_attempts=max_attempts,
        timeout_seconds=timeout_seconds,
        sleeps=sleeps,
    )
    return GeminiClient(api_key=Secret("x"), client=fake, policy=policy, get_now=lambda: NOW)


def _api_error(status: int) -> APIError:
    return APIError(status, {"error": {"message": "boom", "status": "X"}})


def _contents(fake) -> list:
    return fake.models.generate_content.call_args.kwargs["contents"]


def test_chat_returns_response_text():

    fake = _fake_genai_client("hello from gemini")
    gc = _client(fake)
    out = gc.chat(model="gemini-2.0-flash", messages=[{"role": "user", "content": "hi"}])
    assert out == "hello from gemini"


def test_chat_passes_model_through():

    fake = _fake_genai_client()
    gc = _client(fake)
    gc.chat(model="gemini-2.0-flash", messages=[{"role": "user", "content": "hi"}])
    assert fake.models.generate_content.call_args.kwargs["model"] == "gemini-2.0-flash"


def test_chat_translates_roles():

    fake = _fake_genai_client()
    gc = _client(fake)
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
    gc = _client(fake)
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
    gc = _client(fake)
    gc.chat(model="m", messages=[{"role": "user", "content": "hi"}])
    parts = _contents(fake)[-1]["parts"]
    assert all("inline_data" not in p for p in parts)


def test_chat_wraps_sdk_error_as_gemini_error():

    fake = _fake_genai_client(raise_exc=RuntimeError("boom"))
    gc = _client(fake)
    with pytest.raises(GeminiError):
        gc.chat(model="m", messages=[{"role": "user", "content": "hi"}])


def test_none_text_response_raises_gemini_error():

    fake = _fake_genai_client(response_text=None)
    gc = _client(fake)
    with pytest.raises(GeminiError):
        gc.chat(model="m", messages=[{"role": "user", "content": "hi"}])


def test_gemini_error_is_llm_error():

    assert issubclass(GeminiError, LLMError)


def test_ollama_error_is_llm_error():

    assert issubclass(OllamaError, LLMError)


def test_gemini_client_satisfies_chat_client_protocol():

    assert isinstance(_client(MagicMock()), ChatClient)


class TestWhichSdkFailuresAreWorthRepeating:
    """A hosted model is a third party like any other, and it fails like one."""

    def test_a_rate_limit_is_worth_another_attempt(self):
        assert gemini_transient_check(get_now=lambda: NOW)(_api_error(429)).retry

    def test_a_server_failure_is_worth_another_attempt(self):
        assert gemini_transient_check(get_now=lambda: NOW)(_api_error(503)).retry

    def test_a_rejected_prompt_is_not_asked_again(self):
        """A 400 fails identically however politely it is repeated."""
        assert not gemini_transient_check(get_now=lambda: NOW)(_api_error(400)).retry

    @pytest.mark.parametrize(
        "error",
        [httpx.TimeoutException("too slow"), httpx.ConnectError("refused")],
        ids=["timeout", "connect"],
    )
    def test_a_transport_failure_carries_no_status_and_is_still_transient(self, error):
        """The SDK speaks httpx underneath, and these are the commonest failures
        it has. They carry no status, so a check reading only `.code` would call
        the most retryable thing there is permanent."""
        advice = gemini_transient_check(get_now=lambda: NOW)(error)
        assert advice.retry
        assert advice.retry_after_seconds is None

    def test_our_own_error_is_not_a_reason_to_ask_again(self):
        assert not gemini_transient_check(get_now=lambda: NOW)(
            GeminiError("response contained no text")
        ).retry


class TestPoliteness:
    """The SDK call goes through the shared policy, like every other caller."""

    def test_a_transient_failure_is_tried_again(self):
        """The load-bearing one: the check must see the SDK's error.

        `chat` converts failures into `GeminiError` for its callers, and that
        conversion has to happen *outside* the policy. Inside, the check would be
        handed our own wrapper, find no status on it, and call every transient
        failure permanent — retry configured, wired, and dead.
        """
        fake = _fake_genai_client()
        fake.models.generate_content.side_effect = [_api_error(503), FakeResponse("ok")]

        assert _client(fake).chat(model="m", messages=MESSAGES) == "ok"
        assert fake.models.generate_content.call_count == 2

    def test_the_servers_own_wait_is_honoured(self):
        """It named a number, so we use its number rather than our schedule."""
        response = httpx.Response(429, headers={"Retry-After": "30"})
        error = APIError(429, {"error": {"message": "slow down", "status": "X"}}, response)
        fake = _fake_genai_client()
        fake.models.generate_content.side_effect = [error, FakeResponse("ok")]
        sleeps: list[float] = []

        _client(fake, sleeps=sleeps).chat(model="m", messages=MESSAGES)

        assert sleeps == [pytest.approx(30.0)]

    def test_a_rejected_prompt_is_asked_exactly_once(self):
        fake = _fake_genai_client(raise_exc=_api_error(400))

        with pytest.raises(GeminiError):
            _client(fake).chat(model="m", messages=MESSAGES)

        assert fake.models.generate_content.call_count == 1

    def test_it_stops_at_the_configured_number_of_attempts(self):
        fake = _fake_genai_client(raise_exc=_api_error(503))

        with pytest.raises(GeminiError):
            _client(fake, max_attempts=2).chat(model="m", messages=MESSAGES)

        assert fake.models.generate_content.call_count == 2

    def test_the_sdk_error_survives_as_the_cause(self):
        """Wrapping is for the caller's benefit; it must not lose what happened."""
        original = _api_error(500)
        fake = _fake_genai_client(raise_exc=original)

        with pytest.raises(GeminiError) as caught:
            _client(fake).chat(model="m", messages=MESSAGES)

        assert caught.value.__cause__ is original

    def test_the_configured_timeout_reaches_the_sdk(self):
        """The timeout has one home — the policy — and it is per call, because
        that is where the policy applies it."""
        fake = _fake_genai_client()

        _client(fake, timeout_seconds=12.5).chat(model="m", messages=MESSAGES)

        config = fake.models.generate_content.call_args.kwargs["config"]
        assert config["http_options"]["timeout"] == 12500

    def test_an_empty_reply_is_not_asked_again(self):
        """A model that answered with nothing answered; it is not a transport
        failure, and repeating the prompt spends the allowance on a refusal."""
        fake = _fake_genai_client(response_text=None)

        with pytest.raises(GeminiError):
            _client(fake).chat(model="m", messages=MESSAGES)

        assert fake.models.generate_content.call_count == 1

    def test_a_second_identical_prompt_is_really_asked(self):
        """`NullCache` is a decision: extraction already skips on
        `extraction_input_hash` one layer up, which avoids the call rather than
        replaying an answer. A prompt cache here would serve a stale reply to a
        question somebody meant to ask again."""
        fake = _fake_genai_client("first")

        client = _client(fake)
        client.chat(model="m", messages=MESSAGES)
        client.chat(model="m", messages=MESSAGES)

        assert fake.models.generate_content.call_count == 2

    def test_a_client_cannot_be_built_without_a_policy(self):
        """Not optional: `| None = None` is how the refit was built, typed, and
        then never forwarded for a fortnight."""
        missing_the_policy = {"api_key": Secret("x"), "client": _fake_genai_client()}
        with pytest.raises(TypeError):
            GeminiClient(**missing_the_policy)


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
        client = _client(object())

        parts = self._parts(client, b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

        assert parts[0]["inline_data"]["mime_type"] == "image/png"

    def test_a_jpeg_is_still_labelled_jpeg(self):
        client = _client(object())

        parts = self._parts(client, b"\xff\xd8\xff\xe0" + b"\x00" * 8)

        assert parts[0]["inline_data"]["mime_type"] == "image/jpeg"

    def test_a_webp_is_labelled_webp(self):
        client = _client(object())

        parts = self._parts(client, b"RIFF\x24\x00\x00\x00WEBP" + b"\x00" * 8)

        assert parts[0]["inline_data"]["mime_type"] == "image/webp"

    def test_each_image_is_labelled_for_itself(self):
        """A mixed batch is the case a single hardcoded label cannot serve."""
        client = _client(object())
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
        client = _client(object())
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8

        parts = self._parts(client, png)

        assert parts[0]["inline_data"]["data"] == png
