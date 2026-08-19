"""Thin client for the Google Gemini API, exposing the ChatClient interface.

Translates the neutral ``[{"role", "content"}]`` message shape used by the
extraction/disambiguation providers into Gemini's ``generate_content`` contents,
so a ``GeminiClient`` is a drop-in for ``OllamaClient`` behind ``ChatClient``.

**A hosted model is a third party like any other**, so every call goes through
the shared request policy — throttled, retried, backed off, timed out from
config. Ollama is exempt because it is *localhost*, never because it is "the
model client": that phrasing survives a provider swap and would silently exempt
a hosted API.

This is the one caller reached through a vendor SDK rather than ``requests``,
which is why the policy wraps a **call** and not a URL. Configuring retry on the
SDK instead was considered and rejected — it would be the same bug in a new
costume, one provider's private notion of politeness with nothing else able to
see it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

import httpx

from src.network.http import api_status_transient_check
from src.network.policy import RequestPolicy
from src.network.protocols import (
    RETRY_WITH_BACKOFF,
    NullCache,
    RetryAdvice,
    TransientCheck,
)
from src.utils.images import sniff_mime_type
from src.utils.chat_client import LLMError
from src.utils.secret import Secret

#: What the policy is assigned to in `network.hosts`. A constant rather than a
#: literal at the call site, the same way `OPEN_METEO_HOST` is, so the config
#: entry and the caller cannot drift apart.
GEMINI_HOST = "generativelanguage.googleapis.com"


class GeminiError(LLMError):
    """Raised when a Gemini API call fails."""


def gemini_transient_check(*, get_now: Callable[[], datetime]) -> TransientCheck:
    """Which of the SDK's failures are worth another attempt.

    Two vocabularies, because the SDK has two. An ``APIError`` carries the status
    the service answered with, which `api_status_transient_check` already judges.
    Underneath, the SDK speaks httpx, and a timeout or a refused connection
    arrives as an httpx error carrying **no status at all** — so a check reading
    only a status would call the most retryable failures there are permanent.

    The SDK's own retry configuration classifies exactly this pair, and it is off
    by default; this is that judgement made once, where every provider's is.

    Args:
        get_now: Injected clock, for a `Retry-After` given as an HTTP date.
    """
    by_status = api_status_transient_check(get_now=get_now)

    def check(error: BaseException) -> RetryAdvice:
        if isinstance(error, (httpx.TimeoutException, httpx.ConnectError)):
            return RETRY_WITH_BACKOFF
        return by_status(error)

    return check


class GeminiClient:
    """Wraps Gemini ``generate_content`` behind the ChatClient interface.

    The model is supplied per call (via ``chat``), mirroring ``OllamaClient`` —
    the client holds only auth/transport state.

    Args:
        api_key: Google AI API key.
        policy: The shared request policy. Required and keyword-only: an
            optional dependency is one that arrives nowhere — the refit was
            built, typed and never forwarded for a fortnight behind a
            ``| None = None``.
        get_now: Injected clock, for reading a `Retry-After` date.
        client: Preconstructed ``google.genai`` client. Injected in tests; when
            omitted a real client is built lazily (requires the ``google-genai``
            package and a valid ``api_key``).
    """

    def __init__(
        self,
        api_key: Secret,
        *,
        policy: RequestPolicy,
        get_now: Callable[[], datetime],
        client: Any = None,
    ) -> None:
        self._policy = policy
        self._is_transient = gemini_transient_check(get_now=get_now)
        if client is not None:
            self._client = client
        else:
            from google import genai

            # No `http_options` timeout here: the policy hands one to each
            # attempt, and a lifetime or a limit with two homes is one that will
            # disagree with itself.
            self._client = genai.Client(api_key=api_key.expose_secret())

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        images: list[bytes] | None = None,
    ) -> str:
        """Send a chat request and return the reply text.

        Args:
            model: Gemini model name (e.g. 'gemini-2.0-flash').
            messages: Chat turns with 'role' ('user'/'assistant') and 'content'.
            images: Optional raw image bytes attached to the last user turn.

        Returns:
            The model's reply as a string.

        Raises:
            GeminiError: On any SDK/network failure or an empty response.
        """
        contents = self._build_contents(messages, images)

        # The wrap is *outside* the policy, and that is the whole of the design.
        # Inside it, the transient check would be handed a `GeminiError` — our
        # own type, carrying no status — and would judge every rate limit and
        # every 503 permanent. Retry would be configured, wired, and dead.
        try:
            return self._policy.call(
                host=GEMINI_HOST,
                perform=lambda timeout: self._generate(model, contents, timeout),
                is_transient=self._is_transient,
                cache=NullCache(
                    reason="a prompt is not a cacheable resource, and extraction "
                    "already skips on extraction_input_hash one layer up — which "
                    "avoids the call rather than replaying its answer"
                ),
                label="gemini",
            )
        except GeminiError:
            raise
        except Exception as exc:
            raise GeminiError(f"Gemini request failed: {exc}") from exc

    def _generate(self, model: str, contents: list[dict[str, Any]], timeout: float) -> str:
        """One attempt, at the timeout this attempt was given.

        Raises the SDK's own error untouched, so the policy's check can read the
        status off it. An empty reply is *not* a transport failure — the model
        answered — so it is raised as ours and the check declines to repeat it.
        """
        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            # Milliseconds, per the SDK. Per call rather than per client because
            # that is where the policy applies it.
            config={"http_options": {"timeout": int(timeout * 1000)}},
        )

        text = getattr(response, "text", None)
        if text is None:
            raise GeminiError("Gemini response contained no text")
        return str(text)

    @staticmethod
    def _build_contents(
        messages: list[dict[str, Any]],
        images: list[bytes] | None,
    ) -> list[dict[str, Any]]:
        """Translate neutral messages into Gemini contents; attach images last."""
        contents: list[dict[str, Any]] = []
        for message in messages:
            role = "model" if message["role"] == "assistant" else "user"
            parts: list[dict[str, Any]] = [{"text": message["content"]}]
            contents.append({"role": role, "parts": parts})

        if images:
            if not contents or contents[-1]["role"] != "user":
                contents.append({"role": "user", "parts": []})
            for image in images:
                contents[-1]["parts"].append(
                    # Sniffed per image, not hardcoded: the fetcher takes
                    # whatever a listing links to, so a PNG sent as JPEG is a
                    # label the model may reject or misread.
                    {
                        "inline_data": {
                            "mime_type": sniff_mime_type(image),
                            "data": image,
                        }
                    }
                )

        return contents
