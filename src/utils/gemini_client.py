"""Thin client for the Google Gemini API, exposing the ChatClient interface.

Translates the neutral ``[{"role", "content"}]`` message shape used by the
extraction/disambiguation providers into Gemini's ``generate_content`` contents,
so a ``GeminiClient`` is a drop-in for ``OllamaClient`` behind ``ChatClient``.
"""

from __future__ import annotations

from typing import Any

from src.utils.chat_client import LLMError


class GeminiError(LLMError):
    """Raised when a Gemini API call fails."""


class GeminiClient:
    """Wraps Gemini ``generate_content`` behind the ChatClient interface.

    The model is supplied per call (via ``chat``), mirroring ``OllamaClient`` —
    the client holds only auth/transport state.

    Args:
        api_key: Google AI API key.
        timeout: Request timeout in seconds.
        client: Preconstructed ``google.genai`` client. Injected in tests; when
            omitted a real client is built lazily (requires the ``google-genai``
            package and a valid ``api_key``).
    """

    def __init__(self, api_key: str, timeout: int = 60, *, client: Any = None) -> None:
        self._timeout = timeout
        if client is not None:
            self._client = client
        else:
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=timeout * 1000),
            )

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
        try:
            response = self._client.models.generate_content(model=model, contents=contents)
        except Exception as exc:
            raise GeminiError(f"Gemini request failed: {exc}") from exc

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
                    {"inline_data": {"mime_type": "image/jpeg", "data": image}}
                )

        return contents
