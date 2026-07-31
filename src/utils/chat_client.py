"""Structural protocol for chat-capable LLM clients.

Any client exposing a compatible ``chat()`` method satisfies this protocol, so
extraction and disambiguation providers can be driven by any backend (Ollama,
Gemini, ...) without depending on a concrete client class.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class LLMError(Exception):
    """Base error for any chat/LLM provider failure.

    Provider-specific client errors (``OllamaError``, ``GeminiError``, ...) inherit
    from this so callers can handle any backend with a single ``except LLMError``.
    """


@runtime_checkable
class ChatClient(Protocol):
    """A client that can send a chat request and return the reply text."""

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        images: list[bytes] | None = None,
    ) -> str:
        """Send a chat request and return the assistant's reply as a string."""
        ...
