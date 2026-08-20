"""Structural protocol for chat-capable LLM clients.

Any client exposing a compatible ``chat()`` method satisfies this protocol, so
extraction and disambiguation providers can be driven by any backend (Ollama,
Gemini, ...) without depending on a concrete client class.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

#: How long a *generation* is worth waiting for, named in `network.patience`.
#:
#: Declared here rather than in either client because it describes the ask and
#: not the answerer: a model producing tokens takes minutes whoever runs it,
#: where an embedding from the same host takes milliseconds. Any `ChatClient`
#: names this, so moving extraction to another provider changes which host's
#: spacing applies and nothing about how long we are willing to wait.
GENERATION_PATIENCE = "generation"


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
