"""Embedding generation behind a swappable provider interface.

Preference lines, event tags, and event summaries all become vectors through the
same one-method interface, so the embedding backend can be replaced without
touching any caller.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.utils.chat_client import LLMError


class EmbeddingError(LLMError):
    """Raised when an embedding cannot be generated.

    Inherits from LLMError so callers already catching model-provider failures
    handle embedding failures with the same except clause.
    """


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into a vector."""

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for the given text."""
        ...


class EmbeddingClient(Protocol):
    """Transport capable of generating embeddings (e.g. OllamaClient)."""

    def embed(self, model: str, text: str) -> list[float]:
        """Return an embedding for text using the named model."""
        ...


class OllamaEmbeddingProvider:
    """Generates embeddings via an Ollama-compatible client.

    Args:
        client: Transport exposing embed(model, text). Injected so tests
            substitute a fake and never reach the network.
        model: Embedding model name.
    """

    def __init__(
        self, client: EmbeddingClient, model: str = "nomic-embed-text"
    ) -> None:
        self._client = client
        self._model = model

    def embed(self, text: str) -> list[float]:
        """Embed a single piece of text.

        Args:
            text: Text to embed. Must be non-blank — a blank string carries no
                signal and would silently pollute similarity scores.

        Returns:
            The embedding vector.

        Raises:
            EmbeddingError: If text is blank, the provider fails, or the
                provider returns an empty vector.
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        try:
            vector = self._client.embed(model=self._model, text=text)
        except LLMError as exc:
            raise EmbeddingError(
                f"Embedding failed for model {self._model!r}: {exc}"
            ) from exc

        if not vector:
            raise EmbeddingError(
                f"Model {self._model!r} returned an empty vector for {text!r}"
            )

        return vector
