"""Embedding generation behind a swappable provider interface.

Preference lines, event tags, and event summaries all become vectors through the
same one-method interface, so the embedding backend can be replaced without
touching any caller.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.config import DEFAULT_EMBEDDING_MODEL
from src.utils.chat_client import LLMError


class EmbeddingError(LLMError):
    """Raised when an embedding cannot be generated.

    Inherits from LLMError so callers already catching model-provider failures
    handle embedding failures with the same except clause.
    """


@runtime_checkable
class Embedder(Protocol):
    """Turns text into a vector."""

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for the given text."""
        ...


class EmbeddingClient(Protocol):
    """Transport capable of generating embeddings (e.g. OllamaClient)."""

    def embed(self, model: str, text: str) -> list[float]:
        """Return an embedding for text using the named model."""
        ...


class EmbeddingProvider:
    """Generates embeddings through an injected embedding client.

    Which model, and whose, is the composition root's decision: extraction and
    embedding are separate slots and need not be answered by the same provider.

    Args:
        client: Transport exposing embed(model, text). Injected so tests
            substitute a fake and never reach the network.
        model: Embedding model name.
    """

    def __init__(
        self, client: EmbeddingClient, model: str = DEFAULT_EMBEDDING_MODEL
    ) -> None:
        self._client = client
        self._model = model

    def embed(self, text: str) -> list[float]:
        """Embed a single piece of text.

        Text is lowercased first. `nomic-embed-text` has an uncased vocabulary,
        so capitalised words fall through to a single unknown token: measured,
        "Karaoke", "Trivia", "Bingo", "Death" and "Music" all return the *same*
        vector, while their lowercase forms are properly distinct (0.40–0.64
        apart). Folding here — the one point every embedding passes through —
        guarantees both sides of every comparison are treated alike.

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
            vector = self._client.embed(model=self._model, text=text.lower())
        except LLMError as exc:
            raise EmbeddingError(
                f"Embedding failed for model {self._model!r}: {exc}"
            ) from exc

        if not vector:
            raise EmbeddingError(
                f"Model {self._model!r} returned an empty vector for {text!r}"
            )

        return vector
