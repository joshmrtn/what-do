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


class RefusingEmbeddingProvider:
    """An embedding provider that raises rather than calling a model.

    What the read path holds. `CLAUDE.md`: "The CLI shall make no LLM calls
    during interactive use." Network is expressly permitted — a forecast is
    milliseconds — but a model is not, and an embedding *is* a model call.

    Nothing on the read path should need one. Tags change only through
    extraction, which the read path does not run, so `embedding_input_hash` is
    invariant and `EmbeddingStage` skips every event by its own rule; preference
    lines come from a cache keyed on their text. So this is not a limitation
    bolted on — it is the assertion that the assumption holds, and the one place
    it can fail loudly instead of quietly taking minutes.

    The caller's job is to treat the failure as "fall back to what is stored",
    never as an error worth losing the listing over.
    """

    def embed(self, text: str) -> list[float]:
        """Always raises. See the class docstring.

        Raises:
            EmbeddingError: Always.
        """
        raise EmbeddingError(
            "the read path may not call an embedding model; "
            "this needs a batch run to embed"
        )
