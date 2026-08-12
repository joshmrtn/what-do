"""Embedding pipeline stage.

Turns each event's tags and summary into vectors, ready for semantic
deduplication and similarity scoring.
"""

from __future__ import annotations

import hashlib

from typing import Callable

from src.models.event import Event
from src.scoring.embeddings import EmbeddingError, EmbeddingProvider
from src.utils.logging import StructuredLogger
from src.utils.vectors import encode_vector


def embedding_input_hash(event: Event) -> str:
    """Digest of the tags and summary the vectors are built from.

    Both are extraction's output, so a re-extraction that changes them changes
    this, and the embeddings follow without needing to know extraction ran.
    """
    text = "\n".join([*(tag.text for tag in event.tags), event.summary or ""])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingStage:
    """Attaches tag and summary embeddings to each event.

    Vectors for identical text are generated once per run — tags like
    "live music" recur across many events, and each embedding is a round trip
    to the model. `preload` extends that across runs: a vector is a pure
    function of its text and the model, so a tag embedded on any previous night
    never needs embedding again. Measured, 5,289 tag instances share 1,258
    distinct tags, and a second night sees almost none of them for the first
    time.

    An event is re-embedded when its tags or summary have changed since the
    vectors were built. Skipping on "has vectors" alone would leave an event
    re-extracted into different tags still carrying the old ones' vectors,
    which misranks it silently.

    Unlike preference embeddings, a failure here is not fatal: one unscorable
    event costs one recommendation, whereas a missing preference silently
    re-scores the whole batch. Failed events are flagged and left for the
    scoring layer to skip.

    Args:
        provider: Embedding provider.
        logger: Structured logger for embedding failures.
        preload: Returns vectors already computed for this model, keyed by text.
            None starts every run with an empty memo, which is what a fresh
            database wants.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        logger: StructuredLogger,
        preload: Callable[[], dict[str, bytes]] | None = None,
    ) -> None:
        self._provider = provider
        self._logger = logger
        self._preload = preload

    def process(self, events: list[Event]) -> list[Event]:
        """Embed the tags and summary of each event.

        Args:
            events: Events carrying extracted tags and summaries.

        Returns:
            The same list, with embeddings attached in place.
        """
        memo: dict[str, bytes] = self._preload() if self._preload is not None else {}
        if memo:
            self._logger.info(
                f"reusing {len(memo)} tag vector(s) from earlier runs",
                component="embedding_stage",
                duration_ms=0,
            )

        for event in events:
            self._embed_event(event, memo)

        return events

    def _embed_event(self, event: Event, memo: dict[str, bytes]) -> None:
        """Embed one event's tags and summary, updating it in place."""
        digest = embedding_input_hash(event)
        if event.tag_embeddings and event.embedding_input_hash == digest:
            return  # already embedded, from these exact tags and summary

        if not event.tags:
            event.metadata["embedding_skipped"] = True
            return

        try:
            # Built separately and assigned only on success, so a failure
            # partway through cannot leave the event scored against a subset
            # of its own tags.
            vectors = [self._vector(tag.text, memo) for tag in event.tags]
        except EmbeddingError as exc:
            self._fail(event, f"tags: {exc}")
            return

        event.attach_tag_embeddings(vectors)

        if not event.summary or not event.summary.strip():
            event.embedding_input_hash = digest
            return

        try:
            event.summary_embedding = self._vector(event.summary, memo)
        except EmbeddingError as exc:
            # Tag embeddings are kept — they are the primary signal, and the
            # summary is only a weighted supporting term. The hash stays unset
            # so the next run retries the summary rather than treating a
            # half-embedded event as done.
            self._fail(event, f"summary: {exc}")
            return

        event.embedding_input_hash = digest

    def _vector(self, text: str, memo: dict[str, bytes]) -> bytes:
        """Return the encoded vector for text, reusing it across the run."""
        if text not in memo:
            memo[text] = encode_vector(self._provider.embed(text))
        return memo[text]

    def _fail(self, event: Event, detail: str) -> None:
        """Flag an event as unembeddable and log why."""
        event.metadata["embedding_failed"] = True
        self._logger.error(
            f"Embedding failed for event {event.event_id} ({detail})",
            component="embedding_stage",
            duration_ms=0,
        )
