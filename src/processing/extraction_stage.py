"""LLM extraction pipeline stage (Pass 1).

Converts enriched Event objects into semantically structured events
by extracting tags, summary, and filling missing fields via LLM.
"""

from __future__ import annotations

from typing import Any

from src.models.event import Event
from src.processing.extraction import ExtractionError, ExtractionProvider
from src.processing.image_fetcher import ImageFetchError, ImageFetcher


class ExtractionStage:
    """Pipeline stage that runs LLM Pass 1 extraction on each event.

    Events with tags already populated are bypassed — their existing
    tags and summary pass through unchanged. This handles synthetic events
    and any future pre-tagged source without special-casing.

    Args:
        provider: LLM extraction provider.
        image_fetcher: Fetcher for event images (None if multimodal not needed).
        logger: Structured logger instance.
    """

    def __init__(
        self,
        provider: ExtractionProvider,
        image_fetcher: ImageFetcher | None,
        logger: Any,
    ) -> None:
        self._provider = provider
        self._image_fetcher = image_fetcher
        self._logger = logger

    def process(self, events: list[Event]) -> list[Event]:
        """Run extraction on each event that needs it.

        Args:
            events: List of enriched events.

        Returns:
            Same list with tags, summary, and optional field fills applied.
        """
        for event in events:
            if event.tags:
                continue
            self._extract(event)
        return events

    def _extract(self, event: Event) -> None:
        """Run extraction on a single event, updating it in place."""
        image_bytes = self._fetch_image(event)

        text = "\n".join(filter(None, [event.title, event.description]))

        try:
            result = self._provider.extract(text, image_bytes=image_bytes)
        except ExtractionError as exc:
            self._logger.error(
                f"LLM extraction failed for event {event.event_id}: {exc}",
                component="extraction_stage",
                duration_ms=0,
            )
            event.metadata["llm_extraction_failed"] = True
            return

        event.tags = result.tags
        event.summary = result.summary

        if event.title is None:
            event.title = result.title
        if event.venue is None:
            event.venue = result.venue
        if event.start_time is None:
            event.start_time = result.start_time
        if event.end_time is None:
            event.end_time = result.end_time

    def _fetch_image(self, event: Event) -> bytes | None:
        """Fetch image bytes for the event if an image URL is present."""
        if not event.image_url or self._image_fetcher is None:
            return None

        try:
            image_bytes = self._image_fetcher.fetch(event.image_url)
            event.image_bytes = image_bytes
            return image_bytes
        except ImageFetchError as exc:
            self._logger.warning(
                f"Image fetch failed for event {event.event_id} ({event.image_url}): {exc}",
                component="extraction_stage",
                duration_ms=0,
            )
            return None
