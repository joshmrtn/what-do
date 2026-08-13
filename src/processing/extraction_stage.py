"""LLM extraction pipeline stage (Pass 1).

Converts enriched Event objects into semantically structured events
by extracting tags, summary, and filling missing fields via LLM.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from src.models.event import Event
from src.processing.extraction_input import (
    extraction_input,
    extraction_input_hash,
    input_hash,
)
from src.processing.extraction import ExtractionError, ExtractionProvider
from src.processing.image_fetcher import ImageFetchError, ImageFetcher


def _has_authored_tags(event: Event) -> bool:
    """Whether this event's tags were written by us rather than extracted.

    Exempt for the same reason synthetic activities are: the tags are authored,
    so there is nothing for a model to improve and everything for it to invent.
    A hash rule alone cannot see this — the input changes, the authored output
    should not.
    """
    return bool(event.metadata.get("authored_tags"))


class ExtractionStage:
    """Pipeline stage that runs LLM Pass 1 extraction on each event.

    An event is extracted when the hash of its input text differs from the one
    stored on it, so an edited description is picked up rather than skipped
    forever — the old check was "does it already have tags?", which could not
    tell an edited event from a finished one, nor a valid empty-tag result from
    an extraction that never ran.

    Synthetic events are never extracted. Their tags come from hand-written
    config, so running Pass 1 over them would overwrite what a person wrote.
    That exemption is provenance, not state, so it reads from `is_synthetic`
    rather than from anything this stage stores.

    Args:
        provider: LLM extraction provider.
        image_fetcher: Fetcher for event images (None if multimodal not needed).
        logger: Structured logger instance.
        get_now: Injectable clock; supplies the reference date the model uses to
            resolve relative dates in event text.
        save_fn: Called with each event as soon as it is extracted, so model
            time already spent survives a run that does not finish. On the batch
            VM one extraction takes minutes, which makes a full pass long enough
            that saving only at the end risks losing all of it. Called only for
            real model calls — a pass that skips everything on its hash writes
            nothing.
    """

    def __init__(
        self,
        provider: ExtractionProvider,
        image_fetcher: ImageFetcher | None,
        logger: Any,
        get_now: Callable[[], datetime] = datetime.now,
        save_fn: Callable[[Event], None] | None = None,
    ) -> None:
        self._provider = provider
        self._image_fetcher = image_fetcher
        self._logger = logger
        self._get_now = get_now
        self._save_fn = save_fn

    def set_save_fn(self, save_fn: Callable[[Event], None] | None) -> None:
        """Set where checkpoints go, or None to disable them.

        The orchestrator owns every save point, so it decides this rather than
        the stage — and a dry run sets None because it persists nothing at all.
        """
        self._save_fn = save_fn

    def process(self, events: list[Event]) -> list[Event]:
        """Run extraction on each event that needs it.

        Args:
            events: List of enriched events.

        Returns:
            Same list with tags, summary, and optional field fills applied.
        """
        extracted = 0
        for event in events:
            if event.is_synthetic or _has_authored_tags(event):
                continue
            text = extraction_input(event)
            if event.extraction_input_hash == input_hash(text):
                continue
            self._extract(event, text)
            extracted += 1
            self._checkpoint(event, extracted)

        return events

    def _checkpoint(self, event: Event, extracted: int) -> None:
        """Persist the event just extracted, never ending the stage.

        A failing save must not throw away the model time it was protecting —
        the next checkpoint, or the orchestrator's own save, may well succeed.
        """
        if self._save_fn is None:
            return
        try:
            self._save_fn(event)
        except Exception as exc:  # noqa: BLE001 — a checkpoint is best-effort
            self._logger.error(
                f"extraction checkpoint failed after {extracted} events: {exc}",
                component="extraction_stage",
                duration_ms=0,
            )

    def _extract(self, event: Event, text: str) -> None:
        """Run extraction on a single event, updating it in place."""
        image_bytes = self._fetch_image(event)

        try:
            result = self._provider.extract(
                text, image_bytes=image_bytes, reference_date=self._get_now()
            )
        except ExtractionError as exc:
            self._logger.error(
                f"LLM extraction failed for event {event.event_id}: {exc}",
                component="extraction_stage",
                duration_ms=0,
            )
            event.metadata["llm_extraction_failed"] = True
            return

        # Recorded only here, past the failure path, so a failed run stays
        # distinguishable from a finished one and is retried.
        event.extraction_input_hash = input_hash(text)

        # Copied off the result rather than read from the provider, so it
        # describes the attempt that actually answered. Written beside the hash
        # and for the same reason: a failed re-extraction leaves both alone, so
        # a row keeps the provenance of the tags it still has. The skip path
        # never reaches here at all, which is what stops a normal night — where
        # almost every event skips — from blanking what it recorded before.
        event.extraction_model = result.model
        event.extraction_prompt_version = result.prompt_version

        # Through `replace_tags` rather than by assignment: almost every event a
        # batch extracts arrives from storage carrying vectors for its stored
        # tags, and those describe tags it is about to stop having.
        event.replace_tags(result.tags)
        # A source that states everything it knows authors its own summary, and
        # the model can only invent past it. NSNO publishes one line per event;
        # asked to summarise `Trivia` under a `Karaoke & trivia` heading it
        # produced "an evening of karaoke and trivia" for events that were only
        # ever trivia — and that text is embedded and drives dedup pass 2.
        if not event.metadata.get("authored_summary"):
            event.replace_summary(result.summary)
        event.setting = result.setting

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
