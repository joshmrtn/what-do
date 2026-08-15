"""LLM extraction pipeline stage (Pass 1).

Converts enriched Event objects into semantically structured events
by extracting tags, summary, and filling missing fields via LLM.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable

from src.models.event import Event
from src.normalization.normalizer import normalize_venue
from src.processing.extraction_input import (
    extraction_input,
    extraction_input_hash,
    input_hash,
)
from src.models.tag import Tag
from src.processing.extraction import (
    ExtractionError,
    ExtractionProvider,
    ExtractionResult,
)
from src.processing.image_fetcher import ImageFetchError, ImageFetcher


def _extraction_order(events: list[Event]) -> list[Event]:
    """Soonest first, undated last, as a new list.

    A budget means some events go unextracted, so which ones is a decision
    rather than an accident: tonight's must be ready tomorrow however deep the
    backlog runs. It also gives the unavailable-provider path the right failure
    mode for free, since whatever a run does not reach is furthest out.

    Sorted on a POSIX timestamp rather than on the datetime: sources genuinely
    differ in whether they state an offset — JSON-LD does, HTML listings do not
    — and comparing an aware value with a naive one raises. Reading a naive
    value as local time is a guess about *ordering* only, never about the data.

    Undated last because such an event is usually a standing listing rather
    than an imminent one. It still ranks inline; this is only which of our own
    work we do first.
    """

    def key(event: Event) -> tuple[int, float]:
        if event.start_time is None:
            return (1, 0.0)
        return (0, event.start_time.timestamp())

    return sorted(events, key=key)


def _has_authored_tags(event: Event) -> bool:
    """Whether this event's tags were written by us rather than extracted.

    Exempt for the same reason synthetic activities are: the tags are authored,
    so there is nothing for a model to improve and everything for it to invent.
    A hash rule alone cannot see this — the input changes, the authored output
    should not.
    """
    return bool(event.metadata.get("authored_tags"))


def _with_category_fallback(
    event: Event, result: ExtractionResult
) -> tuple[list[Tag], str | None]:
    """The listing's own heading, when extraction produced no usable tag.

    `music` for a bare performer name under a Music heading beats both invention
    and silence, and it is safe by construction: `_CARRIED_CATEGORIES` already
    withholds `Other` and `Karaoke & trivia`, so a heading only ever reaches
    `listing_category` when it says something the title does not.

    Weight 1.0 because weight is *centrality*, not confidence — it is the only
    thing known about the event, so it is entirely central. How little is known
    is `tag_confidence`'s job, and on a one-tag event it will already be low.

    Recorded in the degradation rather than passing silently, because the refit
    fits against what a *model* does: a fallback tag it never emitted would
    teach the curve that it did.
    """
    category = event.metadata.get("listing_category")
    if result.tags or not category:
        return result.tags, result.degradation

    note = "fell back to listing category"
    reason = f"{result.degradation}; {note}" if result.degradation else note
    return [Tag(text=category.casefold(), weight=1.0)], reason


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
        budget_minutes: How long this stage may spend on model calls, or None
            for no bound. Extraction is the only stage measured in minutes an
            event, so it is the only one worth bounding: a cold start against a
            45-day horizon was measured at 14.9h, and changing `extraction_input`
            re-extracts the whole corpus. Events the budget defers write no hash
            and are picked up by the next run.
    """

    def __init__(
        self,
        provider: ExtractionProvider,
        image_fetcher: ImageFetcher | None,
        logger: Any,
        get_now: Callable[[], datetime] = datetime.now,
        save_fn: Callable[[Event], None] | None = None,
        budget_minutes: float | None = None,
    ) -> None:
        self._provider = provider
        self._image_fetcher = image_fetcher
        self._logger = logger
        self._get_now = get_now
        self._save_fn = save_fn
        self._budget_minutes = budget_minutes
        self._scope_fn: Callable[[Event], bool] | None = None
        #: How many events the last `process` wanted to extract and could
        #: not, because the budget ran out. Describes the run just
        #: finished, so it is reset at the top of every pass.
        self.deferred = 0
        #: How many the last `process` skipped because ranking could never use
        #: them. Same lifetime, and the pair reads as one sentence: what the
        #: budget could not buy, and what it should never have been asked to.
        self.out_of_scope = 0

    def set_save_fn(self, save_fn: Callable[[Event], None] | None) -> None:
        """Set where checkpoints go, or None to disable them.

        The orchestrator owns every save point, so it decides this rather than
        the stage — and a dry run sets None because it persists nothing at all.
        """
        self._save_fn = save_fn

    def set_scope_fn(self, scope_fn: Callable[[Event], bool] | None) -> None:
        """Set which events are worth spending model time on.

        The orchestrator owns this for the same reason it owns `set_save_fn`:
        the predicate depends on the run date, which `--run-date` may override,
        so composition cannot build it. Left unset, every event is extracted.

        It must be the *ranking* scope. An event ranking will discard is one no
        extraction can help, and passing a different predicate here would spend
        the budget on events the run then throws away — which is the failure
        this exists to stop.
        """
        self._scope_fn = scope_fn

    def process(self, events: list[Event]) -> list[Event]:
        """Run extraction on each event that needs it.

        Args:
            events: List of enriched events.

        Returns:
            Same list with tags, summary, and optional field fills applied.
        """
        extracted = 0
        self.deferred = 0
        self.out_of_scope = 0
        deadline = self._deadline()

        # Iterated in priority order, returned in the caller's. The orchestrator
        # hands this same list to embedding and to the save, so re-ordering it
        # is a side effect nothing asked for.
        for event in _extraction_order(events):
            if event.is_synthetic or _has_authored_tags(event):
                continue
            text = extraction_input(event)
            if event.extraction_input_hash == input_hash(text):
                continue
            # After the hash check, not before it, so the count means "stale and
            # out of scope" — the work actually saved. Before it, the same skip
            # would also count every past event already extracted: ~470 against
            # a real saving of 124, a number describing nothing. Hashing a short
            # string is free; a count that can be trusted is not.
            if self._scope_fn is not None and not self._scope_fn(event):
                self.out_of_scope += 1
                continue
            # After the skips, so an event that never reaches the model costs
            # nothing against a budget denominated in model time. Before the
            # call rather than during it: one event may overshoot, which is far
            # cheaper than abandoning minutes already spent.
            if deadline is not None and self._get_now() >= deadline:
                self.deferred += 1
                continue
            self._extract(event, text)
            extracted += 1
            self._checkpoint(event, extracted=extracted)

        if self.deferred:
            self._logger.info(
                f"extraction budget spent after {extracted} events; "
                f"{self.deferred} deferred to the next run",
                component="extraction_stage",
                duration_ms=0,
            )

        if self.out_of_scope:
            self._logger.info(
                f"skipped {self.out_of_scope} event(s) ranking cannot use",
                component="extraction_stage",
                duration_ms=0,
            )

        return events

    def _deadline(self) -> datetime | None:
        """When this stage must stop starting new work, or None if unbounded."""
        if self._budget_minutes is None:
            return None
        return self._get_now() + timedelta(minutes=self._budget_minutes)

    def _checkpoint(self, event: Event, *, extracted: int) -> None:
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
            # Only "no reply at all" reaches here. A reply that fell short comes
            # back as a result carrying its `degradation`, because that is
            # deterministic and re-running it buys nothing; this is not, so the
            # event keeps no hash and the next run tries again.
            self._logger.error(
                f"LLM extraction unavailable for event {event.event_id}: {exc}",
                component="extraction_stage",
                duration_ms=0,
            )
            return

        if result.degradation is not None:
            self._logger.warning(
                f"LLM extraction degraded for event {event.event_id}: {result.degradation}",
                component="extraction_stage",
                duration_ms=0,
            )

        # Copied off the result rather than read from the provider, so it
        # describes the attempt that actually answered. Written beside the hash
        # and for the same reason: a failed re-extraction leaves both alone, so
        # a row keeps the provenance of the tags it still has. The skip path
        # never reaches here at all, which is what stops a normal night — where
        # almost every event skips — from blanking what it recorded before.
        event.extraction_model = result.model
        event.extraction_prompt_version = result.prompt_version

        tags, degradation = _with_category_fallback(event, result)
        event.extraction_degradation = degradation

        # Through `replace_tags` rather than by assignment: almost every event a
        # batch extracts arrives from storage carrying vectors for its stored
        # tags, and those describe tags it is about to stop having.
        event.replace_tags(tags)
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
            # Through the same canonicaliser normalization uses. A venue the
            # model supplies is arriving after normalization has run, so
            # without this it reaches storage in the model's own casing while
            # every venue from a listing is title-cased — two spellings of one
            # place, which then fail `venues_match` and never deduplicate.
            event.venue = normalize_venue(result.venue)
        if event.start_time is None:
            event.start_time = result.start_time
        if event.end_time is None:
            event.end_time = result.end_time

        # Hashed *after* the fills, not before, because extraction can change
        # its own input: the venue it fills is part of `extraction_input`, so a
        # hash taken beforehand describes text the event no longer has and the
        # next run re-extracts for a change we made ourselves. Still past the
        # unavailable path, so a run that never got an answer stays retryable —
        # and a degradation is an answer, so it writes the hash like any other.
        # The observation itself, beside its digest and on the same terms. The
        # hash makes a re-run skippable; the text is what any later refit or
        # model has to learn from, and it cannot be recovered afterwards —
        # rebuilding it needs the builder and the fields as they were, and both
        # move. Length stored rather than derived on read, so it stays true even
        # if the text is ever trimmed.
        recorded = extraction_input(event)
        event.extraction_input = recorded
        event.extraction_input_chars = len(recorded)
        event.extraction_input_hash = input_hash(recorded)

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
