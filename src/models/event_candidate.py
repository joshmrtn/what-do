"""EventCandidate data model — raw discovered event information."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.models.tag import Tag


@dataclass
class EventCandidate:
    """Raw event information as returned by an ingestion source adapter."""

    id: str
    source: str
    source_type: str
    discovered_at: datetime
    url: str | None = None
    image_url: str | None = None
    raw_published_at: datetime | None = None
    title: str | None = None
    description: str | None = None
    venue: str | None = None
    location: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    #: How much is known about *when* — see `TIMINGS`. A source that gives a
    #: date but no hour says so here rather than letting a placed start read as
    #: a real one.
    timing: str = "exact"
    #: Structured facts a source states about an event that are not prose about
    #: it. A listing's section heading belongs here and not in `description`:
    #: putting a taxonomy label where event copy goes is what led LLM Pass 1 to
    #: read `Karaoke & trivia` as a claim about the event and tag every trivia
    #: night `karaoke`. Merged into `Event.metadata` by normalization.
    metadata: dict[str, Any] = field(default_factory=dict)
    #: A summary the source itself supports, composed rather than inferred.
    #: Terse sources state everything they know in one line; a model asked to
    #: summarise that can only repeat it or invent past it.
    summary: str | None = None
    #: Tags the adapter can author outright, for events whose title *is* the
    #: activity. Paired with `metadata["authored_tags"]`, which exempts the
    #: event from extraction entirely.
    tags: list[Tag] = field(default_factory=list)
    #: When a source was last seen publishing this listing, as against
    #: `discovered_at`, which is when we first met it. One restamped field
    #: cannot answer both, and was silently answering the second while being
    #: read as the first (#27).
    #:
    #: An adapter never sets this: what it holds is a single *observation*, so
    #: first and last are the same instant and `__post_init__` says so. The
    #: split is a property of the stored history, and the repository is what
    #: resolves it — `discovered_at` survives a re-fetch, this does not.
    last_seen_at: datetime | None = None

    def __post_init__(self) -> None:
        """A candidate nobody has stored yet was first seen when it was seen."""
        if self.last_seen_at is None:
            self.last_seen_at = self.discovered_at
        self._canonicalise_instants()

    def _canonicalise_instants(self) -> None:
        """Hold every timestamp in UTC, whatever offset the publisher used.

        A feed's choice of wire format is its own convention and stops at the
        boundary, like every other source quirk. Google Calendar emits
        `DTSTART:…Z` for some events and `DTSTART;TZID=…` for others in one
        file, so the raw layer carried three offsets and `for_window` compared
        them as text — and text order only matches chronological order when the
        offset is fixed. UTC is the only representation where the two agree.

        `Event` goes the other way, to the *local* zone, because that is where
        local reasoning lives and `.date()` reads the zone rather than the text.

        **A naive value is left exactly as it is, deliberately.** It carries no
        instant to convert, and this type has no zone to convert it with —
        guessing one here would misplace an event silently.

        Naivety is *resolved* one layer out, at ingestion, where the configured
        zone is known: `_resolve_naivety` reads a bare timestamp as local and
        records `assumed_zone` in the candidate's metadata, so the raw layer
        holds an instant and the fact that we supplied its offset. Ingestion is
        the seam because it is the single funnel every fetched candidate passes
        through, so no adapter can bypass it (#30).

        This stays tolerant rather than strict for two reasons: ingestion must
        not crash on a naive candidate — the failure that killed the first live
        fetch — and a candidate constructed outside that funnel, in a test or a
        one-off script, is not wrong, merely unresolved.
        """
        for name in ("discovered_at", "last_seen_at", "start_time",
                     "end_time", "raw_published_at"):
            value: datetime | None = getattr(self, name)
            if value is None or value.utcoffset() is None:
                continue
            setattr(self, name, value.astimezone(timezone.utc))
