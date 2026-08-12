"""A handle discovered from post text, on its way to becoming a source."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

#: Newly discovered, not yet trusted enough to fetch from.
PROBATIONARY = "probationary"
#: Fetched from on every run.
ACTIVE = "active"
#: Classified as a person rather than a venue, and left alone.
DISCARDED = "discarded"


@dataclass(frozen=True)
class CandidateEntity:
    """One discovered handle and everything known about why it is here.

    Discovery works by mention: a handle named in enough posts, by enough
    sources we already trust, earns its way to `active`. That is why the
    counters matter — they are the evidence for a promotion, and `mention_sources`
    is what stops one chatty account promoting its own friends, since a mention
    only counts once per source.

    Attributes:
        entity_id: Stable id, generated once on first sighting.
        handle: The social handle, including its `@`.
        state: `probationary`, `active`, or `discarded`.
        depth: How many hops from a seed this was found at.
        mention_count: Distinct sources that have mentioned it.
        mention_sources: Which sources those were, so a repeat does not recount.
        llm_classification: `venue` or `person`, once disambiguation has run.
        discovery_context: The post text it was first seen in. First one wins —
            later sightings do not overwrite it, because the earliest context is
            the one nearest the discovery that justified keeping the handle.
        promoted_venue_id: Set once a venue row exists for it.
        created_at: First sighting.
        updated_at: Last change.
    """

    entity_id: str
    handle: str
    state: str = PROBATIONARY
    depth: int = 0
    mention_count: int = 0
    mention_sources: list[str] = field(default_factory=list)
    llm_classification: str | None = None
    discovery_context: str | None = None
    promoted_venue_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
