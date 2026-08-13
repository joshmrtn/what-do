"""The text LLM Pass 1 runs over, and the hash that decides whether it re-runs.

Its own module, and deliberately dependency-free. It began in
`extraction_stage.py`, but ranking needs the same definition to judge how many
tags an input could have earned — and importing the stage to get it dragged
`image_fetcher`, and so `requests`, into the CLI's import graph, which is the
one path that must not meet the network layer.

Two callers must agree on this text exactly. Extraction hashes it to decide
whether to spend minutes on an event again; ranking measures it to decide how
much a thin tag list means. A second, drifting definition would make an event
re-extract while being judged against the length of something else.
"""

from __future__ import annotations

import hashlib

from src.models.event import Event


def extraction_input(event: Event) -> str:
    """The text LLM Pass 1 runs over, and the thing whose hash gates a re-run.

    Everything the source told us about the event, each fact on its own labelled
    line where a label is what makes it legible. The prompt names those labels
    and says what they are — a section heading is a listing site's taxonomy, a
    venue is a place — which is the whole reason neither is folded into the
    description as prose.

    The venue and city were omitted for months while `_compose_summary` built
    exactly that string one function away. `Steve Dennis` was the entire input
    for an event we knew was at Lobsta Land in Gloucester, and the model
    answered by nulling every field, honestly: a performer's name supports no
    tag, and the anti-invention rule forbids guessing one. Measured on the real
    model, adding this line turned those all-null replies into real tags.
    """
    place = ", ".join(p for p in (event.venue, event.location) if p)
    category = event.metadata.get("listing_category")
    parts = [
        event.title,
        # Above the description: what it is, where it is, then what it says.
        f"Venue: {place}" if place else None,
        event.description,
        f"Event category: {category}" if category else None,
    ]
    return "\n".join(filter(None, parts))


def input_hash(text: str) -> str:
    """Stable digest of an extraction input."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extraction_input_hash(event: Event) -> str:
    """The digest an event must carry for extraction to consider it done."""
    return input_hash(extraction_input(event))
