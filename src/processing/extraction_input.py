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

    A listing's section heading is appended as its own labelled line rather than
    folded into the description. The prompt names that label and says what it is
    — a taxonomy from a listing site, not a claim about this event — which is
    the whole reason it stopped being stored as prose.
    """
    category = event.metadata.get("listing_category")
    parts = [
        event.title,
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
