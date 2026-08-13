"""Unit tests for the text LLM Pass 1 runs over.

Its own file because it is its own module, and for the reason that module
exists: extraction hashes this text to decide whether to spend minutes on an
event again, and ranking measures it to decide how much a thin tag list means.
Two callers must agree on it exactly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.models.event import Event
from src.processing.extraction_input import extraction_input, extraction_input_hash

_NOW = datetime(2026, 8, 13, tzinfo=timezone.utc)


def _event(**kwargs: Any) -> Event:
    defaults: dict[str, Any] = dict(
        event_id="evt-1",
        source_event_candidates=[],
        source_type="northshorenightout",
        created_at=_NOW,
        updated_at=_NOW,
    )
    defaults.update(kwargs)
    return Event(**defaults)


def test_a_title_alone_is_the_whole_input():
    assert extraction_input(_event(title="Jazz Night")) == "Jazz Night"


def test_the_description_follows_the_title():
    event = _event(title="Jazz Night", description="Live jazz at the waterfront.")

    assert extraction_input(event) == "Jazz Night\nLive jazz at the waterfront."


def test_the_category_is_labelled():
    """A section heading is a taxonomy from a listing site, not a claim about
    this event, and the prompt can only say so about a field it can see."""
    event = _event(title="Josh Schulz", metadata={"listing_category": "Music"})

    assert extraction_input(event) == "Josh Schulz\nEvent category: Music"


def test_the_hash_follows_the_text():
    a = _event(title="Jazz Night")
    b = _event(title="Jazz Night", description="Now with a description.")

    assert extraction_input_hash(a) != extraction_input_hash(b)


class TestTheVenueLine:
    """`Steve Dennis` was the entire input for an event we knew was at Lobsta
    Land in Gloucester, while `_compose_summary` built that very string one
    function away. Measured on the real model: inputs that had returned all
    nulls returned real tags once the venue was there.
    """

    def test_the_venue_and_city_travel_as_a_labelled_line(self):
        """Labelled, like the category, so the prompt can name the field and
        say what it is — a place, not an activity. Prose would also give the
        echo rule nothing to anchor to."""
        event = _event(title="Steve Dennis", venue="Lobsta Land", location="Gloucester")

        assert extraction_input(event) == "Steve Dennis\nVenue: Lobsta Land, Gloucester"

    def test_a_venue_with_no_city_still_travels(self):
        event = _event(title="Steve Dennis", venue="Lobsta Land")

        assert extraction_input(event) == "Steve Dennis\nVenue: Lobsta Land"

    def test_a_city_with_no_venue_still_travels(self):
        event = _event(title="Steve Dennis", location="Gloucester")

        assert extraction_input(event) == "Steve Dennis\nVenue: Gloucester"

    def test_an_event_with_neither_gains_no_line(self):
        """An empty `Venue:` label would be a field the model must interpret as
        meaning nothing, which is worse than its absence."""
        assert extraction_input(_event(title="Steve Dennis")) == "Steve Dennis"

    def test_the_venue_line_sits_above_the_description(self):
        """Reading order: what it is, where it is, then what it says."""
        event = _event(
            title="Trivia", venue="The James", location="Essex", description="Doors at 7."
        )

        assert extraction_input(event) == "Trivia\nVenue: The James, Essex\nDoors at 7."

    def test_the_category_still_comes_last(self):
        event = _event(
            title="Steve Dennis",
            venue="Lobsta Land",
            location="Gloucester",
            metadata={"listing_category": "Music"},
        )

        assert extraction_input(event).endswith("Event category: Music")

    def test_adding_a_venue_changes_the_hash(self):
        """The whole corpus re-extracts because of this, which is intended —
        but it must be visible rather than a surprise on a Tuesday night."""
        without = _event(title="Steve Dennis")
        with_venue = _event(title="Steve Dennis", venue="Lobsta Land")

        assert extraction_input_hash(without) != extraction_input_hash(with_venue)
