"""Unit tests for stable candidate id derivation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.ingestion.candidate_id import (
    content_identity,
    derive_candidate_id,
    derive_content_id,
)
from src.ingestion.id_churn import content_key
from src.models.event_candidate import EventCandidate

START = datetime(2026, 8, 22, 23, 0, tzinfo=timezone.utc)


def test_same_material_yields_same_id():
    """The whole point: a refetch of the same item resolves to the same id."""
    first = derive_candidate_id("apify", "post_abc123")
    second = derive_candidate_id("apify", "post_abc123")
    assert first == second


def test_different_material_yields_different_id():
    assert derive_candidate_id("apify", "post_abc") != derive_candidate_id("apify", "post_xyz")


def test_source_type_prefixes_the_id():
    assert derive_candidate_id("apify", "post_abc").startswith("apify:")


def test_same_material_under_different_source_types_differs():
    """Two sources describing one upstream item must not collide on id."""
    assert derive_candidate_id("picuki", "abc") != derive_candidate_id("dumpor", "abc")


def test_composite_material_is_order_sensitive():
    """A film id and a showtime are not interchangeable."""
    assert derive_candidate_id("v", "film_1", "20:30") != derive_candidate_id("v", "20:30", "film_1")


def test_none_parts_are_tolerated():
    derived = derive_candidate_id("apify", None, "post_abc")
    assert derived.startswith("apify:")


def test_none_is_distinct_from_empty_string_position():
    """A missing part must not silently shift the remaining material."""
    assert derive_candidate_id("v", None, "b") != derive_candidate_id("v", "b", None)


def test_datetime_material_is_accepted():
    when = datetime(2025, 6, 16, 20, 30, tzinfo=timezone.utc)
    assert derive_candidate_id("v", "film_1", when) == derive_candidate_id("v", "film_1", when)


def test_id_is_fixed_length_regardless_of_material_size():
    short = derive_candidate_id("apify", "a")
    long = derive_candidate_id("apify", "a" * 5000)
    assert len(short) == len(long)


def test_rejects_material_with_no_content():
    """All-empty material would collapse every such item onto one id."""
    with pytest.raises(ValueError):
        derive_candidate_id("apify", None, "")


class TestTheContentKeyIsSharedWithTheChurnDetector:
    """The detector decides *"this is the same listing"*; the latched id
    derivation has to collapse exactly the listings it counted. Two copies of
    that rule would diverge silently and perfectly: the detector would report
    0% churn while the ids churned, each answering with its own key.
    """

    def _pair(self, **overrides: object) -> tuple[object, object]:
        base: dict = {
            "source": "nsno_ics",
            "title": "Isabel Stover",
            "venue": "The Joy Nest",
            "start": START,
        }
        other = dict(base)
        other.update(overrides)
        return base, other

    @pytest.mark.parametrize(
        "overrides, same_listing",
        [
            ({}, True),
            # Canonicalisation the detector already applies. A re-cased or
            # re-articled republish is the same listing to it, so it must be the
            # same id too — or the latch fires and the duplicates survive it.
            ({"title": "ISABEL STOVER"}, True),
            ({"venue": "Rhumb Line"}, False),
            ({"venue": "the joy nest"}, True),
            # Genuinely different listings.
            ({"title": "Someone Else"}, False),
            ({"venue": "Somewhere Else"}, False),
            ({"start": START.replace(day=23)}, False),
            # The feed, not the category — two feeds may cover one event, and
            # each publishes its own listing of it.
            ({"source": "nsno_listing"}, False),
        ],
    )
    def test_same_key_iff_same_id(self, overrides: dict, same_listing: bool) -> None:
        base, other = self._pair(**overrides)

        keys_agree = content_identity(**base) == content_identity(**other)
        ids_agree = derive_content_id(**base) == derive_content_id(**other)

        assert keys_agree is same_listing
        assert ids_agree is same_listing

    def test_the_detector_reads_the_same_key(self) -> None:
        """`content_key` takes a candidate and must not paraphrase the rule."""
        candidate = EventCandidate(
            id="whatever",
            source="nsno_ics",
            source_type="northshorenightout",
            title="ISABEL STOVER",
            venue="the joy nest",
            start_time=START,
            discovered_at=START,
        )

        assert content_key(candidate) == content_identity(
            source="nsno_ics",
            title="Isabel Stover",
            venue="The Joy Nest",
            start=START,
        )

    def test_a_missing_title_and_venue_still_yields_an_id(self) -> None:
        """A start alone is thin, but it is material — and raising here would
        crash ingestion on a listing the publisher simply wrote badly."""
        assert derive_content_id(
            source="nsno_ics", title=None, venue=None, start=START
        )
