"""Unit tests for the Event model."""

from datetime import datetime, timezone

import pytest

from src.models.event import Event
from src.models.timing import ALL_DAY, EXACT, TIMINGS, UNKNOWN
from src.models.source_type import SYNTHETIC
from src.models.tag import Tag
from src.storage.events import validate_tag_vectors


def _now() -> datetime:
    return datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def test_event_construction_minimal():
    """Event can be constructed with only required fields."""
    event = Event(
        event_id="abc-123",
        source_event_candidates=["cand-1"],
        source_type="apify",
        created_at=_now(),
        updated_at=_now(),
    )
    assert event.event_id == "abc-123"
    assert event.source_type == "apify"


def test_event_source_event_candidates_is_list_of_strings():
    event = Event(
        event_id="abc-123",
        source_event_candidates=["cand-1", "cand-2"],
        source_type="apify",
        created_at=_now(),
        updated_at=_now(),
    )
    assert event.source_event_candidates == ["cand-1", "cand-2"]


def test_event_optional_fields_default_none():
    event = Event(
        event_id="abc-123",
        source_event_candidates=["cand-1"],
        source_type="apify",
        created_at=_now(),
        updated_at=_now(),
    )
    assert event.url is None
    assert event.image_url is None
    assert event.image_bytes is None
    assert event.title is None
    assert event.venue is None
    assert event.description is None
    assert event.location is None
    assert event.start_time is None
    assert event.end_time is None
    assert event.summary is None
    assert event.summary_embedding is None
    assert event.weather is None
    assert event.astronomical_data is None


def test_event_tags_default_empty_list():
    event = Event(
        event_id="abc-123",
        source_event_candidates=["cand-1"],
        source_type="apify",
        created_at=_now(),
        updated_at=_now(),
    )
    assert event.tags == []


def test_event_tag_embeddings_default_empty_list():
    event = Event(
        event_id="abc-123",
        source_event_candidates=["cand-1"],
        source_type="apify",
        created_at=_now(),
        updated_at=_now(),
    )
    assert event.tag_embeddings == []


def test_event_metadata_defaults_empty_dict():
    event = Event(
        event_id="abc-123",
        source_event_candidates=["cand-1"],
        source_type="apify",
        created_at=_now(),
        updated_at=_now(),
    )
    assert event.metadata == {}


def test_event_metadata_not_shared_across_instances():
    """Default metadata dicts must be independent per instance."""
    a = Event(
        event_id="a",
        source_event_candidates=[],
        source_type="apify",
        created_at=_now(),
        updated_at=_now(),
    )
    b = Event(
        event_id="b",
        source_event_candidates=[],
        source_type="apify",
        created_at=_now(),
        updated_at=_now(),
    )
    a.metadata["x"] = 1
    assert "x" not in b.metadata


def test_event_tags_not_shared_across_instances():
    """Default tags lists must be independent per instance."""
    a = Event(
        event_id="a",
        source_event_candidates=[],
        source_type="apify",
        created_at=_now(),
        updated_at=_now(),
    )
    b = Event(
        event_id="b",
        source_event_candidates=[],
        source_type="apify",
        created_at=_now(),
        updated_at=_now(),
    )
    a.tags.append(Tag(text="music"))
    assert b.tags == []


def test_event_full_construction():
    """Event with all fields set round-trips correctly."""
    start = datetime(2025, 6, 15, 20, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 6, 15, 23, 0, 0, tzinfo=timezone.utc)
    event = Event(
        event_id="full-1",
        source_event_candidates=["c1", "c2"],
        source_type="cinema_veezi",
        url="https://example.com/event",
        image_url="https://example.com/img.jpg",
        title="Jazz Night",
        venue="The Vault",
        description="A great jazz show.",
        location="Salem, MA",
        start_time=start,
        end_time=end,
        tags=[Tag(text="jazz", weight=1.0), Tag(text="live music", weight=0.6)],
        summary="An evening of jazz at The Vault.",
        tag_embeddings=[b"fake-bytes"],
        summary_embedding=b"more-fake-bytes",
        weather={"condition": "clear"},
        astronomical_data={"sunset": "20:30"},
        metadata={"source": "test"},
        created_at=_now(),
        updated_at=_now(),
    )
    assert event.title == "Jazz Night"
    assert event.venue == "The Vault"
    assert event.tags == [Tag(text="jazz", weight=1.0), Tag(text="live music", weight=0.6)]
    assert event.start_time == start
    assert len(event.source_event_candidates) == 2


def test_a_synthetic_event_is_identified_as_such():
    """Provenance, not a stored flag: `source_type` already carries this fact."""
    event = Event(
        event_id="e1",
        source_event_candidates=[],
        source_type=SYNTHETIC,
        created_at=_now(),
        updated_at=_now(),
    )

    assert event.is_synthetic is True


def test_a_scraped_event_is_not_synthetic():
    event = Event(
        event_id="e1",
        source_event_candidates=[],
        source_type="apify",
        created_at=_now(),
        updated_at=_now(),
    )

    assert event.is_synthetic is False


class TestTagsAndTheirVectors:
    """A vector describes the tag it was built from, so the two move together.

    The pairing is positional, which makes a length mismatch silently drop the
    tail. Enforcing it only where events are written left the one place that can
    break it — the mutation site — unguarded, and a re-extraction returning a
    different number of tags took down a whole night's batch.
    """

    def _embedded(self) -> Event:
        """An event as it arrives from storage: tags with a vector each."""
        event = Event(
            event_id="e1",
            source_event_candidates=[],
            source_type="northshorenightout",
            created_at=_now(),
            updated_at=_now(),
        )
        event.replace_tags([Tag(text="karaoke"), Tag(text="bar")])
        event.attach_tag_embeddings([b"v-karaoke", b"v-bar"])
        return event

    def test_replacing_the_tags_drops_the_old_vectors(self):
        """The crash of 2026-08-12, in miniature.

        Re-extraction returns fewer tags than the stored vectors describe. The
        vectors are worthless — they describe tags the event no longer has —
        and keeping them is what made the event unwritable.
        """
        event = self._embedded()

        event.replace_tags([Tag(text="trivia")])

        assert event.tags == [Tag(text="trivia")]
        assert event.tag_embeddings == []

    def test_replacing_the_tags_leaves_a_writable_event(self):
        """Tags with no vectors is the ordinary state between the two stages.

        Which is the whole point: the expensive extraction survives, and only
        the cheap embedding is re-paid.
        """
        event = self._embedded()

        event.replace_tags([Tag(text="trivia")])

        validate_tag_vectors(event)  # must not raise

    def test_vectors_that_cannot_pair_are_refused(self):
        event = self._embedded()

        with pytest.raises(ValueError, match=r"2 tag\(s\) but 1 vector\(s\)"):
            event.attach_tag_embeddings([b"only-one"])

    def test_a_refused_attach_leaves_the_previous_vectors_alone(self):
        """Rejecting a bad list must not also destroy a good one."""
        event = self._embedded()

        with pytest.raises(ValueError):
            event.attach_tag_embeddings([b"only-one"])

        assert event.tag_embeddings == [b"v-karaoke", b"v-bar"]

    def test_re_embedding_after_a_replacement_pairs_with_the_new_tags(self):
        event = self._embedded()

        event.replace_tags([Tag(text="trivia")])
        event.attach_tag_embeddings([b"v-trivia"])

        assert event.tag_embeddings == [b"v-trivia"]
        validate_tag_vectors(event)

    def test_replacing_the_summary_drops_its_vector(self):
        """The same mistake one field over.

        Nothing validates a summary vector — a single blob has nothing to pair
        against — so a stale one is invisible rather than fatal.
        """
        event = self._embedded()
        event.replace_summary("An evening of karaoke.")
        event.summary_embedding = b"v-summary"

        event.replace_summary("A trivia night.")

        assert event.summary == "A trivia night."
        assert event.summary_embedding is None


def _timed(**overrides) -> Event:
    return Event(
        event_id="evt-1",
        source_event_candidates=[],
        source_type="northshorenightout",
        created_at=_now(),
        updated_at=_now(),
        **overrides,
    )


class TestTiming:
    """A date with no time is two different facts, and they read differently.

    `VALUE=DATE` in a calendar genuinely means all day. A listing that omits the
    hour means nobody has said yet. Collapsing them makes one label a lie.
    """

    def test_timing_defaults_to_exact(self):
        assert _timed().timing == EXACT

    def test_an_all_day_event_records_that(self):
        assert _timed(timing=ALL_DAY).timing == "all_day"

    def test_an_unpublished_time_records_that(self):
        assert _timed(timing=UNKNOWN).timing == "unknown"

    def test_only_an_exact_timing_states_a_clock_time(self):
        """A placed start exists so the night window can position the event.

        Everything downstream has to know it was placed, not published.
        """
        assert _timed().states_a_time is True
        assert _timed(timing=ALL_DAY).states_a_time is False
        assert _timed(timing=UNKNOWN).states_a_time is False

    def test_every_timing_is_a_declared_value(self):
        assert set(TIMINGS) == {"exact", "all_day", "unknown"}
