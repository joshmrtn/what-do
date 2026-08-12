"""Unit tests for MoonRssSource's reading of a feed item."""

from __future__ import annotations

import io
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pytest

from src.storage.memory.http_cache import InMemoryHttpCache
from src.config import FeedConfig
from src.ingestion.calendars.moon_source import MoonRssSource
from src.ingestion.rss import RssItem
from src.models.timing import EXACT, UNKNOWN
from src.storage.db import init_db
from src.utils.logging import get_logger

EASTERN = ZoneInfo("America/New_York")
FIXED_NOW = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return MoonRssSource(
        config=FeedConfig(name="moon", url="https://www.moon-ns.org/shows?format=rss", source_type="moon"),
        http_cache=InMemoryHttpCache(),
        get_now=lambda: FIXED_NOW,
        logger=get_logger("test", stream=io.StringIO()),
        timezone_name="America/New_York",
        day_starts_at=time(4, 0),
    )


def _item(title: str, description: str = "") -> RssItem:
    return RssItem(title=title, link="https://www.moon-ns.org/shows/x", description=description)


class TestDate:
    def test_reads_the_date_from_the_title(self, source) -> None:
        event = source.interpret(
            _item(
                "6/27/26: Viraya, Missed Opportunities at Felt Fanatic",
                "Catch Viraya on 6/27/26 at 6pm at Felt Fanatic",
            )
        )

        assert event.start == datetime(2026, 6, 27, 18, 0, tzinfo=EASTERN)

    def test_reads_a_four_digit_year(self, source) -> None:
        event = source.interpret(_item("6/5/2026 inplainsight, Film & Gender"))

        assert event.start.date() == datetime(2026, 6, 5).date()

    def test_reads_a_date_that_is_not_at_the_start(self, source) -> None:
        """Titles carry markers ahead of the date: `*** Moved to Faces ***`."""
        event = source.interpret(_item("*** Moved to Faces Brewery *** 6/17/26 Shiver., Pil"))

        assert event.start.date() == datetime(2026, 6, 17).date()

    def test_an_item_with_no_date_is_refused(self, source) -> None:
        # A guessed date is worse than a missing event.
        assert source.interpret(_item("Salem Arts Festival")) is None

    def test_the_publication_date_is_never_the_event_date(self, source) -> None:
        item = RssItem(
            title="Salem Arts Festival",
            published_at=datetime(2026, 5, 20, tzinfo=timezone.utc),
        )

        assert source.interpret(item) is None

    def test_the_start_is_placed_in_the_configured_zone(self, source) -> None:
        event = source.interpret(_item("6/27/26: A Show", "Catch it on 6/27/26 at 6pm at BitBar"))

        assert event.start.tzinfo is not None
        assert event.start.utcoffset() is not None


class TestTime:
    def test_reads_the_time_from_the_description(self, source) -> None:
        event = source.interpret(_item("6/27/26: A Show", "Catch it on 6/27/26 at 6pm at BitBar"))

        assert event.start.hour == 18
        assert event.timing == EXACT

    def test_reads_a_time_with_minutes(self, source) -> None:
        event = source.interpret(_item("6/27/26: A Show", "Catch it at 7:30pm at BitBar"))

        assert (event.start.hour, event.start.minute) == (19, 30)
        assert event.timing == EXACT

    def test_reads_a_morning_time(self, source) -> None:
        event = source.interpret(_item("6/27/26: A Show", "Doors at 11am at BitBar"))

        assert event.start.hour == 11

    def test_an_item_with_no_time_is_placed_at_the_day_start(self, source) -> None:
        """A date with no published hour is not the same as an all-day event."""
        event = source.interpret(_item("6/25/26: Cherubhead, Weatherman"))

        assert event.timing == UNKNOWN
        assert (event.start.hour, event.start.minute) == (4, 0)

    def test_an_image_only_description_still_yields_the_date(self, source) -> None:
        """Some items are a Squarespace figure and nothing else."""
        event = source.interpret(
            _item("6/25/26: Cherubhead", '<figure class="sqs-block-image-figure"><img/></figure>')
        )

        assert event.start.date() == datetime(2026, 6, 25).date()
        assert event.timing == UNKNOWN


class TestVenue:
    def test_reads_the_venue_the_description_template_names(self, source) -> None:
        event = source.interpret(
            _item("6/27/26: A Show", "Catch A Show on 6/27/26 at 6pm at Felt Fanatic")
        )

        assert event.venue == "Felt Fanatic"

    def test_a_venue_split_across_lines_is_still_read(self, source) -> None:
        event = source.interpret(_item("6/27/26: A Show", "Catch it at 6pm at Felt \nFanatic"))

        assert event.venue == "Felt Fanatic"

    def test_a_trailing_run_of_prose_is_not_a_venue(self, source) -> None:
        """`at 7pm at Faces Malden - ALL AGES | $15 Day of Show` names no venue."""
        event = source.interpret(
            _item("6/17/26: A Show", "Doors at 7pm at Faces Malden - ALL AGES | $15 Day of Show")
        )

        assert event.venue is None

    def test_the_title_is_never_read_for_a_venue(self, source) -> None:
        """`5/29/26 at Gulu Gulu Cafe: Tiny the Bear` would yield the lineup."""
        event = source.interpret(_item("6/29/26 at Gulu Gulu Cafe: Tiny the Bear, Yes, Chef!"))

        assert event.venue is None

    def test_an_item_with_no_description_has_no_venue(self, source) -> None:
        event = source.interpret(_item("6/25/26: A Show"))

        assert event.venue is None


class TestTitle:
    def test_a_leading_date_is_dropped_from_the_title(self, source) -> None:
        event = source.interpret(_item("6/27/26: Viraya, Missed Opportunities at Felt Fanatic"))

        assert event.title == "Viraya, Missed Opportunities at Felt Fanatic"

    def test_a_leading_date_with_no_colon_is_dropped_too(self, source) -> None:
        event = source.interpret(_item("6/6/26 Sunset Mission, DHARA, Mutineer"))

        assert event.title == "Sunset Mission, DHARA, Mutineer"

    def test_a_date_inside_the_title_is_left_alone(self, source) -> None:
        event = source.interpret(_item("*** Moved to Faces Brewery *** 6/17/26 Shiver."))

        assert event.title == "*** Moved to Faces Brewery *** 6/17/26 Shiver."

    def test_the_description_reaches_extraction_as_text(self, source) -> None:
        event = source.interpret(
            _item("6/27/26: A Show", "<p>Catch it at 6pm at BitBar</p>")
        )

        assert event.description == "Catch it at 6pm at BitBar"


class TestCancellation:
    def test_a_cancelled_show_is_refused(self, source) -> None:
        """The feed marks these `*** CANCELED*** ` and leaves the date in place."""
        assert source.interpret(_item("*** CANCELED*** 6/14/2026 NAGLY Benefit Show")) is None

    def test_the_british_spelling_is_refused_too(self, source) -> None:
        assert source.interpret(_item("***CANCELLED*** 6/14/26 A Show")) is None

    def test_a_leading_marker_with_a_separator_is_refused(self, source) -> None:
        """The other convention seen in the wild: `CANCELED - <title>`."""
        assert source.interpret(_item("CANCELED - Pop Up Library 6/14/26")) is None

    def test_a_moved_show_is_kept(self, source) -> None:
        """Moved is not cancelled — it is still happening."""
        event = source.interpret(_item("*** Moved to Deep Cuts *** 6/17/26 Vallory Falls"))

        assert event is not None

    def test_a_show_merely_named_for_cancelling_is_kept(self, source) -> None:
        """The word is only a verdict inside the feed's own marker."""
        event = source.interpret(_item("6/20/26 Cancelled Culture: A Comedy Show"))

        assert event is not None
        assert event.title == "Cancelled Culture: A Comedy Show"

    def test_a_band_with_the_word_in_its_name_is_kept(self, source) -> None:
        event = source.interpret(_item("6/20/26 The Cancelled, Dagwood, Replica City"))

        assert event is not None

    def test_a_show_promising_it_is_never_cancelled_is_kept(self, source) -> None:
        event = source.interpret(_item("6/20/26 Rain or Shine — Never Cancelled Tour"))

        assert event is not None
