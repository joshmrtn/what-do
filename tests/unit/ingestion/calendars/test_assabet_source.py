"""Unit tests for AssabetRssSource's reading of a feed item."""

from __future__ import annotations

import io
from datetime import datetime, time, timedelta, timezone

import pytest
from unittest.mock import MagicMock

from src.config import FeedConfig
from src.ingestion.calendars.assabet_source import AssabetRssSource
from src.ingestion.rss import RssItem
from src.models.timing import EXACT
from src.storage.sqlite.connection import init_db
from src.utils.logging import get_logger
from tests.support.network import fetcher_for

EASTERN = timezone(timedelta(hours=-4))
FIXED_NOW = datetime(2026, 8, 8, 10, 0, tzinfo=timezone.utc)
FEED_URL = "https://salempl.assabetinteractive.com/calendar/upcoming-events.rss"

#: The four structural lines every Assabet item opens with, then its prose.
IN_LIBRARY = (
    "<p>Saturday, August 8, 2026 10:30—11:00 AM</p>"
    "<p>Children's Program Room - Ground Floor</p>"
    "<p>The Salem Public Library</p>"
    "<p>370 Essex St, Salem, MA, 01970</p>"
    "<p>An adaptive sensory-friendly song and movement program.</p>"
    "<p>For ages 3-8 with siblings and caregiver(s).</p>"
)

OFF_SITE = (
    "<p>Thursday, August 13, 2026 4:00—4:30 PM</p>"
    "<p>Salem Farmers' Market - 32 Derby Square, Salem, MA 01970</p>"
    "<p>Community Visits</p>"
    "<p>, Salem, MA, 01970</p>"
    "<p>Join us for a weekly storytime at the Farmers' Market downtown!</p>"
)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return AssabetRssSource(
        config=FeedConfig(
            name="salempl",
            url=FEED_URL,
            source_type="salempl",
            city="Salem",
        ),
        fetcher=fetcher_for(MagicMock(), urls=FEED_URL, now=FIXED_NOW),
        get_now=lambda: FIXED_NOW,
        logger=get_logger("test", stream=io.StringIO()),
        timezone_name="America/New_York",
        day_starts_at=time(4, 0),
    )


def _item(
    title: str = "Family Circle Time",
    description: str = IN_LIBRARY,
    published: datetime | None = datetime(2026, 8, 8, 10, 30, tzinfo=EASTERN),
) -> RssItem:
    return RssItem(
        title=title,
        link="https://salempl.assabetinteractive.com/calendar/family-circle-time",
        description=description,
        published_at=published,
    )


class TestStart:
    def test_the_publication_date_is_the_event_start(self, source) -> None:
        """The opposite of MOON, where pubDate is when the show was announced."""
        event = source.interpret(_item())

        assert event.start == datetime(2026, 8, 8, 10, 30, tzinfo=EASTERN)
        assert event.timing == EXACT

    def test_an_item_with_no_publication_date_is_refused(self, source) -> None:
        # Nothing else in the item carries a machine-readable date.
        assert source.interpret(_item(published=None)) is None

    def test_a_naive_publication_date_is_placed_in_the_configured_zone(self, source) -> None:
        event = source.interpret(_item(published=datetime(2026, 8, 8, 10, 30)))

        assert event.start.utcoffset() is not None


class TestVenue:
    def test_an_in_library_event_takes_the_organisation_as_its_venue(self, source) -> None:
        event = source.interpret(_item())

        assert event.venue == "The Salem Public Library"

    def test_an_off_site_event_takes_the_place_it_is_actually_at(self, source) -> None:
        """`Community Visits` is a programme, not somewhere you can go."""
        event = source.interpret(_item(description=OFF_SITE))

        assert event.venue == "Salem Farmers' Market"

    def test_an_address_on_the_place_line_is_trimmed_off(self, source) -> None:
        description = OFF_SITE.replace(
            "Salem Farmers' Market - 32 Derby Square, Salem, MA 01970",
            "Mayor Jean Levesque Community Life Center - 401 Bridge St, Salem, MA 01970",
        )

        event = source.interpret(_item(description=description))

        assert event.venue == "Mayor Jean Levesque Community Life Center"

    def test_a_room_name_is_not_trimmed_at_its_hyphen(self, source) -> None:
        """`Children's Program Room - Ground Floor` names no address."""
        description = IN_LIBRARY.replace(
            "<p>370 Essex St, Salem, MA, 01970</p>", "<p>, Salem, MA, 01970</p>"
        )

        event = source.interpret(_item(description=description))

        assert event.venue == "Children's Program Room - Ground Floor"

    def test_an_item_with_no_structure_falls_back_to_the_configured_venue(self, source) -> None:
        event = source.interpret(_item(description="<p>Just some prose</p>"))

        assert event.venue is None


class TestDescription:
    def test_the_structural_preamble_is_dropped(self, source) -> None:
        """Date, room, organisation and address are already structured fields."""
        event = source.interpret(_item())

        assert event.description is not None
        assert "370 Essex St" not in event.description
        assert "Saturday, August 8" not in event.description
        assert event.description.startswith("An adaptive sensory-friendly")

    def test_the_whole_prose_is_kept(self, source) -> None:
        event = source.interpret(_item())

        assert "For ages 3-8" in event.description

    def test_an_item_with_no_prose_has_no_description(self, source) -> None:
        preamble = (
            "<p>Saturday, August 8, 2026 10:30—11:00 AM</p>"
            "<p>Children's Program Room</p><p>The Salem Public Library</p>"
            "<p>370 Essex St, Salem, MA, 01970</p>"
        )

        event = source.interpret(_item(description=preamble))

        assert event.description is None


class TestCancellation:
    def test_a_cancelled_event_is_refused(self, source) -> None:
        """This feed writes `CANCELED - <title>`, as MOON writes `*** CANCELED***`."""
        assert source.interpret(_item(title="CANCELED - Pop Up Library")) is None

    def test_an_event_merely_named_for_cancelling_is_kept(self, source) -> None:
        assert source.interpret(_item(title="Cancelled Plans Book Club")) is not None
