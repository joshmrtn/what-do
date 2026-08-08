"""Do617 parsing against real captured venue pages.

No network: the fixtures are real responses from Do617 venue pages, captured on
2026-08-07 with scripts and styles emptied. They are the only place the parser
meets the site's actual chrome — navigation, footer and sidebar all carry
`itemprop` and `name` attributes of their own.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.ingestion.aggregators.do617_listing import parse_do617

FIXTURES = Path(__file__).parent.parent / "fixtures"

EASTERN = timezone(timedelta(hours=-4))


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


class TestGuluGulu:
    def test_reads_every_event_on_the_page(self) -> None:
        page = parse_do617(_fixture("do617_gulu_gulu.html"))

        assert len(page.events) == 25

    def test_reads_the_first_event_whole(self) -> None:
        page = parse_do617(_fixture("do617_gulu_gulu.html"))

        event = page.events[0]
        assert event.title == "Eva James - Live Music No Cover"
        assert event.start == datetime(2026, 8, 7, 20, 0, tzinfo=EASTERN)
        assert event.end == datetime(2026, 8, 7, 23, 0, tzinfo=EASTERN)
        assert event.permalink == "/events/2026/8/7/eva-james-live-music-no-cover-tickets"
        assert event.venue == "Gulu-Gulu Cafe"
        assert event.venue_slug == "gulu-gulu-cafe"
        assert event.street == "247 Essex St"
        assert event.city == "Salem"
        assert event.region == "MA"
        assert event.latitude == 42.5650452

    def test_events_are_ascending_and_within_the_listed_range(self) -> None:
        page = parse_do617(_fixture("do617_gulu_gulu.html"))

        starts = [event.start for event in page.events]
        assert starts == sorted(starts)
        assert starts[-1] == datetime(2026, 9, 2, 18, 0, tzinfo=EASTERN)

    def test_every_event_has_a_distinct_permalink(self) -> None:
        page = parse_do617(_fixture("do617_gulu_gulu.html"))

        permalinks = [event.permalink for event in page.events]
        assert len(set(permalinks)) == len(permalinks)

    def test_site_chrome_never_becomes_an_event(self) -> None:
        page = parse_do617(_fixture("do617_gulu_gulu.html"))

        assert all(event.venue == "Gulu-Gulu Cafe" for event in page.events)
        assert all(event.permalink.startswith("/events/") for event in page.events)

    def test_finds_the_next_page(self) -> None:
        page = parse_do617(_fixture("do617_gulu_gulu.html"))

        assert page.next_page_url == "/venues/gulu-gulu-cafe?page=2"


class TestSecondPage:
    def test_continues_where_the_first_page_stopped(self) -> None:
        first = parse_do617(_fixture("do617_gulu_gulu.html"))
        second = parse_do617(_fixture("do617_gulu_gulu_page2.html"))

        assert len(second.events) == 19
        assert second.events[0].start > first.events[-1].start

    def test_shares_no_events_with_the_first_page(self) -> None:
        first = {event.permalink for event in parse_do617(_fixture("do617_gulu_gulu.html")).events}
        second = {
            event.permalink for event in parse_do617(_fixture("do617_gulu_gulu_page2.html")).events
        }

        assert first & second == set()

    def test_the_last_page_offers_no_next(self) -> None:
        page = parse_do617(_fixture("do617_gulu_gulu_page2.html"))

        assert page.next_page_url is None


class TestEmptyVenue:
    def test_a_venue_with_no_listings_yields_no_events(self) -> None:
        # Koto's page is valid and carries the venue's address, but Do617 has
        # nothing upcoming for it. That is a normal state, not a failure.
        page = parse_do617(_fixture("do617_koto.html"))

        assert page.events == []
        assert page.next_page_url is None
