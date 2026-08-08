"""Unit tests for the Do617 schema.org microdata parser."""

from datetime import datetime, timedelta, timezone

from src.ingestion.aggregators.do617_listing import parse_do617

EASTERN = timezone(timedelta(hours=-4))


def _card(
    permalink: str = "/events/2026/8/7/eva-james-live-music-no-cover-tickets",
    title: str = "Eva James - Live Music No Cover",
    start: str = "2026-08-07T20:00-0400",
    end: str | None = "2026-08-07T23:00-0400",
    venue: str | None = "Gulu-Gulu Cafe",
    venue_slug: str | None = "gulu-gulu-cafe",
    street: str | None = "247 Essex St",
    city: str | None = "Salem",
    region: str | None = "MA",
    geo: bool = True,
    category: str = "food-drink",
) -> str:
    end_html = f'<meta itemprop="endDate" datetime="{end}" content="{end}"/>' if end else ""
    geo_html = (
        """
        <span itemprop="geo" itemscope itemtype="http://schema.org/GeoCoordinates">
          <meta itemprop="latitude" content="42.5650452">
          <meta itemprop="longitude" content="-70.86270890000002" />
        </span>"""
        if geo
        else ""
    )
    venue_link = (
        f'<a href="/venues/{venue_slug}" itemprop="url"><span itemprop="name">{venue}</span></a>'
        if venue
        else ""
    )
    address = "".join(
        f'<meta itemprop="{prop}" content="{value}" />'
        for prop, value in (
            ("streetAddress", street),
            ("addressLocality", city),
            ("addressRegion", region),
        )
        if value is not None
    )
    return f"""
    <div class="ds-listing event-card ds-event-category-{category}"
         data-permalink="{permalink}" itemprop="event" itemscope
         itemtype="http://schema.org/Event">
      <a href="{permalink}" itemprop="url" class="ds-listing-event-title url summary">
        <span class="ds-byline"></span>
        <span class="ds-listing-event-title-text" itemprop="name">{title}</span>
      </a>
      <div class="ds-listing-details-container">
        <div class="ds-listing-details">
          <div class="ds-venue-name" itemprop="location" itemscope
               itemtype="http://schema.org/Place">
            {venue_link}
            <span itemprop="address" itemscope itemtype="http://schema.org/PostalAddress">
              {address}
            </span>
            {geo_html}
          </div>
          <div class="ds-event-time dtstart"> 8:00PM </div>
          <meta itemprop="startDate" datetime="{start}" content="{start}"/>
          {end_html}
        </div>
      </div>
    </div>"""


def _page(*cards: str, next_page: str | None = None) -> str:
    nav = (
        f'<a href="{next_page}" class="ds-next-page" rel="next"><span>Next Page</span></a>'
        if next_page
        else ""
    )
    return f"""<html><body>
      <div class="ds-listing-header"><span itemprop="name">Ignore this heading</span></div>
      <div class="ds-listings">{''.join(cards)}</div>
      <div class="ds-pagination">{nav}</div>
    </body></html>"""


class TestParsing:
    def test_reads_one_card(self) -> None:
        page = parse_do617(_page(_card()))

        assert len(page.events) == 1
        event = page.events[0]
        assert event.permalink == "/events/2026/8/7/eva-james-live-music-no-cover-tickets"
        assert event.title == "Eva James - Live Music No Cover"
        assert event.venue == "Gulu-Gulu Cafe"
        assert event.venue_slug == "gulu-gulu-cafe"
        assert event.street == "247 Essex St"
        assert event.city == "Salem"
        assert event.region == "MA"
        assert event.latitude == 42.5650452
        assert event.longitude == -70.86270890000002

    def test_start_keeps_the_offset_the_source_stated(self) -> None:
        page = parse_do617(_page(_card()))

        start = page.events[0].start
        assert start.tzinfo is not None
        assert start == datetime(2026, 8, 7, 20, 0, tzinfo=EASTERN)
        assert page.events[0].end == datetime(2026, 8, 7, 23, 0, tzinfo=EASTERN)

    def test_reads_every_card_in_page_order(self) -> None:
        page = parse_do617(
            _page(
                _card(permalink="/events/2026/8/7/one", start="2026-08-07T20:00-0400"),
                _card(permalink="/events/2026/8/9/two", start="2026-08-09T14:00-0400"),
                _card(permalink="/events/2026/9/2/three", start="2026-09-02T18:00-0400"),
            )
        )

        assert [event.permalink for event in page.events] == [
            "/events/2026/8/7/one",
            "/events/2026/8/9/two",
            "/events/2026/9/2/three",
        ]

    def test_venue_name_never_captures_the_event_title(self) -> None:
        # Both the title and the venue are `itemprop="name"`; only nesting
        # inside `itemprop="location"` tells them apart.
        page = parse_do617(_page(_card(title="Gulu-Gulu Cafe Presents")))

        event = page.events[0]
        assert event.title == "Gulu-Gulu Cafe Presents"
        assert event.venue == "Gulu-Gulu Cafe"

    def test_ignores_names_outside_any_card(self) -> None:
        page = parse_do617(_page(_card()))

        assert [event.title for event in page.events] == ["Eva James - Live Music No Cover"]


class TestDegrading:
    def test_a_venue_page_with_no_events_is_not_an_error(self) -> None:
        page = parse_do617(_page())

        assert page.events == []
        assert page.next_page_url is None

    def test_a_card_without_an_end_time_keeps_its_start(self) -> None:
        page = parse_do617(_page(_card(end=None)))

        assert page.events[0].end is None
        assert page.events[0].start == datetime(2026, 8, 7, 20, 0, tzinfo=EASTERN)

    def test_a_card_without_geo_keeps_everything_else(self) -> None:
        page = parse_do617(_page(_card(geo=False)))

        event = page.events[0]
        assert event.latitude is None
        assert event.longitude is None
        assert event.venue == "Gulu-Gulu Cafe"

    def test_a_card_without_a_venue_link_keeps_its_title(self) -> None:
        page = parse_do617(_page(_card(venue=None, venue_slug=None)))

        event = page.events[0]
        assert event.venue is None
        assert event.venue_slug is None
        assert event.title == "Eva James - Live Music No Cover"

    def test_a_card_with_no_start_is_dropped(self) -> None:
        # A guessed date is worse than a missing event.
        page = parse_do617(_page(_card(start="")))

        assert page.events == []

    def test_a_card_with_an_unparseable_start_is_dropped(self) -> None:
        page = parse_do617(_page(_card(start="next Thursday")))

        assert page.events == []

    def test_an_unparseable_end_does_not_lose_the_event(self) -> None:
        page = parse_do617(_page(_card(end="whenever")))

        assert len(page.events) == 1
        assert page.events[0].end is None

    def test_a_card_without_a_permalink_is_dropped(self) -> None:
        # Identity comes from the site's own key; without it there is none.
        page = parse_do617(_page(_card(permalink="")))

        assert page.events == []


class TestPagination:
    def test_finds_the_next_page_link(self) -> None:
        page = parse_do617(_page(_card(), next_page="/venues/gulu-gulu-cafe?page=2"))

        assert page.next_page_url == "/venues/gulu-gulu-cafe?page=2"

    def test_a_last_page_has_no_next_link(self) -> None:
        page = parse_do617(_page(_card()))

        assert page.next_page_url is None
