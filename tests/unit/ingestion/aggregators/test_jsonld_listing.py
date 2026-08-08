"""Unit tests for the schema.org JSON-LD event parser."""

from datetime import datetime, timedelta, timezone

from src.ingestion.aggregators.jsonld_listing import parse_jsonld_events

EASTERN = timezone(timedelta(hours=-4))


def _event(
    name: str = "Drop-in Art Making",
    start: str | None = "2026-08-08T11:00:00-04:00",
    end: str | None = "2026-08-08T15:00:00-04:00",
    url: str = "https://www.pem.org/events/drop-in-art-making",
    venue: str | None = "Peabody Essex Museum",
    street: str | None = "161 Essex Street, Salem, MA 01970",
    description: str | None = "Drop-in Art Making - Event",
    status: str | None = None,
) -> str:
    parts = [f'"@type": "Event"', f'"name": {name!r}'.replace("'", '"')]
    if start is not None:
        parts.append(f'"startDate": "{start}"')
    if end is not None:
        parts.append(f'"endDate": "{end}"')
    parts.append(f'"url": "{url}"')
    if description is not None:
        parts.append(f'"description": "{description}"')
    if status is not None:
        parts.append(f'"eventStatus": "{status}"')
    if venue is not None:
        address = f', "address": {{"@type": "PostalAddress", "streetAddress": "{street}"}}' if street else ""
        parts.append(
            f'"location": {{"@type": "Place", "name": "{venue}"{address}}}'
        )
    return "{" + ", ".join(parts) + "}"


def _page(*events: str, wrapper: str = "collection") -> str:
    items = ", ".join(
        f'{{"@type": "ListItem", "position": {i + 1}, "item": {event}}}'
        for i, event in enumerate(events)
    )
    if wrapper == "collection":
        block = (
            '{"@context": "https://schema.org", "@type": "CollectionPage", '
            f'"itemListElement": [{items}]}}'
        )
    elif wrapper == "nested":
        block = (
            '{"@context": "https://schema.org", "@type": "CollectionPage", '
            f'"mainEntity": {{"@type": "ItemList", "itemListElement": [{items}]}}}}'
        )
    else:
        block = "[" + ", ".join(events) + "]"
    return f"""<html><head>
      <script type="application/ld+json">
        {{"@type": "Organization", "name": "Not an event"}}
      </script>
      <script type="application/ld+json">{block}</script>
    </head><body><p>Ignore me</p></body></html>"""


class TestParsing:
    def test_reads_one_event(self) -> None:
        events = parse_jsonld_events(_page(_event()))

        assert len(events) == 1
        event = events[0]
        assert event.title == "Drop-in Art Making"
        assert event.url == "https://www.pem.org/events/drop-in-art-making"
        assert event.venue == "Peabody Essex Museum"
        assert event.description == "Drop-in Art Making - Event"

    def test_keeps_the_offset_the_source_stated(self) -> None:
        events = parse_jsonld_events(_page(_event()))

        assert events[0].start == datetime(2026, 8, 8, 11, 0, tzinfo=EASTERN)
        assert events[0].end == datetime(2026, 8, 8, 15, 0, tzinfo=EASTERN)

    def test_reads_the_street_address(self) -> None:
        events = parse_jsonld_events(_page(_event()))

        assert events[0].address == "161 Essex Street, Salem, MA 01970"

    def test_reads_every_event_in_order(self) -> None:
        events = parse_jsonld_events(
            _page(
                _event(name="First", start="2026-08-08T11:00:00-04:00"),
                _event(name="Second", start="2026-08-09T13:00:00-04:00"),
            )
        )

        assert [e.title for e in events] == ["First", "Second"]

    def test_reads_a_list_nested_under_main_entity(self) -> None:
        events = parse_jsonld_events(_page(_event(), wrapper="nested"))

        assert len(events) == 1

    def test_reads_a_bare_array_of_events(self) -> None:
        events = parse_jsonld_events(_page(_event(), wrapper="bare"))

        assert len(events) == 1

    def test_ignores_blocks_that_hold_no_events(self) -> None:
        """Sites carry Organization and BreadcrumbList blocks alongside."""
        events = parse_jsonld_events(_page(_event()))

        assert [e.title for e in events] == ["Drop-in Art Making"]


class TestDegrading:
    def test_a_page_with_no_structured_data_yields_nothing(self) -> None:
        assert parse_jsonld_events("<html><body>Nothing here</body></html>") == []

    def test_an_unparseable_block_does_not_lose_the_others(self) -> None:
        """Rockport publishes invalid JSON-LD; a neighbour must still be read."""
        page = _page(_event()).replace(
            '{"@type": "Organization", "name": "Not an event"}',
            '{"@type": "Event", "startDate": , "endDate": Friday, April 2}',
        )

        assert len(parse_jsonld_events(page)) == 1

    def test_an_event_without_a_start_is_dropped(self) -> None:
        assert parse_jsonld_events(_page(_event(start=None))) == []

    def test_an_event_with_an_empty_start_is_dropped(self) -> None:
        """Exactly what Rockport publishes: `"startDate": ""`."""
        assert parse_jsonld_events(_page(_event(start=""))) == []

    def test_an_event_without_an_end_keeps_its_start(self) -> None:
        events = parse_jsonld_events(_page(_event(end=None)))

        assert events[0].end is None
        assert events[0].start == datetime(2026, 8, 8, 11, 0, tzinfo=EASTERN)

    def test_an_event_without_a_location_keeps_its_title(self) -> None:
        events = parse_jsonld_events(_page(_event(venue=None)))

        assert events[0].venue is None
        assert events[0].title == "Drop-in Art Making"

    def test_an_event_without_a_name_is_dropped(self) -> None:
        page = _page(_event()).replace('"name": "Drop-in Art Making", ', "", 1)

        assert parse_jsonld_events(page) == []


class TestCancellation:
    def test_a_cancelled_event_is_dropped(self) -> None:
        """schema.org states this outright, so no marker guessing is needed."""
        page = _page(_event(status="https://schema.org/EventCancelled"))

        assert parse_jsonld_events(page) == []

    def test_a_scheduled_event_is_kept(self) -> None:
        page = _page(_event(status="https://schema.org/EventScheduled"))

        assert len(parse_jsonld_events(page)) == 1

    def test_a_postponed_event_is_dropped(self) -> None:
        page = _page(_event(status="https://schema.org/EventPostponed"))

        assert parse_jsonld_events(page) == []
