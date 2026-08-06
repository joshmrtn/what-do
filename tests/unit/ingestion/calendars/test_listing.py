"""Unit tests for the HTML event-listing parser."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from src.ingestion.calendars.listing import ListingEntry, parse_listing

TODAY = date(2026, 8, 5)


def _page(*paragraphs: str) -> str:
    return "<body><div class='sqs-html-content'>" + "".join(paragraphs) + "</div></body>"


def _heading(text: str, large: bool = False) -> str:
    cls = ' class="sqsrte-large"' if large else ""
    return f"<p{cls}><strong>{text}</strong></p>"


def _event(text: str) -> str:
    return f'<p class="sqsrte-small">{text}</p>'


def test_parses_a_single_event():

    html = _page(
        _heading("Wednesday, August 5", large=True),
        _heading("Music"),
        _event("6:30 PM - Jazz Night - Joy Nest - Newburyport"),
    )

    entries = parse_listing(html, today=TODAY)

    assert entries == [
        ListingEntry(
            title="Jazz Night",
            venue="Joy Nest",
            city="Newburyport",
            category="Music",
            start=datetime(2026, 8, 5, 18, 30),
            url=None,
        )
    ]


def test_empty_page_yields_nothing():

    assert parse_listing(_page(), today=TODAY) == []


def test_times_convert_from_twelve_hour_clock():

    html = _page(
        _heading("Wednesday, August 5", large=True),
        _event("12:00 PM - Noon Show - V - C"),
        _event("12:30 AM - Midnight Show - V - C"),
        _event("11:45 PM - Late Show - V - C"),
    )

    starts = [e.start for e in parse_listing(html, today=TODAY)]

    assert starts == [
        datetime(2026, 8, 5, 12, 0),
        datetime(2026, 8, 5, 0, 30),
        datetime(2026, 8, 5, 23, 45),
    ]


def test_date_heading_applies_to_every_following_event():

    html = _page(
        _heading("Wednesday, August 5", large=True),
        _event("7:00 PM - First - V - C"),
        _heading("Thursday, August 6", large=True),
        _event("7:00 PM - Second - V - C"),
    )

    entries = parse_listing(html, today=TODAY)

    assert entries[0].start.date() == date(2026, 8, 5)
    assert entries[1].start.date() == date(2026, 8, 6)


def test_category_heading_applies_until_the_next_one():

    html = _page(
        _heading("Wednesday, August 5", large=True),
        _heading("Music"),
        _event("7:00 PM - A Band - V - C"),
        _heading("Sports"),
        _event("8:00 PM - A Game - V - C"),
    )

    entries = parse_listing(html, today=TODAY)

    assert [e.category for e in entries] == ["Music", "Sports"]


def test_category_resets_on_a_new_day():
    """A day with no category heading must not inherit yesterday's."""

    html = _page(
        _heading("Wednesday, August 5", large=True),
        _heading("Music"),
        _event("7:00 PM - A Band - V - C"),
        _heading("Thursday, August 6", large=True),
        _event("8:00 PM - Something - V - C"),
    )

    entries = parse_listing(html, today=TODAY)

    assert entries[0].category == "Music"
    assert entries[1].category is None


def test_event_link_is_captured():
    """The listing carries per-event links the calendar feed does not have."""

    html = _page(
        _heading("Wednesday, August 5", large=True),
        _event(
            '7:00 PM -<a href="https://venue.test/e/1"><u>Alex Anthony</u></a>'
            " - Minglewood Harborside - Gloucester"
        ),
    )

    entry = parse_listing(html, today=TODAY)[0]

    assert entry.title == "Alex Anthony"
    assert entry.venue == "Minglewood Harborside"
    assert entry.url == "https://venue.test/e/1"


def test_missing_space_after_the_dash_still_parses():
    """Linked titles render as `7:00 PM -Title`, with no space after the dash."""

    html = _page(
        _heading("Wednesday, August 5", large=True),
        _event("7:00 PM -Matt Rich &amp; Guest - The Rhumb Line - Gloucester"),
    )

    entry = parse_listing(html, today=TODAY)[0]

    assert entry.title == "Matt Rich & Guest"
    assert entry.venue == "The Rhumb Line"
    assert entry.city == "Gloucester"


def test_title_containing_a_dash_keeps_venue_and_city_correct():
    """Venue and city are the last two segments; anything before is the title."""

    html = _page(
        _heading("Wednesday, August 5", large=True),
        _event("7:00 PM - Sing-Along - Movie Night - The Cabot - Beverly"),
    )

    entry = parse_listing(html, today=TODAY)[0]

    assert entry.title == "Sing-Along - Movie Night"
    assert entry.venue == "The Cabot"
    assert entry.city == "Beverly"


def test_event_without_a_venue_segment_is_kept():
    """Losing venue attribution is acceptable; losing the event is not."""

    html = _page(
        _heading("Wednesday, August 5", large=True),
        _event("7:00 PM - Mystery Event"),
    )

    entry = parse_listing(html, today=TODAY)[0]

    assert entry.title == "Mystery Event"
    assert entry.venue is None
    assert entry.city is None


def test_events_before_any_date_heading_are_skipped():
    """A time with no date cannot be placed, and a guessed date would be a lie."""

    html = _page(_event("7:00 PM - Orphan - V - C"))

    assert parse_listing(html, today=TODAY) == []


def test_paragraphs_that_are_not_events_are_ignored():

    html = _page(
        _heading("Wednesday, August 5", large=True),
        "<p>Welcome to our listings!</p>",
        "<p></p>",
        _event("7:00 PM - Real - V - C"),
    )

    entries = parse_listing(html, today=TODAY)

    assert [e.title for e in entries] == ["Real"]


def test_weekday_name_selects_the_year_over_mere_nearness():
    """August 5 is a Wednesday in 2026 and a Thursday in 2027.

    A "Thursday" heading read in July 2026 therefore means 2027, even though
    2026 is the nearer year — the weekday is a checksum on the date.
    """
    html = _page(
        _heading("Thursday, August 5", large=True),
        _event("7:00 PM - Show - V - C"),
    )

    entry = parse_listing(html, today=date(2026, 7, 20))[0]

    assert entry.start.date() == date(2027, 8, 5)


def test_nearby_heading_resolves_to_the_current_year():

    html = _page(
        _heading("Wednesday, August 5", large=True),
        _event("7:00 PM - Show - V - C"),
    )

    entry = parse_listing(html, today=date(2026, 7, 20))[0]

    assert entry.start.date() == date(2026, 8, 5)


def test_a_long_stale_heading_is_dropped_rather_than_guessed():
    """Months-old headings mean the page is stale; placing them would be a lie."""

    html = _page(
        _heading("Wednesday, August 5", large=True),
        _event("7:00 PM - Show - V - C"),
    )

    assert parse_listing(html, today=date(2026, 12, 20)) == []


def test_january_listing_read_in_december_rolls_to_next_year():
    """The listing looks forward, so a nearby past date is the wrong reading."""

    html = _page(
        _heading("Friday, January 1", large=True),
        _event("7:00 PM - New Year - V - C"),
    )

    entry = parse_listing(html, today=date(2026, 12, 20))[0]

    assert entry.start.date() == date(2027, 1, 1)


def test_entities_in_venue_and_city_are_resolved():

    html = _page(
        _heading("Wednesday, August 5", large=True),
        _event("7:00 PM - Show - O&#39;Neill&#39;s - Salem"),
    )

    entry = parse_listing(html, today=TODAY)[0]

    assert entry.venue == "O'Neill's"


def test_unparseable_date_heading_does_not_place_later_events():

    html = _page(
        _heading("Sometime Soon", large=True),
        _event("7:00 PM - Show - V - C"),
    )

    assert parse_listing(html, today=TODAY) == []
