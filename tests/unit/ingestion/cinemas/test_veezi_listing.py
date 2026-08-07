"""Unit tests for the Veezi public sessions parser.

Pure: HTML and a reference date in, showings out. No network, no clock.
"""

from datetime import date, datetime

import pytest

from src.ingestion.cinemas.veezi_listing import parse_sessions

TODAY = date(2026, 8, 7)
TOKEN = "mz33me119qs1fn6sympyf41n2w"


def _film(title: str, *date_blocks: str, censor: str = "R") -> str:
    blocks = "".join(date_blocks)
    return f"""
    <div class="film " id="" name="">
      <div class="poster-container">
        <img class="poster" src="/Media/Poster?siteToken={TOKEN}&amp;code=1" alt="{title}" />
      </div>
      <div>
        <h3 class="title">
            {title}
        </h3>
        <p><span class="censor">{censor}</span>for reasons.</p>
        <div class="sessions">{blocks}</div>
      </div>
    </div>
    """


def _day(heading: str, *sessions: tuple[str, str]) -> str:
    items = "".join(
        f'<li><a href="https://ticketing.useast.veezi.com/purchase/{sid}'
        f'?siteToken={TOKEN}"><time>{when}</time></a></li>'
        for sid, when in sessions
    )
    return (
        f'<div class="date-container"><h4 class="date">{heading}</h4>'
        f'<ul class="session-times">{items}</ul></div>'
    )


def _page(*films: str) -> str:
    return f"<html><body>{''.join(films)}</body></html>"


class TestParsing:
    def test_a_showing_becomes_one_session(self):
        html = _page(_film("The Odyssey", _day("Friday 7, August", ("38750", "3:30 PM"))))

        sessions = parse_sessions(html, TODAY)

        assert len(sessions) == 1
        assert sessions[0].title == "The Odyssey"
        assert sessions[0].start == datetime(2026, 8, 7, 15, 30)

    def test_every_showtime_of_a_day_is_kept(self):
        html = _page(
            _film(
                "The Odyssey",
                _day("Friday 7, August", ("38750", "3:30 PM"), ("38754", "7:00 PM")),
            )
        )

        assert [s.start.hour for s in parse_sessions(html, TODAY)] == [15, 19]

    def test_the_session_id_comes_from_the_booking_link(self):
        """The source's own key — 84 ids for 84 showings, with no collisions."""
        html = _page(_film("The Odyssey", _day("Friday 7, August", ("38750", "3:30 PM"))))

        assert parse_sessions(html, TODAY)[0].session_id == "38750"

    def test_the_booking_url_is_kept(self):
        html = _page(_film("The Odyssey", _day("Friday 7, August", ("38750", "3:30 PM"))))

        assert parse_sessions(html, TODAY)[0].url.endswith(f"purchase/38750?siteToken={TOKEN}")

    def test_a_film_showing_on_several_days_yields_one_session_each(self):
        html = _page(
            _film(
                "The Odyssey",
                _day("Friday 7, August", ("1", "3:30 PM")),
                _day("Saturday 8, August", ("2", "4:00 PM")),
            )
        )

        assert [s.start.date() for s in parse_sessions(html, TODAY)] == [
            date(2026, 8, 7), date(2026, 8, 8)
        ]

    def test_midnight_and_noon_are_not_confused(self):
        html = _page(
            _film(
                "Late Show",
                _day("Friday 7, August", ("1", "12:00 PM"), ("2", "12:30 AM")),
            )
        )

        assert [s.start.hour for s in parse_sessions(html, TODAY)] == [12, 0]


class TestDuplicates:
    """The page lists the same showing more than once — 60 of 144 rows.

    Deduping on the session id is what makes the count honest.
    """

    def test_a_repeated_showing_is_emitted_once(self):
        block = _day("Friday 7, August", ("38750", "3:30 PM"))
        html = _page(_film("The Odyssey", block), _film("The Odyssey", block))

        assert len(parse_sessions(html, TODAY)) == 1

    def test_distinct_showings_of_one_film_all_survive(self):
        html = _page(
            _film(
                "The Odyssey",
                _day("Friday 7, August", ("38750", "3:30 PM"), ("38754", "7:00 PM")),
            )
        )

        assert len(parse_sessions(html, TODAY)) == 2


class TestYearResolution:
    """Headings omit the year, so the weekday name is the checksum."""

    def test_a_heading_late_in_the_year_resolves_forward(self):
        """1 January 2027 is a Friday, and that is the checksum doing the work."""
        html = _page(_film("Winter Film", _day("Friday 1, January", ("1", "7:00 PM"))))

        assert parse_sessions(html, date(2026, 12, 20))[0].start.date() == date(2027, 1, 1)

    def test_a_heading_whose_weekday_does_not_match_is_skipped(self):
        """A weekday that fits no nearby year is corrupt, not a date to guess at."""
        html = _page(_film("Impossible", _day("Monday 7, August", ("1", "7:00 PM"))))

        assert parse_sessions(html, TODAY) == []


class TestMalformedInput:
    def test_a_session_without_a_booking_link_is_skipped(self):
        """No id means no stable identity, and inventing one duplicates nightly."""
        html = _page(
            _film("No Link", '<div class="date-container"><h4 class="date">Friday 7, August</h4>'
                             '<ul class="session-times"><li><time>7:00 PM</time></li></ul></div>')
        )

        assert parse_sessions(html, TODAY) == []

    def test_a_film_without_a_title_is_skipped(self):
        html = _page(
            '<div class="film "><div class="sessions">'
            + _day("Friday 7, August", ("1", "7:00 PM"))
            + "</div></div>"
        )

        assert parse_sessions(html, TODAY) == []

    def test_an_unparseable_time_is_skipped_without_losing_the_rest(self):
        html = _page(
            _film(
                "Mixed",
                _day("Friday 7, August", ("1", "half past six"), ("2", "7:00 PM")),
            )
        )

        assert [s.session_id for s in parse_sessions(html, TODAY)] == ["2"]

    def test_an_empty_page_yields_nothing(self):
        assert parse_sessions("<html><body></body></html>", TODAY) == []

    def test_entities_in_a_title_are_resolved(self):
        html = _page(_film("Tom &amp; Jerry", _day("Friday 7, August", ("1", "7:00 PM"))))

        assert parse_sessions(html, TODAY)[0].title == "Tom & Jerry"
