"""Unit tests for The Cabot's listing parser."""

from datetime import date, datetime

from src.ingestion.cinemas.cabot_listing import parse_cabot

TODAY = date(2026, 8, 7)


def _item(
    event_id: str = "20591",
    day: str = "<span>7</span>",
    month: str = "Aug",
    time_html: str = '<div class="time"> 8:00pm</div>',
    title: str = "Garrison Keillor Tonight",
    genres: tuple[str, ...] = ("Music",),
    slug: str = "garrison-keillor-tonight",
    subtitle: str = "",
) -> str:
    genre_html = "".join(f'<div class="genre">{g}</div>' for g in genres)
    sub = f'<p class="h5">{subtitle}</p>' if subtitle else ""
    return f"""
    <div class="event_item" id="event_item_{event_id}">
      <div class="event_item_inner">
        <div class="event_thumb"><a href="https://thecabot.org/event/{slug}/"></a></div>
        <div class="event_info">
          <div class="event_date">{day} {month} {time_html}</div>
          <div class="event_text">{genre_html}<p class="h4">{title}</p>{sub}</div>
        </div>
        <div class="event_btn"><a class="btn" href="https://thecabot.org/event/{slug}/">Buy Tickets</a></div>
      </div>
    </div>"""


def _page(*items: str, showing: str = "Showing 1-10 of 88 events") -> str:
    return f"""<html><body><div class="events_holder">
      <p class="results_count">{showing}</p>{''.join(items)}</div></body></html>"""


class TestParsing:
    def test_an_event_is_parsed(self):
        events = parse_cabot(_page(_item()), TODAY)

        assert len(events) == 1
        assert events[0].title == "Garrison Keillor Tonight"
        assert events[0].start == datetime(2026, 8, 7, 20, 0)

    def test_the_id_comes_from_the_listing(self):
        """A WordPress post id — the site's own key, stable across runs."""
        assert parse_cabot(_page(_item(event_id="20591")), TODAY)[0].event_id == "20591"

    def test_the_event_page_link_is_kept(self):
        assert parse_cabot(_page(_item()), TODAY)[0].url.endswith("/garrison-keillor-tonight/")

    def test_genres_are_collected(self):
        events = parse_cabot(_page(_item(genres=("Music", "Comedy"))), TODAY)

        assert events[0].genres == ["Music", "Comedy"]

    def test_a_morning_time_is_not_read_as_evening(self):
        events = parse_cabot(_page(_item(time_html='<div class="time">10:30am</div>')), TODAY)

        assert events[0].start.hour == 10

    def test_an_event_with_no_time_keeps_its_date(self):
        """The day is known even when the hour is not."""
        events = parse_cabot(_page(_item(time_html="")), TODAY)

        assert events[0].start.date() == date(2026, 8, 7)
        assert events[0].time_known is False

    def test_a_subtitle_is_captured(self):
        events = parse_cabot(_page(_item(subtitle="INDIGO PARK TOUR")), TODAY)

        assert events[0].subtitle == "INDIGO PARK TOUR"
        assert events[0].off_site is False

    def test_an_off_site_marker_is_read(self):
        """The subtitle holds a venue or a tour name; only the marker tells which."""
        item = _item(subtitle="Off Cabot - 9 Wallis St, Beverly").replace(
            '<div class="event_thumb">',
            '<div class="event_thumb"><img class="off_cabot_logo" alt="Off Cabot Event">',
        )

        assert parse_cabot(_page(item), TODAY)[0].off_site is True


class TestDateRanges:
    """`11 - 25 Aug` is a run, not a single night."""

    def test_a_range_yields_a_start_and_an_end(self):
        events = parse_cabot(
            _page(_item(day='<span class="smaller">11 - 25</span>', time_html="")), TODAY
        )

        assert events[0].start.date() == date(2026, 8, 11)
        assert events[0].end.date() == date(2026, 8, 25)

    def test_a_single_date_has_no_end(self):
        assert parse_cabot(_page(_item()), TODAY)[0].end is None


class TestYearResolution:
    """Dates carry no year and no weekday, so ordering is the only signal."""

    def test_the_listing_is_read_as_ascending(self):
        events = parse_cabot(
            _page(
                _item(event_id="1", day="<span>20</span>", month="Dec"),
                _item(event_id="2", day="<span>5</span>", month="Jan"),
            ),
            TODAY,
        )

        assert [e.start.year for e in events] == [2026, 2027]

    def test_a_date_already_past_is_read_as_next_year(self):
        events = parse_cabot(_page(_item(day="<span>3</span>", month="Feb")), TODAY)

        assert events[0].start.year == 2027

    def test_a_date_just_past_stays_this_year(self):
        """A listing may still show something from a few days ago."""
        events = parse_cabot(_page(_item(day="<span>1</span>", month="Aug")), TODAY)

        assert events[0].start.year == 2026


class TestPagination:
    def test_the_total_is_read_from_the_results_count(self):
        page = _page(_item(), showing="Showing 1-10 of 88 events")

        assert parse_cabot(page, TODAY, want_total=True)[1] == 88

    def test_a_missing_results_count_reports_no_total(self):
        page = _page(_item(), showing="")

        assert parse_cabot(page, TODAY, want_total=True)[1] is None


class TestMalformedInput:
    def test_an_event_without_a_title_is_skipped(self):
        broken = _item().replace('<p class="h4">Garrison Keillor Tonight</p>', "")

        assert parse_cabot(_page(broken), TODAY) == []

    def test_an_unreadable_month_is_skipped_without_losing_the_rest(self):
        events = parse_cabot(
            _page(_item(event_id="1", month="Smarch"), _item(event_id="2")), TODAY
        )

        assert [e.event_id for e in events] == ["2"]

    def test_an_empty_page_yields_nothing(self):
        assert parse_cabot("<html><body></body></html>", TODAY) == []

    def test_entities_in_a_title_are_resolved(self):
        events = parse_cabot(_page(_item(title="Rock &amp; Roll")), TODAY)

        assert events[0].title == "Rock & Roll"
