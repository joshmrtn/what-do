"""Unit tests for the RSS feed source base class."""

from __future__ import annotations

import io
from datetime import datetime, time, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.storage.memory.http_cache import InMemoryHttpCache
from src.config import FeedConfig
from src.ingestion.calendars.rss_source import RssEvent, RssFeedSource
from src.ingestion.rss import RssItem
from src.models.event_candidate import EventCandidate
from src.models.timing import EXACT, UNKNOWN
from src.storage.sqlite.connection import init_db
from src.utils.logging import get_logger
from tests.support.network import fetcher_for

#: 02:00 in New York, so the night in progress began the previous day.
FIXED_NOW = datetime(2026, 8, 7, 6, 0, tzinfo=timezone.utc)
URL = "https://example.org/shows?format=rss"
EASTERN = timezone(timedelta(hours=-4))


@pytest.fixture
def cache():
    """No database: these tests are about conditional requests."""
    return InMemoryHttpCache()


def _feed(*items: str) -> str:
    return (
        '<?xml version="1.0"?><rss version="2.0"><channel>'
        f'<title>A Venue</title>{"".join(items)}</channel></rss>'
    )


def _item(
    title: str = "A Show",
    link: str = "https://example.org/shows/a-show",
    guid: str = "abc123",
    pub_date: str = "Wed, 20 May 2026 19:12:56 +0000",
) -> str:
    return (
        f"<item><title>{title}</title><link>{link}</link>"
        f"<guid>{guid}</guid><pubDate>{pub_date}</pubDate>"
        "<description>Some prose</description></item>"
    )


class _FakeSession:
    def __init__(self, body: str) -> None:
        self._body = body
        self.requested: list[str] = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.requested.append(url)
        response = MagicMock()
        response.status_code = 200
        response.text = self._body
        response.headers = {}
        response.raise_for_status.return_value = None
        return response


class _Stub(RssFeedSource):
    """Places every item at a fixed moment, unless told to refuse it."""

    def __init__(self, *args, start=None, refuse=(), timing=EXACT, **kwargs):
        super().__init__(*args, **kwargs)
        self._start = start or datetime(2026, 8, 7, 20, 0, tzinfo=EASTERN)
        self._refuse = refuse
        self._timing = timing

    def interpret(self, item: RssItem) -> RssEvent | None:
        if item.title in self._refuse:
            return None
        return RssEvent(
            title=item.title,
            start=self._start,
            timing=self._timing,
            venue="Felt Fanatic",
            description="a description",
        )


def _make_source(cache, body, horizon_days=45, **stub_kwargs):
    http = _FakeSession(body)
    source = _Stub(
        config=FeedConfig(name="moon", url=URL, source_type="moon"),
        fetcher=fetcher_for(http, urls=URL, http_cache=cache, now=FIXED_NOW),
        get_now=lambda: FIXED_NOW,
        logger=get_logger("test", stream=io.StringIO()),
        timezone_name="America/New_York",
        horizon_days=horizon_days,
        day_starts_at=time(4, 0),
        **stub_kwargs,
    )
    return source, http


class TestMapping:
    def test_returns_event_candidates(self, cache):
        source, _ = _make_source(cache, _feed(_item()))

        results = source.fetch()

        assert len(results) == 1
        assert isinstance(results[0], EventCandidate)

    def test_the_id_comes_from_the_items_guid(self, cache):
        source, _ = _make_source(cache, _feed(_item(guid="5f2a1b")))

        assert source.fetch()[0].id == "moon:5f2a1b"

    def test_the_url_comes_from_the_items_link(self, cache):
        source, _ = _make_source(cache, _feed(_item(link="https://example.org/shows/x")))

        assert source.fetch()[0].url == "https://example.org/shows/x"

    def test_the_publication_date_is_the_announcement_not_the_start(self, cache):
        """pubDate is when the show was posted, often weeks before it runs."""
        source, _ = _make_source(cache, _feed(_item()))

        candidate = source.fetch()[0]
        assert candidate.raw_published_at == datetime(2026, 5, 20, 19, 12, 56, tzinfo=timezone.utc)
        assert candidate.start_time == datetime(2026, 8, 7, 20, 0, tzinfo=EASTERN)

    def test_the_interpreted_fields_are_carried(self, cache):
        source, _ = _make_source(cache, _feed(_item(title="Fat Randy")))

        candidate = source.fetch()[0]
        assert candidate.title == "Fat Randy"
        assert candidate.venue == "Felt Fanatic"
        assert candidate.description == "a description"

    def test_the_interpreted_timing_is_carried(self, cache):
        source, _ = _make_source(cache, _feed(_item()), timing=UNKNOWN)

        assert source.fetch()[0].timing == UNKNOWN

    def test_reads_every_item(self, cache):
        source, _ = _make_source(cache, _feed(_item(guid="a"), _item(guid="b")))

        assert [c.id for c in source.fetch()] == ["moon:a", "moon:b"]


class TestInterpretation:
    def test_an_item_the_subclass_refuses_is_dropped(self, cache):
        body = _feed(_item(title="datable", guid="a"), _item(title="undatable", guid="b"))
        source, _ = _make_source(cache, body, refuse=("undatable",))

        assert [c.id for c in source.fetch()] == ["moon:a"]

    def test_a_feed_of_nothing_but_refusals_yields_nothing(self, cache):
        source, _ = _make_source(cache, _feed(_item(title="undatable")), refuse=("undatable",))

        assert source.fetch() == []


class TestWindow:
    def test_an_event_past_the_horizon_is_dropped(self, cache):
        source, _ = _make_source(
            cache, _feed(_item()), start=datetime(2026, 12, 1, 20, 0, tzinfo=EASTERN)
        )

        assert source.fetch() == []

    def test_an_event_before_the_night_floor_is_dropped(self, cache):
        source, _ = _make_source(
            cache, _feed(_item()), start=datetime(2026, 6, 27, 18, 0, tzinfo=EASTERN)
        )

        assert source.fetch() == []

    def test_an_event_earlier_tonight_is_kept(self, cache):
        source, _ = _make_source(
            cache, _feed(_item()), start=datetime(2026, 8, 6, 20, 0, tzinfo=EASTERN)
        )

        assert len(source.fetch()) == 1


class TestFailure:
    def test_a_body_that_is_not_a_feed_raises(self, cache):
        source, _ = _make_source(cache, "<html><body>Sorry</body></html>")

        with pytest.raises(ValueError):
            source.fetch()
