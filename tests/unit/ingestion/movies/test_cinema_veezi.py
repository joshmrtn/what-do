"""Unit tests for CinemaVeeziAdapter."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.ingestion.movies.cinema_veezi import CinemaVeeziAdapter
from src.models.event_candidate import EventCandidate

FIXED_NOW = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
SHOWTIME = datetime(2025, 6, 16, 20, 30, 0, tzinfo=timezone.utc)

_VEEZI_RESPONSE = [
    {
        "FilmTitle": "The Grand Illusion",
        "ShowDateTime": SHOWTIME.isoformat(),
        "CinemaName": "Cinema Salem",
        "SynopsisShort": "A classic French film.",
        "PosterUrl": "https://cdn.veezi.com/poster.jpg",
        "ScheduledFilmId": "film_001",
    },
]


def _make_adapter(response=None):

    mock_session = MagicMock()
    mock_session.get.return_value.json.return_value = response or _VEEZI_RESPONSE
    mock_session.get.return_value.raise_for_status.return_value = None

    return CinemaVeeziAdapter(
        api_key="fake-veezi-key",
        session=mock_session,
        get_now=lambda: FIXED_NOW,
    )


def test_returns_event_candidates():

    results = _make_adapter().fetch()
    assert len(results) == 1
    assert isinstance(results[0], EventCandidate)


def test_source_type_is_cinema_veezi():
    assert _make_adapter().fetch()[0].source_type == "cinema_veezi"


def test_raw_published_at_is_none():
    """Movie showtimes have no post date; raw_published_at must be None."""
    assert _make_adapter().fetch()[0].raw_published_at is None


def test_start_time_populated():
    result = _make_adapter().fetch()[0]
    assert result.start_time == SHOWTIME


def test_title_populated():
    assert _make_adapter().fetch()[0].title == "The Grand Illusion"


def test_venue_populated():
    assert _make_adapter().fetch()[0].venue == "Cinema Salem"


def test_discovered_at_uses_get_now():
    assert _make_adapter().fetch()[0].discovered_at == FIXED_NOW


def test_raises_on_http_error():

    mock_session = MagicMock()
    mock_session.get.return_value.raise_for_status.side_effect = Exception("HTTP 401")

    adapter = CinemaVeeziAdapter(
        api_key="bad-key",
        session=mock_session,
        get_now=lambda: FIXED_NOW,
    )
    with pytest.raises(Exception):
        adapter.fetch()


def test_id_is_stable_across_fetches():
    """A refetch of the same session must reuse its id, not mint a new one."""
    adapter = _make_adapter()
    assert adapter.fetch()[0].id == adapter.fetch()[0].id


def test_id_is_stable_across_runs_on_different_days():
    """The id must not drift with the clock, or every nightly run duplicates."""
    later = datetime(2025, 9, 1, 3, 0, 0, tzinfo=timezone.utc)
    first = _make_adapter().fetch()[0]

    mock_session = MagicMock()
    mock_session.get.return_value.json.return_value = _VEEZI_RESPONSE
    mock_session.get.return_value.raise_for_status.return_value = None
    second = CinemaVeeziAdapter(
        api_key="fake-veezi-key",
        session=mock_session,
        get_now=lambda: later,
    ).fetch()[0]

    assert first.id == second.id


def test_two_showtimes_of_one_film_get_distinct_ids():
    """ScheduledFilmId repeats across sessions; the showtime disambiguates."""
    later = datetime(2025, 6, 16, 23, 0, 0, tzinfo=timezone.utc)
    response = [
        _VEEZI_RESPONSE[0],
        dict(_VEEZI_RESPONSE[0], ShowDateTime=later.isoformat()),
    ]
    results = _make_adapter(response).fetch()
    assert results[0].id != results[1].id


def test_distinct_films_get_distinct_ids():
    response = [
        _VEEZI_RESPONSE[0],
        dict(_VEEZI_RESPONSE[0], ScheduledFilmId="film_002", FilmTitle="Another Picture"),
    ]
    results = _make_adapter(response).fetch()
    assert results[0].id != results[1].id


def test_falls_back_to_stable_id_without_a_scheduled_film_id():
    """Missing natural key still yields a repeatable id, never a fresh uuid."""
    session = {k: v for k, v in _VEEZI_RESPONSE[0].items() if k != "ScheduledFilmId"}
    assert _make_adapter([session]).fetch()[0].id == _make_adapter([session]).fetch()[0].id
