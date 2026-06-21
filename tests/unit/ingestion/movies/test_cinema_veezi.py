"""Unit tests for CinemaVeeziAdapter."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

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
    from src.ingestion.movies.cinema_veezi import CinemaVeeziAdapter

    mock_session = MagicMock()
    mock_session.get.return_value.json.return_value = response or _VEEZI_RESPONSE
    mock_session.get.return_value.raise_for_status.return_value = None

    return CinemaVeeziAdapter(
        api_key="fake-veezi-key",
        session=mock_session,
        get_now=lambda: FIXED_NOW,
    )


def test_returns_event_candidates():
    from src.models.event_candidate import EventCandidate

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
    from src.ingestion.movies.cinema_veezi import CinemaVeeziAdapter

    mock_session = MagicMock()
    mock_session.get.return_value.raise_for_status.side_effect = Exception("HTTP 401")

    adapter = CinemaVeeziAdapter(
        api_key="bad-key",
        session=mock_session,
        get_now=lambda: FIXED_NOW,
    )
    with pytest.raises(Exception):
        adapter.fetch()
