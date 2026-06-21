"""Unit tests for AmcAdapter."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

FIXED_NOW = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
SHOWTIME = datetime(2025, 6, 16, 19, 0, 0, tzinfo=timezone.utc)

_AMC_RESPONSE = {
    "data": {
        "getMoviesAndShowtimes": [
            {
                "movie": {
                    "name": "Alien: Romulus",
                    "synopsis": "Terror in space.",
                    "posterSrc": "https://cdn.amctheatres.com/poster.jpg",
                    "id": "amc_movie_001",
                },
                "showtimes": [
                    {
                        "showDateTimeUtc": SHOWTIME.isoformat(),
                        "theatre": {"name": "AMC Methuen 20"},
                        "id": "amc_show_001",
                    }
                ],
            }
        ]
    }
}


def _make_adapter(response=None):
    from src.ingestion.movies.amc import AmcAdapter

    mock_session = MagicMock()
    mock_session.post.return_value.json.return_value = response or _AMC_RESPONSE
    mock_session.post.return_value.raise_for_status.return_value = None

    return AmcAdapter(
        api_key="fake-amc-key",
        postal_code="01970",
        session=mock_session,
        get_now=lambda: FIXED_NOW,
    )


def test_returns_event_candidates():
    from src.models.event_candidate import EventCandidate

    results = _make_adapter().fetch()
    assert len(results) == 1
    assert isinstance(results[0], EventCandidate)


def test_source_type_is_amc():
    assert _make_adapter().fetch()[0].source_type == "amc"


def test_raw_published_at_is_none():
    """Movie showtimes have no post date; raw_published_at must be None."""
    assert _make_adapter().fetch()[0].raw_published_at is None


def test_start_time_populated():
    assert _make_adapter().fetch()[0].start_time == SHOWTIME


def test_title_populated():
    assert _make_adapter().fetch()[0].title == "Alien: Romulus"


def test_venue_populated():
    assert _make_adapter().fetch()[0].venue == "AMC Methuen 20"


def test_discovered_at_uses_get_now():
    assert _make_adapter().fetch()[0].discovered_at == FIXED_NOW


def test_raises_on_http_error():
    from src.ingestion.movies.amc import AmcAdapter

    mock_session = MagicMock()
    mock_session.post.return_value.raise_for_status.side_effect = Exception("HTTP 403")

    adapter = AmcAdapter(
        api_key="bad-key",
        postal_code="01970",
        session=mock_session,
        get_now=lambda: FIXED_NOW,
    )
    with pytest.raises(Exception):
        adapter.fetch()
