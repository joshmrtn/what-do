"""Unit tests for AmcAdapter."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.ingestion.movies.amc import AMC_HOST, AmcAdapter
from src.models.event_candidate import EventCandidate
from tests.support.network import fetcher_policy

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


def _make_adapter(response=None, content_ids=False):

    mock_session = MagicMock()
    mock_session.post.return_value.json.return_value = response or _AMC_RESPONSE
    mock_session.post.return_value.raise_for_status.return_value = None

    return AmcAdapter(
        api_key="fake-amc-key",
        postal_code="01970",
        session=mock_session,
        policy=fetcher_policy(urls=f"https://{AMC_HOST}/graphql", now=FIXED_NOW),
        get_now=lambda: FIXED_NOW,
        uses_content_id=lambda source: content_ids,
    )


def test_returns_event_candidates():

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

    mock_session = MagicMock()
    mock_session.post.return_value.raise_for_status.side_effect = Exception("HTTP 403")

    adapter = AmcAdapter(
        api_key="bad-key",
        postal_code="01970",
        session=mock_session,
        policy=fetcher_policy(urls=f"https://{AMC_HOST}/graphql", now=FIXED_NOW),
        get_now=lambda: FIXED_NOW,
        uses_content_id=lambda source: False,
    )
    with pytest.raises(Exception):
        adapter.fetch()


def _response_with_showtimes(showtimes):
    entry = _AMC_RESPONSE["data"]["getMoviesAndShowtimes"][0]
    return {
        "data": {
            "getMoviesAndShowtimes": [{"movie": entry["movie"], "showtimes": showtimes}]
        }
    }


def test_id_is_stable_across_fetches():
    """A refetch of the same showtime must reuse its id, not mint a new one."""
    adapter = _make_adapter()
    assert adapter.fetch()[0].id == adapter.fetch()[0].id


def test_id_is_stable_across_runs_on_different_days():
    """The id must not drift with the clock, or every nightly run duplicates."""
    later = datetime(2025, 9, 1, 3, 0, 0, tzinfo=timezone.utc)
    first = _make_adapter().fetch()[0]

    mock_session = MagicMock()
    mock_session.post.return_value.json.return_value = _AMC_RESPONSE
    mock_session.post.return_value.raise_for_status.return_value = None
    second = AmcAdapter(
        api_key="fake-amc-key",
        postal_code="01970",
        session=mock_session,
        policy=fetcher_policy(urls=f"https://{AMC_HOST}/graphql", now=later),
        get_now=lambda: later,
        uses_content_id=lambda source: False,
    ).fetch()[0]

    assert first.id == second.id


def test_two_showtimes_of_one_movie_get_distinct_ids():
    base = _AMC_RESPONSE["data"]["getMoviesAndShowtimes"][0]["showtimes"][0]
    later = datetime(2025, 6, 16, 22, 0, 0, tzinfo=timezone.utc)
    response = _response_with_showtimes(
        [base, dict(base, id="amc_show_002", showDateTimeUtc=later.isoformat())]
    )
    results = _make_adapter(response).fetch()
    assert results[0].id != results[1].id


def test_falls_back_to_stable_id_without_a_showtime_id():
    """Missing natural key still yields a repeatable id, never a fresh uuid."""
    base = _AMC_RESPONSE["data"]["getMoviesAndShowtimes"][0]["showtimes"][0]
    response = _response_with_showtimes([{k: v for k, v in base.items() if k != "id"}])
    assert _make_adapter(response).fetch()[0].id == _make_adapter(response).fetch()[0].id


class TestWhenTheShowtimeIdsAreNotTrusted:
    """A showtime is a listing — a film, at a theatre, at a time — so the shared
    listing key describes it exactly, and AMC can latch like any other source.
    """

    def test_a_re_minted_showtime_id_yields_the_same_id(self):
        first = _make_adapter(content_ids=True).fetch()[0]

        renumbered = _response_with_showtimes([
            {
                "showDateTimeUtc": SHOWTIME.isoformat(),
                "theatre": {"name": "AMC Methuen 20"},
                "id": "amc_show_999",
            }
        ])
        later = _make_adapter(response=renumbered, content_ids=True).fetch()[0]

        assert first.id == later.id

    def test_the_showtime_id_is_not_in_the_id_at_all(self):
        publisher_keyed = _make_adapter().fetch()[0]
        content_keyed = _make_adapter(content_ids=True).fetch()[0]

        assert publisher_keyed.id != content_keyed.id
