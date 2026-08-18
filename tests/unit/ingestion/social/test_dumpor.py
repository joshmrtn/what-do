"""Unit tests for DumporAdapter."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from unittest.mock import MagicMock

import requests

import pytest

from src.ingestion.social.dumpor import DumporAdapter
from src.models.event_candidate import EventCandidate
from tests.support.network import fetcher_for

FIXED_NOW = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
PUBLISHED_AT = datetime(2025, 6, 10, 18, 0, 0, tzinfo=timezone.utc)

_DUMPOR_RESPONSE = [
    {
        "shortcode": "dump_xyz",
        "caption_text": "Open mic Friday at 9pm",
        "taken_at_timestamp": int(PUBLISHED_AT.timestamp()),
        "permalink": "https://dumpor.com/p/xyz",
        "display_url": "https://cdn.dumpor.com/img.jpg",
    },
]


def _json_session(payload):
    """A session answering with JSON text, which is what the transport returns."""
    session = MagicMock()
    response = session.get.return_value
    response.status_code = 200
    response.text = json.dumps(payload)
    response.headers = {}
    response.raise_for_status.return_value = None
    return session


def _make_adapter(response=None):

    return DumporAdapter(
        handles=["@testvenue"],
        fetcher=fetcher_for(
            _json_session(response or _DUMPOR_RESPONSE),
            urls="https://dumpor.com/api/user/testvenue",
            now=FIXED_NOW,
        ),
        get_now=lambda: FIXED_NOW,
    )


def test_returns_event_candidates():

    results = _make_adapter().fetch()
    assert len(results) == 1
    assert isinstance(results[0], EventCandidate)


def test_source_type_is_dumpor():
    assert _make_adapter().fetch()[0].source_type == "dumpor"


def test_raw_published_at_from_unix_timestamp():
    result = _make_adapter().fetch()[0]
    assert result.raw_published_at is not None
    assert result.raw_published_at == PUBLISHED_AT


def test_discovered_at_uses_get_now():
    assert _make_adapter().fetch()[0].discovered_at == FIXED_NOW


def test_raises_on_http_error():

    session = MagicMock()
    session.get.return_value.raise_for_status.side_effect = requests.HTTPError("500")

    adapter = DumporAdapter(
        handles=["@testvenue"],
        fetcher=fetcher_for(
            session,
            urls="https://dumpor.com/api/user/testvenue",
            now=FIXED_NOW,
            max_attempts=1,
        ),
        get_now=lambda: FIXED_NOW,
    )
    with pytest.raises(Exception):
        adapter.fetch()


def test_id_is_stable_across_fetches():
    """A refetch of the same post must reuse its id, not mint a new one."""
    adapter = _make_adapter()
    assert adapter.fetch()[0].id == adapter.fetch()[0].id


def test_id_is_stable_across_runs_on_different_days():
    """The id must not drift with the clock, or every nightly run duplicates."""
    later = datetime(2025, 9, 1, 3, 0, 0, tzinfo=timezone.utc)
    first = _make_adapter().fetch()[0]

    second = DumporAdapter(
        handles=["@testvenue"],
        fetcher=fetcher_for(_json_session(_DUMPOR_RESPONSE), urls="https://dumpor.com/api/user/testvenue", now=later),
        get_now=lambda: later,
    ).fetch()[0]

    assert first.id == second.id


def test_distinct_posts_get_distinct_ids():
    response = [
        dict(_DUMPOR_RESPONSE[0], shortcode="one", permalink="https://dumpor.com/p/one"),
        dict(_DUMPOR_RESPONSE[0], shortcode="two", permalink="https://dumpor.com/p/two"),
    ]
    results = _make_adapter(response).fetch()
    assert results[0].id != results[1].id


def test_id_survives_an_edited_caption():
    """The post's own id identifies it; the caption is not part of that."""
    original = _make_adapter().fetch()[0]
    edited = _make_adapter(
        [dict(_DUMPOR_RESPONSE[0], caption_text="Open mic Friday at 10pm")]
    ).fetch()[0]
    assert original.id == edited.id


def test_falls_back_to_stable_id_without_a_shortcode():
    """Missing natural key still yields a repeatable id, never a fresh uuid."""
    post = {k: v for k, v in _DUMPOR_RESPONSE[0].items() if k != "shortcode"}
    assert _make_adapter([post]).fetch()[0].id == _make_adapter([post]).fetch()[0].id
