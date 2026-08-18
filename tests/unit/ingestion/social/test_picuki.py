"""Unit tests for PicukiAdapter."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from unittest.mock import MagicMock

import requests

import pytest

from src.ingestion.social.picuki import PicukiAdapter
from src.models.event_candidate import EventCandidate
from tests.support.network import fetcher_for

FIXED_NOW = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
PUBLISHED_AT = datetime(2025, 6, 10, 18, 0, 0, tzinfo=timezone.utc)

_PICUKI_RESPONSE = [
    {
        "post_id": "picuki_abc",
        "text": "Live music tonight at 8pm!",
        "date": PUBLISHED_AT.isoformat(),
        "link": "https://www.picuki.com/post/abc",
        "image": "https://cdn.picuki.com/img.jpg",
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

    return PicukiAdapter(
        handles=["@testvenue"],
        fetcher=fetcher_for(
            _json_session(response or _PICUKI_RESPONSE),
            urls="https://www.picuki.com/profile/testvenue",
            now=FIXED_NOW,
        ),
        get_now=lambda: FIXED_NOW,
    )


def test_returns_event_candidates():

    adapter = _make_adapter()
    results = adapter.fetch()

    assert len(results) == 1
    assert isinstance(results[0], EventCandidate)


def test_source_type_is_picuki():
    adapter = _make_adapter()
    assert adapter.fetch()[0].source_type == "picuki"


def test_raw_published_at_populated():
    adapter = _make_adapter()
    assert adapter.fetch()[0].raw_published_at == PUBLISHED_AT


def test_discovered_at_uses_get_now():
    adapter = _make_adapter()
    assert adapter.fetch()[0].discovered_at == FIXED_NOW


def test_raises_on_http_error():

    session = MagicMock()
    session.get.return_value.raise_for_status.side_effect = requests.HTTPError("503")

    adapter = PicukiAdapter(
        handles=["@testvenue"],
        fetcher=fetcher_for(session, urls="https://www.picuki.com/profile/testvenue", now=FIXED_NOW, max_attempts=1),
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

    second = PicukiAdapter(
        handles=["@testvenue"],
        fetcher=fetcher_for(_json_session(_PICUKI_RESPONSE), urls="https://www.picuki.com/profile/testvenue", now=later),
        get_now=lambda: later,
    ).fetch()[0]

    assert first.id == second.id


def test_distinct_posts_get_distinct_ids():
    response = [
        dict(_PICUKI_RESPONSE[0], post_id="one", link="https://www.picuki.com/post/one"),
        dict(_PICUKI_RESPONSE[0], post_id="two", link="https://www.picuki.com/post/two"),
    ]
    results = _make_adapter(response).fetch()
    assert results[0].id != results[1].id


def test_id_survives_edited_text():
    """The post's own id identifies it; the caption is not part of that."""
    original = _make_adapter().fetch()[0]
    edited = _make_adapter([dict(_PICUKI_RESPONSE[0], text="Live music tonight at 9pm!")]).fetch()[0]
    assert original.id == edited.id


def test_falls_back_to_stable_id_without_a_post_id():
    """Missing natural key still yields a repeatable id, never a fresh uuid."""
    post = {k: v for k, v in _PICUKI_RESPONSE[0].items() if k != "post_id"}
    assert _make_adapter([post]).fetch()[0].id == _make_adapter([post]).fetch()[0].id
