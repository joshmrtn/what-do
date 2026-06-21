"""Unit tests for PicukiAdapter."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

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


def _make_adapter(response=None):
    from src.ingestion.social.picuki import PicukiAdapter

    mock_session = MagicMock()
    mock_session.get.return_value.json.return_value = response or _PICUKI_RESPONSE
    mock_session.get.return_value.raise_for_status.return_value = None

    return PicukiAdapter(
        handles=["@testvenue"],
        session=mock_session,
        get_now=lambda: FIXED_NOW,
    )


def test_returns_event_candidates():
    from src.models.event_candidate import EventCandidate

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
    from src.ingestion.social.picuki import PicukiAdapter

    mock_session = MagicMock()
    mock_session.get.return_value.raise_for_status.side_effect = Exception("HTTP 503")

    adapter = PicukiAdapter(
        handles=["@testvenue"],
        session=mock_session,
        get_now=lambda: FIXED_NOW,
    )
    with pytest.raises(Exception):
        adapter.fetch()
