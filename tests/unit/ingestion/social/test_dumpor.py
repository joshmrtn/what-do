"""Unit tests for DumporAdapter."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

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


def _make_adapter(response=None):
    from src.ingestion.social.dumpor import DumporAdapter

    mock_session = MagicMock()
    mock_session.get.return_value.json.return_value = response or _DUMPOR_RESPONSE
    mock_session.get.return_value.raise_for_status.return_value = None

    return DumporAdapter(
        handles=["@testvenue"],
        session=mock_session,
        get_now=lambda: FIXED_NOW,
    )


def test_returns_event_candidates():
    from src.models.event_candidate import EventCandidate

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
    from src.ingestion.social.dumpor import DumporAdapter

    mock_session = MagicMock()
    mock_session.get.return_value.raise_for_status.side_effect = Exception("HTTP 500")

    adapter = DumporAdapter(
        handles=["@testvenue"],
        session=mock_session,
        get_now=lambda: FIXED_NOW,
    )
    with pytest.raises(Exception):
        adapter.fetch()
