"""Unit tests for ApifyAdapter."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

FIXED_NOW = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
PUBLISHED_AT = datetime(2025, 6, 10, 18, 0, 0, tzinfo=timezone.utc)

# Minimal Apify Instagram scraper response fixture
_APIFY_RESPONSE = [
    {
        "id": "post_abc123",
        "caption": "Jazz Night this Friday! @jazzclub",
        "timestamp": PUBLISHED_AT.isoformat(),
        "url": "https://www.instagram.com/p/abc123/",
        "displayUrl": "https://cdn.example.com/img.jpg",
        "locationName": "The Vault Lounge",
    },
]


def _make_adapter(response=None):
    from src.ingestion.social.apify import ApifyAdapter

    mock_session = MagicMock()
    mock_session.get.return_value.json.return_value = response or _APIFY_RESPONSE
    mock_session.get.return_value.raise_for_status.return_value = None

    return ApifyAdapter(
        api_key="fake-key",
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


def test_source_type_is_apify():
    adapter = _make_adapter()
    result = adapter.fetch()[0]
    assert result.source_type == "apify"


def test_raw_published_at_populated():
    adapter = _make_adapter()
    result = adapter.fetch()[0]
    assert result.raw_published_at is not None
    assert result.raw_published_at == PUBLISHED_AT


def test_discovered_at_uses_get_now():
    adapter = _make_adapter()
    result = adapter.fetch()[0]
    assert result.discovered_at == FIXED_NOW


def test_description_populated_from_caption():
    adapter = _make_adapter()
    result = adapter.fetch()[0]
    assert result.description == "Jazz Night this Friday! @jazzclub"


def test_image_url_populated():
    adapter = _make_adapter()
    result = adapter.fetch()[0]
    assert result.image_url == "https://cdn.example.com/img.jpg"


def test_raises_on_http_error():
    from src.ingestion.social.apify import ApifyAdapter

    mock_session = MagicMock()
    mock_session.get.return_value.raise_for_status.side_effect = Exception("HTTP 429")

    adapter = ApifyAdapter(
        api_key="fake-key",
        handles=["@testvenue"],
        session=mock_session,
        get_now=lambda: FIXED_NOW,
    )
    with pytest.raises(Exception):
        adapter.fetch()
