"""Unit tests for ImageFetcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from datetime import datetime, timezone

import pytest
import requests as req

from src.processing.image_fetcher import DATA_DERIVED_POLICY, HttpImageFetcher, ImageFetchError, ImageFetcher
from tests.support.network import fetcher_policy


def _make_response(status_code: int, content: bytes = b""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    return resp


IMAGE_URL = "http://cdn.example.com/image.png"
NOW = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)


def _image_fetcher(session, *, max_attempts: int = 1) -> HttpImageFetcher:
    """The real fetcher over a faked session.

    The policy is named at the call site — an image URL points at whatever CDN
    a venue happens to use, so those hosts cannot be listed in config. That is
    a decision somebody made, not a catch-all.
    """
    return HttpImageFetcher(
        session,
        fetcher_policy(
            urls=IMAGE_URL, now=NOW, max_attempts=max_attempts, policy_name=DATA_DERIVED_POLICY
        ),
    )


def test_fetch_returns_bytes_on_200():
    session = MagicMock()
    session.get.return_value = _make_response(200, content=b"\x89PNG image data")

    assert _image_fetcher(session).fetch(IMAGE_URL) == b"\x89PNG image data"


def test_the_timeout_comes_from_the_policy():
    """It was a default argument no test ever set, so production was the only
    path that chose one."""
    session = MagicMock()
    session.get.return_value = _make_response(200, content=b"x")

    _image_fetcher(session).fetch(IMAGE_URL)

    assert session.get.call_args.kwargs["timeout"] == pytest.approx(30.0)


def test_fetch_raises_on_non_200():
    session = MagicMock()
    session.get.return_value = _make_response(404)

    with pytest.raises(ImageFetchError, match="404"):
        _image_fetcher(session).fetch(IMAGE_URL)


def test_fetch_raises_on_timeout():
    session = MagicMock()
    session.get.side_effect = req.Timeout("timed out")

    with pytest.raises(ImageFetchError, match="timed out"):
        _image_fetcher(session).fetch(IMAGE_URL)


def test_fetch_raises_on_connection_error():
    session = MagicMock()
    session.get.side_effect = req.ConnectionError("refused")

    with pytest.raises(ImageFetchError, match="refused"):
        _image_fetcher(session).fetch(IMAGE_URL)


def test_a_transient_failure_is_retried():
    """Nothing retried an image before; a CDN blip lost the image outright."""
    session = MagicMock()
    session.get.side_effect = [
        req.ConnectionError("blip"),
        _make_response(200, content=b"second time"),
    ]

    assert _image_fetcher(session, max_attempts=2).fetch(IMAGE_URL) == b"second time"


def test_mock_fetcher_satisfies_abc():

    class MockFetcher(ImageFetcher):
        def fetch(self, url: str) -> bytes:
            return b"fake"

    fetcher = MockFetcher()
    assert fetcher.fetch("http://x.com/img.jpg") == b"fake"
