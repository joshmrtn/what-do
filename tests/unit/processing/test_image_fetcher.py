"""Unit tests for ImageFetcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests as req

from src.processing.image_fetcher import HttpImageFetcher, ImageFetchError, ImageFetcher


def _make_response(status_code: int, content: bytes = b""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    return resp


def test_fetch_returns_bytes_on_200():

    fetcher = HttpImageFetcher(timeout=10)
    mock_resp = _make_response(200, content=b"\x89PNG image data")

    with patch("requests.get", return_value=mock_resp):
        result = fetcher.fetch("http://example.com/image.png")

    assert result == b"\x89PNG image data"


def test_fetch_raises_on_non_200():

    fetcher = HttpImageFetcher(timeout=10)
    mock_resp = _make_response(404)

    with patch("requests.get", return_value=mock_resp):
        with pytest.raises(ImageFetchError, match="404"):
            fetcher.fetch("http://example.com/missing.jpg")


def test_fetch_raises_on_timeout():

    fetcher = HttpImageFetcher(timeout=10)

    with patch("requests.get", side_effect=req.Timeout("timed out")):
        with pytest.raises(ImageFetchError, match="timed out"):
            fetcher.fetch("http://example.com/image.jpg")


def test_fetch_raises_on_connection_error():

    fetcher = HttpImageFetcher(timeout=10)

    with patch("requests.get", side_effect=req.ConnectionError("refused")):
        with pytest.raises(ImageFetchError, match="refused"):
            fetcher.fetch("http://example.com/image.jpg")


def test_mock_fetcher_satisfies_abc():

    class MockFetcher(ImageFetcher):
        def fetch(self, url: str) -> bytes:
            return b"fake"

    fetcher = MockFetcher()
    assert fetcher.fetch("http://x.com/img.jpg") == b"fake"
