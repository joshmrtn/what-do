"""Image fetching abstraction for multimodal LLM extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

import requests


class ImageFetchError(Exception):
    """Raised when an image cannot be fetched."""


class ImageFetcher(ABC):
    """Fetches raw image bytes from a URL."""

    @abstractmethod
    def fetch(self, url: str) -> bytes:
        """Fetch image bytes from a URL.

        Args:
            url: The image URL to fetch.

        Returns:
            Raw image bytes.

        Raises:
            ImageFetchError: On network error or non-200 response.
        """


class HttpImageFetcher(ImageFetcher):
    """Fetches images over HTTP using requests.

    Args:
        timeout: Request timeout in seconds.
    """

    def __init__(self, timeout: int = 15) -> None:
        self._timeout = timeout

    def fetch(self, url: str) -> bytes:
        """Fetch image bytes from a URL.

        Args:
            url: The image URL to fetch.

        Returns:
            Raw image bytes.

        Raises:
            ImageFetchError: On network error or non-200 response.
        """
        try:
            resp = requests.get(url, timeout=self._timeout)
        except requests.Timeout as exc:
            raise ImageFetchError(f"timed out fetching {url}: {exc}") from exc
        except requests.ConnectionError as exc:
            raise ImageFetchError(f"refused connecting to {url}: {exc}") from exc

        if resp.status_code != 200:
            raise ImageFetchError(f"HTTP {resp.status_code} fetching {url}")

        return resp.content
