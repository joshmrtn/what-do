"""Image fetching abstraction for multimodal LLM extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from datetime import datetime, timezone
from urllib.parse import urlsplit

import requests

from src.network.http import requests_transient_check
from src.network.policy import RequestPolicy
from src.network.protocols import NullCache


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


#: The policy for hosts that arrive from fetched data. An image URL points at
#: whatever CDN a venue happens to use, so these hosts cannot be enumerated in
#: config — but naming the policy **here, at the call site built to use it**
#: keeps it a decision somebody made rather than a catch-all that any
#: unassigned host quietly falls into.
DATA_DERIVED_POLICY = "data_derived"


class HttpImageFetcher(ImageFetcher):
    """Fetches images over HTTP, through the request policy."""

    def __init__(self, session: requests.Session, policy: RequestPolicy) -> None:
        """
        Args:
            session: Injected HTTP session, so tests never reach the network.
            policy: Throttle, retry and timeout. Used directly rather than
                through `HttpFetcher`, because an image is bytes and the
                document fetcher deals in text.
        """
        self._session = session
        self._policy = policy
        self._is_transient = requests_transient_check(get_now=lambda: datetime.now(timezone.utc))

    def fetch(self, url: str) -> bytes:
        """Fetch image bytes from a URL.

        Args:
            url: The image URL to fetch. Its host arrived from fetched data, so
                the policy is named rather than looked up by host.

        Returns:
            Raw image bytes.

        Raises:
            ImageFetchError: On network error or non-200 response.
        """
        host = urlsplit(url).hostname or url
        try:
            return self._policy.call(
                host=host,
                perform=lambda timeout: self._download(url, timeout),
                is_transient=self._is_transient,
                # An image at a URL does not change, so it would cache well —
                # but nothing here re-requests one: the bytes are consumed by
                # the caller in the same pass. Recorded rather than left as an
                # absence somebody has to infer.
                cache=NullCache(
                    reason="image bytes are consumed by the caller in the same "
                    "pass, so there is no second request to serve"
                ),
                label="image",
                policy=DATA_DERIVED_POLICY,
            )
        except requests.Timeout as exc:
            raise ImageFetchError(f"timed out fetching {url}: {exc}") from exc
        except requests.ConnectionError as exc:
            raise ImageFetchError(f"refused connecting to {url}: {exc}") from exc
        except requests.HTTPError as exc:
            raise ImageFetchError(f"HTTP error fetching {url}: {exc}") from exc

    def _download(self, url: str, timeout: float) -> bytes:
        """One attempt. Raises so the policy can decide about trying again."""
        resp = self._session.get(url, timeout=timeout)
        if resp.status_code != 200:
            raise ImageFetchError(f"HTTP {resp.status_code} fetching {url}")
        return bytes(resp.content)
