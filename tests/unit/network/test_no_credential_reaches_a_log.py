"""A failed request never writes a credential to the log.

The regression guard on the value layer. It asserts on the *rendered log line*
rather than on any adapter's internals, so it keeps working when an adapter
changes how it authenticates — which is what makes it survive the move to
bearer headers rather than being rewritten by it.

**It is a regression guard and not a driver, and the distinction was earned.**
Written first as the red step for "credentials are minted as `Secret`", it
reported seven passing tests against a codebase with the scrub hook deleted
entirely. The reason is worth keeping: it handed each adapter a `Secret` while
the signatures still said `str`, so `urlencode` called `str()` on it, the
placeholder went into the query string, and the sentinel never reached the URL
at all. The adapters were protected by the test's own choice of type. A test
that supplies the safety it is checking for cannot fail.

What made it bite is the adapters calling `expose_secret()` — the value reaches
the URL, `requests` puts the URL inside `HTTPError`, and only the scrub at the
log boundary keeps it out of the line. Mutation-tested at that point: removing
`scrub` from `_JSONFormatter.format` turns Apify and TMDb red and leaves AMC
green, which is the discrimination the paragraph below claims.

**Why the failure is built the way `requests` builds it.** The credential does
not reach the log through anything we write: `RequestPolicy._log_giving_up` logs
`host`, `label` and `str(error)`, and never a URL. It is `requests` that puts
the fully-built URL — query string included — inside `HTTPError`'s message. A
hand-made `HTTPError("boom")` carries no URL at all, so a test using one would
pass against a codebase leaking on every failed request.

**Why the logger is real.** Scrubbing happens as the line is written. A
recording double captures the message before that and would report a leak
production does not have, so the only stand-in here is the stream.

AMC is in the list as the control: its credential has always travelled in a
header, so it must pass from the first run. An assertion that cannot tell AMC
from the others is not measuring anything.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import urlencode

import pytest
import requests

from src.enrichment.movies import TMDB_HOST, TMDbProvider
from src.ingestion.movies.amc import AMC_HOST, AmcAdapter
from src.ingestion.social.apify import APIFY_HOST, ApifyAdapter
from src.storage.memory.movie_cache import InMemoryMovieCache
from src.utils.logging import get_logger
from src.utils.secret import Secret
from tests.support.network import fetcher_for, fetcher_policy

FIXED_NOW = datetime(2026, 8, 19, 2, 0, 0, tzinfo=timezone.utc)

#: Long enough to be registered, and unmistakable in a haystack.
SENTINEL = "sentinel-credential-4d9f2a7c1e"


def _failing_session(status: int = 401) -> Any:
    """A session that fails the way `requests` fails.

    The URL is assembled from `params` exactly as `requests` assembles it, and
    the error is raised by a real `Response`, so the message carries whatever
    the caller put in the query string. Nothing here hardcodes the leak — it
    reproduces the mechanism and lets the adapter decide whether there is one.
    """

    def _fail(url: str, **kwargs: Any) -> Any:
        params = kwargs.get("params")
        response = requests.Response()
        response.status_code = status
        response.reason = "Unauthorized"
        response.url = f"{url}?{urlencode(params)}" if params else url
        response._content = b'{"status_message": "Invalid API key"}'
        response.raise_for_status()
        return response  # pragma: no cover - raise_for_status always raises

    session = MagicMock()
    session.get.side_effect = _fail
    session.post.side_effect = _fail
    return session


def _log_to() -> tuple[Any, io.StringIO]:
    """A real structured logger over a stream, and the stream to read back."""
    stream = io.StringIO()
    return get_logger(f"test.credentials.{id(stream)}", stream=stream), stream


def _lines(stream: io.StringIO) -> list[str]:
    stream.seek(0)
    return [json.loads(line)["message"] for line in stream if line.strip()]


def _drive_apify(logger: Any) -> None:
    ApifyAdapter(
        Secret(SENTINEL),
        ["somebar"],
        fetcher_for(
            _failing_session(),
            urls=f"https://{APIFY_HOST}/v2/acts",
            now=FIXED_NOW,
            logger=logger,
        ),
        get_now=lambda: FIXED_NOW,
    ).fetch()


def _drive_tmdb(logger: Any) -> None:
    TMDbProvider(
        Secret(SENTINEL),
        session=_failing_session(),
        policy=fetcher_policy(
            urls=f"https://{TMDB_HOST}/3", now=FIXED_NOW, logger=logger
        ),
        movie_cache=InMemoryMovieCache(),
        cache_ttl=timedelta(days=7),
        get_now=lambda: FIXED_NOW,
    ).fetch("Dune", 2021)


def _drive_amc(logger: Any) -> None:
    AmcAdapter(
        Secret(SENTINEL),
        "01970",
        session=_failing_session(),
        policy=fetcher_policy(
            urls=f"https://{AMC_HOST}/graphql", now=FIXED_NOW, logger=logger
        ),
        get_now=lambda: FIXED_NOW,
        uses_content_id=lambda source: False,
    ).fetch()


#: Every adapter that holds a credential, and how to make it fail.
DRIVERS = {"apify": _drive_apify, "tmdb": _drive_tmdb, "amc": _drive_amc}


@pytest.mark.parametrize("provider", sorted(DRIVERS))
def test_a_failed_request_logs_no_credential(provider: str) -> None:
    logger, stream = _log_to()

    try:
        DRIVERS[provider](logger)
    except Exception:  # noqa: BLE001 - the failure is the point; the log is the assertion
        pass

    written = "\n".join(_lines(stream))
    assert SENTINEL not in written, f"{provider} wrote its credential to the log"


@pytest.mark.parametrize("provider", sorted(DRIVERS))
def test_the_failure_is_still_reported(provider: str) -> None:
    """The guard must not be satisfied by a log line that never got written."""
    logger, stream = _log_to()

    try:
        DRIVERS[provider](logger)
    except Exception:  # noqa: BLE001
        pass

    assert _lines(stream), f"{provider} logged nothing, so the guard proved nothing"


def test_the_sentinel_would_be_visible_if_it_leaked():
    """The assertion can fail: an unprotected value in the same path is found.

    Without this, every case above passes equally well against a logger that
    writes nothing and a scrub that redacts everything.
    """
    logger, stream = _log_to()
    session = _failing_session()

    fetcher = fetcher_for(
        session, urls="https://example.test/thing", now=FIXED_NOW, logger=logger
    )
    with pytest.raises(requests.HTTPError):
        fetcher.get(
            "https://example.test/thing",
            label="control",
            params={"q": "not-a-secret-9f3a2c7e5b"},
            cache_key="https://example.test/thing",
        )

    assert "not-a-secret-9f3a2c7e5b" in "\n".join(_lines(stream))
