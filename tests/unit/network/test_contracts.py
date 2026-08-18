"""One set of assertions run against every implementation of the two seams.

`RequestPolicy` takes exactly two things injected per provider — a transient
check and a cache strategy — and both are the kind of thing that rots quietly.
A second implementation that *almost* honours the interface fails nowhere: the
policy calls it, gets something plausible, and carries on.

That is why this file is parametrised rather than written per implementation.
It is the mechanism that has kept `InMemory*Repository` honest while the stage
fakes drifted, and every new provider adds its implementation to the lists here
rather than bringing tests of its own.

**The contracts are deliberately weak.** A cache that stores nothing is a
legitimate implementation — `NullCache` is a decision, not a stub — so the
universal contract cannot say "what goes in comes out". What it *can* say is
that nothing comes out that never went in, which is the property the policy
actually relies on when it returns a cached value instead of calling.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import requests

from src.network.http import HttpDocument, UrlCache, requests_transient_check
from src.network.protocols import CacheStrategy, NullCache, TransientCheck
from src.storage.memory.http_cache import InMemoryHttpCache

NOW = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)
VALUE = HttpDocument(body="a body", etag='"abc"', last_modified=None)


# ---------------------------------------------------------------------------
# Every CacheStrategy implementation
# ---------------------------------------------------------------------------


def _null_cache() -> CacheStrategy[HttpDocument]:
    return NullCache(reason="a prompt is not a cacheable resource")


def _url_cache() -> CacheStrategy[HttpDocument]:
    return UrlCache(
        url="https://example.org/events",
        http_cache=InMemoryHttpCache(),
        get_now=lambda: NOW,
        ttl=timedelta(hours=6),
    )


#: Every implementation, including the ones that deliberately keep nothing.
ALL_CACHES = [_null_cache, _url_cache]

#: Those that claim to store. `NullCache` is absent by design, not by omission.
STORING_CACHES = [_url_cache]


@pytest.mark.parametrize("build", ALL_CACHES, ids=lambda b: b.__name__)
def test_an_empty_cache_is_a_miss_not_an_error(build):
    """The policy calls `get` before anything has ever been stored."""
    assert build().get() is None


@pytest.mark.parametrize("build", ALL_CACHES, ids=lambda b: b.__name__)
def test_put_accepts_a_value_and_answers_nothing(build):
    """`put` is fire-and-forget. A strategy that returned a value would tempt a
    caller into reading it, and half of them would then be wrong."""
    assert build().put(VALUE) is None


@pytest.mark.parametrize("build", ALL_CACHES, ids=lambda b: b.__name__)
def test_nothing_comes_out_that_never_went_in(build):
    """The property the policy actually leans on.

    It returns a cached value *instead of* calling, so a strategy that invented
    one would put a fabricated response into the pipeline with nothing
    downstream able to tell.
    """
    cache = build()
    cache.put(VALUE)
    got = cache.get()
    assert got is None or got == VALUE


@pytest.mark.parametrize("build", STORING_CACHES, ids=lambda b: b.__name__)
def test_a_storing_cache_returns_what_it_was_given(build):
    cache = build()
    cache.put(VALUE)
    assert cache.get() == VALUE


@pytest.mark.parametrize("build", STORING_CACHES, ids=lambda b: b.__name__)
def test_a_storing_cache_keeps_the_last_write(build):
    """A refetch replaces; it does not accumulate versions behind the key."""
    cache = build()
    cache.put(VALUE)
    newer = HttpDocument(body="newer", etag='"def"', last_modified=None)
    cache.put(newer)
    assert cache.get() == newer


# ---------------------------------------------------------------------------
# Every TransientCheck implementation
# ---------------------------------------------------------------------------


def _requests_check() -> TransientCheck:
    return requests_transient_check(get_now=lambda: NOW)


#: Phase 5 adds the SDK check here. One implementation pins little; the list is
#: the point, because the second one arrives already held to this.
ALL_CHECKS = [_requests_check]

#: Failures every transport can suffer, plus one that is ours rather than any
#: transport's.
ANY_ERROR = [
    ValueError("not a transport failure"),
    TypeError("ours"),
    RuntimeError("unclassifiable"),
    requests.Timeout("too slow"),
    requests.ConnectionError("refused"),
    KeyboardInterrupt(),
]


@pytest.mark.parametrize("build", ALL_CHECKS, ids=lambda b: b.__name__)
@pytest.mark.parametrize("error", ANY_ERROR, ids=lambda e: type(e).__name__)
def test_a_check_answers_for_any_failure_at_all(build, error):
    """It runs inside the policy's `except`, so raising there would replace the
    real failure with one from the code deciding what to do about it."""
    advice = build()(error)
    assert isinstance(advice.retry, bool)


@pytest.mark.parametrize("build", ALL_CHECKS, ids=lambda b: b.__name__)
@pytest.mark.parametrize("error", ANY_ERROR, ids=lambda e: type(e).__name__)
def test_a_wait_is_never_negative(build, error):
    """The policy sleeps on this number."""
    seconds = build()(error).retry_after_seconds
    assert seconds is None or seconds >= 0


@pytest.mark.parametrize("build", ALL_CHECKS, ids=lambda b: b.__name__)
@pytest.mark.parametrize("error", ANY_ERROR, ids=lambda e: type(e).__name__)
def test_no_wait_is_named_for_something_we_will_not_retry(build, error):
    """A delay attached to a refusal describes a wait that never happens, and
    reads to anyone debugging as though the server asked for one."""
    advice = build()(error)
    if not advice.retry:
        assert advice.retry_after_seconds is None


@pytest.mark.parametrize("build", ALL_CHECKS, ids=lambda b: b.__name__)
def test_an_error_from_no_transport_is_not_retried(build):
    """Our own bug fails identically however politely it is repeated."""
    assert not build()(TypeError("ours")).retry
