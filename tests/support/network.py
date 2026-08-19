"""Builds a real `HttpFetcher` around a faked session.

Every calendar-style adapter fetches through the network policy now, so its
tests need one wired up. Wiring it in ten places would give ten slightly
different politeness policies in the tests — the exact failure the adapter
exists to prevent in `src/`.

Only the **session** is a double here. The throttle, the policy, the cache
strategy and the fetcher are all the real objects, so a test exercises the real
retry, the real conditional request and the real lifetime. The transport is the
boundary, and it is the only thing standing in for something else.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit

from src.config import NetworkConfig, NetworkPolicy
from src.network.http import HttpFetcher
from src.network.policy import RequestPolicy
from src.network.throttle import InMemoryThrottle
from src.storage.memory.http_cache import InMemoryHttpCache

#: Named after what it covers, like every policy in `config.yaml`. Attempts are
#: deliberately more than one, so a test that means to exercise retry can.
TEST_POLICY = "test_sources"


def network_for(
    urls: str | Iterable[str],
    *,
    cache_ttl: timedelta | None = timedelta(hours=6),
    max_attempts: int = 3,
    timeout_seconds: float = 30.0,
    policy_name: str = TEST_POLICY,
) -> NetworkConfig:
    """One policy, assigned to every host in `urls`.

    Named `network_for` rather than `test_network`: pytest collects any
    module-level `test_*` it can see, so the second name made an imported
    factory look like a failing test wherever it was used.

    Hosts are assigned rather than defaulted, because there is no default to
    fall back on — that is the design, and a test config that quietly acquired
    one would stop testing it.
    """
    if isinstance(urls, str):
        urls = [urls]
    hosts = {}
    for url in urls:
        host = urlsplit(url).hostname
        if host is not None:
            hosts[host] = policy_name

    return NetworkConfig(
        policies={
            policy_name: NetworkPolicy(
                min_interval_seconds=0.0,
                timeout_seconds=timeout_seconds,
                max_attempts=max_attempts,
                backoff_base_seconds=0.0,
                backoff_max_seconds=0.0,
                cache_ttl=cache_ttl,
            )
        },
        hosts=hosts,
    )


def fetcher_policy(
    *,
    urls: str | Iterable[str],
    now: datetime | Callable[[], datetime],
    cache_ttl: timedelta | None = timedelta(hours=6),
    max_attempts: int = 3,
    timeout_seconds: float = 30.0,
    sleeps: list[float] | None = None,
    policy_name: str = TEST_POLICY,
    logger: Any = None,
) -> RequestPolicy:
    """A real policy over a test network config.

    For a provider that speaks JSON rather than documents: it brings its own
    cache strategy — keyed on what identifies *its* request, which the policy
    never sees — so it needs the policy, not the fetcher.
    """
    clock: Callable[[], datetime] = now if callable(now) else (lambda: now)
    network = network_for(
        urls,
        cache_ttl=cache_ttl,
        max_attempts=max_attempts,
        timeout_seconds=timeout_seconds,
        policy_name=policy_name,
    )
    record = sleeps.append if sleeps is not None else (lambda seconds: None)

    return RequestPolicy(
        network=network,
        throttle=InMemoryThrottle(get_now=clock, sleep=record),
        sleep=record,
        random=lambda: 0.5,
        logger=logger,
    )


def fetcher_for(
    session: Any,
    *,
    urls: str | Iterable[str],
    http_cache: Any = None,
    now: datetime | Callable[[], datetime],
    cache_ttl: timedelta | None = timedelta(hours=6),
    max_attempts: int = 3,
    sleeps: list[float] | None = None,
    logger: Any = None,
) -> HttpFetcher:
    """A real fetcher whose only stand-in is the session.

    Args:
        session: The faked transport.
        urls: Every URL the source under test will ask for; their hosts get the
            test policy.
        http_cache: Where bodies and validators go. A fresh in-memory one by
            default.
        now: Fixed instant, or a clock.
        cache_ttl: The policy's lifetime. `None` declares a `never`.
        max_attempts: Total attempts before giving up.
        sleeps: Collects what the policy would have waited, for a test that
            cares. Real time is never spent either way.
        logger: Where the policy reports retries and failures. A test asserting
            on what reaches a log wants a real `StructuredLogger` over a
            `StringIO` here, not a recording double — credentials are scrubbed
            as the line is written, so a double that captures the message
            earlier would report a leak that production does not have.
    """
    clock: Callable[[], datetime] = now if callable(now) else (lambda: now)

    return HttpFetcher(
        session=session,
        network=network_for(urls, cache_ttl=cache_ttl, max_attempts=max_attempts),
        policy=fetcher_policy(
            urls=urls, now=clock, cache_ttl=cache_ttl,
            max_attempts=max_attempts, sleeps=sleeps, logger=logger,
        ),
        http_cache=http_cache if http_cache is not None else InMemoryHttpCache(),
        get_now=clock,
    )
