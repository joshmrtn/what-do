"""Builds the one way out of this process to somebody else's server.

Every composition root — the batch, the read path, the CLI — goes through here,
so there is one place that decides what a request costs and one place to change
it. Building a policy inline at each root is how eleven slightly different
politeness settings appear, which is the failure the adapter exists to prevent.

`time.sleep` and `random.random` are injected rather than reached for inside the
policy, because a default only production ever reaches is untested by
construction — the shape that killed the first live fetch through
`get_now=datetime.now`.
"""

from __future__ import annotations

import random
import time
from datetime import datetime
from typing import Callable

import requests

from src.config import AppConfig
from src.enrichment.air_quality import AIR_QUALITY_HOST, OpenMeteoAirQualityProvider
from src.enrichment.weather import OPEN_METEO_HOST, OpenMeteoProvider
from src.network.http import HttpFetcher
from src.network.policy import RequestPolicy
from src.network.throttle import InMemoryThrottle
from src.storage.protocols import HttpCache, DayCache
from src.utils.logging import StructuredLogger


def build_request_policy(
    config: AppConfig,
    *,
    get_now: Callable[[], datetime],
    logger: StructuredLogger | None = None,
) -> RequestPolicy:
    """The throttle, retry schedule and timeouts every caller shares.

    One throttle across every provider, because two callers pointed at one host
    are one conversation from the server's side.
    """
    return RequestPolicy(
        network=config.network,
        throttle=InMemoryThrottle(get_now=get_now, sleep=time.sleep),
        sleep=time.sleep,
        random=random.random,
        logger=logger,
    )


def build_http_fetcher(
    config: AppConfig,
    *,
    http_cache: HttpCache,
    get_now: Callable[[], datetime],
    policy: RequestPolicy | None = None,
    logger: StructuredLogger | None = None,
) -> HttpFetcher:
    """One polite conditional GET, shared by every source that fetches a document.

    Args:
        config: Read for the host policies.
        http_cache: Where bodies and validators are kept, so politeness
            survives a restart.
        get_now: Injected clock.
        policy: Share one with the other providers in the same process. Built
            here when a caller has no other use for it.
        logger: Structured logger.
    """
    return HttpFetcher(
        session=requests.Session(),
        network=config.network,
        policy=policy
        if policy is not None
        else build_request_policy(config, get_now=get_now, logger=logger),
        http_cache=http_cache,
        get_now=get_now,
        logger=logger,
    )


def build_weather_provider(
    config: AppConfig,
    *,
    weather_cache: DayCache,
    get_now: Callable[[], datetime],
    policy: RequestPolicy | None = None,
    logger: StructuredLogger | None = None,
) -> OpenMeteoProvider:
    """The forecast provider, with its lifetime read from its host's policy.

    The lifetime has one home — `network.policies.open_meteo.cache_ttl_seconds`
    — and it is resolved here rather than inside the provider so that an
    unassigned host fails at composition, where the message can name it, rather
    than on the first fetch of a nightly run.
    """
    return OpenMeteoProvider(
        session=requests.Session(),
        policy=policy
        if policy is not None
        else build_request_policy(config, get_now=get_now, logger=logger),
        weather_cache=weather_cache,
        cache_ttl=config.network.for_host(OPEN_METEO_HOST).cache_ttl,
        get_now=get_now,
    )


def build_air_quality_provider(
    config: AppConfig,
    *,
    air_quality_cache: DayCache,
    get_now: Callable[[], datetime],
    policy: RequestPolicy | None = None,
    logger: StructuredLogger | None = None,
) -> OpenMeteoAirQualityProvider:
    """The air quality provider, with the cache it has never had.

    Its host is not the forecast's. They belong to one provider and share a
    policy, but they are separate services with **different horizons**, and the
    bound that expresses that stays with the caller.
    """
    return OpenMeteoAirQualityProvider(
        session=requests.Session(),
        policy=policy
        if policy is not None
        else build_request_policy(config, get_now=get_now, logger=logger),
        air_quality_cache=air_quality_cache,
        cache_ttl=config.network.for_host(AIR_QUALITY_HOST).cache_ttl,
        get_now=get_now,
    )
