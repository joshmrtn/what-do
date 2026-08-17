"""The politeness policy: throttle, retry with backoff, timeout, cache, logging.

The policy is deliberately ignorant of transport. It takes a callable that
performs the request, so a vendor SDK goes through it unchanged — configuring
retry on the SDK instead would be the same bug in a new costume, two places to
remember politeness.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from src.config import ConfigError, NetworkConfig, NetworkPolicy
from src.network.policy import RequestPolicy
from src.network.protocols import DO_NOT_RETRY, RETRY_WITH_BACKOFF, RetryAdvice
from src.network.throttle import InMemoryThrottle

HOST = "api.example.com"
START = datetime(2026, 8, 17, 2, 0, tzinfo=timezone.utc)


class FakeClock:
    def __init__(self) -> None:
        self.now = START
        self.slept: list[float] = []

    def get_now(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += timedelta(seconds=seconds)


class RecordingCache:
    """Returns what it was seeded with and records what it was handed.

    A spy, not a mirror: it makes no claim about how any real cache keys itself,
    which is the caller's business and differs per provider.
    """

    def __init__(self, stored: Any = None) -> None:
        self.stored = stored
        self.puts: list[Any] = []

    def get(self) -> Any:
        return self.stored

    def put(self, value: Any) -> None:
        self.puts.append(value)


class RecordingLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def info(self, message: str, *, component: str = "", duration_ms: int = 0) -> None:
        self.messages.append(("info", message))

    def warning(self, message: str, *, component: str = "", duration_ms: int = 0) -> None:
        self.messages.append(("warning", message))

    def error(self, message: str, *, component: str = "", duration_ms: int = 0) -> None:
        self.messages.append(("error", message))


POLICY = "test_policy"


def _limits(
    *,
    min_interval_seconds: float = 1.0,
    timeout_seconds: float = 30.0,
    max_attempts: int = 3,
    backoff_base_seconds: float = 1.0,
    backoff_max_seconds: float = 60.0,
    cache_ttl: timedelta | None = timedelta(hours=1),
) -> NetworkPolicy:
    """A complete policy, for tests about one field at a time.

    `NetworkPolicy` itself has no defaults — a default is a policy nobody
    decided, applied to a host nobody considered. These live in the test because
    a test about backoff should not have to restate a timeout to say what it means.
    """
    return NetworkPolicy(
        min_interval_seconds=min_interval_seconds,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        backoff_base_seconds=backoff_base_seconds,
        backoff_max_seconds=backoff_max_seconds,
        cache_ttl=cache_ttl,
    )


def _network(**overrides: Any) -> NetworkConfig:
    """A config declaring one policy, with `HOST` assigned to it."""
    return NetworkConfig(
        policies={POLICY: _limits(**overrides)}, hosts={HOST: POLICY}
    )


def _never_transient(error: BaseException) -> RetryAdvice:
    return DO_NOT_RETRY


def _always_transient(error: BaseException) -> RetryAdvice:
    return RETRY_WITH_BACKOFF


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def logger() -> RecordingLogger:
    return RecordingLogger()


def _policy(clock: FakeClock, logger: RecordingLogger, network: NetworkConfig | None = None):
    return RequestPolicy(
        network=network or _network(),
        throttle=InMemoryThrottle(get_now=clock.get_now, sleep=clock.sleep),
        sleep=clock.sleep,
        random=lambda: 1.0,
        logger=logger,
    )


class TestTheCacheComesFirst:
    def test_a_hit_is_served_without_performing(self, clock, logger):
        performed: list[float] = []

        result = _policy(clock, logger).call(
            host=HOST,
            perform=lambda timeout: performed.append(timeout) or "fresh",
            is_transient=_never_transient,
            cache=RecordingCache(stored="remembered"),
            label="example",
        )

        assert result == "remembered"
        assert performed == []

    def test_a_hit_does_not_spend_the_hosts_interval(self, clock, logger):
        """A cache exists to remove the call, so it must not also pay for one.

        Throttling a hit would make a warm cache as slow as a cold one and
        silently serialise a batch that is doing no network work at all.
        """
        policy = _policy(clock, logger)
        cache = RecordingCache(stored="remembered")

        policy.call(host=HOST, perform=lambda t: "fresh", is_transient=_never_transient,
                    cache=cache, label="example")
        policy.call(host=HOST, perform=lambda t: "fresh", is_transient=_never_transient,
                    cache=cache, label="example")

        assert clock.slept == []

    def test_a_miss_stores_what_it_fetched(self, clock, logger):
        cache = RecordingCache(stored=None)

        _policy(clock, logger).call(
            host=HOST, perform=lambda t: "fresh", is_transient=_never_transient,
            cache=cache, label="example",
        )

        assert cache.puts == ["fresh"]


class TestTheTimeoutReachesTheCall:
    def test_perform_is_handed_the_hosts_timeout(self, clock, logger):
        """A timeout cannot be imposed on an opaque callable from outside, so
        the policy hands its number in and the transport applies it."""
        network = _network(timeout_seconds=45.0)
        seen: list[float] = []

        _policy(clock, logger, network).call(
            host=HOST, perform=lambda timeout: seen.append(timeout),
            is_transient=_never_transient, cache=RecordingCache(), label="example",
        )

        assert seen == [45.0]

    def test_each_host_gets_its_own_timeout(self, clock, logger):
        """A slow venue site and an API are not owed the same patience."""
        network = NetworkConfig(
            policies={
                "api": _limits(timeout_seconds=5.0),
                "web": _limits(timeout_seconds=60.0),
            },
            hosts={HOST: "api", "slow-venue.example": "web"},
        )
        policy = _policy(clock, logger, network)
        seen: list[float] = []

        for host in (HOST, "slow-venue.example"):
            policy.call(
                host=host, perform=lambda timeout: seen.append(timeout),
                is_transient=_never_transient, cache=RecordingCache(), label="example",
            )

        assert seen == [5.0, 60.0]

    def test_an_undeclared_host_is_refused_at_the_call(self, clock, logger):
        """The config error surfaces where the call is made, naming the host.

        A guess here would be the failure the whole section exists to prevent:
        a host nobody set a policy for, quietly borrowing somebody else's.
        """
        with pytest.raises(ConfigError, match="undeclared.example"):
            _policy(clock, logger).call(
                host="undeclared.example", perform=lambda timeout: "fresh",
                is_transient=_never_transient, cache=RecordingCache(), label="example",
            )


class TestACategoryNamedAtTheCallSite:
    """Hosts that arrive from fetched data cannot be assigned in advance.

    An image URL points at whatever CDN a venue uses, so the caller names the
    policy it was built to use. The throttle still spaces per host — it just
    reads its numbers from a policy nobody had to enumerate hosts for.
    """

    def test_a_named_policy_is_used_instead_of_a_host_assignment(self, clock, logger):
        network = NetworkConfig(
            policies={"data_derived": _limits(timeout_seconds=7.0)}, hosts={}
        )
        seen: list[float] = []

        _policy(clock, logger, network).call(
            host="cdn.unknown.example", policy="data_derived",
            perform=lambda timeout: seen.append(timeout),
            is_transient=_never_transient, cache=RecordingCache(), label="image",
        )

        assert seen == [7.0]

    def test_spacing_is_still_per_host(self, clock, logger):
        """Two images from one CDN wait; two from different CDNs do not.

        Sharing a policy must not collapse into sharing a queue, or every image
        in a batch would serialise behind whichever CDN was slowest.
        """
        network = NetworkConfig(
            policies={"data_derived": _limits(min_interval_seconds=2.0)}, hosts={}
        )
        policy = _policy(clock, logger, network)

        for host in ("cdn-a.example", "cdn-b.example", "cdn-a.example"):
            policy.call(
                host=host, policy="data_derived", perform=lambda timeout: "bytes",
                is_transient=_never_transient, cache=RecordingCache(), label="image",
            )

        assert clock.slept == [pytest.approx(2.0)]

    def test_an_undeclared_policy_name_is_refused(self, clock, logger):
        with pytest.raises(ConfigError, match="images"):
            _policy(clock, logger).call(
                host="cdn.unknown.example", policy="images",
                perform=lambda timeout: "bytes",
                is_transient=_never_transient, cache=RecordingCache(), label="image",
            )


class TestRetry:
    def test_a_transient_failure_is_tried_again(self, clock, logger):
        attempts: list[int] = []

        def perform(timeout: float) -> str:
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("503")
            return "fresh"

        result = _policy(clock, logger).call(
            host=HOST, perform=perform, is_transient=_always_transient,
            cache=RecordingCache(), label="example",
        )

        assert result == "fresh"
        assert len(attempts) == 3

    def test_a_permanent_failure_is_not_tried_again(self, clock, logger):
        """A bad request repeated politely is still a bad request."""
        attempts: list[int] = []

        def perform(timeout: float) -> str:
            attempts.append(1)
            raise RuntimeError("404")

        with pytest.raises(RuntimeError, match="404"):
            _policy(clock, logger).call(
                host=HOST, perform=perform, is_transient=_never_transient,
                cache=RecordingCache(), label="example",
            )

        assert len(attempts) == 1

    def test_exhausting_the_attempts_raises_the_last_error(self, clock, logger):
        network = _network(max_attempts=3)
        attempts: list[int] = []

        def perform(timeout: float) -> str:
            attempts.append(1)
            raise RuntimeError(f"attempt {len(attempts)}")

        with pytest.raises(RuntimeError, match="attempt 3"):
            _policy(clock, logger, network).call(
                host=HOST, perform=perform, is_transient=_always_transient,
                cache=RecordingCache(), label="example",
            )

        assert len(attempts) == 3

    def test_a_failed_call_stores_nothing(self, clock, logger):
        """Caching a failure would serve it back for the whole TTL."""
        cache = RecordingCache()

        with pytest.raises(RuntimeError):
            _policy(clock, logger).call(
                host=HOST, perform=_raising, is_transient=_never_transient,
                cache=cache, label="example",
            )

        assert cache.puts == []

    def test_every_attempt_is_throttled_not_just_the_first(self, clock, logger):
        """A retry storm is the least polite moment to stop being polite.

        Acquiring once outside the loop would let a failing host be hammered
        `max_attempts` times back to back with only the backoff between.
        """
        network = _network(
                max_attempts=2, min_interval_seconds=4.0,
                backoff_base_seconds=1.0, backoff_max_seconds=60.0,
            )

        with pytest.raises(RuntimeError):
            _policy(clock, logger, network).call(
                host=HOST, perform=_raising, is_transient=_always_transient,
                cache=RecordingCache(), label="example",
            )

        # 1.0s of backoff, then the throttle waiting out the rest of its 4s.
        assert clock.slept == [pytest.approx(1.0), pytest.approx(3.0)]


class TestBackoff:
    def test_it_grows_exponentially(self, clock, logger):
        network = _network(
                max_attempts=4, min_interval_seconds=0.0,
                backoff_base_seconds=1.0, backoff_max_seconds=60.0,
            )

        with pytest.raises(RuntimeError):
            _policy(clock, logger, network).call(
                host=HOST, perform=_raising, is_transient=_always_transient,
                cache=RecordingCache(), label="example",
            )

        assert clock.slept == [pytest.approx(1.0), pytest.approx(2.0), pytest.approx(4.0)]

    def test_it_is_capped(self, clock, logger):
        """Uncapped, a dead host stalls the batch for hours instead of failing."""
        network = _network(
                max_attempts=5, min_interval_seconds=0.0,
                backoff_base_seconds=10.0, backoff_max_seconds=25.0,
            )

        with pytest.raises(RuntimeError):
            _policy(clock, logger, network).call(
                host=HOST, perform=_raising, is_transient=_always_transient,
                cache=RecordingCache(), label="example",
            )

        assert clock.slept == [
            pytest.approx(10.0), pytest.approx(20.0),
            pytest.approx(25.0), pytest.approx(25.0),
        ]

    def test_jitter_never_drops_below_half_the_backoff(self, clock, logger):
        """Full jitter can round to no wait at all, which is impolite at exactly
        the moment the server asked for room."""
        network = _network(
                max_attempts=2, min_interval_seconds=0.0,
                backoff_base_seconds=8.0, backoff_max_seconds=60.0,
            )
        policy = RequestPolicy(
            network=network,
            throttle=InMemoryThrottle(get_now=clock.get_now, sleep=clock.sleep),
            sleep=clock.sleep,
            random=lambda: 0.0,
            logger=logger,
        )

        with pytest.raises(RuntimeError):
            policy.call(host=HOST, perform=_raising, is_transient=_always_transient,
                        cache=RecordingCache(), label="example")

        assert clock.slept == [pytest.approx(4.0)]

    def test_retry_after_overrides_the_computed_backoff(self, clock, logger):
        """The server named a number; honour it rather than guessing past it."""
        network = _network(
                max_attempts=2, min_interval_seconds=0.0,
                backoff_base_seconds=1.0, backoff_max_seconds=60.0,
            )

        def advises_retry_after(error: BaseException) -> RetryAdvice:
            return RetryAdvice(retry=True, retry_after_seconds=17.0)

        with pytest.raises(RuntimeError):
            _policy(clock, logger, network).call(
                host=HOST, perform=_raising, is_transient=advises_retry_after,
                cache=RecordingCache(), label="example",
            )

        assert clock.slept == [pytest.approx(17.0)]

    def test_retry_after_is_not_jittered(self, clock, logger):
        """Jittering a number the server chose is second-guessing it."""
        network = _network(
                max_attempts=2, min_interval_seconds=0.0,
                backoff_base_seconds=1.0, backoff_max_seconds=60.0,
            )
        policy = RequestPolicy(
            network=network,
            throttle=InMemoryThrottle(get_now=clock.get_now, sleep=clock.sleep),
            sleep=clock.sleep,
            random=lambda: 0.0,
            logger=logger,
        )

        with pytest.raises(RuntimeError):
            policy.call(
                host=HOST, perform=_raising,
                is_transient=lambda e: RetryAdvice(retry=True, retry_after_seconds=17.0),
                cache=RecordingCache(), label="example",
            )

        assert clock.slept == [pytest.approx(17.0)]


class TestLogging:
    def test_a_retry_says_who_was_asked_and_how_long_we_waited(self, clock, logger):
        """One place that can answer "what did we ask whom, and how often"."""
        network = _network(max_attempts=2, min_interval_seconds=0.0)

        with pytest.raises(RuntimeError):
            _policy(clock, logger, network).call(
                host=HOST, perform=_raising, is_transient=_always_transient,
                cache=RecordingCache(), label="example",
            )

        retries = [m for level, m in logger.messages if level == "warning"]
        assert len(retries) == 1
        assert HOST in retries[0]


def _raising(timeout: float) -> str:
    raise RuntimeError("503 unavailable")
