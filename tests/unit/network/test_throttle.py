"""The per-host throttle: spacing requests without a real clock or a real sleep."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.network.throttle import InMemoryThrottle

START = datetime(2026, 8, 17, 2, 0, tzinfo=timezone.utc)


class FakeClock:
    """An injected clock whose `sleep` advances it, as a real one would.

    Time is an external boundary, so faking it is the injectable-time rule
    rather than an exception to "a double may not reimplement".
    """

    def __init__(self) -> None:
        self.now = START
        self.slept: list[float] = []

    def get_now(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def throttle(clock: FakeClock) -> InMemoryThrottle:
    return InMemoryThrottle(get_now=clock.get_now, sleep=clock.sleep)


def test_the_first_request_to_a_host_does_not_wait(throttle, clock):
    throttle.acquire("api.example.com", min_interval_seconds=2.0)

    assert clock.slept == []


def test_a_second_request_waits_out_the_remainder(throttle, clock):
    throttle.acquire("api.example.com", min_interval_seconds=2.0)
    clock.now += timedelta(seconds=0.5)

    throttle.acquire("api.example.com", min_interval_seconds=2.0)

    assert clock.slept == [pytest.approx(1.5)]


def test_a_request_after_the_interval_does_not_wait(throttle, clock):
    throttle.acquire("api.example.com", min_interval_seconds=2.0)
    clock.now += timedelta(seconds=5)

    throttle.acquire("api.example.com", min_interval_seconds=2.0)

    assert clock.slept == []


def test_hosts_are_spaced_independently(throttle, clock):
    """A slow venue website must not make TMDb wait, and the reverse."""
    throttle.acquire("slow-venue.example", min_interval_seconds=10.0)

    throttle.acquire("api.themoviedb.org", min_interval_seconds=0.05)

    assert clock.slept == []


def test_two_callers_of_one_host_are_one_conversation(throttle, clock):
    """The reason the throttle is per host and shared, not per provider.

    Weather and air quality are separate providers pointed at the same
    Open-Meteo host; from the server's side they are one client.
    """
    throttle.acquire("api.open-meteo.com", min_interval_seconds=1.0)

    throttle.acquire("api.open-meteo.com", min_interval_seconds=1.0)

    assert clock.slept == [pytest.approx(1.0)]


def test_spacing_is_measured_from_the_wait_that_just_ended(throttle, clock):
    """Three back-to-back calls space evenly rather than collapsing.

    Recording the time observed *before* sleeping would leave the third call
    thinking the interval had already elapsed, so a burst of N would pay one
    wait between them all instead of N-1.
    """
    for _ in range(3):
        throttle.acquire("api.example.com", min_interval_seconds=2.0)

    assert clock.slept == [pytest.approx(2.0), pytest.approx(2.0)]


def test_a_zero_interval_never_waits(throttle, clock):
    """Localhost is exempt by having no interval, not by a special case."""
    throttle.acquire("localhost", min_interval_seconds=0.0)
    throttle.acquire("localhost", min_interval_seconds=0.0)

    assert clock.slept == []
