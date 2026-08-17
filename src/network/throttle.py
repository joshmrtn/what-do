"""Spacing requests so a host sees a conversation rather than a burst."""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Protocol


class Throttle(Protocol):
    """Blocks until a host may politely be asked again.

    The interval is passed in rather than held, so the throttle stays ignorant
    of config and one instance can serve every host at its own rate.
    """

    def acquire(self, host: str, *, min_interval_seconds: float) -> None:
        """Return once `min_interval_seconds` has passed since the last request."""
        ...


class InMemoryThrottle:
    """Per-host spacing held in memory for the life of the process.

    In-memory rather than persisted because the two processes are naturally
    separated — the batch runs at ~02:00 and the CLI is used by hand hours later
    — so the combined-rate risk is real but low. Persisting it is a swap behind
    `Throttle`, not a rewrite, if that stops being true.
    """

    def __init__(
        self,
        *,
        get_now: Callable[[], datetime],
        sleep: Callable[[float], None],
    ) -> None:
        """
        Args:
            get_now: Injected clock.
            sleep: Injected delay, so tests never wait in real time.
        """
        self._get_now = get_now
        self._sleep = sleep
        self._last_request: dict[str, datetime] = {}

    def acquire(self, host: str, *, min_interval_seconds: float) -> None:
        """Wait out whatever is left of this host's interval.

        The timestamp recorded is the one observed *after* any wait, not before
        it. Recording the earlier one would leave a third back-to-back call
        believing the interval had already elapsed, so a burst of N would pay a
        single wait between all of them instead of N-1.
        """
        if min_interval_seconds <= 0:
            self._last_request[host] = self._get_now()
            return

        now = self._get_now()
        last = self._last_request.get(host)

        if last is not None:
            remaining = min_interval_seconds - (now - last).total_seconds()
            if remaining > 0:
                self._sleep(remaining)
                now = self._get_now()

        self._last_request[host] = now
