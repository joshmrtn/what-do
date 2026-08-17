"""The two things a provider must bring, and nothing else.

The policy — throttle, retry, backoff, timeout, logging — inspects no URL and
knows no transport. Exactly two concerns are transport-specific, and both are
injected once per provider rather than remembered at each call site:

* **is this failure worth another attempt?** an HTTP status for `requests`, the
  SDK's own exception types for a vendor client;
* **what does this caller cache, and how is it keyed?** TMDb on normalised title
  and year, weather on `(date, latitude, longitude)`, a feed on its URL. A
  single shared table would express the third and mangle the first two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryAdvice:
    """Whether to try again, and whether the server said when.

    Neither field has a default. A default would let a predicate stay silent
    about the thing it exists to decide, and the quiet answer would be indistinguishable
    from a considered one.
    """

    retry: bool
    #: Seconds the server asked us to wait, from `Retry-After`. `None` means it
    #: offered no opinion and the caller should back off on its own schedule.
    retry_after_seconds: float | None


#: A failure that will fail identically next time. A bad request repeated
#: politely is still a bad request.
DO_NOT_RETRY = RetryAdvice(retry=False, retry_after_seconds=None)

#: Worth another attempt, on our own exponential schedule.
RETRY_WITH_BACKOFF = RetryAdvice(retry=True, retry_after_seconds=None)


class TransientCheck(Protocol):
    """Reads one transport's failures and says whether to try again."""

    def __call__(self, error: BaseException) -> RetryAdvice:
        """Advice for a failed attempt."""
        ...


class CacheStrategy(Protocol[T]):
    """A caller's own cache, already bound to the key of this one call.

    Bound rather than keyed here because the policy cannot know what identifies
    a request: it never sees a URL, let alone a normalised film title. The
    caller builds the strategy where its keying and its TTL are both obvious,
    which is the only place either is a real decision.
    """

    def get(self) -> T | None:
        """What was stored for this key and is still fresh, or None."""
        ...

    def put(self, value: T) -> None:
        """Store this result under the key the strategy was built with."""
        ...


@dataclass(frozen=True)
class NullCache(Generic[T]):
    """A caller that deliberately caches nothing, with the reason recorded.

    `reason` is required. A null cache is a real decision — a prompt is not a
    cacheable resource the way a forecast is, and extraction already skips on
    `extraction_input_hash` one layer up, which avoids the call rather than
    replaying its answer. A null cache that goes unexplained is indistinguishable
    from a caller that simply forgot.
    """

    reason: str

    def get(self) -> T | None:
        """Always a miss."""
        return None

    def put(self, value: T) -> None:
        """Discarded, for the reason this strategy records."""
        return None
