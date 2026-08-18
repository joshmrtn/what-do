"""What a movie provider said, including that it had nothing.

Exists because **a miss must be cacheable**. `RequestPolicy.call` reads `None`
from a cache strategy as "nothing stored" and calls anyway, so a provider that
answered `None` for an unknown title would ask again on every run, for ever —
the forecast-horizon failure in a different costume, and TMDb is the provider
where a miss is the *common* answer.

A `MovieLookup` is an object whether or not it found anything, so the absence
can be stored and served like any other answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MovieLookup:
    """One answer from a movie provider.

    `metadata` is None when the provider was asked and had nothing — which is a
    real answer, not a failure. A failure raises instead, so the policy can
    decide whether it is worth another attempt.
    """

    metadata: dict[str, Any] | None

    @property
    def found(self) -> bool:
        """Whether the provider recognised the title."""
        return self.metadata is not None
