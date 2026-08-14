"""What a source published, as it published it."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class CandidateVersion:
    """One distinct observed content for a candidate.

    Frozen, because a retained observation that can be edited is the thing this
    exists to prevent.
    """

    candidate_id: str
    #: Digest of `payload`. Also the second half of the primary key, which is
    #: what makes an unchanged re-fetch a no-op rather than a read followed by
    #: a decision.
    content_hash: str
    #: When this content was **first** seen. `EventCandidate.last_seen_at`
    #: already answers "are we still seeing it", and a version that moved its
    #: own stamp could not say when the text actually changed.
    observed_at: datetime
    #: The published fields, exactly as stored. A dict rather than columns so
    #: adding a field to `EventCandidate` cannot leave the history behind — the
    #: payload and the fingerprint are both built from one list.
    payload: dict[str, Any]
