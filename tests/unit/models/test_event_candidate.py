"""An EventCandidate holds instants, in one representation.

The raw layer is the only place a publisher's own choice of wire format used to
survive: Google Calendar's `basic.ics` emits `DTSTART:…Z` for some events and
`DTSTART;TZID=…` for others *in the same feed*, so `event_candidates` carried
three different offsets and `for_window` compared them as text. Text order only
matches chronological order when every value shares one offset.

`Event` is deliberately the other way round — `_normalize_timestamp` puts it in
the *local* zone, because that is where local reasoning belongs and `.date()`
and `strftime` read the zone rather than the text. Two conventions, one
boundary between them.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.models.event_candidate import EventCandidate

_EASTERN = timezone(timedelta(hours=-4))
_UTC_NOON = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def _candidate(**kwargs) -> EventCandidate:
    defaults = dict(id="c1", source="s", source_type="t", discovered_at=_UTC_NOON)
    defaults.update(kwargs)
    return EventCandidate(**defaults)


class TestInstantsAreCanonical:
    def test_an_offset_start_time_is_converted_to_utc(self):
        """Same instant, one representation — not a different time."""
        local = datetime(2026, 8, 14, 20, 30, tzinfo=_EASTERN)

        candidate = _candidate(start_time=local)

        assert candidate.start_time == local
        assert candidate.start_time.utcoffset() == timedelta(0)
        assert candidate.start_time.isoformat() == "2026-08-15T00:30:00+00:00"

    def test_every_timestamp_field_is_canonical(self):
        """One field converted and the rest left alone is the worse bug: the
        comparison looks fixed while the next field to be compared still lies."""
        local = datetime(2026, 8, 14, 20, 30, tzinfo=_EASTERN)

        candidate = _candidate(
            discovered_at=local,
            start_time=local,
            end_time=local + timedelta(hours=2),
            raw_published_at=local - timedelta(days=1),
            last_seen_at=local + timedelta(days=1),
        )

        for field in ("discovered_at", "start_time", "end_time",
                      "raw_published_at", "last_seen_at"):
            value = getattr(candidate, field)
            assert value.utcoffset() == timedelta(0), f"{field} kept its offset"

    def test_a_value_already_in_utc_is_untouched(self):
        assert _candidate(start_time=_UTC_NOON).start_time == _UTC_NOON

    def test_absent_timestamps_stay_absent(self):
        candidate = _candidate(start_time=None, end_time=None, raw_published_at=None)

        assert (candidate.start_time, candidate.end_time) == (None, None)
        assert candidate.raw_published_at is None

    def test_a_naive_start_time_is_left_exactly_as_it_is(self):
        """Tolerated on purpose, not overlooked.

        A naive value carries no instant to convert, and only the adapter knows
        which zone it meant. Ingestion must not crash on one — that is the
        failure that killed the first live fetch — and `_normalize_timestamp`
        resolves it a layer on by attaching the configured zone. Converting
        here would misplace the event silently instead.
        """
        naive = datetime(2026, 8, 14, 20, 30)

        candidate = _candidate(start_time=naive)

        assert candidate.start_time == naive
        assert candidate.start_time.tzinfo is None

    def test_sorting_as_text_now_matches_sorting_as_time(self):
        """The property the whole change exists for. Two events an hour apart,
        published by feeds that disagree about representation: as stored text
        the earlier one used to sort second."""
        earlier = _candidate(id="a", start_time=datetime(2026, 8, 14, 20, 30, tzinfo=_EASTERN))
        later = _candidate(id="b", start_time=datetime(2026, 8, 15, 1, 30, tzinfo=timezone.utc))

        assert earlier.start_time < later.start_time
        assert earlier.start_time.isoformat() < later.start_time.isoformat()
