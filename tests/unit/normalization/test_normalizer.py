"""Unit tests for the normalization engine."""

from __future__ import annotations

import unicodedata
import uuid
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.models.event_candidate import EventCandidate
from src.normalization.normalizer import NormalizationEngine


_TZ = "America/New_York"
_ZONEINFO = ZoneInfo(_TZ)


def _candidate(**kwargs) -> EventCandidate:
    defaults = dict(
        id=str(uuid.uuid4()),
        source="@test_source",
        source_type="apify",
        discovered_at=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        title="Jazz Night",
        description="A great jazz show.",
        start_time=datetime(2025, 6, 15, 20, 0, 0, tzinfo=timezone.utc),
    )
    defaults.update(kwargs)
    return EventCandidate(**defaults)


def _now():
    return datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def _normalize(candidate: EventCandidate) -> NormalizationResult:
    engine = NormalizationEngine(timezone_name=_TZ)
    return engine.normalize([candidate], get_now=_now)


# --- timestamp normalization ---

def test_timezone_aware_start_time_converted_to_config_tz():
    utc_start = datetime(2025, 6, 15, 20, 0, 0, tzinfo=timezone.utc)
    result = _normalize(_candidate(start_time=utc_start))
    assert result.events[0].start_time.tzinfo is not None
    assert result.events[0].start_time == utc_start.astimezone(_ZONEINFO)


def test_timezone_aware_end_time_converted_to_config_tz():
    utc_end = datetime(2025, 6, 15, 23, 0, 0, tzinfo=timezone.utc)
    result = _normalize(_candidate(end_time=utc_end))
    assert result.events[0].end_time.tzinfo is not None
    assert result.events[0].end_time == utc_end.astimezone(_ZONEINFO)


def test_naive_start_time_gets_config_tz_attached():
    naive_start = datetime(2025, 6, 15, 20, 0, 0)  # no tzinfo
    result = _normalize(_candidate(start_time=naive_start))
    event = result.events[0]
    assert event.start_time.tzinfo is not None
    assert event.start_time == naive_start.replace(tzinfo=_ZONEINFO)


def test_none_start_time_stays_none():
    result = _normalize(_candidate(start_time=None))
    assert result.events[0].start_time is None


def test_none_end_time_stays_none():
    result = _normalize(_candidate(end_time=None))
    assert result.events[0].end_time is None


# --- venue name normalization ---

def test_venue_title_cased():
    result = _normalize(_candidate(venue="the vault lounge"))
    assert result.events[0].venue == "The Vault Lounge"


def test_venue_article_suffix_moved_to_prefix():
    result = _normalize(_candidate(venue="vault, the"))
    assert result.events[0].venue == "The Vault"


def test_venue_article_suffix_case_insensitive():
    result = _normalize(_candidate(venue="VAULT LOUNGE, THE"))
    assert result.events[0].venue == "The Vault Lounge"


def test_venue_already_canonical_unchanged():
    result = _normalize(_candidate(venue="The Vault Lounge"))
    assert result.events[0].venue == "The Vault Lounge"


def test_venue_none_stays_none():
    result = _normalize(_candidate(venue=None))
    assert result.events[0].venue is None


def test_venue_a_suffix_moved_to_prefix():
    result = _normalize(_candidate(venue="Coffee Shop, A"))
    assert result.events[0].venue == "A Coffee Shop"


def test_venue_an_suffix_moved_to_prefix():
    result = _normalize(_candidate(venue="Honest Pub, An"))
    assert result.events[0].venue == "An Honest Pub"


# --- text normalization ---

def test_description_excess_whitespace_stripped():
    result = _normalize(_candidate(description="  lots   of   spaces  "))
    assert result.events[0].description == "lots of spaces"


def test_description_non_breaking_space_replaced():
    result = _normalize(_candidate(description="hello\xa0world"))
    assert result.events[0].description == "hello world"


def test_description_nfc_normalized():
    # e + combining acute = é in NFD form
    nfd = unicodedata.normalize("NFD", "café")
    result = _normalize(_candidate(description=nfd))
    assert result.events[0].description == "café"
    assert unicodedata.is_normalized("NFC", result.events[0].description)


def test_title_whitespace_normalized():
    result = _normalize(_candidate(title="  Jazz   Night  "))
    assert result.events[0].title == "Jazz Night"


def test_none_description_stays_none():
    result = _normalize(_candidate(description=None))
    assert result.events[0].description is None


# --- malformed record policy ---

def test_missing_both_title_and_start_time_discarded():
    cand = _candidate(title=None, start_time=None)
    result = _normalize(cand)
    assert len(result.events) == 0
    assert len(result.discards) == 1


def test_discard_includes_candidate_reference():
    cand = _candidate(title=None, start_time=None, id="bad-cand-id")
    result = _normalize(cand)
    assert result.discards[0].candidate.id == "bad-cand-id"


def test_discard_includes_reason():
    cand = _candidate(title=None, start_time=None)
    result = _normalize(cand)
    assert "title" in result.discards[0].reason
    assert "start_time" in result.discards[0].reason


def test_missing_only_title_flagged_not_discarded():
    result = _normalize(_candidate(title=None))
    assert len(result.events) == 1
    assert result.events[0].metadata.get("missing_title") is True


def test_missing_only_start_time_flagged_not_discarded():
    result = _normalize(_candidate(start_time=None))
    assert len(result.events) == 1
    assert result.events[0].metadata.get("missing_start_time") is True


def test_complete_record_has_no_missing_flags():
    result = _normalize(_candidate())
    assert "missing_title" not in result.events[0].metadata
    assert "missing_start_time" not in result.events[0].metadata


def test_discard_count_correct_for_multiple_candidates():
    good = _candidate(title="Good Event")
    bad = _candidate(title=None, start_time=None)
    engine = NormalizationEngine(timezone_name=_TZ)
    result = engine.normalize([good, bad], get_now=_now)
    assert len(result.events) == 1
    assert len(result.discards) == 1


# --- output structure ---

def test_normalized_event_has_source_candidate_id():
    cand = _candidate()
    result = _normalize(cand)
    assert cand.id in result.events[0].source_event_candidates


def test_normalized_event_source_type_matches_candidate():
    result = _normalize(_candidate(source_type="cinema_veezi"))
    assert result.events[0].source_type == "cinema_veezi"


def test_normalized_event_has_event_id():
    result = _normalize(_candidate())
    assert result.events[0].event_id != ""


def test_normalized_event_timestamps_set():
    result = _normalize(_candidate())
    assert result.events[0].created_at == _now()
    assert result.events[0].updated_at == _now()


def test_multiple_candidates_all_normalized():
    candidates = [_candidate(title=f"Event {i}") for i in range(3)]
    engine = NormalizationEngine(timezone_name=_TZ)
    result = engine.normalize(candidates, get_now=_now)
    assert len(result.events) == 3
    assert len(result.discards) == 0
