"""Unit tests for FailoverChain."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.utils.logging import get_logger


def _make_logger():
    return get_logger("test_failover", stream=io.StringIO())


def _make_candidate(n: int):
    from src.models.event_candidate import EventCandidate

    return EventCandidate(
        id=f"id-{n}",
        source="@src",
        source_type="apify",
        discovered_at=datetime.now(timezone.utc),
    )


def _stub_source(candidates):
    """Return a mock IngestionSource that yields given candidates."""
    from src.ingestion.source import IngestionSource

    src = MagicMock(spec=IngestionSource)
    src.fetch.return_value = candidates
    return src


def _failing_source(exc=None):
    """Return a mock IngestionSource that always raises."""
    from src.ingestion.source import IngestionSource

    src = MagicMock(spec=IngestionSource)
    src.fetch.side_effect = exc or RuntimeError("provider down")
    return src


# ---------------------------------------------------------------------------
# FailoverChain tests
# ---------------------------------------------------------------------------


def test_single_source_success():
    from src.ingestion.failover import FailoverChain

    candidates = [_make_candidate(1), _make_candidate(2)]
    chain = FailoverChain(sources=[_stub_source(candidates)], logger=_make_logger())
    result = chain.fetch_all()
    assert result == candidates


def test_primary_fails_secondary_used():
    from src.ingestion.failover import FailoverChain

    candidates = [_make_candidate(1)]
    chain = FailoverChain(
        sources=[_failing_source(), _stub_source(candidates)],
        logger=_make_logger(),
    )
    result = chain.fetch_all()
    assert result == candidates


def test_primary_and_secondary_fail_tertiary_used():
    from src.ingestion.failover import FailoverChain

    candidates = [_make_candidate(1)]
    chain = FailoverChain(
        sources=[_failing_source(), _failing_source(), _stub_source(candidates)],
        logger=_make_logger(),
    )
    result = chain.fetch_all()
    assert result == candidates


def test_all_sources_fail_returns_empty():
    from src.ingestion.failover import FailoverChain

    chain = FailoverChain(
        sources=[_failing_source(), _failing_source()],
        logger=_make_logger(),
    )
    result = chain.fetch_all()
    assert result == []


def test_empty_source_list_returns_empty():
    from src.ingestion.failover import FailoverChain

    chain = FailoverChain(sources=[], logger=_make_logger())
    result = chain.fetch_all()
    assert result == []


def test_first_success_stops_chain():
    """Once a source succeeds, later sources are never called."""
    from src.ingestion.failover import FailoverChain

    good = _stub_source([_make_candidate(1)])
    never_called = _stub_source([_make_candidate(2)])

    chain = FailoverChain(sources=[good, never_called], logger=_make_logger())
    chain.fetch_all()

    never_called.fetch.assert_not_called()
