"""Unit tests for the accumulated churn evidence and the one-way latch."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.ingestion.id_churn import ChurnTally
from src.storage.identity_state import IdentityState
from src.storage.sqlite.identity_state import SqliteIdentityStateRepository
from src.storage.sqlite.connection import init_db

NOW = datetime(2026, 8, 18, 6, 0, tzinfo=timezone.utc)
LATER = datetime(2026, 8, 19, 6, 0, tzinfo=timezone.utc)


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "event_hub.db"
    init_db(path)
    return SqliteIdentityStateRepository(path)


class TestAccumulatingEvidence:
    """Evidence accumulates and never resets. A streak would be reset by a quiet
    night — measured, `northshorenightout` held its UIDs for a whole day on
    2026-08-17 — and a feed that churns every *other* night would never reach
    two consecutive qualifying runs at all, accumulating duplicates for ever
    while the latch waited."""

    def test_an_unknown_source_starts_empty(self, repo):
        state = repo.get("northshorenightout")

        assert state.churn_evidence == 0
        assert state.qualifying_runs == 0
        assert state.latched_at is None

    def test_a_qualifying_run_adds_its_churned_count(self, repo):
        repo.record("nsno", ChurnTally(seen_before=115, churned=115), at=NOW)

        state = repo.get("nsno")
        assert state.churn_evidence == 115
        assert state.qualifying_runs == 1

    def test_evidence_accumulates_across_runs(self, repo):
        repo.record("nsno", ChurnTally(seen_before=4, churned=4), at=NOW)
        repo.record("nsno", ChurnTally(seen_before=4, churned=4), at=LATER)

        state = repo.get("nsno")
        assert state.churn_evidence == 8
        assert state.qualifying_runs == 2

    def test_a_quiet_night_does_not_reset_the_evidence(self, repo):
        """The 2026-08-17 case. The publisher held its UIDs for one day and the
        detector read a clean 0%; a streak counter would have started over."""
        repo.record("nsno", ChurnTally(seen_before=115, churned=115), at=NOW)
        repo.record("nsno", ChurnTally(seen_before=114, churned=0), at=LATER)

        state = repo.get("nsno")
        assert state.churn_evidence == 115
        assert state.qualifying_runs == 1

    def test_a_clean_source_accumulates_nothing(self, repo):
        repo.record("capeanncinema", ChurnTally(seen_before=211, churned=0), at=NOW)

        assert repo.get("capeanncinema").churn_evidence == 0

    def test_an_unmeasurable_run_is_not_evidence_either_way(self, repo):
        """`rate` is `None` — a first run, or a source whose candidates are not
        listings. Neither says anything about identity."""
        repo.record("somevenue", ChurnTally(seen_before=0, churned=0), at=NOW)

        state = repo.get("somevenue")
        assert state.churn_evidence == 0
        assert state.qualifying_runs == 0

    def test_sources_accumulate_separately(self, repo):
        repo.record("nsno", ChurnTally(seen_before=115, churned=115), at=NOW)
        repo.record("capeanncinema", ChurnTally(seen_before=211, churned=0), at=NOW)

        assert repo.get("nsno").churn_evidence == 115
        assert repo.get("capeanncinema").churn_evidence == 0


class TestTheLatchIsOneWay:
    def test_latching_is_recorded_with_when(self, repo):
        repo.latch("nsno", at=NOW)

        assert repo.get("nsno").latched_at == NOW

    def test_a_latched_source_stays_latched_through_later_runs(self, repo):
        """Once content-keyed, churn reads 0% *by construction*. Anything that
        re-evaluated in both directions would oscillate, re-keying and minting
        duplicates each way."""
        repo.latch("nsno", at=NOW)
        repo.record("nsno", ChurnTally(seen_before=115, churned=0), at=LATER)

        assert repo.get("nsno").latched_at == NOW

    def test_latching_twice_keeps_the_first_time(self, repo):
        repo.latch("nsno", at=NOW)
        repo.latch("nsno", at=LATER)

        assert repo.get("nsno").latched_at == NOW

    def test_the_latched_set_is_readable_in_one_query(self, repo):
        repo.latch("nsno", at=NOW)
        repo.record("capeanncinema", ChurnTally(seen_before=211, churned=0), at=NOW)

        assert repo.latched() == {"nsno"}


class TestItSurvivesARestart:
    def test_state_is_read_back_from_the_database(self, tmp_path):
        path = tmp_path / "event_hub.db"
        init_db(path)
        SqliteIdentityStateRepository(path).record(
            "nsno", ChurnTally(seen_before=115, churned=115), at=NOW
        )

        reopened = SqliteIdentityStateRepository(path).get("nsno")

        assert reopened == IdentityState(
            source="nsno",
            churn_evidence=115,
            qualifying_runs=1,
            latched_at=None,
            updated_at=NOW,
        )
