"""Unit tests for the one-way churn latch."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.config import SourcesConfig
from src.ingestion.id_churn import ChurnTally
from src.ingestion.latch import MIN_EVIDENCE, MIN_RUNS, arm_latches, latched_rule
from src.storage.identity_state import IdentityState
from src.storage.sqlite.identity_state import SqliteIdentityStateRepository
from src.storage.sqlite.connection import init_db

NOW = datetime(2026, 8, 18, 6, 0, tzinfo=timezone.utc)


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "event_hub.db"
    init_db(path)
    return SqliteIdentityStateRepository(path)


def _churning(n: int = 115) -> ChurnTally:
    return ChurnTally(seen_before=n, churned=n)


class TestWhenTheLatchFires:
    def test_decisive_evidence_over_enough_runs_fires(self, repo):
        for day in range(MIN_RUNS):
            arm_latches(
                {"nsno": _churning()},
                state=repo,
                sources=SourcesConfig(),
                at=NOW + timedelta(days=day),
            )

        assert repo.latched() == {"nsno"}

    def test_one_run_is_not_enough_however_decisive(self, repo):
        """A single anomalous run should not permanently change how a source is
        keyed. Everything here is irreversible."""
        arm_latches(
            {"nsno": _churning(10_000)}, state=repo, sources=SourcesConfig(), at=NOW
        )

        assert repo.latched() == set()

    def test_enough_runs_is_not_enough_without_enough_evidence(self, repo):
        """The counterpart to the rule above, and the one nothing else pins: a
        handful of re-minted listings spread over a couple of nights is thin.
        Both bars have to be cleared, and each is load-bearing on its own."""
        for day in range(MIN_RUNS):
            arm_latches(
                {"tiny": _churning(1)},
                state=repo,
                sources=SourcesConfig(),
                at=NOW + timedelta(days=day),
            )

        assert repo.get("tiny").qualifying_runs >= MIN_RUNS
        assert repo.get("tiny").churn_evidence < MIN_EVIDENCE
        assert repo.latched() == set()

    def test_a_small_feed_latches_later_rather_than_never(self, repo):
        """Three live feeds never see more than five listings a night. A per-run
        sample gate does not make them slower to latch, it makes it impossible —
        they would churn at 100% for ever with nothing acting on it."""
        for day in range(MIN_EVIDENCE):
            arm_latches(
                {"tiny": _churning(1)},
                state=repo,
                sources=SourcesConfig(),
                at=NOW + timedelta(days=day),
            )

        assert repo.latched() == {"tiny"}

    def test_a_clean_source_never_latches(self, repo):
        for day in range(50):
            arm_latches(
                {"capeanncinema": ChurnTally(seen_before=211, churned=0)},
                state=repo,
                sources=SourcesConfig(),
                at=NOW + timedelta(days=day),
            )

        assert repo.latched() == set()

    def test_a_feed_churning_every_other_night_still_latches(self, repo):
        """The case that killed the streak rule. `northshorenightout` held its
        UIDs for a whole day on 2026-08-17, and a source doing that every other
        night would never reach two *consecutive* qualifying runs — accumulating
        duplicates for ever while the latch waited."""
        for day in range(MIN_EVIDENCE * 4):
            tally = _churning(1) if day % 2 == 0 else ChurnTally(120, 0)
            arm_latches(
                {"alternating": tally},
                state=repo,
                sources=SourcesConfig(),
                at=NOW + timedelta(days=day),
            )

        assert repo.latched() == {"alternating"}


class TestPinningSuppressesTheActionNotTheObservation:
    def _pinned(self) -> SourcesConfig:
        return SourcesConfig(identity={"nsno": "publisher"})

    def test_a_pinned_source_never_latches(self, repo):
        for day in range(MIN_EVIDENCE * 2):
            arm_latches(
                {"nsno": _churning()},
                state=repo,
                sources=self._pinned(),
                at=NOW + timedelta(days=day),
            )

        assert repo.latched() == set()

    def test_its_evidence_is_still_accumulated(self, repo):
        """So a wrong pin is visible rather than silent — the user's call."""
        arm_latches(
            {"nsno": _churning()}, state=repo, sources=self._pinned(), at=NOW
        )

        assert repo.get("nsno").churn_evidence == 115

    def test_it_is_reported_as_churning_despite_the_pin(self, repo):
        for day in range(MIN_RUNS):
            report = arm_latches(
                {"nsno": _churning()},
                state=repo,
                sources=self._pinned(),
                at=NOW + timedelta(days=day),
            )

        assert report.pinned_but_churning == ["nsno"]
        assert report.latched == []


class TestTheRuleTheAdaptersRead:
    def test_a_latched_source_reads_as_content_keyed(self, repo):
        repo.latch("nsno", at=NOW)

        rule = latched_rule(SourcesConfig(), state=repo)

        assert rule("nsno") is True

    def test_an_unlatched_source_keeps_its_publisher_ids(self, repo):
        rule = latched_rule(SourcesConfig(), state=repo)

        assert rule("capeanncinema") is False

    def test_config_still_wins_where_it_speaks(self, repo):
        rule = latched_rule(SourcesConfig(identity={"nsno": "content"}), state=repo)

        assert rule("nsno") is True

    def test_the_state_is_read_once_not_per_candidate(self, repo):
        """Every candidate in a run asks this. A query per candidate would be
        thousands of round trips for an answer that cannot change mid-run."""
        repo.latch("nsno", at=NOW)
        rule = latched_rule(SourcesConfig(), state=repo)

        repo.latch("capeanncinema", at=NOW)

        assert rule("capeanncinema") is False


class TestTheReport:
    def test_it_names_what_fired_and_when(self, repo):
        for day in range(MIN_RUNS):
            report = arm_latches(
                {"nsno": _churning()},
                state=repo,
                sources=SourcesConfig(),
                at=NOW + timedelta(days=day),
            )

        assert report.latched == ["nsno"]

    def test_a_quiet_run_reports_nothing(self, repo):
        report = arm_latches(
            {"capeanncinema": ChurnTally(seen_before=211, churned=0)},
            state=repo,
            sources=SourcesConfig(),
            at=NOW,
        )

        assert report.latched == []
        assert report.pinned_but_churning == []
