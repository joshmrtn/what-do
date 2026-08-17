"""Contract every RescoreRepository implementation must satisfy.

Nothing is destroyed: a night rescored three times keeps three rows, and
`latest_for` reads the most recent without the others ceasing to exist.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import pytest

from src.models.rescore import Rescore
from src.storage.memory.rescores import InMemoryRescoreRepository
from src.storage.sqlite.connection import init_db
from src.storage.sqlite.rescores import SqliteRescoreRepository

RUN_DATE = date(2026, 8, 17)
_AT = datetime(2026, 8, 17, 20, 0, tzinfo=timezone.utc)


@pytest.fixture(params=["sqlite", "memory"])
def repo(request, tmp_path):
    """One repository per implementation, so every test below runs twice."""
    if request.param == "sqlite":
        path = tmp_path / "rescores.db"
        init_db(path)
        return SqliteRescoreRepository(path)
    return InMemoryRescoreRepository()


#: "Not supplied", so a test can ask for a rescore that genuinely carries no
#: forecast. Defaulting the parameter to None made the two indistinguishable and
#: the no-forecast test silently asserted the default instead.
_UNSET = object()


def _rescore(
    rescored_at: datetime = _AT,
    run_date: date = RUN_DATE,
    events_rescored: int = 81,
    forecast_issued_at: Any = _UNSET,
) -> Rescore:
    return Rescore(
        run_date=run_date,
        rescored_at=rescored_at,
        events_rescored=events_rescored,
        forecast_issued_at=(
            _AT - timedelta(minutes=5)
            if forecast_issued_at is _UNSET
            else forecast_issued_at
        ),
    )


class TestRecord:
    def test_a_recorded_rescore_is_the_latest_for_its_run(self, repo):
        repo.record(_rescore())

        assert repo.latest_for(RUN_DATE) is not None

    def test_every_field_round_trips_at_a_non_default_value(self, repo):
        """A field left at its default passes against a column that is missing."""
        issued = datetime(2026, 8, 17, 6, 30, tzinfo=timezone.utc)
        repo.record(_rescore(events_rescored=1169, forecast_issued_at=issued))

        stored = repo.latest_for(RUN_DATE)

        assert stored.run_date == RUN_DATE
        assert stored.rescored_at == _AT
        assert stored.events_rescored == 1169
        assert stored.forecast_issued_at == issued

    def test_a_rescore_with_no_forecast_round_trips(self, repo):
        """An all-indoor listing has no forecast behind it, and that is not an error."""
        repo.record(_rescore(forecast_issued_at=None))

        assert repo.latest_for(RUN_DATE).forecast_issued_at is None

    def test_a_second_rescore_does_not_replace_the_first(self, repo):
        """Nothing is destroyed — "how often did this move?" stays answerable."""
        repo.record(_rescore(rescored_at=_AT))
        repo.record(_rescore(rescored_at=_AT + timedelta(hours=1)))

        assert len(repo.for_run(RUN_DATE)) == 2


class TestLatestFor:
    def test_a_run_never_rescored_has_none(self, repo):
        assert repo.latest_for(RUN_DATE) is None

    def test_the_latest_is_the_most_recent_not_the_last_written(self, repo):
        repo.record(_rescore(rescored_at=_AT + timedelta(hours=2)))
        repo.record(_rescore(rescored_at=_AT))

        assert repo.latest_for(RUN_DATE).rescored_at == _AT + timedelta(hours=2)

    def test_another_run_date_does_not_answer_for_this_one(self, repo):
        """Each night's provenance is its own."""
        repo.record(_rescore(run_date=date(2026, 8, 16)))

        assert repo.latest_for(RUN_DATE) is None


class TestForRun:
    def test_rescores_come_back_newest_first(self, repo):
        repo.record(_rescore(rescored_at=_AT))
        repo.record(_rescore(rescored_at=_AT + timedelta(hours=3)))
        repo.record(_rescore(rescored_at=_AT + timedelta(hours=1)))

        stamps = [row.rescored_at for row in repo.for_run(RUN_DATE)]

        assert stamps == sorted(stamps, reverse=True)

    def test_a_run_never_rescored_is_empty(self, repo):
        assert repo.for_run(RUN_DATE) == []
