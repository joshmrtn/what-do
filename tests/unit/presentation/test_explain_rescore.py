"""What `--explain` says about a ranking that was recomputed after its batch.

Without this the numbers on the page are unattributable: a run date holds a
score produced by a forecast its own `run_history` row does not describe, and
nothing on screen says which computation the reader is looking at.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from src.models.event import Event
from src.models.event_score import EventScore
from src.models.ranking import Ranking
from src.models.rescore import Rescore
from src.presentation.render import render_explanation

RUN_DATE = date(2026, 8, 17)
TZ = timezone(timedelta(hours=-4))
NOW = datetime(2026, 8, 17, 20, 0, tzinfo=TZ)


def _event() -> Event:
    return Event(
        event_id="evt-1",
        source_event_candidates=[],
        source_type="instagram",
        created_at=NOW,
        updated_at=NOW,
        title="Harbour Fireworks",
        venue="The Pier",
        start_time=datetime(2026, 8, 17, 21, 0, tzinfo=TZ),
    )


def _score() -> EventScore:
    return EventScore(
        event_id="evt-1",
        run_date=RUN_DATE,
        base_score=0.5,
        tag_confidence=1.0,
        match="yes",
        reasons=[],
    )


def _ranking() -> Ranking:
    return Ranking(
        event_id="evt-1",
        run_date=RUN_DATE,
        weather_adjustment=0.12,
        final_score=0.87,
        rank=3,
    )


def _rescore(**overrides) -> Rescore:
    fields = {
        "run_date": RUN_DATE,
        "rescored_at": NOW,
        "events_rescored": 81,
        "forecast_issued_at": NOW - timedelta(minutes=4),
        "preference_revision_id": "rev-9",
        **overrides,
    }
    return Rescore(**fields)


def _explain(rescore: Rescore | None) -> str:
    return render_explanation(
        _event(), _score(), _ranking(), total=1169, rescore=rescore
    )


def test_a_batch_only_ranking_says_nothing_extra():
    """Most rankings are the batch's, and saying so on each would be noise."""
    out = _explain(None)

    assert "rescored" not in out.lower()


def test_a_rescored_ranking_says_so_and_when():
    out = _explain(_rescore())

    assert "rescored" in out.lower()
    assert "20:00" in out


def test_the_forecast_it_used_is_named():
    """The reason the numbers moved, stated beside the numbers."""
    out = _explain(_rescore(forecast_issued_at=datetime(2026, 8, 17, 19, 30, tzinfo=TZ)))

    assert "19:30" in out


def test_a_rescore_with_no_forecast_still_reports_itself():
    """An all-indoor night rescores and gains nothing, which is still a fact."""
    out = _explain(_rescore(forecast_issued_at=None))

    assert "rescored" in out.lower()


def test_the_rescore_line_survives_an_unranked_event():
    """A superseded event has no placement, and must not lose the header."""
    out = render_explanation(
        _event(), None, None, total=1169, rescore=_rescore()
    )

    assert "not ranked" in out
