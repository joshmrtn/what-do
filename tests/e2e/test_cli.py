"""End-to-end CLI tests against a populated database.

These run the real argument parsing, the real SQLite reads and the real
renderer. Only the clock is injected, because "today" has to be fixed for the
fixtures to mean anything.
"""

import io
import socket
import time
from datetime import date, datetime, timedelta, timezone
from datetime import time as time_of_day
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

from src.models.event import Event
from src.models.recommendation import Recommendation, make_recommendation_id
from src.presentation.cli import ViewSettings, run
from src.scoring.similarity import Reason
from src.storage.db import init_db
from src.storage.events import save_events
from src.storage.recommendations import save_recommendations

TZ = timezone(timedelta(hours=-4))
RUN_DATE = date(2025, 6, 21)
NOW = datetime(2025, 6, 21, 17, 0, tzinfo=TZ)

#: Pinned rather than read from `config/config.yaml`. The CLI falls back to the
#: system timezone when no config exists, which would move the night under a
#: fresh clone or a UTC machine and make these assertions environmental.
VIEW = ViewSettings(zone=ZoneInfo("America/New_York"), day_starts_at=time_of_day(4, 0))
TOMORROW = date(2025, 6, 22)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if the CLI opens a socket.

    The whole design rests on query time being local-only. A regression that
    reintroduced a network call would otherwise show up as nothing worse than a
    slow command.
    """

    def _refuse(*args, **kwargs):
        raise AssertionError("the CLI must not open a network connection")

    monkeypatch.setattr(socket, "socket", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)


def _event(event_id: str, title: str, start: datetime | None, venue: str = "The Dive Bar") -> Event:
    sunset = datetime(2025, 6, 21, 20, 15, tzinfo=TZ)
    return Event(
        event_id=event_id,
        source_event_candidates=[f"cand-{event_id}"],
        source_type="instagram",
        created_at=NOW,
        updated_at=NOW,
        title=title,
        venue=venue,
        start_time=start,
        astronomical_data={"sunset": sunset.isoformat()},
    )


def _recommendation(event_id: str, rank: int, tier: str, score: float) -> Recommendation:
    return Recommendation(
        recommendation_id=make_recommendation_id(RUN_DATE, event_id),
        event_id=event_id,
        run_date=RUN_DATE,
        base_score=score,
        weather_adjustment=0.05,
        tag_confidence=1.0,
        final_score=score,
        match="yes",
        tier=tier,
        rank=rank,
        reasons=[
            Reason(
                factor="like_similarity",
                matched_preference="karaoke night",
                similarity=0.87,
                contribution=score,
                direction="positive",
                tag="karaoke",
            )
        ],
    )


def _at(hour: int, day: int = 21) -> datetime:
    return datetime(2025, 6, day, hour, 0, tzinfo=TZ)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Ten ranked events spanning two days, plus one with no start time."""
    path = tmp_path / "event_hub.db"
    init_db(path)

    events = [
        _event("t1", "Karaoke Night", _at(20)),
        _event("t2", "Open Mic", _at(19)),
        _event("t3", "Late Jazz", _at(22)),
        _event("t4", "Afternoon Market", _at(14)),
        _event("t5", "Dull Mixer", _at(18)),
        _event("t6", "Duller Meetup", _at(19)),
        _event("m1", "Tomorrow Gig", _at(20, day=22)),
        _event("m2", "Tomorrow Film", _at(21, day=22)),
        _event("m3", "Tomorrow Market", _at(11, day=22)),
        _event("u1", "Open Studio Weekend", None),
    ]
    save_events(events, path)

    recommendations = [
        _recommendation("t1", 1, "top_pick", 0.81),
        _recommendation("t2", 2, "top_pick", 0.72),
        _recommendation("t3", 3, "top_pick", 0.64),
        _recommendation("u1", 4, "top_pick", 0.58),
        _recommendation("m1", 5, "top_pick", 0.55),
        _recommendation("m2", 6, "worth_considering", 0.31),
        _recommendation("t4", 7, "worth_considering", 0.22),
        _recommendation("m3", 8, "worth_considering", 0.14),
        _recommendation("t5", 9, "everything_else", 0.02),
        _recommendation("t6", 10, "everything_else", -0.30),
    ]
    save_recommendations(recommendations, path)

    return path


def _invoke(
    db_path: Path, *argv: str, now: datetime | None = None
) -> tuple[int, str, str]:
    """Run the CLI as a user would.

    `--db` leads because argparse takes top-level options before a subcommand;
    `what-do add-source @x --db ...` is a usage error, as it is for git.
    """
    stdout, stderr = io.StringIO(), io.StringIO()
    code = run(
        ["--db", str(db_path), *argv],
        get_now=lambda: now if now is not None else NOW,
        stdout=stdout,
        stderr=stderr,
        load_view_settings=lambda: VIEW,
    )
    return code, stdout.getvalue(), stderr.getvalue()


def test_default_view_shows_only_todays_events(db_path):
    _, out, _ = _invoke(db_path)

    assert "Karaoke Night" in out
    assert "Tomorrow Gig" not in out
    assert "Tomorrow Film" not in out


def test_default_view_keeps_undated_events(db_path):
    _, out, _ = _invoke(db_path)

    assert "Open Studio Weekend" in out
    assert "UNDATED" in out


def test_default_view_separates_the_tiers(db_path):
    _, out, _ = _invoke(db_path)

    assert out.index("TOP PICKS") < out.index("WORTH CONSIDERING")
    assert "Afternoon Market" in out


def test_default_view_folds_the_bottom_tier_but_reports_its_size(db_path):
    _, out, _ = _invoke(db_path)

    assert "Dull Mixer" not in out
    assert "2 more" in out


def test_all_expands_the_bottom_tier(db_path):
    _, out, _ = _invoke(db_path, "--all")

    assert "Dull Mixer" in out
    assert "Duller Meetup" in out


def test_reasons_are_shown(db_path):
    _, out, _ = _invoke(db_path)

    assert "karaoke night" in out


def test_verbose_shows_the_score_behind_the_tier(db_path):
    _, out, _ = _invoke(db_path, "-v")

    assert "+0.81" in out


def test_time_filter_narrows_to_the_window(db_path):
    _, out, _ = _invoke(db_path, "--time", "19:30-23:00")

    assert "Karaoke Night" in out
    assert "Late Jazz" in out
    assert "Afternoon Market" not in out


def test_after_sunset_drops_daytime_events(db_path):
    _, out, _ = _invoke(db_path, "--after-sunset")

    assert "Late Jazz" in out
    assert "Afternoon Market" not in out
    assert "Open Mic" not in out


def test_raw_shows_every_event_across_every_day(db_path):
    _, out, _ = _invoke(db_path, "--raw")

    assert "Karaoke Night" in out
    assert "Tomorrow Gig" in out
    assert "Dull Mixer" in out
    assert "Open Studio Weekend" in out


def test_raw_bypasses_ranking(db_path):
    _, out, _ = _invoke(db_path, "--raw")

    assert "TOP PICKS" not in out
    assert "karaoke night" not in out


def test_an_empty_database_is_not_an_error(tmp_path):
    path = tmp_path / "empty.db"
    init_db(path)

    code, out, err = _invoke(path)

    assert code == 0
    assert err == ""
    assert "batch" in out.lower()


def test_a_database_that_does_not_exist_yet_is_not_a_crash(tmp_path, monkeypatch):
    """The state before the very first batch run must not be a stack trace."""
    missing = tmp_path / "database" / "event_hub.db"
    monkeypatch.setattr("src.presentation.cli.DEFAULT_DB_PATH", missing)
    stdout, stderr = io.StringIO(), io.StringIO()

    code = run([], get_now=lambda: NOW, stdout=stdout, stderr=stderr)

    assert code == 0
    assert "batch" in stdout.getvalue().lower()
    assert stderr.getvalue() == ""
    assert not missing.exists(), "reading must not leave an empty database behind"


def test_a_zero_byte_database_file_is_treated_as_not_ready(tmp_path, monkeypatch):
    """sqlite3.connect creates one on any stray read, so it must not look real."""
    empty = tmp_path / "event_hub.db"
    empty.touch()
    monkeypatch.setattr("src.presentation.cli.DEFAULT_DB_PATH", empty)
    stdout, stderr = io.StringIO(), io.StringIO()

    code = run([], get_now=lambda: NOW, stdout=stdout, stderr=stderr)

    assert code == 0
    assert "batch" in stdout.getvalue().lower()


def test_an_explicitly_named_database_that_is_missing_is_a_usage_error(tmp_path):
    """A typo in --db must not look like an empty database."""
    stdout, stderr = io.StringIO(), io.StringIO()

    code = run(
        ["--db", str(tmp_path / "typo.db")],
        get_now=lambda: NOW,
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert "typo.db" in stderr.getvalue()


def test_raw_on_a_missing_database_is_also_handled(tmp_path):
    stdout, stderr = io.StringIO(), io.StringIO()

    code = run(
        ["--db", str(tmp_path / "typo.db"), "--raw"],
        get_now=lambda: NOW,
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert "typo.db" in stderr.getvalue()


def test_add_source_writes_to_the_seeds_file(tmp_path, db_path):
    seeds = tmp_path / "seeds.yaml"

    code, out, _ = _invoke(db_path, "add-source", "@smoketest", "--seeds-file", str(seeds))

    assert code == 0
    assert yaml.safe_load(seeds.read_text())["handles"] == ["@smoketest"]


def test_the_default_view_returns_in_under_a_second(db_path):
    started = time.perf_counter()
    _invoke(db_path)
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0, f"CLI took {elapsed:.3f}s; it reads precomputed rows only"


def test_raw_view_returns_in_under_a_second(db_path):
    started = time.perf_counter()
    _invoke(db_path, "--raw")
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0, f"CLI took {elapsed:.3f}s; it reads precomputed rows only"


def test_after_midnight_still_shows_the_evening_in_progress(db_path):
    """End to end for #18: asking at 00:30 answers about the night in progress."""
    _, out, _ = _invoke(db_path, now=datetime(2025, 6, 22, 0, 30, tzinfo=TZ))

    assert "Karaoke Night" in out
    assert "Tomorrow Gig" not in out
    assert "Saturday 21 June" in out
