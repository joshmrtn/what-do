"""End-to-end CLI tests against a populated database.

These run the real argument parsing, the real SQLite reads and the real
renderer. Only the clock is injected, because "today" has to be fixed for the
fixtures to mean anything.
"""

import io
import re
import socket
import time
from datetime import date, datetime, timedelta, timezone
from datetime import time as time_of_day
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

from src.models.event import Event
from src.models.event_score import EventScore
from src.models.ranking import Ranking
from src.config import ViewConfig
from src.presentation.cli import ViewSettings, run
from src.scoring.similarity import Reason
from src.storage.sqlite.connection import init_db
from src.storage.events import save_events
from src.storage.sqlite.rankings import SqliteRankingRepository
from src.storage.sqlite.scores import SqliteScoreRepository

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


def _recommendation(
    event_id: str, rank: int, score: float, extra_reasons: int = 0
) -> tuple[EventScore, Ranking]:
    """One event's verdict and its placement, which are now stored apart.

    `extra_reasons` exists for the reason-limit tests, which need an event
    carrying more reasons than the limit under test. Zero everywhere else, so no
    other assertion shifts.
    """
    return (
        EventScore(
            event_id=event_id,
            run_date=RUN_DATE,
            base_score=score,
            tag_confidence=1.0,
            match="yes",
            reasons=[
                Reason(
                    factor="like_similarity",
                    matched_preference="karaoke night",
                    similarity=0.87,
                    contribution=score,
                    direction="positive",
                    tag="karaoke",
                ),
                *[
                    Reason(
                        factor="like_similarity",
                        matched_preference=f"extra preference {n}",
                        similarity=0.5,
                        contribution=0.1,
                        direction="positive",
                        tag=f"extra{n}",
                    )
                    for n in range(extra_reasons)
                ],
            ],
        ),
        Ranking(
            event_id=event_id,
            run_date=RUN_DATE,
            weather_adjustment=0.05,
            final_score=score,
            rank=rank,
        ),
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
        _recommendation("t1", 1, 0.81, extra_reasons=1),
        _recommendation("t2", 2, 0.72),
        _recommendation("t3", 3, 0.64),
        _recommendation("u1", 4, 0.58),
        _recommendation("m1", 5, 0.55),
        _recommendation("m2", 6, 0.31),
        _recommendation("t4", 7, 0.22),
        _recommendation("m3", 8, 0.14),
        _recommendation("t5", 9, 0.02),
        _recommendation("t6", 10, -0.30),
    ]
    SqliteScoreRepository(path).save([score for score, _ in recommendations])
    SqliteRankingRepository(path).save([ranking for _, ranking in recommendations])

    return path


def _invoke_add_source(seeds: Path, *argv: str) -> tuple[int, str, str]:
    """Run `add-source` against a throwaway seeds file.

    Never the real one: this command *writes*, and probing it by hand during the
    flag audit put `'@   '` into `data/seeds.yaml`.
    """
    stdout, stderr = io.StringIO(), io.StringIO()
    code = run(
        ["add-source", *argv, "--seeds-file", str(seeds)],
        stdout=stdout,
        stderr=stderr,
    )
    return code, stdout.getvalue(), stderr.getvalue()


def _invoke(
    db_path: Path,
    *argv: str,
    now: datetime | None = None,
    view: ViewSettings | None = None,
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
        load_view_settings=lambda: view if view is not None else VIEW,
    )
    return code, stdout.getvalue(), stderr.getvalue()


def test_default_view_shows_only_todays_events(db_path):
    _, out, _ = _invoke(db_path)

    assert "Karaoke Night" in out
    assert "Tomorrow Gig" not in out
    assert "Tomorrow Film" not in out


def test_default_view_keeps_undated_events(db_path):
    """Ranked among the rest, with the column saying the time is unpublished."""
    _, out, _ = _invoke(db_path)

    assert "Open Studio Weekend" in out
    assert "time TBC" in out


def test_the_view_is_one_ranked_list_with_no_bands(db_path):
    _, out, _ = _invoke(db_path)

    for banner in ("TOP PICKS", "WORTH CONSIDERING", "EVERYTHING ELSE", "UNDATED"):
        assert banner not in out


def test_the_default_view_is_cut_at_ten_and_counts_the_rest(db_path):
    _, out, _ = _invoke(db_path, "--limit", "3")

    assert "Dull Mixer" not in out
    assert "ranked lower (--all)" in out


def test_all_shows_every_ranked_event(db_path):
    _, out, _ = _invoke(db_path, "--all")

    assert "Dull Mixer" in out
    assert "Duller Meetup" in out
    assert "ranked lower (--all)" not in out


def test_reasons_are_shown(db_path):
    _, out, _ = _invoke(db_path)

    assert "karaoke night" in out


def test_verbose_shows_the_score_the_order_was_built_on(db_path):
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


def test_raw_shows_a_superseded_event_and_says_what_absorbed_it(db_path):
    """The real path, against a real database, because that is where it broke.

    `--raw` reached past the repository to the one reader that ignores
    supersession, as a *default argument* — so every test injected its own
    loader and production ran the unfiltered path. Filtering here instead would
    make `--raw` the one view that cannot show what the batch actually did;
    including silently makes a reader take a merged-away duplicate for real.
    """
    loser = _event("dead", "Wood & Bone", _at(19))
    loser.superseded_by = "t1"
    loser.merged_by = "semantic"
    loser.merge_similarity = 0.926
    save_events([loser], db_path)

    _, out, _ = _invoke(db_path, "--raw")

    assert "Wood & Bone" in out
    assert "superseded by t1" in out
    assert "semantic" in out
    assert "0.926" in out
    assert "1 superseded" in out


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


class TestExplain:
    """One event, accounted for, through the real parser and real reads."""

    def test_a_rank_from_the_list_explains_that_event(self, db_path):
        code, out, _ = _invoke(db_path, "--explain", "1")

        assert code == 0
        assert "Karaoke Night" in out
        assert "karaoke night" in out  # the matched preference line

    def test_part_of_a_title_explains_that_event(self, db_path):
        code, out, _ = _invoke(db_path, "--explain", "open mic")

        assert code == 0
        assert "Open Mic" in out

    def test_an_ambiguous_title_lists_the_matches_and_fails(self, db_path):
        """Rather than guessing. Reading the wrong event's explanation and
        believing it is worse than being asked to be specific."""
        code, out, err = _invoke(db_path, "--explain", "tomorrow")

        assert code == 1
        assert "matches" in err
        assert "Tomorrow Gig" in err
        assert out == ""

    def test_an_unknown_selector_fails_with_a_usable_message(self, db_path):
        code, _, err = _invoke(db_path, "--explain", "no such event")

        assert code == 1
        assert "no event matches" in err.lower()

    def test_a_superseded_event_explains_what_it_can(self, db_path):
        """It has no score and no ranking, so there is no placement to account
        for — but "why was this merged away" is exactly the question `--raw`
        marking raises, and this is where it gets answered."""
        loser = _event("dead", "Wood & Bone", _at(19))
        loser.superseded_by = "t1"
        loser.merged_by = "semantic"
        loser.merge_similarity = 0.926
        save_events([loser], db_path)

        code, out, _ = _invoke(db_path, "--explain", "wood")

        assert code == 0
        assert "not ranked" in out.lower()
        assert "superseded by t1" in out
        assert "0.926" in out

    def test_explain_does_not_disturb_the_default_view(self, db_path):
        """The list stays clean. That was settled twice and this must not be the
        change that walks it back."""
        _, out, _ = _invoke(db_path)

        assert "score " not in out
        assert "not recorded" not in out


class TestOtherNights:
    """Days that were ranked and stored but unreachable (#19).

    The batch ranks every event in a run, spanning the whole horizon. The CLI
    could only ever show one of those days — `--run-date` selects which *batch*
    to read, not which night to display — so tomorrow's events were scored,
    ranked, persisted and invisible. `--raw` was the only way to see them, and
    it bypasses ranking entirely.
    """

    def test_a_named_date_shows_that_night(self, db_path):
        code, out, _ = _invoke(db_path, "--date", TOMORROW.isoformat())

        assert code == 0
        assert "Tomorrow Gig" in out
        assert "Karaoke Night" not in out

    def test_the_heading_names_the_night_being_shown(self, db_path):
        _, out, _ = _invoke(db_path, "--date", TOMORROW.isoformat())

        assert TOMORROW.strftime("%A %-d %B") in out

    def test_a_malformed_date_is_a_usage_error(self, db_path):
        code, _, err = _invoke(db_path, "--date", "next tuesday")

        assert code == 1
        assert "YYYY-MM-DD" in err

    def test_a_night_with_nothing_ranked_says_so(self, db_path):
        """Rather than rendering an empty list, which reads like a failure."""
        code, out, _ = _invoke(db_path, "--date", "2025-12-25")

        assert code == 0
        assert "No events" in out

    def test_several_nights_render_as_separate_sections(self, db_path):
        """Each night gets its own heading and its own ranking, because the
        question "what is on Saturday" is per-night."""
        code, out, _ = _invoke(db_path, "--days", "2")

        assert code == 0
        assert RUN_DATE.strftime("%A %-d %B") in out
        assert TOMORROW.strftime("%A %-d %B") in out
        assert out.index(RUN_DATE.strftime("%A %-d %B")) < out.index(
            TOMORROW.strftime("%A %-d %B")
        )

    def test_days_starts_from_tonight(self, db_path):
        code, out, _ = _invoke(db_path, "--days", "1")

        assert code == 0
        assert "Karaoke Night" in out
        assert "Tomorrow Gig" not in out

    def test_days_must_be_positive(self, db_path):
        code, _, err = _invoke(db_path, "--days", "0")

        assert code == 1
        assert "--days" in err

    def test_date_and_days_together_is_a_usage_error(self, db_path):
        """They answer the same question two ways, so accepting both would mean
        silently ignoring one."""
        code, _, err = _invoke(db_path, "--date", TOMORROW.isoformat(), "--days", "3")

        assert code == 1
        assert "--date" in err and "--days" in err


class TestUpcoming:
    """One cross-day leaderboard: what is coming up that I should plan for (#23).

    Distinct from `--days`, which is per-night sections. This is a single
    ordered list across the whole window, because the question is "what is worth
    putting in the calendar", not "what is on Saturday".

    Valid only because scores are never normalised per batch — that CLAUDE.md
    rule is what lets scores from different nights sort against each other
    honestly, and it is what makes this a filter rather than a re-ranking.
    """

    def test_it_starts_after_tonight(self, db_path):
        """`--upcoming` reads as "after tonight", and the default view already
        answers tonight. Including it would make the busiest night crowd out
        the far-out events this exists to surface — and tonight is the one night
        that needs no planning at all."""
        code, out, _ = _invoke(db_path, "--upcoming")

        assert code == 0
        assert "Tomorrow Gig" in out
        assert "Karaoke Night" not in out

    def test_it_spans_several_nights(self, db_path):
        _, out, _ = _invoke(db_path, "--upcoming", "--limit", "20")

        assert "Tomorrow Gig" in out
        assert "Tomorrow Market" in out

    def test_it_is_one_list_not_per_night_sections(self, db_path):
        _, out, _ = _invoke(db_path, "--upcoming")

        assert RUN_DATE.strftime("%A %-d %B") not in out
        assert TOMORROW.strftime("%A %-d %B") not in out

    def test_it_orders_by_score_not_by_date(self, db_path):
        """The whole point. A better-scored event belongs above a worse one
        whichever night it falls on, or this is just a longer listing."""
        _, out, _ = _invoke(db_path, "--upcoming", "--limit", "20")

        assert out.index("Tomorrow Gig") < out.index("Tomorrow Market")

    def test_the_window_is_configurable(self, db_path):
        """`--upcoming 1` is the single night after tonight."""
        _, out, _ = _invoke(db_path, "--upcoming", "1")

        assert "Tomorrow Gig" in out
        assert "Karaoke Night" not in out

    def test_each_row_says_which_day_it_is_on(self, db_path):
        """In a per-night view the heading answers that. Here there is no
        heading, so a row without its date is unplaceable."""
        _, out, _ = _invoke(db_path, "--upcoming")

        assert TOMORROW.strftime("%a") in out

    def test_what_is_cut_is_still_counted(self, db_path):
        """The standing rule: a hidden event is invisible, a counted one is a
        flag away."""
        _, out, _ = _invoke(db_path, "--upcoming", "--limit", "2")

        assert "more" in out

    def test_it_cannot_be_combined_with_a_night_selector(self, db_path):
        """`--upcoming`, `--date` and `--days` all choose what is shown, so a
        pair would mean silently ignoring one."""
        code, _, err = _invoke(db_path, "--upcoming", "--date", TOMORROW.isoformat())

        assert code == 1
        assert "--upcoming" in err and "--date" in err

    def test_days_must_be_positive(self, db_path):
        code, _, err = _invoke(db_path, "--upcoming", "0")

        assert code == 1
        assert "--upcoming" in err

    def test_a_negative_day_count_is_also_rejected(self, db_path):
        """Not obvious, and it was briefly broken. A bare `--upcoming` has to
        record *something* the parser can carry until config is loaded, and any
        number chosen for that is a number a person can type — `-1` silently
        meant "use the default" instead of reporting the mistake, exactly as `0`
        had before it. The sentinel has to come from outside the value domain.
        """
        code, out, err = _invoke(db_path, "--upcoming", "-1")

        assert code == 1
        assert "--upcoming" in err
        assert out == ""


class TestConfiguredViewNumbers:
    """The view's numbers come from config, not from constants in the source.

    Three are defaults a flag can override, so they earn their place here by
    sparing a preference from being retyped every invocation. `long_span_hours`
    has no flag at all and decides real behaviour (#31).
    """

    def _view(self, **kwargs) -> ViewSettings:
        return ViewSettings(
            zone=VIEW.zone,
            day_starts_at=VIEW.day_starts_at,
            view=ViewConfig(**kwargs),
        )

    def test_the_configured_limit_decides_how_many_are_shown(self, db_path):
        _, out, _ = _invoke(db_path, view=self._view(limit=2))

        assert len(re.findall(r"^  \d+\. ", out, re.M)) == 2

    def test_what_the_configured_limit_cuts_is_still_counted(self, db_path):
        _, out, _ = _invoke(db_path, view=self._view(limit=2))

        assert "more event" in out

    def test_an_explicit_flag_still_beats_the_configured_limit(self, db_path):
        _, out, _ = _invoke(db_path, "--limit", "1", view=self._view(limit=5))

        assert len(re.findall(r"^  \d+\. ", out, re.M)) == 1

    def test_the_configured_window_decides_how_far_upcoming_reaches(self, db_path):
        """A bare `--upcoming` takes its span from config rather than a
        constant — which is the whole difference between a default and a
        number baked into the source."""
        _, near, _ = _invoke(db_path, "--upcoming", view=self._view(upcoming_days=1))
        _, far, _ = _invoke(db_path, "--upcoming", view=self._view(upcoming_days=30))

        assert "Tomorrow Gig" in near
        assert "Tomorrow Gig" in far

    def test_the_configured_reason_limit_decides_how_many_reasons_show(self, db_path):
        """`t1` carries a second reason for this, via `extra_reasons` — every
        other event keeps exactly one, so no other assertion shifts."""
        _, one, _ = _invoke(db_path, "--limit", "1", view=self._view(reason_limit=1))
        _, two, _ = _invoke(db_path, "--limit", "1", view=self._view(reason_limit=2))

        assert one.count("<-") == 1
        assert two.count("<-") == 2


class TestInputThatCannotBeHonoured:
    """Values a person can type that were silently reinterpreted.

    All the same shape as the `--upcoming` sentinel: an input inside the domain
    the user can express, given a meaning they did not ask for. The failure is
    never a crash — it is a listing that looks right.
    """

    def test_a_zero_limit_is_refused_rather_than_replaced(self, db_path):
        """`args.limit or default` reads 0 as "unset", so asking for none
        silently produced the default ten."""
        code, out, err = _invoke(db_path, "--limit", "0")

        assert code == 1
        assert "--limit" in err
        assert out == ""

    def test_a_negative_limit_is_refused_rather_than_slicing_from_the_end(self, db_path):
        """The worst of the family, because it looks like it worked: `pairs[:-5]`
        drops the *last* five and renders everything else, so the listing is
        plausible and wrong, and the "+ N more" count no longer describes it."""
        code, out, err = _invoke(db_path, "--limit", "-5")

        assert code == 1
        assert "--limit" in err
        assert out == ""

    def test_an_empty_explain_selector_is_refused(self, db_path):
        """An empty string is falsy, so `--explain ''` fell through to the
        default view — the one command that must never quietly answer a
        different question."""
        code, out, err = _invoke(db_path, "--explain", "")

        assert code == 1
        assert "--explain" in err
        assert out == ""

    def test_a_whitespace_only_explain_selector_is_refused(self, db_path):
        code, _, err = _invoke(db_path, "--explain", "   ")

        assert code == 1
        assert "--explain" in err

    def test_a_night_with_nothing_on_still_says_which_night(self, db_path):
        """`--days` renders one section per night, and an empty one dropped its
        heading — so a run of quiet nights became indistinguishable repetitions
        of "No events to show." with nothing saying which was which."""
        _, out, _ = _invoke(db_path, "--days", "4")

        quiet = RUN_DATE + timedelta(days=3)
        assert quiet.strftime("%A %-d %B") in out


class TestBlankInputIsNotInput:
    """An empty or whitespace-only value is not a value.

    The same family as `--explain ''` — a falsy string skips the `if args.x:`
    that was meant to detect absence, so a flag the user typed is treated as one
    they did not. Found by probing the whole flag surface after the `--upcoming`
    sentinel, rather than waiting for one to bite.
    """

    def test_a_blank_time_window_is_refused(self, db_path):
        """It rendered an unfiltered listing — the filter silently not applied
        is worse than no filter, because the output looks filtered."""
        code, out, err = _invoke(db_path, "--time", "")

        assert code == 1
        assert "--time" in err
        assert out == ""

    def test_a_blank_date_is_refused(self, db_path):
        code, out, err = _invoke(db_path, "--date", "   ")

        assert code == 1
        assert "--date" in err
        assert out == ""

    def test_a_blank_handle_is_not_added_to_seeds(self, tmp_path):
        """The worst of them: `add-source '   '` wrote `'@   '` into seeds.yaml
        and reported success. A whitespace handle is a real entry in a real file
        that discovery would then try to fetch."""
        seeds = tmp_path / "seeds.yaml"
        seeds.write_text("handles: []\nvenues: []\n")

        code, out, err = _invoke_add_source(seeds, "   ")

        assert code == 1
        assert "handle" in err.lower()
        assert yaml.safe_load(seeds.read_text())["handles"] == []

    def test_a_blank_venue_name_is_not_added_to_seeds(self, tmp_path):
        seeds = tmp_path / "seeds.yaml"
        seeds.write_text("handles: []\nvenues: []\n")

        code, _, err = _invoke_add_source(seeds, "--venue", "  ", "--address", "1 High St")

        assert code == 1
        assert yaml.safe_load(seeds.read_text())["venues"] == []

    def test_a_real_handle_is_still_added(self, tmp_path):
        """The guard must not make the command useless."""
        seeds = tmp_path / "seeds.yaml"
        seeds.write_text("handles: []\nvenues: []\n")

        code, out, _ = _invoke_add_source(seeds, "@jazzclub")

        assert code == 0
        assert yaml.safe_load(seeds.read_text())["handles"] == ["@jazzclub"]
