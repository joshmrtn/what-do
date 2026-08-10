"""Unit tests for CLI argument handling and dispatch."""

import io
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
import yaml

from src.models.event import Event
from src.models.recommendation import Recommendation
from src.presentation.cli import ViewSettings, run
from src.scoring.similarity import Reason

TZ = timezone(timedelta(hours=-4))
TODAY = date(2025, 6, 21)
NOW = datetime(2025, 6, 21, 17, 0, tzinfo=TZ)


def _event(
    event_id: str,
    title: str,
    start: datetime | None,
    sunset: datetime | None = None,
) -> Event:
    return Event(
        event_id=event_id,
        source_event_candidates=[],
        source_type="instagram",
        created_at=NOW,
        updated_at=NOW,
        title=title,
        venue="Somewhere",
        start_time=start,
        astronomical_data={"sunset": sunset.isoformat()} if sunset else None,
    )


def _pair(
    event_id: str,
    title: str,
    start: datetime | None,
    tier: str = "top_pick",
    rank: int = 1,
    sunset: datetime | None = None,
) -> tuple[Recommendation, Event]:
    recommendation = Recommendation(
        recommendation_id=f"{TODAY.isoformat()}:{event_id}",
        event_id=event_id,
        run_date=TODAY,
        base_score=0.42,
        weather_adjustment=0.05,
        tag_confidence=1.0,
        final_score=0.68,
        match="yes",
        tier=tier,
        rank=rank,
        reasons=[
            Reason(
                factor="like_similarity",
                matched_preference="karaoke night",
                similarity=0.87,
                contribution=0.8,
                direction="positive",
                tag="karaoke",
            )
        ],
    )
    return recommendation, _event(event_id, title, start, sunset)


SUNSET = datetime(2025, 6, 21, 20, 15, tzinfo=TZ)

PAIRS = [
    _pair("a", "Tonight Early", datetime(2025, 6, 21, 18, 0, tzinfo=TZ), rank=1, sunset=SUNSET),
    _pair("b", "Tonight Late", datetime(2025, 6, 21, 21, 0, tzinfo=TZ), rank=2, sunset=SUNSET),
    _pair("c", "Tomorrow", datetime(2025, 6, 22, 20, 0, tzinfo=TZ), rank=3, sunset=SUNSET),
    _pair("d", "Undated", None, rank=4),
    _pair("e", "Low Ranked", datetime(2025, 6, 21, 19, 0, tzinfo=TZ), tier="everything_else",
          rank=5, sunset=SUNSET),
]

ALL_EVENTS = [e for _, e in PAIRS]

ZONE = ZoneInfo("America/New_York")
VIEW = ViewSettings(zone=ZONE, day_starts_at=time(4, 0))


class _Harness:
    """Captures a CLI invocation's streams and what it asked the database for."""

    def __init__(self, pairs=None, events=None, db_ready=True, now=None, view=None):
        self.pairs = PAIRS if pairs is None else pairs
        self.events = ALL_EVENTS if events is None else events
        self.db_ready = db_ready
        self.now = NOW if now is None else now
        self.view = VIEW if view is None else view
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.requested_run_date: date | None = None
        self.load_ranked_calls = 0
        self.load_events_calls = 0

    def _load_ranked(self, db_path, run_date=None, tier_for=None):
        self.load_ranked_calls += 1
        self.requested_run_date = run_date
        self.tier_for = tier_for
        return self.pairs

    def _load_events(self, db_path):
        self.load_events_calls += 1
        return self.events

    def invoke(self, *argv: str) -> int:
        return run(
            list(argv),
            get_now=lambda: self.now,
            load_view_settings=lambda: self.view,
            stdout=self.stdout,
            stderr=self.stderr,
            load_pairs=self._load_ranked,
            load_all_events=self._load_events,
            db_ready=lambda _: self.db_ready,
        )

    @property
    def out(self) -> str:
        return self.stdout.getvalue()

    @property
    def err(self) -> str:
        return self.stderr.getvalue()


class TestDefaultView:
    def test_exits_zero(self):
        harness = _Harness()

        assert harness.invoke() == 0

    def test_shows_only_todays_events(self):
        harness = _Harness()
        harness.invoke()

        assert "Tonight Early" in harness.out
        assert "Tomorrow" not in harness.out

    def test_shows_undated_events(self):
        """Not knowing when it is does not mean losing it."""
        harness = _Harness()
        harness.invoke()

        assert "Undated" in harness.out

    def test_folds_the_bottom_tier_but_keeps_its_count_visible(self):
        harness = _Harness()
        harness.invoke()

        assert "Low Ranked" not in harness.out
        assert "1 more" in harness.out

    def test_all_expands_the_bottom_tier(self):
        harness = _Harness()
        harness.invoke("--all")

        assert "Low Ranked" in harness.out

    def test_heading_names_the_day_being_shown(self):
        harness = _Harness()
        harness.invoke()

        assert "21" in harness.out and "June" in harness.out

    def test_verbose_shows_score_components(self):
        harness = _Harness()
        harness.invoke("-v")

        assert "base" in harness.out

    def test_reads_the_latest_run_by_default(self):
        harness = _Harness()
        harness.invoke()

        assert harness.requested_run_date is None

    def test_run_date_pins_an_earlier_run(self):
        harness = _Harness()
        harness.invoke("--run-date", "2025-06-20")

        assert harness.requested_run_date == date(2025, 6, 20)

    def test_a_malformed_run_date_is_a_usage_error(self):
        harness = _Harness()

        assert harness.invoke("--run-date", "last tuesday") == 1
        assert "run-date" in harness.err


class TestTimeFilter:
    def test_keeps_only_events_in_the_window(self):
        harness = _Harness()
        harness.invoke("--time", "20:30-23:30")

        assert "Tonight Late" in harness.out
        assert "Tonight Early" not in harness.out

    def test_drops_undated_events(self):
        """A timing filter cannot make a claim about an event with no time."""
        harness = _Harness()
        harness.invoke("--time", "20:30-23:30")

        assert "Undated" not in harness.out

    def test_a_malformed_window_exits_one(self):
        harness = _Harness()

        assert harness.invoke("--time", "tonight") == 1

    def test_a_malformed_window_explains_itself_on_stderr(self):
        harness = _Harness()
        harness.invoke("--time", "tonight")

        assert "HH:MM-HH:MM" in harness.err
        assert harness.out == ""


class TestAfterSunset:
    def test_keeps_only_events_after_sunset(self):
        harness = _Harness()
        harness.invoke("--after-sunset")

        assert "Tonight Late" in harness.out
        assert "Tonight Early" not in harness.out

    def test_composes_with_the_time_filter(self):
        harness = _Harness()
        harness.invoke("--after-sunset", "--time", "18:00-23:30")

        assert "Tonight Late" in harness.out
        assert "Tonight Early" not in harness.out


class TestRawMode:
    def test_lists_every_event_regardless_of_date(self):
        harness = _Harness()
        harness.invoke("--raw")

        assert "Tomorrow" in harness.out
        assert "Low Ranked" in harness.out

    def test_bypasses_ranking_entirely(self):
        harness = _Harness()
        harness.invoke("--raw")

        assert harness.load_ranked_calls == 0
        assert harness.load_events_calls == 1
        assert "TOP PICKS" not in harness.out

    def test_ignores_the_time_filter(self):
        """--raw is the escape hatch; it must not quietly apply a filter."""
        harness = _Harness()
        harness.invoke("--raw", "--time", "20:30-23:30")

        assert "Tonight Early" in harness.out


class TestEmptyDatabase:
    def test_no_recommendations_is_not_an_error(self):
        harness = _Harness(pairs=[])

        assert harness.invoke() == 0

    def test_it_points_at_the_batch_rather_than_failing_silently(self):
        harness = _Harness(pairs=[])
        harness.invoke()

        assert "batch" in harness.out.lower()

    def test_an_empty_raw_view_says_so(self):
        harness = _Harness(events=[])
        harness.invoke("--raw")

        assert "No events" in harness.out

    def test_an_uninitialised_database_is_reported_before_any_read(self):
        """No batch has run yet: say so rather than reading a table that is absent."""
        harness = _Harness(db_ready=False)

        assert harness.invoke() == 0
        assert harness.load_ranked_calls == 0
        assert "batch" in harness.out.lower()

    def test_an_uninitialised_database_short_circuits_raw_too(self):
        harness = _Harness(db_ready=False)

        assert harness.invoke("--raw") == 0
        assert harness.load_events_calls == 0


class TestColor:
    def test_no_ansi_when_output_is_not_a_terminal(self):
        harness = _Harness()
        harness.invoke()

        assert "\033[" not in harness.out


class TestAddSource:
    def test_writes_a_handle_to_seeds(self, tmp_path):
        seeds = tmp_path / "seeds.yaml"
        harness = _Harness()

        assert harness.invoke("add-source", "@jazzclub", "--seeds-file", str(seeds)) == 0
        assert "@jazzclub" in yaml.safe_load(seeds.read_text())["handles"]

    def test_adding_the_same_handle_twice_does_not_duplicate_it(self, tmp_path):
        seeds = tmp_path / "seeds.yaml"
        harness = _Harness()
        harness.invoke("add-source", "@jazzclub", "--seeds-file", str(seeds))
        harness.invoke("add-source", "@jazzclub", "--seeds-file", str(seeds))

        assert yaml.safe_load(seeds.read_text())["handles"] == ["@jazzclub"]

    def test_re_adding_a_handle_is_a_friendly_message_not_an_error(self, tmp_path):
        seeds = tmp_path / "seeds.yaml"
        harness = _Harness()
        harness.invoke("add-source", "@jazzclub", "--seeds-file", str(seeds))

        assert harness.invoke("add-source", "@jazzclub", "--seeds-file", str(seeds)) == 0
        assert "already" in harness.out.lower()
        assert harness.err == ""

    def test_a_handle_without_an_at_sign_is_normalised(self, tmp_path):
        seeds = tmp_path / "seeds.yaml"
        harness = _Harness()
        harness.invoke("add-source", "jazzclub", "--seeds-file", str(seeds))

        assert yaml.safe_load(seeds.read_text())["handles"] == ["@jazzclub"]

    def test_writes_a_venue_with_its_address(self, tmp_path):
        seeds = tmp_path / "seeds.yaml"
        harness = _Harness()
        harness.invoke(
            "add-source", "--venue", "The Dive Bar", "--address", "1 Main St",
            "--seeds-file", str(seeds),
        )

        venues = yaml.safe_load(seeds.read_text())["venues"]
        assert venues == [{"name": "The Dive Bar", "address": "1 Main St"}]

    def test_adding_the_same_venue_twice_does_not_duplicate_it(self, tmp_path):
        seeds = tmp_path / "seeds.yaml"
        harness = _Harness()
        args = ("add-source", "--venue", "The Dive Bar", "--address", "1 Main St",
                "--seeds-file", str(seeds))
        harness.invoke(*args)
        harness.invoke(*args)

        assert len(yaml.safe_load(seeds.read_text())["venues"]) == 1

    def test_a_venue_without_an_address_is_a_usage_error(self, tmp_path):
        seeds = tmp_path / "seeds.yaml"
        harness = _Harness()

        assert harness.invoke("add-source", "--venue", "The Dive Bar",
                              "--seeds-file", str(seeds)) == 1
        assert "address" in harness.err.lower()

    def test_no_arguments_at_all_is_a_usage_error(self, tmp_path):
        seeds = tmp_path / "seeds.yaml"
        harness = _Harness()

        assert harness.invoke("add-source", "--seeds-file", str(seeds)) == 1

    def test_an_existing_seeds_file_is_preserved(self, tmp_path):
        seeds = tmp_path / "seeds.yaml"
        seeds.write_text(yaml.dump({"handles": ["@existing"], "venues": []}))
        harness = _Harness()
        harness.invoke("add-source", "@new", "--seeds-file", str(seeds))

        assert yaml.safe_load(seeds.read_text())["handles"] == ["@existing", "@new"]

    def test_add_source_never_touches_the_database(self, tmp_path):
        harness = _Harness()
        harness.invoke("add-source", "@jazzclub", "--seeds-file", str(tmp_path / "s.yaml"))

        assert harness.load_ranked_calls == 0
        assert harness.load_events_calls == 0


class TestNightWindow:
    """#18: the day comes from the configured zone, and rolls over at 04:00."""

    def test_uses_the_configured_zone_not_the_system_date(self):
        """22:00 in New York is already tomorrow in UTC.

        Taking the date from the machine would show tomorrow's listing while
        the user is still in tonight.
        """
        harness = _Harness(now=datetime(2025, 6, 22, 2, 0, tzinfo=timezone.utc))

        harness.invoke()

        assert "Tonight Early" in harness.out
        assert "Tomorrow" not in harness.out

    def test_after_midnight_still_shows_the_evening_in_progress(self):
        """The defect in #18: at 00:30 the calendar flipped and the answer emptied."""
        harness = _Harness(now=datetime(2025, 6, 22, 0, 30, tzinfo=ZONE))

        harness.invoke()

        assert "Tonight Early" in harness.out
        assert "Tonight Late" in harness.out
        assert "Tomorrow" not in harness.out

    def test_heading_names_the_night_not_the_calendar_date(self):
        harness = _Harness(now=datetime(2025, 6, 22, 0, 30, tzinfo=ZONE))

        harness.invoke()

        assert "Saturday 21 June" in harness.out

    def test_after_the_rollover_the_next_night_begins(self):
        harness = _Harness(now=datetime(2025, 6, 22, 4, 30, tzinfo=ZONE))

        harness.invoke()

        assert "Tomorrow" in harness.out
        assert "Tonight Early" not in harness.out

    def test_a_midnight_rollover_restores_calendar_days(self):
        harness = _Harness(
            now=datetime(2025, 6, 22, 0, 30, tzinfo=ZONE),
            view=ViewSettings(zone=ZONE, day_starts_at=time(0, 0)),
        )

        harness.invoke()

        assert "Tonight Early" not in harness.out

    def test_undated_events_survive_the_window(self):
        """A missing start time is a gap in what we know, not evidence about when."""
        harness = _Harness()

        harness.invoke()

        assert "Undated" in harness.out


class TestViewSettingsFallback:
    """The CLI must stay usable when there is no config to read."""

    def test_warning_goes_to_stderr_and_leaves_stdout_clean(self):
        harness = _Harness(
            view=ViewSettings(
                zone=ZONE, day_starts_at=time(4, 0), warning="Warning: no usable config"
            )
        )

        exit_code = harness.invoke()

        assert exit_code == 0
        assert "Warning: no usable config" in harness.err
        assert "Warning" not in harness.out

    def test_no_warning_emitted_when_config_loaded(self):
        harness = _Harness()

        harness.invoke()

        assert harness.err == ""

    def test_no_warning_when_the_fallback_changed_nothing(self):
        """`--raw` never consults the zone, so a guess there is not worth saying."""
        harness = _Harness(
            view=ViewSettings(
                zone=ZONE, day_starts_at=time(4, 0), warning="Warning: no usable config"
            )
        )

        harness.invoke("--raw")

        assert harness.err == ""
