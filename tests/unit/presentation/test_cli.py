"""Unit tests for CLI argument handling and dispatch."""

import io
import re
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
import yaml

from src.models.event import Event
from src.models.event_score import EventScore
from src.models.ranked_event import RankedEvent
from src.models.ranking import Ranking
from src.config import (
    DEFAULT_EMBEDDING_MODEL,
    AppConfig,
    ConfigError,
    FeedConfig,
    LocationConfig,
    ScrapingConfig,
    SourcesConfig,
    VenueDiscoveryConfig,
)
from src.presentation.cli import ViewSettings, default_view_settings, run
from src.scoring.similarity import Reason

TZ = timezone(timedelta(hours=-4))
TODAY = date(2025, 6, 21)
NOW = datetime(2025, 6, 21, 17, 0, tzinfo=TZ)


def _event(
    event_id: str,
    title: str,
    start: datetime | None,
    sunset: datetime | None = None,
    source_type: str = "instagram",
    url: str | None = None,
) -> Event:
    return Event(
        event_id=event_id,
        source_event_candidates=[],
        source_type=source_type,
        created_at=NOW,
        updated_at=NOW,
        title=title,
        venue="Somewhere",
        start_time=start,
        url=url,
        astronomical_data={"sunset": sunset.isoformat()} if sunset else None,
    )


def _pair(
    event_id: str,
    title: str,
    start: datetime | None,
    rank: int = 1,
    sunset: datetime | None = None,
    source_type: str = "instagram",
    url: str | None = None,
    run_date: date | None = None,
) -> RankedEvent:
    run_date = TODAY if run_date is None else run_date
    return RankedEvent(
        event=_event(event_id, title, start, sunset, source_type, url),
        score=EventScore(
            event_id=event_id,
            run_date=run_date,
            base_score=0.42,
            tag_confidence=1.0,
            match="yes",
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
        ),
        ranking=Ranking(
            event_id=event_id,
            run_date=run_date,
            weather_adjustment=0.05,
            final_score=0.68,
            rank=rank,
        ),
    )


SUNSET = datetime(2025, 6, 21, 20, 15, tzinfo=TZ)

PAIRS = [
    _pair("a", "Tonight Early", datetime(2025, 6, 21, 18, 0, tzinfo=TZ), rank=1, sunset=SUNSET),
    _pair("b", "Tonight Late", datetime(2025, 6, 21, 21, 0, tzinfo=TZ), rank=2, sunset=SUNSET),
    _pair("c", "Tomorrow", datetime(2025, 6, 22, 20, 0, tzinfo=TZ), rank=3, sunset=SUNSET),
    _pair("d", "Undated", None, rank=4),
    _pair("e", "Low Ranked", datetime(2025, 6, 21, 19, 0, tzinfo=TZ), rank=5, sunset=SUNSET),
]

ALL_EVENTS = [ranked.event for ranked in PAIRS]

ZONE = ZoneInfo("America/New_York")
VIEW = ViewSettings(zone=ZONE, day_starts_at=time(4, 0))


def _config_with(sources: SourcesConfig) -> AppConfig:
    """Minimal real AppConfig — only the view loader's inputs need to be true."""
    return AppConfig(
        location=LocationConfig(
            latitude=0.0,
            longitude=0.0,
            postal_code="00000",
            search_radius_miles=25.0,
            timezone="America/New_York",
        ),
        scraping=ScrapingConfig(),
        venue_discovery=VenueDiscoveryConfig(),
        sources=sources,
    )


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
        self.requested_embedding_model: str | None = None
        self.load_ranked_calls = 0
        self.load_events_calls = 0

    def _load_ranked(self, db_path, run_date=None, embedding_model=None):
        self.load_ranked_calls += 1
        self.requested_run_date = run_date
        self.requested_embedding_model = embedding_model
        return self.pairs

    def _load_events(self, db_path, **kwargs):
        """`**kwargs` deliberately: a double exists to record and forward, and
        the instant it declares the shape of what it forwards it has started
        reimplementing. This broke when the loader gained `embedding_model`."""
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

    def test_the_list_is_cut_at_the_limit_and_the_rest_counted(self):
        harness = _Harness()
        harness.invoke("--limit", "1")

        shown = re.findall(r"^  \d+\. ", harness.out, re.M)
        assert len(shown) == 1
        assert "3 more" in harness.out

    def test_all_shows_every_ranked_event(self):
        harness = _Harness()
        harness.invoke("--all")

        assert "ranked lower (--all)" not in harness.out

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

    def test_refuses_a_time_filter_rather_than_ignoring_it(self):
        """`--raw` is the escape hatch; it must not quietly apply a filter.

        Inverted 2026-08-14: it must not quietly *ignore* one either. Printing
        1667 unfiltered events under a typed `--time` is a listing that looks
        filtered, which is the failure mode this whole audit was about. Making
        `--raw` actually filter is wanted and tracked, and is not a matter of
        passing the value through — the filters take ranked pairs and `--raw`
        reads events.
        """
        harness = _Harness()
        code = harness.invoke("--raw", "--time", "20:30-23:30")

        assert code == 1
        assert "--raw" in harness.err and "--time" in harness.err
        assert harness.out == ""


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



class TestSourceAttributionWiring:
    """The site map reaches the renderer, and it comes from config."""

    def test_the_views_site_map_reaches_the_rendered_output(self):
        view = ViewSettings(
            zone=ZONE,
            day_starts_at=time(4, 0),
            source_urls={"northshorenightout": "https://northshorenightout.com/"},
        )
        pair = _pair(
            "evt-nsno",
            "Trivia",
            datetime(2025, 6, 21, 20, 0, tzinfo=TZ),
            source_type="northshorenightout",
        )
        harness = _Harness(pairs=[pair], view=view)

        assert harness.invoke() == 0
        assert "source: https://northshorenightout.com/" in harness.out

    def test_default_view_settings_builds_the_map_from_config(self, monkeypatch):
        config = _config_with(
            SourcesConfig(
                html_calendars=[
                    FeedConfig(
                        name="nsno",
                        url="https://calendar.google.com/calendar/ical/x/basic.ics",
                        source_type="northshorenightout",
                        site_url="https://northshorenightout.com/",
                    )
                ]
            )
        )
        monkeypatch.setattr("src.presentation.cli.load_config", lambda: config)

        assert default_view_settings().source_urls == {
            "northshorenightout": "https://northshorenightout.com/"
        }

    def test_an_unusable_config_degrades_to_no_attribution(self, monkeypatch):
        def boom():
            raise ConfigError("no config")

        monkeypatch.setattr("src.presentation.cli.load_config", boom)

        settings = default_view_settings()

        assert settings.source_urls == {}
        assert settings.warning is not None


class TestAStaleRankingIsAnnounced:
    """The 2026-08-12 failure mode: today's heading over yesterday's order.

    The batch died before ranking, `latest_run_date()` returned the previous
    night, and the CLI presented it as current. Nothing in the output gave the
    listing's age away.
    """

    def _stale(self, days: int) -> list[RankedEvent]:
        when = TODAY - timedelta(days=days)
        return [
            _pair("a", "Tonight Early", datetime(2025, 6, 21, 18, 0, tzinfo=TZ),
                  rank=1, sunset=SUNSET, run_date=when),
        ]

    def test_a_current_ranking_says_nothing(self):
        harness = _Harness()

        harness.invoke()

        assert "old" not in harness.err

    def test_a_day_old_ranking_is_announced(self):
        harness = _Harness(pairs=self._stale(1))

        harness.invoke()

        assert "1 day old" in harness.err
        assert "2025-06-20" in harness.err

    def test_the_listing_is_still_shown(self):
        """A failed batch must not also cost the listing."""
        harness = _Harness(pairs=self._stale(1))

        assert harness.invoke() == 0
        assert "Tonight Early" in harness.out

    def test_the_notice_stays_off_stdout(self):
        """So a piped or paged listing is unchanged, as every other warning is."""
        harness = _Harness(pairs=self._stale(3))

        harness.invoke()

        assert "3 days old" in harness.err
        assert "days old" not in harness.out


class TestTheViewReadsUnderTheConfiguredModel:
    """The two composition roots must agree on which vectors exist.

    The batch passed `config.models.embeddings` to the event repository; the CLI
    built its own and took the default. Identical only while `config.yaml` names
    the model that happens to be `DEFAULT_EMBEDDING_MODEL` — change that line and
    the batch writes vectors under the new name while the view reads under the
    old one, finding none, with nothing to say so.
    """

    def test_the_configured_model_reaches_the_loader(self):
        view = ViewSettings(
            zone=ZONE, day_starts_at=time(4, 0), embedding_model="some-other-model"
        )
        harness = _Harness(view=view)

        harness.invoke()

        assert harness.requested_embedding_model == "some-other-model"

    def test_an_unreadable_config_falls_back_to_the_default(self):
        """A fresh clone has no config and must still render, as the zone does."""
        harness = _Harness()

        harness.invoke()

        assert harness.requested_embedding_model == DEFAULT_EMBEDDING_MODEL


class TestUpcomingKeepsRankOrder:
    """Grouping by night to anchor the window must not reorder the list.

    The e2e fixture cannot show this: its upcoming events all fall on one night,
    so there is only ever one group and the order survives by accident. Two
    nights with interleaved ranks is what tells the difference.
    """

    def test_ranks_stay_ascending_across_nights(self):
        harness = _Harness()
        # Both after tonight, since that is what `--upcoming` covers. The
        # worse-ranked one is listed first, so night-bucket order and rank order
        # disagree — otherwise the assertion holds by accident.
        harness.pairs = [
            _pair("far", "Farther Night", datetime(2025, 6, 23, 21, 0, tzinfo=TZ), rank=9),
            _pair("near", "Nearer Night", datetime(2025, 6, 22, 21, 0, tzinfo=TZ), rank=2),
        ]

        harness.invoke("--upcoming", "--time", "20:00-23:59")

        assert harness.out.index("Nearer Night") < harness.out.index("Farther Night")


class TestUpcomingAppliesAfterSunset:
    """Discriminating where the e2e fixture cannot: it gives every event the
    same sunset, so nothing is ever dropped by it there."""

    def test_an_event_without_sunset_data_is_dropped(self):
        harness = _Harness()
        tomorrow = datetime(2025, 6, 22, 21, 0, tzinfo=TZ)
        harness.pairs = [
            _pair("has", "Has Sunset", tomorrow, sunset=datetime(2025, 6, 22, 20, 15, tzinfo=TZ)),
            _pair("none", "No Sunset", tomorrow, rank=2, sunset=None),
        ]

        harness.invoke("--upcoming", "--after-sunset")

        assert "Has Sunset" in harness.out
        assert "No Sunset" not in harness.out
