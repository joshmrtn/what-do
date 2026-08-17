"""Tests for what the read path can tell the reader about how stale it is.

Two questions, kept apart because they have different answers and different
fixes: *how old is the forecast this ranking used?* and *have the preferences
moved since it was scored?*
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.models.event import Event
from src.models.preference_revision import PreferenceRevision
from src.presentation.freshness import (
    PREFERENCES_CHANGED,
    PREFERENCES_UNCHANGED,
    PREFERENCES_UNKNOWN,
    freshness_notice,
    latest_forecast,
    preference_state,
)

NOW = datetime(2026, 8, 17, 20, 0, tzinfo=timezone.utc)
TTL = timedelta(minutes=60)


def _event(issued_at: datetime | None) -> Event:
    weather = (
        None
        if issued_at is None
        else {
            "sampled_hour": 20,
            "forecast": {"issued_at": issued_at.isoformat(), "hour": {}, "day_series": []},
            "observed": None,
        }
    )
    return Event(
        event_id="evt",
        source_event_candidates=[],
        source_type="instagram",
        created_at=NOW,
        updated_at=NOW,
        title="Test Event",
        weather=weather,
    )


def _revision(content_hash: str) -> PreferenceRevision:
    return PreferenceRevision(captured_at=NOW, content_hash=content_hash, lines=[])


class TestLatestForecast:
    def test_no_events_means_no_forecast(self):
        assert latest_forecast([]) is None

    def test_events_with_no_weather_mean_no_forecast(self):
        """Indoor events and anything beyond the horizon carry none."""
        assert latest_forecast([_event(None), _event(None)]) is None

    def test_the_newest_issue_time_wins(self):
        """The listing is as fresh as the freshest forecast behind it."""
        older = NOW - timedelta(hours=18)
        newer = NOW - timedelta(hours=2)

        assert latest_forecast([_event(older), _event(newer)]) == newer

    def test_an_event_without_weather_does_not_hide_one_with_it(self):
        issued = NOW - timedelta(hours=3)

        assert latest_forecast([_event(None), _event(issued)]) == issued


class TestPreferenceState:
    def test_matching_hashes_are_unchanged(self):
        assert preference_state("abc", _revision("abc")) == PREFERENCES_UNCHANGED

    def test_differing_hashes_are_changed(self):
        assert preference_state("abc", _revision("def")) == PREFERENCES_CHANGED

    def test_no_recorded_revision_is_unknown_not_unchanged(self):
        """An absent record and a matching one must never look the same.

        This is the 1e defect's exact shape: a feature that was never wired
        reports the same silence as one that ran and found nothing wrong. Every
        run stored before the writer existed is in this state.
        """
        assert preference_state("abc", None) == PREFERENCES_UNKNOWN


class TestFreshnessNotice:
    def _notice(self, *, issued_at=None, preferences=PREFERENCES_UNCHANGED):
        return freshness_notice(
            forecast_issued_at=issued_at,
            now=NOW,
            ttl=TTL,
            preferences=preferences,
        )

    def test_a_fresh_forecast_and_unchanged_preferences_say_nothing(self):
        assert self._notice(issued_at=NOW - timedelta(minutes=10)) is None

    def test_a_forecast_past_the_ttl_is_reported_with_its_age(self):
        notice = self._notice(issued_at=NOW - timedelta(hours=14))

        assert notice is not None
        assert "14 hours" in notice

    def test_an_age_under_two_hours_reads_in_minutes(self):
        """"1 hours old" is the kind of wrong that makes a reader distrust it."""
        notice = self._notice(issued_at=NOW - timedelta(minutes=95))

        assert notice is not None
        assert "95 minutes" in notice

    def test_a_forecast_inside_the_ttl_is_not_mentioned(self):
        assert self._notice(issued_at=NOW - timedelta(minutes=59)) is None

    def test_no_forecast_at_all_is_not_reported_as_stale(self):
        """Nothing to refresh is not the same as something gone off.

        An all-indoor listing has no forecast behind it and is not stale for
        want of one.
        """
        assert self._notice(issued_at=None) is None

    def test_changed_preferences_are_reported(self):
        notice = self._notice(preferences=PREFERENCES_CHANGED)

        assert notice is not None
        assert "preference" in notice.lower()

    def test_unknown_preferences_are_not_reported_as_changed(self):
        """Saying "changed" about a run that recorded nothing would be a guess."""
        notice = self._notice(preferences=PREFERENCES_UNKNOWN)

        assert notice is None

    def test_both_stale_reports_both(self):
        notice = self._notice(
            issued_at=NOW - timedelta(hours=14), preferences=PREFERENCES_CHANGED
        )

        assert notice is not None
        assert "14 hours" in notice
        assert "preference" in notice.lower()

    def test_a_forecast_issued_in_the_future_is_not_stale(self):
        """A clock skew must not read as a fourteen-hour-old forecast."""
        assert self._notice(issued_at=NOW + timedelta(minutes=5)) is None
