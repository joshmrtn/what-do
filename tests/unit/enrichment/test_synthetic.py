"""Unit tests for SyntheticActivityGenerator and parse_time_window."""

import zoneinfo
from datetime import date, datetime, time, timedelta, timezone

import pytest

from src.config import SyntheticActivityRule, SyntheticConditions
from src.enrichment.astronomical import AstronomicalData
from src.enrichment.synthetic import SyntheticActivityGenerator, parse_time_window
from src.models.event import Event

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TZ = zoneinfo.ZoneInfo("America/New_York")
RUN_DATE = date(2025, 6, 21)


def _make_astro(
    *,
    dawn_h=4,
    dawn_m=35,
    sunrise_h=5,
    sunrise_m=8,
    sunset_h=20,
    sunset_m=24,
    dusk_h=20,
    dusk_m=57,
) -> AstronomicalData:
    def dt(h, m):
        return datetime(2025, 6, 21, h, m, tzinfo=TZ)

    return AstronomicalData(
        dawn=dt(dawn_h, dawn_m),
        sunrise=dt(sunrise_h, sunrise_m),
        sunset=dt(sunset_h, sunset_m),
        dusk=dt(dusk_h, dusk_m),
    )


ASTRO = _make_astro()

CLEAR_WEATHER = {
    "temperature_f": 70.0,
    "condition": "clear",
    "precipitation_mm": 0.0,
    "wind_speed_mph": 5.0,
}

RAINY_WEATHER = {
    "temperature_f": 55.0,
    "condition": "rain",
    "precipitation_mm": 8.0,
    "wind_speed_mph": 12.0,
}


def _rule(
    name="Evening walk",
    min_temp_f=None,
    max_temp_f=None,
    weather_conditions=None,
    time_window=None,
    tags=None,
    summary="A pleasant evening walk",
) -> SyntheticActivityRule:
    return SyntheticActivityRule(
        name=name,
        conditions=SyntheticConditions(
            min_temp_f=min_temp_f,
            max_temp_f=max_temp_f,
            weather=weather_conditions or [],
            time_window=time_window,
        ),
        tags=tags or ["outdoor", "walking"],
        summary=summary,
    )


GEN = SyntheticActivityGenerator()

# ---------------------------------------------------------------------------
# Condition checks
# ---------------------------------------------------------------------------


def test_all_conditions_met_returns_event():
    rule = _rule(min_temp_f=55.0, max_temp_f=85.0, weather_conditions=["clear"])
    events = GEN.generate([rule], RUN_DATE, CLEAR_WEATHER, ASTRO)
    assert len(events) == 1


def test_min_temp_not_met_returns_no_event():
    rule = _rule(min_temp_f=75.0)  # weather is 70°F — below threshold
    events = GEN.generate([rule], RUN_DATE, CLEAR_WEATHER, ASTRO)
    assert len(events) == 0


def test_max_temp_not_met_returns_no_event():
    rule = _rule(max_temp_f=65.0)  # weather is 70°F — above threshold
    events = GEN.generate([rule], RUN_DATE, CLEAR_WEATHER, ASTRO)
    assert len(events) == 0


def test_weather_condition_not_met_returns_no_event():
    rule = _rule(weather_conditions=["clear"])
    events = GEN.generate([rule], RUN_DATE, RAINY_WEATHER, ASTRO)
    assert len(events) == 0


def test_weather_none_returns_no_event():
    rule = _rule(min_temp_f=55.0, weather_conditions=["clear"])
    events = GEN.generate([rule], RUN_DATE, weather=None, astro=ASTRO)
    assert len(events) == 0


def test_no_conditions_always_generates_event():
    rule = _rule()  # no min/max/weather constraints
    events = GEN.generate([rule], RUN_DATE, CLEAR_WEATHER, ASTRO)
    assert len(events) == 1


def test_no_conditions_generates_event_even_with_none_weather():
    rule = _rule()  # no weather constraint → None weather is fine
    events = GEN.generate([rule], RUN_DATE, weather=None, astro=ASTRO)
    assert len(events) == 1


def test_two_rules_both_satisfied_returns_two_events():
    r1 = _rule(name="Walk", tags=["outdoor"])
    r2 = _rule(name="Bike", tags=["outdoor", "cycling"])
    events = GEN.generate([r1, r2], RUN_DATE, CLEAR_WEATHER, ASTRO)
    assert len(events) == 2


def test_two_rules_one_fails_returns_one_event():
    r1 = _rule(name="Walk", min_temp_f=55.0)  # passes
    r2 = _rule(name="Swim", min_temp_f=80.0)  # fails (70°F)
    events = GEN.generate([r1, r2], RUN_DATE, CLEAR_WEATHER, ASTRO)
    assert len(events) == 1
    assert events[0].title == "Walk"


def test_weather_condition_allows_multiple_acceptable_conditions():
    rule = _rule(weather_conditions=["clear", "partly_cloudy"])
    events = GEN.generate([rule], RUN_DATE, CLEAR_WEATHER, ASTRO)
    assert len(events) == 1


# ---------------------------------------------------------------------------
# Generated Event fields
# ---------------------------------------------------------------------------


def test_event_source_type_is_synthetic():
    rule = _rule()
    events = GEN.generate([rule], RUN_DATE, CLEAR_WEATHER, ASTRO)
    assert events[0].source_type == "synthetic"


def test_event_tags_match_rule():
    rule = _rule(tags=["outdoor", "walking", "low_key"])
    events = GEN.generate([rule], RUN_DATE, CLEAR_WEATHER, ASTRO)
    assert events[0].tags == ["outdoor", "walking", "low_key"]


def test_event_summary_matches_rule():
    rule = _rule(summary="Take a stroll at golden hour")
    events = GEN.generate([rule], RUN_DATE, CLEAR_WEATHER, ASTRO)
    assert events[0].summary == "Take a stroll at golden hour"


def test_event_title_is_rule_name():
    rule = _rule(name="Evening walk")
    events = GEN.generate([rule], RUN_DATE, CLEAR_WEATHER, ASTRO)
    assert events[0].title == "Evening walk"


def test_event_id_is_deterministic():
    rule = _rule(name="Evening walk")
    e1 = GEN.generate([rule], RUN_DATE, CLEAR_WEATHER, ASTRO)[0]
    e2 = GEN.generate([rule], RUN_DATE, CLEAR_WEATHER, ASTRO)[0]
    assert e1.event_id == e2.event_id


def test_event_id_format():
    rule = _rule(name="Evening walk")
    events = GEN.generate([rule], RUN_DATE, CLEAR_WEATHER, ASTRO)
    assert events[0].event_id == f"synthetic:evening_walk:{RUN_DATE}"


def test_different_rule_names_produce_different_ids():
    r1 = _rule(name="Walk")
    r2 = _rule(name="Bike")
    e1 = GEN.generate([r1], RUN_DATE, CLEAR_WEATHER, ASTRO)[0]
    e2 = GEN.generate([r2], RUN_DATE, CLEAR_WEATHER, ASTRO)[0]
    assert e1.event_id != e2.event_id


def test_different_dates_produce_different_ids():
    rule = _rule(name="Walk")
    e1 = GEN.generate([rule], date(2025, 6, 21), CLEAR_WEATHER, ASTRO)[0]
    e2 = GEN.generate([rule], date(2025, 6, 22), CLEAR_WEATHER, ASTRO)[0]
    assert e1.event_id != e2.event_id


# ---------------------------------------------------------------------------
# Time window — no window → full day
# ---------------------------------------------------------------------------


def test_no_time_window_spans_full_day():
    rule = _rule(time_window=None)
    events = GEN.generate([rule], RUN_DATE, CLEAR_WEATHER, ASTRO)
    e = events[0]
    assert e.start_time is not None
    assert e.end_time is not None
    assert e.start_time.hour == 0 and e.start_time.minute == 0
    assert e.end_time.hour == 23 and e.end_time.minute == 59 and e.end_time.second == 59


# ---------------------------------------------------------------------------
# parse_time_window
# ---------------------------------------------------------------------------


def test_parse_bounded_window():
    start, end = parse_time_window("sunset_minus_1h to sunset_plus_2h", ASTRO)
    assert start == ASTRO.sunset - timedelta(hours=1)
    assert end == ASTRO.sunset + timedelta(hours=2)


def test_parse_after_sunset():
    start, end = parse_time_window("after sunset", ASTRO)
    assert start == ASTRO.sunset
    assert end.hour == 23 and end.minute == 59 and end.second == 59


def test_parse_before_sunrise():
    start, end = parse_time_window("before sunrise", ASTRO)
    assert start.hour == 0 and start.minute == 0 and start.second == 0
    assert end == ASTRO.sunrise


def test_parse_after_dawn():
    start, end = parse_time_window("after dawn", ASTRO)
    assert start == ASTRO.dawn
    assert end.hour == 23 and end.minute == 59 and end.second == 59


def test_parse_before_dusk():
    start, end = parse_time_window("before dusk", ASTRO)
    assert start.hour == 0 and start.minute == 0 and start.second == 0
    assert end == ASTRO.dusk


def test_parse_anchor_minus_offset():
    start, end = parse_time_window("sunrise_minus_30m to sunrise_plus_1h", ASTRO)
    assert start == ASTRO.sunrise - timedelta(minutes=30)
    assert end == ASTRO.sunrise + timedelta(hours=1)


def test_parse_result_is_timezone_aware():
    start, end = parse_time_window("after sunset", ASTRO)
    assert start.tzinfo is not None
    assert end.tzinfo is not None


def test_parse_bounded_window_used_in_generated_event():
    rule = _rule(time_window="sunset_minus_1h to sunset_plus_2h")
    events = GEN.generate([rule], RUN_DATE, CLEAR_WEATHER, ASTRO)
    e = events[0]
    assert e.start_time == ASTRO.sunset - timedelta(hours=1)
    assert e.end_time == ASTRO.sunset + timedelta(hours=2)
