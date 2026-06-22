"""Synthetic activity generator and time-window DSL parser."""

import re
from datetime import date, datetime, timedelta
from typing import Callable

from src.config import SyntheticActivityRule, SyntheticConditions
from src.enrichment.astronomical import AstronomicalData
from src.models.event import Event

# ---------------------------------------------------------------------------
# Time-window DSL
# ---------------------------------------------------------------------------

_ANCHOR_RE = re.compile(
    r"^(sunrise|sunset|dawn|dusk)(?:_(plus|minus)_(\d+(?:\.\d+)?)(h|m))?$"
)


def _get_anchor(name: str, astro: AstronomicalData) -> datetime:
    """Return the named astronomical anchor datetime."""
    return getattr(astro, name)


def _parse_anchor_expr(expr: str, astro: AstronomicalData) -> datetime:
    """Parse 'sunset', 'sunset_plus_1h', 'sunrise_minus_30m', etc."""
    m = _ANCHOR_RE.match(expr.strip())
    if not m:
        raise ValueError(f"Cannot parse anchor expression: {expr!r}")

    base = _get_anchor(m.group(1), astro)

    if m.group(2) is None:
        return base

    amount = float(m.group(3))
    delta = timedelta(hours=amount) if m.group(4) == "h" else timedelta(minutes=amount)
    return base + delta if m.group(2) == "plus" else base - delta


def parse_time_window(
    window_str: str,
    astro: AstronomicalData,
) -> tuple[datetime, datetime]:
    """Parse a time-window DSL string into (start, end) timezone-aware datetimes.

    Supported forms:
        "after <anchor>"                    → anchor to 23:59:59 local
        "before <anchor>"                   → 00:00:00 local to anchor
        "<anchor_expr> to <anchor_expr>"    → bounded range with optional offsets
    """
    s = window_str.strip()

    if s.startswith("after "):
        anchor = _get_anchor(s[6:].strip(), astro)
        end = anchor.replace(hour=23, minute=59, second=59, microsecond=0)
        return anchor, end

    if s.startswith("before "):
        anchor = _get_anchor(s[7:].strip(), astro)
        start = anchor.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, anchor

    if " to " in s:
        left, right = s.split(" to ", 1)
        return _parse_anchor_expr(left, astro), _parse_anchor_expr(right, astro)

    raise ValueError(f"Cannot parse time window: {window_str!r}")


# ---------------------------------------------------------------------------
# Condition evaluation
# ---------------------------------------------------------------------------


def _conditions_met(conditions: SyntheticConditions, weather: dict | None) -> bool:
    """Return True if all weather-based conditions in the rule are satisfied."""
    needs_weather = (
        conditions.min_temp_f is not None
        or conditions.max_temp_f is not None
        or bool(conditions.weather)
    )

    if needs_weather and weather is None:
        return False

    if weather is not None:
        temp = weather["temperature_f"]
        if conditions.min_temp_f is not None and temp < conditions.min_temp_f:
            return False
        if conditions.max_temp_f is not None and temp > conditions.max_temp_f:
            return False
        if conditions.weather and weather["condition"] not in conditions.weather:
            return False

    return True


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class SyntheticActivityGenerator:
    """Generates synthetic Event objects from config rules and environmental conditions."""

    def generate(
        self,
        rules: list[SyntheticActivityRule],
        run_date: date,
        weather: dict | None,
        astro: AstronomicalData,
        get_now: Callable[[], datetime] = datetime.now,
    ) -> list[Event]:
        """Return fully-formed Events for every rule whose conditions are satisfied.

        Args:
            rules: Synthetic activity rules from config.
            run_date: The batch run date (synthetic events are generated for today only).
            weather: Today's weather dict, or None if unavailable.
            astro: Astronomical data for run_date.
            get_now: Injectable clock (defaults to datetime.now).

        Returns:
            List of Event objects, one per satisfied rule.
        """
        now = get_now()
        tz = astro.sunrise.tzinfo
        events: list[Event] = []

        for rule in rules:
            if not _conditions_met(rule.conditions, weather):
                continue

            event_id = f"synthetic:{rule.name.lower().replace(' ', '_')}:{run_date}"

            if rule.conditions.time_window:
                start_time, end_time = parse_time_window(rule.conditions.time_window, astro)
            else:
                start_time = datetime(run_date.year, run_date.month, run_date.day, 0, 0, 0, tzinfo=tz)
                end_time = datetime(run_date.year, run_date.month, run_date.day, 23, 59, 59, tzinfo=tz)

            events.append(
                Event(
                    event_id=event_id,
                    source_event_candidates=[],
                    source_type="synthetic",
                    created_at=now,
                    updated_at=now,
                    title=rule.name,
                    tags=list(rule.tags),
                    summary=rule.summary,
                    start_time=start_time,
                    end_time=end_time,
                )
            )

        return events
