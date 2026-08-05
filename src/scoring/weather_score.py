"""Turns an event's stored weather into a signed adjustment on its score.

Comfort is computed here, at ranking time, rather than during enrichment. The
enrichment stage deliberately persists raw readings instead of a derived
verdict, so retuned curves can rescore history without refetching anything —
storing an adjustment alongside the readings would throw that away.

Applicability is strict and has two non-obvious cases. An indoor event is never
adjusted, because the weather has no bearing on it. An event with no weather at
all is also never adjusted: beyond the forecast horizon we do not know what the
night holds, and not knowing is not a reason to demote something.
"""

from __future__ import annotations

from typing import Any

from src.config import WeatherConfig
from src.enrichment.comfort import compute_comfort
from src.models.event import Event
from src.scoring.similarity import Reason

WEATHER_FACTOR = "weather_adjustment"

#: Only events actually held outdoors are exposed to the weather.
OUTDOOR = "outdoor"


def select_readings(weather: dict[str, Any] | None) -> dict[str, Any] | None:
    """Pick the readings to score from an event's stored weather record.

    Prefers `observed` over `forecast`, so an event backfilled with what
    actually happened rescores against reality rather than against a prediction
    that may have been issued days earlier.

    Args:
        weather: The event's persisted weather record, or None.

    Returns:
        The readings to score, or None when nothing usable is stored.
    """
    if not weather:
        return None

    observed = weather.get("observed")
    if observed:
        return dict(observed)

    forecast = weather.get("forecast") or {}
    hour = forecast.get("hour")
    return dict(hour) if hour else None


def weather_adjustment(event: Event, config: WeatherConfig) -> tuple[float, Reason | None]:
    """Score one event's weather.

    Args:
        event: An enriched event carrying `setting` and `weather`.
        config: Comfort curves, condition penalties, and adjustment caps.

    Returns:
        The signed adjustment and the Reason explaining it, or (0.0, None) when
        weather does not apply to this event.
    """
    if event.setting != OUTDOOR:
        return 0.0, None

    readings = select_readings(event.weather)
    if readings is None:
        return 0.0, None

    result = compute_comfort(readings, config)
    if not result.factors:
        # Nothing was scorable — an empty record, or readings the curves do not
        # cover. Treated as "unknown", not as "bad".
        return 0.0, None

    return result.adjustment, Reason(
        factor=WEATHER_FACTOR,
        # This Reason is reused from the semantic scorer, where these two fields
        # mean "which preference matched" and "how closely". Here they carry the
        # readings used and the raw comfort they produced, which is the same
        # question — what was compared, and how well did it do.
        matched_preference=result.detail,
        similarity=result.comfort,
        contribution=result.adjustment,
        direction="positive" if result.adjustment >= 0 else "negative",
        tag=None,
    )
