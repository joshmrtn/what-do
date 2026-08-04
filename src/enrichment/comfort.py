"""Weather comfort scoring — maps readings to a signed adjustment on an event's score."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import ComfortCurve, WeatherConfig

#: Reading key holding the categorical condition string.
CONDITION_KEY = "condition"

#: How each reading is rendered in the human-auditable detail line.
_DETAIL_FORMATS: dict[str, str] = {
    "temperature_f": "{:.0f}°F",
    "dew_point_f": "dew point {:.0f}°F",
    "relative_humidity": "{:.0f}% humidity",
    "wind_speed_mph": "wind {:.0f}mph",
    "us_aqi": "AQI {:.0f}",
    "precipitation_mm": "{:.1f}mm precipitation",
}


@dataclass(frozen=True)
class ComfortResult:
    """Weather comfort for one event.

    Args:
        comfort: Aggregate comfort in -1.0..+1.0.
        adjustment: `comfort` scaled by the configured cap for its sign.
        factors: Per-reading comfort values that actually contributed.
        detail: Human-readable summary of the readings used.
    """

    comfort: float = 0.0
    adjustment: float = 0.0
    factors: dict[str, float] = field(default_factory=dict)
    detail: str = ""


def _side_is_unbounded(ideal: float, zero: float, floor: float) -> bool:
    """A side with all three bounds equal imposes no limit in that direction."""
    return ideal == zero == floor


def _ramp(value: float, at_one: float, at_zero: float, at_minus_one: float) -> float:
    """Interpolate along one side of the trapezoid, clamping past the floor."""
    inner_span = at_zero - at_one
    if inner_span == 0:
        inner = 0.0
    else:
        inner = (value - at_one) / inner_span
    if inner <= 1.0:
        return 1.0 - inner

    outer_span = at_minus_one - at_zero
    if outer_span == 0:
        return -1.0
    return max(-1.0, -(value - at_zero) / outer_span)


def curve_value(curve: ComfortCurve, value: float) -> float:
    """Map a single reading through its comfort curve.

    Returns:
        Comfort in -1.0..+1.0; +1.0 anywhere inside the ideal band.
    """
    ideal_lo, ideal_hi = curve.ideal
    if ideal_lo <= value <= ideal_hi:
        return 1.0

    if value > ideal_hi:
        if _side_is_unbounded(ideal_hi, curve.zero[1], curve.floor[1]):
            return 1.0
        return _ramp(value, ideal_hi, curve.zero[1], curve.floor[1])

    if _side_is_unbounded(ideal_lo, curve.zero[0], curve.floor[0]):
        return 1.0
    # Mirror the high side by negating, so one ramp implementation serves both.
    return _ramp(-value, -ideal_lo, -curve.zero[0], -curve.floor[0])


def _suppressed(readings: dict[str, Any], config: WeatherConfig) -> set[str]:
    """Factors standing in for a reading that is present, and so not needed."""
    return {
        name
        for name, curve in config.comfort.items()
        if curve.fallback_for is not None and readings.get(curve.fallback_for) is not None
    }


def _format_detail(factors: dict[str, float], readings: dict[str, Any]) -> str:
    parts = [
        _DETAIL_FORMATS[name].format(readings[name])
        for name in factors
        if name in _DETAIL_FORMATS
    ]
    condition = readings.get(CONDITION_KEY)
    if condition:
        parts.append(str(condition).replace("_", " "))
    return ", ".join(parts)


def compute_comfort(readings: dict[str, Any], config: WeatherConfig) -> ComfortResult:
    """Score weather readings into a signed adjustment.

    Missing readings are dropped and the remaining weights renormalised — never
    treated as zero, so a short air-quality forecast horizon cannot masquerade as
    bad air. Capping factors and condition penalties bound the result from above,
    so a pleasant average can never outvote a thunderstorm.

    Args:
        readings: Weather values keyed by factor name, plus `condition`.
        config: Curves, condition penalties, and adjustment caps.

    Returns:
        ComfortResult; all-zero when nothing is known.
    """
    suppressed = _suppressed(readings, config)

    factors: dict[str, float] = {}
    caps: list[float] = []
    superseded: set[str] = set()
    weighted_total = 0.0
    weight_total = 0.0

    for name, curve in config.comfort.items():
        if name in suppressed:
            continue
        reading = readings.get(name)
        if reading is None:
            continue

        value = curve_value(curve, float(reading))
        factors[name] = value

        if curve.supersedes:
            caps.append(value)
            superseded.update(curve.supersedes)
        else:
            weighted_total += value * curve.weight
            weight_total += curve.weight

    comfort = weighted_total / weight_total if weight_total > 0 else 0.0

    condition = readings.get(CONDITION_KEY)
    if condition is not None and condition not in superseded:
        penalty = config.condition_penalty.get(str(condition))
        # Only a negative penalty caps: 0.0 means "no objection", not "no comfort".
        if penalty is not None and penalty < 0:
            caps.append(penalty)

    for cap in caps:
        comfort = min(comfort, cap)

    scale = (
        config.max_positive_adjustment if comfort >= 0 else config.max_negative_adjustment
    )
    return ComfortResult(
        comfort=comfort,
        adjustment=comfort * scale,
        factors=factors,
        detail=_format_detail(factors, readings),
    )
