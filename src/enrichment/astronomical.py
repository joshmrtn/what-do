"""Astronomical calculator for sunrise, sunset, dawn, and dusk."""

import zoneinfo
from dataclasses import dataclass
from datetime import date, datetime

from astral import LocationInfo
from astral.sun import sun


@dataclass
class AstronomicalData:
    """Timezone-aware solar event times for a given date and location."""

    sunrise: datetime
    sunset: datetime
    dawn: datetime
    dusk: datetime


class AstronomicalCalculator:
    """Calculates solar event times using the astral library (pure, no I/O)."""

    def calculate(
        self, date: date, lat: float, lng: float, tzname: str
    ) -> AstronomicalData:
        """Return solar event times for the given date and location.

        Args:
            date: The date to calculate for.
            lat: Latitude in decimal degrees.
            lng: Longitude in decimal degrees.
            tzname: IANA timezone name (e.g. "America/New_York").

        Returns:
            AstronomicalData with all four times as timezone-aware datetimes.
        """
        tz = zoneinfo.ZoneInfo(tzname)
        loc = LocationInfo(latitude=lat, longitude=lng, timezone=tzname)
        s = sun(loc.observer, date=date, tzinfo=tz)
        return AstronomicalData(
            sunrise=s["sunrise"],
            sunset=s["sunset"],
            dawn=s["dawn"],
            dusk=s["dusk"],
        )
