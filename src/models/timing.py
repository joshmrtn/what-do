"""How much a source told us about when an event happens.

Three states rather than a boolean, because a date with no time is two
different facts. A calendar declaring `VALUE=DATE` means the event genuinely
runs all day; a listing that omits the hour means nobody has published one yet.
Both need a placed start so the night window can position them, and both must
avoid presenting that placed start as though the source had given it — but they
do not read the same to a person deciding whether to go.
"""

from __future__ import annotations

#: A clock time the source actually stated.
EXACT = "exact"

#: A date the source declared as running all day.
ALL_DAY = "all_day"

#: A date whose time was never published.
UNKNOWN = "unknown"

#: The only values `timing` may take.
TIMINGS = (EXACT, ALL_DAY, UNKNOWN)
