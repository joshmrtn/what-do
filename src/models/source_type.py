"""The `source_type` values this project ships.

Not an enumeration of everything valid: `source_type` is an **open set**. A
calendar feed declared in `config.yaml` supplies its own, defaulting to the feed
name, and `similarity.py` already treats the field that way — a dict lookup with
a default rather than an exhaustive match. So this is the set of built-ins, and
its job is to stop five adapters and one generator each spelling their own
literal in their own package.

Every value here except `SYNTHETIC` names *where data came from*. `SYNTHETIC`
names data the system authored itself, which is why it is the one value stages
branch on.
"""

from __future__ import annotations

APIFY = "apify"
PICUKI = "picuki"
DUMPOR = "dumpor"
CINEMA_VEEZI = "cinema_veezi"
AMC = "amc"
SYNTHETIC = "synthetic"

#: Values a config-declared feed may not claim for itself. `synthetic` exempts
#: an event from LLM extraction and from tag-confidence scaling, so a feed
#: named it would silently inherit both — its tags never refreshed, and its thin
#: evidence never discounted.
RESERVED = frozenset({SYNTHETIC})
