"""Whether a source's publisher may be trusted to identify its own listings.

An adapter asks one question — *do I derive this id from content?* — and the
answer is deliberately not its own to make. It comes from config today, and from
config plus a measured, one-way latch once the churn detector can arm itself.

Five adapters key a candidate on something the publisher supplied: the ICS UID,
a Veezi `session_id`, a Cabot `event_id`, an AMC showtime id, a social post id.
Two of those had no alternative path at all, which is the configuration that let
`northshorenightout` reach 860 rows for 156 listings before anybody looked.

**The tri-state is not the adapter's business.** `auto` and `publisher` differ
only in whether the latch may act, which is a question for the latch. Both mean
*use the publisher's id* right now, so the rule handed to an adapter is a plain
boolean.
"""

from __future__ import annotations

from typing import Callable

from src.config import IDENTITY_CONTENT, SourcesConfig

#: Given a `source` — the feed, never the category — does its candidate id come
#: from the listing's content rather than from the publisher's identifier?
ContentIdRule = Callable[[str], bool]


def content_id_rule(sources: SourcesConfig) -> ContentIdRule:
    """The rule as configuration alone states it.

    Args:
        sources: The loaded `sources` config, holding any per-source assignment.

    Returns:
        A rule answering `True` only for sources explicitly set to `content`.
        `auto` answers `False` here — until a latch exists to observe churn,
        trusting the publisher is what `auto` resolves to.
    """

    def rule(source: str) -> bool:
        return sources.identity_for(source) == IDENTITY_CONTENT

    return rule
