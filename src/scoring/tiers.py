"""Tier labels, derived from a score rather than stored with it.

A tier is presentation polish: it names a band, and nothing is ever withheld
because of one. It is a pure function of `final_score` and the configured
thresholds, so storing it alongside the score would let a later threshold change
leave every stored label disagreeing with what the config now says.

Both the ranking engine and the CLI derive labels through here, so they cannot
drift apart.
"""

from __future__ import annotations

#: Deliberately not "excluded": that name invites a `WHERE tier != ...` in the
#: query layer, and the bottom band is folded in the CLI, never hidden.
TOP_PICK = "top_pick"
WORTH_CONSIDERING = "worth_considering"
EVERYTHING_ELSE = "everything_else"

#: Shipped thresholds, mirroring `ScoringConfig`. Used when no config is to
#: hand — the CLI and the ranking engine both pass their configured values.
DEFAULT_TOP_PICKS_MIN = 0.5
DEFAULT_WORTH_CONSIDERING_MIN = 0.1


def tier_for_score(
    final_score: float, top_picks_min: float, worth_considering_min: float
) -> str:
    """Label a score. Presentation only — never a filter.

    Args:
        final_score: The score being labelled.
        top_picks_min: At or above this, the event is a top pick.
        worth_considering_min: At or above this, it is worth considering.

    Returns:
        One of the three tier names.
    """
    if final_score >= top_picks_min:
        return TOP_PICK
    if final_score >= worth_considering_min:
        return WORTH_CONSIDERING
    return EVERYTHING_ELSE


def default_tier(final_score: float) -> str:
    """Label a score using the shipped thresholds."""
    return tier_for_score(
        final_score, DEFAULT_TOP_PICKS_MIN, DEFAULT_WORTH_CONSIDERING_MIN
    )
