"""Rendering for the CLI.

Pure: strings in, one string out. Nothing here reads a clock, touches the
database, or reorders anything — sections are emitted in tier order and events
within them in the rank the batch assigned.

The bottom tier is folded rather than dropped, and its count is always printed.
Thresholds are uncalibrated this early, so a low-ranked event has to stay one
flag away, not disappear.
"""

from __future__ import annotations

from datetime import datetime

from src.models.event import Event
from src.models.recommendation import Recommendation
from src.presentation.filters import RankedPair
from src.scoring.ranking import (
    CONFIDENCE_FACTOR,
    EVERYTHING_ELSE,
    MATCH_FACTOR,
    TOP_PICK,
    WORTH_CONSIDERING,
)
from src.scoring.similarity import DISLIKE_FACTOR, LIKE_FACTOR, Reason
from src.scoring.weather_score import WEATHER_FACTOR

_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"

_SECTION_TITLES = {
    TOP_PICK: "TOP PICKS",
    WORTH_CONSIDERING: "WORTH CONSIDERING",
    EVERYTHING_ELSE: "EVERYTHING ELSE",
}
_UNDATED_TITLE = "UNDATED — timing unconfirmed"

#: Semantic reasons shown per event before `--verbose`. Two is enough to say
#: what an event is; the rest is an audit trail.
_DEFAULT_REASON_LIMIT = 2

_SEMANTIC_FACTORS = (LIKE_FACTOR, DISLIKE_FACTOR)

#: Factors whose `matched_preference` carries a detail string rather than a
#: preference line. See `_format_reason`.
_FACTOR_LABELS = {
    WEATHER_FACTOR: "weather",
    MATCH_FACTOR: "match",
    CONFIDENCE_FACTOR: "thin evidence",
}

_UNTITLED = "(untitled)"


def render_recommendations(
    pairs: list[RankedPair],
    *,
    heading: str | None = None,
    verbose: bool = False,
    show_all: bool = False,
    color: bool = False,
) -> str:
    """Render ranked events as sectioned text.

    Args:
        pairs: Ranked pairs, already filtered and in rank order.
        heading: Optional line above the sections, e.g. the date being shown.
        verbose: Show every reason plus the score components behind the tier.
        show_all: Expand the bottom tier instead of folding it to a count.
        color: Emit ANSI styling. Off when stdout is not a terminal.

    Returns:
        The rendered view, ending in a newline.
    """
    if not pairs:
        return _join([_style("No events to show.", _DIM, color)])

    top = [p for p in pairs if p[0].tier == TOP_PICK and p[1].start_time is not None]
    worth = [p for p in pairs if p[0].tier == WORTH_CONSIDERING and p[1].start_time is not None]
    undated = [p for p in pairs if p[0].tier != EVERYTHING_ELSE and p[1].start_time is None]
    folded = [p for p in pairs if p[0].tier == EVERYTHING_ELSE]

    lines: list[str] = []
    if heading:
        lines += [_style(heading, _BOLD, color), ""]

    for section, title in ((top, _SECTION_TITLES[TOP_PICK]),
                           (worth, _SECTION_TITLES[WORTH_CONSIDERING]),
                           (undated, _UNDATED_TITLE)):
        lines += _render_section(section, title, verbose=verbose, color=color)

    if show_all:
        lines += _render_section(
            folded, _SECTION_TITLES[EVERYTHING_ELSE], verbose=verbose, color=color
        )
    elif folded:
        plural = "" if len(folded) == 1 else "s"
        lines.append(
            _style(f"+ {len(folded)} more event{plural} ranked lower (--all)", _DIM, color)
        )

    return _join(lines)


def render_raw(events: list[Event]) -> str:
    """Render events with no scoring applied, earliest first, undated last.

    The escape hatch from ranking entirely: what the batch collected, before any
    judgement about it.
    """
    if not events:
        return _join(["No events in the database."])

    ordered = sorted(events, key=_start_sort_key)
    lines = [f"{len(events)} event{'' if len(events) == 1 else 's'}", ""]
    for event in ordered:
        when = _format_time(event.start_time) if event.start_time else "  --  "
        lines.append(f"  {when}  {_describe(event)}  [{event.source_type}]")

    return _join(lines)


def _render_section(
    pairs: list[RankedPair], title: str, *, verbose: bool, color: bool
) -> list[str]:
    """Render one titled section, or nothing at all when it is empty."""
    if not pairs:
        return []

    lines = [_style(title, _BOLD, color), ""]
    for recommendation, event in pairs:
        when = _format_time(event.start_time) if event.start_time else ""
        prefix = f"  {recommendation.rank}. "
        lines.append(f"{prefix}{when + '  ' if when else ''}{_describe(event)}")
        if verbose:
            lines.append(_style(f"      {_components(recommendation)}", _DIM, color))
        for reason in _visible_reasons(recommendation.reasons, verbose=verbose):
            lines.append(_style(f"      {_format_reason(reason)}", _DIM, color))
        lines.append("")

    return lines


def _visible_reasons(reasons: list[Reason], *, verbose: bool) -> list[Reason]:
    """Pick which reasons to show, strongest first.

    Default view keeps the two strongest semantic reasons — what the event *is* —
    plus the weather adjustment, which is why an outdoor event moved tonight
    rather than a property of the event at all.
    """
    if verbose:
        return reasons

    semantic = [r for r in reasons if r.factor in _SEMANTIC_FACTORS]
    semantic.sort(key=lambda r: abs(r.contribution), reverse=True)
    weather = [r for r in reasons if r.factor == WEATHER_FACTOR]

    return semantic[:_DEFAULT_REASON_LIMIT] + weather


def _format_reason(reason: Reason) -> str:
    """One reason as a single line, signed by its direction.

    Only the semantic factors were compared against a preference line, so only
    they get the `<-` arrow. Weather, match and confidence reuse the same
    `matched_preference` field to carry a detail string, and writing that as a
    preference match would claim something the scorer never did.
    """
    sign = "+" if reason.direction == "positive" else "-"
    magnitude = f"({reason.contribution:+.2f})"

    if reason.factor in _SEMANTIC_FACTORS:
        subject = reason.tag or "summary"
        return f'{sign} {subject} <- "{reason.matched_preference}" {magnitude}'

    label = _FACTOR_LABELS.get(reason.factor, reason.factor.replace("_", " "))

    return f"{sign} {label}: {reason.matched_preference} {magnitude}"


def _components(recommendation: Recommendation) -> str:
    """The numbers the tier was cut on, for when the tier looks wrong."""
    return (
        f"score {recommendation.final_score:+.2f}  "
        f"base {recommendation.base_score:+.2f}  "
        f"weather {recommendation.weather_adjustment:+.2f}  "
        f"confidence {recommendation.tag_confidence:.2f}  "
        f"match {recommendation.match}"
    )


def _describe(event: Event) -> str:
    """Title and venue, with no dangling separator when the venue is missing."""
    title = event.title or _UNTITLED

    return f"{title} — {event.venue}" if event.venue else title


def _format_time(when: datetime) -> str:
    return when.strftime("%H:%M")


def _start_sort_key(event: Event) -> tuple[int, str]:
    """Sort timed events first by start, then undated ones by title."""
    if event.start_time is None:
        return (1, event.title or "")
    return (0, event.start_time.isoformat())


def _style(text: str, code: str, color: bool) -> str:
    return f"{code}{text}{_RESET}" if color else text


def _join(lines: list[str]) -> str:
    return "\n".join(lines).rstrip("\n") + "\n"
