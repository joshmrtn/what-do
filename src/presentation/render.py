"""Rendering for the CLI.

Pure: strings in, one string out. Nothing here reads a clock, touches the
database, or reorders anything — sections are emitted in tier order and events
within them in the rank the batch assigned.

The bottom tier is folded rather than dropped, and its count is always printed.
Thresholds are uncalibrated this early, so a low-ranked event has to stay one
flag away, not disappear.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from src.models.event import Event
from src.models.recommendation import Recommendation
from src.models.timing import ALL_DAY, UNKNOWN
from src.presentation.filters import RankedPair
from src.scoring.ranking import CONFIDENCE_FACTOR, MATCH_FACTOR
from src.scoring.tiers import EVERYTHING_ELSE, TOP_PICK, WORTH_CONSIDERING
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

#: Shown in the time column instead of a clock time nobody published. Padded to
#: the width of `HH:MM` so the titles beside them stay in one column.
_ALL_DAY_LABEL = "all day"
_UNKNOWN_TIME_LABEL = "time TBC"


def render_recommendations(
    pairs: list[RankedPair],
    *,
    heading: str | None = None,
    verbose: bool = False,
    show_all: bool = False,
    color: bool = False,
    source_urls: Mapping[str, str] | None = None,
) -> str:
    """Render ranked events as sectioned text.

    Args:
        pairs: Ranked pairs, already filtered and in rank order.
        heading: Optional line above the sections, e.g. the date being shown.
        verbose: Show every reason plus the score components behind the tier.
        show_all: Expand the bottom tier instead of folding it to a count.
        color: Emit ANSI styling. Off when stdout is not a terminal.
        source_urls: Human-facing page per `source_type`, used only for events
            carrying no URL of their own.

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
        lines += _render_section(
            section, title, verbose=verbose, color=color, source_urls=source_urls
        )

    if show_all:
        lines += _render_section(
            folded,
            _SECTION_TITLES[EVERYTHING_ELSE],
            verbose=verbose,
            color=color,
            source_urls=source_urls,
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
        when = _when_of(event) or "  --  "
        lines.append(f"  {when}  {_describe(event)}  [{event.source_type}]")

    return _join(lines)


def _render_section(
    pairs: list[RankedPair],
    title: str,
    *,
    verbose: bool,
    color: bool,
    source_urls: Mapping[str, str] | None = None,
) -> list[str]:
    """Render one titled section, or nothing at all when it is empty."""
    if not pairs:
        return []

    lines = [_style(title, _BOLD, color), ""]
    for recommendation, event in pairs:
        when = _when_of(event)
        prefix = f"  {recommendation.rank}. "
        lines.append(f"{prefix}{when + '  ' if when else ''}{_describe(event)}")
        # Directly under the title, ahead of the reasons. The reasons are the
        # batch's justification; the link is what you want next once a title has
        # caught your eye, so it should not sit below two lines of score
        # narrative. Bare and unlabelled — a URL says what it is, and terminals
        # make it clickable.
        if event.url:
            lines.append(_style(f"      {event.url}", _DIM, color))
        # No per-event link: name the source instead. 320 of 359 NSNO events
        # publish no URL, so without this a listing's own error arrives with
        # nothing to check it against and reads as ours. Labelled, because it
        # opens the site rather than the event — an unlabelled URL here would
        # promise more than it delivers.
        elif source_urls and (site := source_urls.get(event.source_type)):
            lines.append(_style(f"      source: {site}", _DIM, color))
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


def _when_of(event: Event) -> str:
    """What to print in the time column.

    A placed start exists so the night window can position an event whose hour
    was never published. Printing it as a clock time would be the most
    convincing kind of wrong, so the label says which it is: a source that
    declared a whole day reads differently from one that simply has not said.
    """
    if event.start_time is None:
        return ""
    if event.timing == ALL_DAY:
        return _ALL_DAY_LABEL
    if event.timing == UNKNOWN:
        return _UNKNOWN_TIME_LABEL

    return _format_time(event.start_time)


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
