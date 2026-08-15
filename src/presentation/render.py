"""Rendering for the CLI.

Pure: strings in, one string out. Nothing here reads a clock, touches the
database, or reorders anything — sections are emitted in tier order and events
within them in the rank the batch assigned.

One list, in the rank the batch assigned. There are no bands and no sections:
the ordering is the whole product, and grouping it into named tiers added a
second, less reliable judgement on top of the only one that matters.

The list is cut at `limit`, and whatever is cut is always counted on screen —
a hidden event is invisible, a counted one is a flag away.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime

from src.models.event import Event
from src.models.event_score import EventScore
from src.models.ranking import Ranking
from src.models.timing import ALL_DAY, UNKNOWN
from src.models.ranked_event import RankedEvent
from src.scoring.ranking import CONFIDENCE_FACTOR, MATCH_FACTOR
from src.scoring.similarity import DISLIKE_FACTOR, LIKE_FACTOR, Reason
from src.scoring.weather_score import WEATHER_FACTOR

_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"

#: Fallback when no caller supplies a limit. `config.view.limit` is the real
#: setting; this exists because the view root's config load is deliberately
#: tolerant and an unreadable config must still produce a usable listing.
DEFAULT_LIMIT = 10

#: Fallback for how many semantic reasons appear before `--verbose`. Two is
#: enough to say what an event is; the rest is an audit trail.
#: `config.view.reason_limit` is the real setting.
DEFAULT_REASON_LIMIT = 2

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


def staleness_notice(run_date: date, tonight: date) -> str | None:
    """Warn that the ranking on screen predates the night it is shown for.

    The order is the product, so a stale one passing as current is the worst
    thing this view can do — and it is invisible, because a batch that dies
    before ranking leaves the previous night's order in place under today's
    heading.

    Deliberately not a refusal. A failed batch must not also cost the listing:
    a day-old ranking is mostly still accurate, and being told its age is what
    makes it safe to read.

    Args:
        run_date: The batch whose ranking is being shown.
        tonight: The night being shown, in the view's own zone.

    Returns:
        The warning, or None when the ranking is current. A run *ahead* of
        tonight is not stale — that is `--run-date` being used deliberately.
    """
    days = (tonight - run_date).days
    if days <= 0:
        return None
    plural = "day" if days == 1 else "days"
    return (
        f"⚠  Showing the ranking from {run_date.isoformat()} — {days} {plural} old.\n"
        f"   No ranking has been produced for tonight; see logs/batch-latest.log"
    )


def render_recommendations(
    pairs: list[RankedEvent],
    *,
    heading: str | None = None,
    verbose: bool = False,
    limit: int | None = DEFAULT_LIMIT,
    color: bool = False,
    source_urls: Mapping[str, str] | None = None,
    show_dates: bool = False,
    reason_limit: int = DEFAULT_REASON_LIMIT,
) -> str:
    """Render ranked events as one list, in the order given.

    Args:
        pairs: Ranked pairs, already filtered and in rank order. Undated events
            are ranked among the rest: a missing start time is a gap in what we
            know, not a reason to set an event aside from the ordering.
        heading: Optional line above the list, e.g. the date being shown.
        verbose: Show every reason plus the score components.
        limit: How many to show; None shows all. Whatever is cut is counted.
        color: Emit ANSI styling. Off when stdout is not a terminal.
        source_urls: Human-facing page per `source_type`, used only for events
            carrying no URL of their own.
        reason_limit: How many semantic reasons to show per event before
            `--verbose`. From `config.view.reason_limit`.
        show_dates: Prefix the time column with the day. For a cross-day list,
            where there is no heading to place a row against — in a per-night
            view the heading already answers it and repeating it is noise.

    Returns:
        The rendered view, ending in a newline.
    """
    if not pairs:
        # The heading survives an empty list. Without it a run of quiet nights
        # under `--days` becomes indistinguishable repetitions of the same
        # sentence, with nothing saying which night each one was.
        empty = _style("No events to show.", _DIM, color)
        if heading:
            return _join([_style(heading, _BOLD, color), "", empty])
        return _join([empty])

    shown = pairs if limit is None else pairs[:limit]
    hidden = len(pairs) - len(shown)

    lines: list[str] = []
    if heading:
        lines += [_style(heading, _BOLD, color), ""]

    for ranked in shown:
        event, score, ranking = ranked.event, ranked.score, ranked.ranking
        when = _when_of(event)
        if show_dates and event.start_time is not None:
            when = f"{event.start_time.strftime('%a %-d')} {when}"
        prefix = f"  {ranking.rank}. "
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
            lines.append(_style(f"      {_components(score, ranking)}", _DIM, color))
        for reason in _visible_reasons(score.reasons, verbose=verbose, limit=reason_limit):
            lines.append(_style(f"      {_format_reason(reason)}", _DIM, color))
        lines.append("")

    if hidden:
        plural = "" if hidden == 1 else "s"
        lines.append(
            _style(f"+ {hidden} more event{plural} ranked lower (--all)", _DIM, color)
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
    superseded = sum(1 for event in events if event.superseded_by is not None)
    # Named in the count, because a total that silently includes merged-away
    # rows misleads in exactly the way listing them unmarked would.
    tally = f"{len(events)} event{'' if len(events) == 1 else 's'}"
    if superseded:
        tally += f" ({superseded} superseded)"
    lines = [tally, ""]
    for event in ordered:
        when = _when_of(event)
        lines.append(f"  {when}  {_describe(event)}  [{event.source_type}]")
        if (mark := _supersession_of(event)) is not None:
            lines.append(f"         {mark}")

    return _join(lines)


def render_explanation(
    event: Event,
    score: EventScore | None,
    ranking: Ranking | None,
    *,
    color: bool = False,
) -> str:
    """Account for one event's placement, in full.

    The default view is deliberately free of score labels and badges — that was
    settled when tiers were removed and reaffirmed when the degradation field
    landed. But the information kept accumulating with nowhere to look at it:
    `score_reasons` alone is thousands of rows of exactly *why is this ranked
    here*. This is the somewhere, reached on purpose rather than decorating the
    list.

    Everything here is read from storage. Nothing is recomputed, which is the
    CLI's whole promise.

    Args:
        event: The event to account for.
        score: Its stored verdict, or None when it has none — a superseded
            event never got one, and that is worth saying rather than hiding.
        ranking: Its placement, or None on the same terms.
        color: Emit ANSI styling.
    """
    lines: list[str] = []
    place = f"  {ranking.rank}. " if ranking is not None else "  "
    when = _when_of(event)
    lines.append(_style(f"{place}{when + '  ' if when else ''}{_describe(event)}", _BOLD, color))
    if event.url:
        lines.append(_style(f"      {event.url}", _DIM, color))
    lines.append("")

    if score is None or ranking is None:
        # Not a failure to explain: an event dedup merged away never got a
        # score, so there is no placement to account for. Saying which is the
        # point — this is the "that merge looks wrong, why?" case.
        lines.append("  not ranked — no score was stored for this event")
    else:
        lines.append(f"  score    {_score_arithmetic(score, ranking)}")

    if (mark := _supersession_of(event)) is not None:
        lines.append(f"  merge    {mark}")

    if event.tags:
        rendered_tags = " · ".join(
            f"{tag.text} {_weight(tag.weight)}" for tag in event.tags
        )
        lines.append(f"  tags     {rendered_tags}")
    if event.summary:
        lines.append(f"  summary  \"{event.summary}\"")
    lines.append(f"  model    {_provenance_of(event)}")

    if score is not None and score.reasons:
        lines.append("")
        lines.append(
            _style("           tag                weight   matched      sim   contribution", _DIM, color)
        )
        weights = {tag.text: tag.weight for tag in event.tags}
        for reason in sorted(score.reasons, key=lambda r: abs(r.contribution), reverse=True):
            lines.append(_explanation_row(reason, weights))

    return _join(lines)


def _score_arithmetic(score: EventScore, ranking: Ranking) -> str:
    """The stored components, in the order they were applied.

    The multiplier is shown as `÷` on a negative base and `×` on a positive one,
    because that is what actually happened: it acts on magnitude with the sign
    preserved, so dividing is what stops the clearest rejections being rewarded.
    From outside it looks like a bug either way, which is the reason to say
    which one ran.
    """
    operator = "×" if score.base_score >= 0 else "÷"
    return (
        f"{ranking.final_score:+.3f}  =  base {score.base_score:+.3f}"
        f"  × confidence {score.tag_confidence:.2f}"
        f"  {operator} match ({score.match})"
        f"  + weather {ranking.weather_adjustment:+.3f}"
    )


def _provenance_of(event: Event) -> str:
    """What produced this event's tags, and how far short the reply fell.

    "not recorded" rather than a blank: most stored events predate provenance,
    and an empty field reads as something the batch failed to write rather than
    something it never had.
    """
    if event.extraction_model is None:
        model = "not recorded"
    else:
        model = event.extraction_model
        if event.extraction_prompt_version:
            model += f", prompt {event.extraction_prompt_version}"
    if event.extraction_degradation:
        model += f"  (degraded: {event.extraction_degradation})"
    return model


def _explanation_row(reason: Reason, weights: dict[str, float]) -> str:
    """One contribution, as a row.

    The summary term carries no tag; it is labelled rather than left blank,
    because an empty column reads as a missing value.
    """
    sign = "+" if reason.direction == "positive" else "−"
    # A tagless row is the summary term *or* a non-semantic factor. Labelling
    # both `(summary)` invents a second summary contribution: measured against
    # the live database, the match classification rendered exactly that way.
    if reason.tag:
        tag = reason.tag
    else:
        tag = _FACTOR_LABELS.get(reason.factor, "(summary)")
    weight = _weight(weights[reason.tag]) if reason.tag in weights else "—"
    return (
        f"    {sign}      {tag:<18} {weight:>4}   "
        f"{reason.matched_preference:<12} {reason.similarity:.3f}     "
        f"{reason.contribution:+.3f}"
    )


def _weight(weight: float) -> str:
    """A tag weight, keeping one decimal but not inventing precision.

    `1` in a column of `0.9` and `0.7` reads as a different quantity, and
    `1.00` claims a precision the model never expressed.
    """
    text = f"{weight:.2f}".rstrip("0")
    return f"{text}0" if text.endswith(".") else text


def _supersession_of(event: Event) -> str | None:
    """How this event was merged away, or None if it still stands.

    A continuation line rather than a suffix: the row stays scannable, and the
    mark reads as being *about* the row above it.
    """
    if event.superseded_by is None:
        return None

    mark = f"superseded by {event.superseded_by}"
    if event.merged_by is None:
        return mark
    # Reconcile matches on shared candidate ids, not on a score, and stores no
    # similarity for that reason. Printing one would misrepresent the merge.
    if event.merge_similarity is None:
        return f"{mark} ({event.merged_by})"
    return f"{mark} ({event.merged_by}, {event.merge_similarity:.3f})"


def _visible_reasons(
    reasons: list[Reason], *, verbose: bool, limit: int = DEFAULT_REASON_LIMIT
) -> list[Reason]:
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

    return semantic[:limit] + weather


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


def _components(score: EventScore, ranking: Ranking) -> str:
    """The numbers behind the order, for when a placement looks wrong."""
    return (
        f"score {ranking.final_score:+.2f}  "
        f"base {score.base_score:+.2f}  "
        f"weather {ranking.weather_adjustment:+.2f}  "
        f"confidence {score.tag_confidence:.2f}  "
        f"match {score.match}"
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

    An event with no start at all reads the same way. It used to be told apart
    by sitting under an `UNDATED` heading; ranked inline among the rest, the
    column is the only thing left to say so.
    """
    if event.start_time is None:
        return _UNKNOWN_TIME_LABEL
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
