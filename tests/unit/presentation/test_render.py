"""Unit tests for CLI rendering."""

from datetime import date, datetime, timedelta, timezone

from src.models.event import Event
from src.models.event_score import EventScore
from src.models.ranked_event import RankedEvent
from src.models.ranking import Ranking
from src.presentation.handles import short_handle
from src.presentation.render import (
    render_explanation,
    render_raw,
    render_recommendations,
    staleness_notice,
)
from src.models.tag import Tag
from src.scoring.similarity import Reason

TZ = timezone(timedelta(hours=-4))
TODAY = date(2025, 6, 21)


def _reason(
    factor: str = "like_similarity",
    preference: str = "karaoke night",
    contribution: float = 0.8,
    tag: str | None = "karaoke",
    direction: str = "positive",
    similarity: float = 0.87,
) -> Reason:
    return Reason(
        factor=factor,
        matched_preference=preference,
        similarity=similarity,
        contribution=contribution,
        direction=direction,
        tag=tag,
    )


def _event(
    event_id: str = "evt-1",
    title: str | None = "Karaoke Night",
    venue: str | None = "The Dive Bar",
    start: datetime | None = None,
    source_type: str = "instagram",
    url: str | None = None,
    timing: str = "exact",
    superseded_by: str | None = None,
    merged_by: str | None = None,
    merge_similarity: float | None = None,
) -> Event:
    return Event(
        event_id=event_id,
        source_event_candidates=[],
        source_type=source_type,
        created_at=datetime(2025, 6, 21, 9, 0),
        updated_at=datetime(2025, 6, 21, 9, 0),
        title=title,
        venue=venue,
        start_time=start,
        url=url,
        timing=timing,
        superseded_by=superseded_by,
        merged_by=merged_by,
        merge_similarity=merge_similarity,
    )


def _ranking(rank: int = 1, final: float = 0.68, weather: float = -0.010) -> Ranking:
    return Ranking(
        event_id="evt-1",
        run_date=TODAY,
        weather_adjustment=weather,
        final_score=final,
        rank=rank,
    )


def _pair(
    event_id: str = "evt-1",
    rank: int = 1,
    reasons: list[Reason] | None = None,
    start: datetime | None = datetime(2025, 6, 21, 20, 0, tzinfo=TZ),
    **event_kwargs,
) -> RankedEvent:
    return RankedEvent(
        event=_event(event_id, start=start, **event_kwargs),
        score=EventScore(
            event_id=event_id,
            run_date=TODAY,
            base_score=0.42,
            tag_confidence=1.0,
            match="yes",
            reasons=reasons if reasons is not None else [_reason()],
        ),
        ranking=Ranking(
            event_id=event_id,
            run_date=TODAY,
            weather_adjustment=0.05,
            final_score=0.68,
            rank=rank,
        ),
    )


class TestRankedList:
    """One list, ordered by the rank the batch assigned. No bands, no sections."""

    def test_events_render_in_the_order_given(self):
        out = render_recommendations(
            [
                _pair("a", rank=1, title="First"),
                _pair("b", rank=2, title="Second"),
                _pair("c", rank=3, title="Third"),
            ]
        )

        assert out.index("First") < out.index("Second") < out.index("Third")

    def test_nothing_is_grouped_into_a_band(self):
        # The whole point of removing tiers: score order is the only structure.
        out = render_recommendations([_pair("a"), _pair("b", rank=2)])

        for banner in ("TOP PICKS", "WORTH CONSIDERING", "EVERYTHING ELSE"):
            assert banner not in out

    def test_heading_is_rendered_when_given(self):
        out = render_recommendations([_pair()], heading="Tonight - Saturday 21 June")

        assert "Tonight - Saturday 21 June" in out

    def test_empty_input_renders_a_friendly_message(self):
        out = render_recommendations([])

        assert "No events" in out

    def test_the_number_is_a_position_in_the_list_not_the_batchs_rank(self):
        """The batch ranks the whole horizon — 1169 events across 89 nights on
        the 08-16 run — and the view shows one night of it, so stored ranks
        arrive full of gaps: 4, 7, 9, 15, 47. The heading says one night and the
        number must not answer a different question."""
        out = render_recommendations([_pair("a", rank=4, title="Fourth")])

        assert "  1. " in out
        assert "4. " not in out

    def test_numbering_runs_one_to_n_over_what_is_shown(self):
        pairs = [
            _pair("a", rank=4, title="First"),
            _pair("b", rank=47, title="Second"),
            _pair("c", rank=1093, title="Third"),
        ]

        out = render_recommendations(pairs)

        assert "  1. " in out and "  2. " in out and "  3. " in out
        assert "47." not in out and "1093." not in out

    def test_the_order_is_untouched(self):
        """A renumbering must not become a reordering. The batch's order is the
        product; this only changes what the rows are called."""
        pairs = [
            _pair("a", rank=4, title="First"),
            _pair("b", rank=47, title="Second"),
        ]

        out = render_recommendations(pairs)

        assert out.index("First") < out.index("Second")


class TestLimit:
    """Ten by default, with the remainder counted rather than silently dropped."""

    def _many(self, count: int) -> list[RankedEvent]:
        return [_pair(f"e{i}", rank=i, title=f"Event {i}") for i in range(1, count + 1)]

    def test_only_the_first_ten_are_shown_by_default(self):
        out = render_recommendations(self._many(12))

        assert "Event 10" in out
        assert "Event 11" not in out

    def test_the_count_of_the_rest_is_always_visible(self):
        out = render_recommendations(self._many(12))

        assert "2 more" in out
        assert "--all" in out

    def test_no_limit_shows_everything(self):
        out = render_recommendations(self._many(12), limit=None)

        assert "Event 12" in out
        assert "ranked lower (--all)" not in out

    def test_no_count_line_when_nothing_is_hidden(self):
        out = render_recommendations(self._many(3))

        assert "ranked lower (--all)" not in out

    def test_a_cut_list_still_numbers_from_one_and_counts_what_it_cut(self):
        """The two must not interfere: renumbering describes what is shown, the
        count describes what is not, and neither is derived from the other."""
        pairs = [
            _pair("a", rank=4, title="First"),
            _pair("b", rank=47, title="Second"),
            _pair("c", rank=71, title="Third"),
            _pair("d", rank=93, title="Fourth"),
            _pair("e", rank=112, title="Fifth"),
        ]

        out = render_recommendations(pairs, limit=3)

        assert "  1. " in out and "  2. " in out and "  3. " in out
        assert "  4. " not in out
        assert "2 more" in out

    def test_an_explicit_limit_is_honoured(self):
        out = render_recommendations(self._many(12), limit=2)

        assert "Event 2" in out
        assert "Event 3" not in out
        assert "10 more" in out


class TestUndatedEvents:
    """An undated event ranks with everything else.

    A separate section would take it out of the ranking, which is the one thing
    the ordering is for. A missing start time is a gap in what we know, not
    evidence about when.
    """

    def test_an_undated_event_ranks_inline_with_the_rest(self):
        out = render_recommendations(
            [
                _pair("a", rank=1, title="Timed Thing"),
                _pair("b", rank=2, start=None, title="Untimed Thing"),
            ]
        )

        assert out.index("Timed Thing") < out.index("Untimed Thing")

    def test_an_undated_event_outranks_a_timed_one_when_it_scores_higher(self):
        out = render_recommendations(
            [
                _pair("a", rank=1, start=None, title="Untimed Thing"),
                _pair("b", rank=2, title="Timed Thing"),
            ]
        )

        assert out.index("Untimed Thing") < out.index("Timed Thing")

    def test_an_undated_event_says_its_time_is_unpublished(self):
        out = render_recommendations([_pair("a", start=None)])

        assert "time TBC" in out

    def test_there_is_no_separate_undated_section(self):
        out = render_recommendations([_pair("a", start=None)])

        assert "UNDATED" not in out

class TestEventLines:
    def test_start_time_is_shown_for_a_timed_event(self):
        out = render_recommendations([_pair("a", start=datetime(2025, 6, 21, 20, 30, tzinfo=TZ))])

        assert "20:30" in out

    def test_title_and_venue_are_shown(self):
        out = render_recommendations([_pair("a", title="Karaoke Night", venue="The Dive Bar")])

        assert "Karaoke Night" in out
        assert "The Dive Bar" in out

    def test_a_missing_venue_is_omitted_without_a_dangling_separator(self):
        out = render_recommendations([_pair("a", title="Karaoke Night", venue=None)])

        assert "Karaoke Night" in out
        assert "Karaoke Night —" not in out

    def test_a_missing_title_renders_a_placeholder(self):
        out = render_recommendations([_pair("a", title=None, venue="The Dive Bar")])

        assert "untitled" in out.lower()

    def test_a_position_is_shown(self):
        out = render_recommendations([_pair("a", rank=4)])

        assert "1." in out


class TestReasons:
    def test_reasons_are_shown_by_default(self):
        out = render_recommendations([_pair("a", reasons=[_reason(preference="karaoke night")])])

        assert "karaoke night" in out

    def test_at_most_two_semantic_reasons_are_shown_by_default(self):
        reasons = [
            _reason(preference="first", contribution=0.9),
            _reason(preference="second", contribution=0.8),
            _reason(preference="third", contribution=0.7),
        ]
        out = render_recommendations([_pair("a", reasons=reasons)])

        assert "third" not in out

    def test_the_strongest_reasons_are_the_ones_kept(self):
        reasons = [
            _reason(preference="weak", contribution=0.1),
            _reason(preference="strong", contribution=0.9),
            _reason(preference="middling", contribution=0.5),
        ]
        out = render_recommendations([_pair("a", reasons=reasons)])

        assert "strong" in out
        assert "middling" in out
        assert "weak" not in out

    def test_a_strong_negative_reason_outranks_a_weak_positive(self):
        """Magnitude decides, not sign — 'why is this low?' is the same question."""
        reasons = [
            _reason(preference="mild plus", contribution=0.1),
            _reason(
                factor="dislike_similarity",
                preference="crowds",
                contribution=-0.9,
                direction="negative",
            ),
        ]
        out = render_recommendations([_pair("a", reasons=reasons)])

        assert "crowds" in out

    def test_verbose_shows_every_reason(self):
        reasons = [
            _reason(preference="first", contribution=0.9),
            _reason(preference="second", contribution=0.8),
            _reason(preference="third", contribution=0.7),
        ]
        out = render_recommendations([_pair("a", reasons=reasons)], verbose=True)

        assert "third" in out

    def test_the_weather_reason_is_shown_by_default(self):
        """Weather is why an outdoor event moved tonight; it is not an also-ran."""
        reasons = [
            _reason(preference="a", contribution=0.9),
            _reason(preference="b", contribution=0.8),
            _reason(
                factor="weather_adjustment",
                preference="clear, 18C",
                contribution=0.05,
                tag=None,
            ),
        ]
        out = render_recommendations([_pair("a", reasons=reasons)])

        assert "clear, 18C" in out

    def test_a_semantic_reason_shows_what_it_matched_against(self):
        out = render_recommendations(
            [_pair("a", reasons=[_reason(tag="karaoke", preference="karaoke night")])]
        )

        assert 'karaoke <- "karaoke night"' in out

    def test_a_non_semantic_reason_is_not_written_as_a_preference_match(self):
        """Weather and match are not compared against a preference line."""
        reasons = [
            _reason(factor="weather_adjustment", preference="clear, 18C", tag=None),
            _reason(factor="match_classification", preference="yes", tag=None),
        ]
        out = render_recommendations([_pair("a", reasons=reasons)], verbose=True)

        assert "<-" not in out
        assert "weather: clear, 18C" in out
        assert "match: yes" in out

    def test_the_tag_is_named_alongside_the_preference_it_matched(self):
        out = render_recommendations(
            [_pair("a", reasons=[_reason(tag="live music", preference="concerts")])]
        )

        assert "live music" in out
        assert "concerts" in out

    def test_an_event_with_no_reasons_renders_without_them(self):
        out = render_recommendations([_pair("a", reasons=[], title="Bare Thing")])

        assert "Bare Thing" in out


class TestVerboseScoreComponents:
    def test_verbose_shows_the_score_the_cut_was_made_on(self):
        """Tier is a label; when it looks wrong the number behind it must be visible."""
        out = render_recommendations([_pair("a")], verbose=True)

        assert "0.68" in out

    def test_verbose_shows_base_score_weather_and_confidence(self):
        out = render_recommendations([_pair("a")], verbose=True)

        assert "0.42" in out
        assert "0.05" in out
        assert "match" in out.lower()

    def test_components_are_not_shown_by_default(self):
        out = render_recommendations([_pair("a")])

        assert "0.42" not in out


class TestColor:
    def test_no_ansi_escapes_when_color_is_off(self):
        out = render_recommendations([_pair("a")], color=False)

        assert "\033[" not in out

    def test_ansi_escapes_appear_when_color_is_on(self):
        out = render_recommendations([_pair("a")], color=True)

        assert "\033[" in out

    def test_content_is_identical_either_way(self):
        plain = render_recommendations([_pair("a")], color=False)
        colored = render_recommendations([_pair("a")], color=True)

        assert "Karaoke Night" in plain
        assert "Karaoke Night" in colored


class TestRenderRaw:
    def test_lists_every_event_with_no_scores(self):
        out = render_raw([_event("a", title="One"), _event("b", title="Two")])

        assert "One" in out
        assert "Two" in out
        assert "TOP PICKS" not in out

    def test_shows_the_source_type(self):
        out = render_raw([_event("a", source_type="cinema_veezi")])

        assert "cinema_veezi" in out

    def test_orders_by_start_time(self):
        events = [
            _event("late", title="Later", start=datetime(2025, 6, 21, 22, 0, tzinfo=TZ)),
            _event("early", title="Earlier", start=datetime(2025, 6, 21, 18, 0, tzinfo=TZ)),
        ]
        out = render_raw(events)

        assert out.index("Earlier") < out.index("Later")

    def test_undated_events_sort_last_and_are_still_listed(self):
        events = [
            _event("undated", title="Whenever", start=None),
            _event("timed", title="At Eight", start=datetime(2025, 6, 21, 20, 0, tzinfo=TZ)),
        ]
        out = render_raw(events)

        assert out.index("At Eight") < out.index("Whenever")

    def test_a_count_is_shown(self):
        out = render_raw([_event("a"), _event("b")])

        assert "2" in out

    def test_every_raw_event_shows_its_handle(self):
        """`--raw` has never had a selector of its own — it prints no ranks, and
        the superseded events it exists to reveal have no ranking row at all. A
        handle is the same string here as in the listing, so what `--raw` makes
        visible, `--explain` can now reach."""
        out = render_raw([_event("a", title="One"), _event("b", title="Two")])

        assert f"#{short_handle('a')}" in out
        assert f"#{short_handle('b')}" in out

    def test_empty_input_renders_a_friendly_message(self):
        assert "No events" in render_raw([])


class TestRenderRawSupersession:
    """`--raw` means every stored event, and that now includes the losers.

    Filtering them would make the raw view the one place that cannot show what
    the batch actually did — and inspecting a merge you disagree with is exactly
    what it is for. Including them silently is the other failure: a reader takes
    a merged-away duplicate for a real event.
    """

    def test_a_superseded_event_is_still_listed(self):
        out = render_raw([_event("loser", title="Wood & Bone", superseded_by="winner")])

        assert "Wood & Bone" in out

    def test_a_superseded_event_says_what_absorbed_it(self):
        out = render_raw(
            [
                _event(
                    "loser",
                    title="Wood & Bone",
                    superseded_by="0bbe43b6",
                    merged_by="semantic",
                    merge_similarity=0.926,
                )
            ]
        )

        assert "superseded by 0bbe43b6" in out
        assert "semantic" in out
        assert "0.926" in out

    def test_a_reconcile_merge_shows_no_similarity(self):
        """Reconcile matches on shared candidate ids, not on a score. Printing a
        number here would misrepresent how the merge was decided — the same
        reason nothing is stored for it."""
        out = render_raw(
            [_event("loser", superseded_by="winner", merged_by="reconcile")]
        )

        assert "superseded by winner" in out
        assert "reconcile" in out
        assert "0.0" not in out

    def test_the_count_says_how_many_were_superseded(self):
        """A total that silently includes merged-away rows is the same lie in
        miniature as omitting them."""
        out = render_raw(
            [
                _event("a"),
                _event("b"),
                _event("loser", superseded_by="a", merged_by="semantic"),
            ]
        )

        assert "3 events" in out
        assert "1 superseded" in out

    def test_a_list_with_no_supersession_says_nothing_about_it(self):
        """The mark must not become decoration on the ordinary view."""
        out = render_raw([_event("a"), _event("b")])

        assert "superseded" not in out
        assert "2 events" in out


LINK = "https://coastalmassbrewing.com/our-events/2026/8/6/music-bingo"


class TestEventUrl:
    """#21: every source stores a URL and none of them were ever shown."""

    def test_the_url_is_rendered(self):
        out = render_recommendations([_pair("a", url=LINK)])

        assert LINK in out

    def test_the_url_precedes_the_reasons(self):
        """The link is what you want next once a title catches your eye.

        Putting it below the score narrative buries the one line that answers
        "what is this actually?".
        """
        out = render_recommendations([_pair("a", url=LINK)])

        assert out.index(LINK) < out.index("karaoke")

    def test_the_url_follows_its_own_title(self):
        out = render_recommendations([_pair("a", title="Music Bingo", url=LINK)])

        assert out.index("Music Bingo") < out.index(LINK)

    def test_an_event_without_a_url_is_unchanged(self):
        """The whole no-URL path must render byte-identically to before."""
        with_none = render_recommendations([_pair("a")])
        explicitly_none = render_recommendations([_pair("a", url=None)])

        assert with_none == explicitly_none
        assert "http" not in with_none

    def test_only_the_event_that_has_one_shows_a_url(self):
        out = render_recommendations(
            [_pair("a", title="Has Link", url=LINK), _pair("b", title="No Link", rank=2)]
        )

        assert out.count(LINK) == 1

    def test_a_url_beyond_the_limit_does_not_leak_into_the_count(self):
        out = render_recommendations(
            [_pair("a"), _pair("b", rank=2, url=LINK)], limit=1
        )

        assert "1 more" in out
        assert LINK not in out

    def test_the_url_shows_once_the_limit_is_lifted(self):
        out = render_recommendations([_pair("b", url=LINK)], limit=None)

        assert LINK in out

    def test_an_undated_event_still_shows_its_url(self):
        out = render_recommendations([_pair("a", start=None, url=LINK)])

        assert LINK in out

    def test_the_raw_view_stays_one_line_per_event(self):
        """`--raw` is the terse escape hatch; a second line per event defeats it."""
        out = render_raw([_event("a", url=LINK)])

        assert LINK not in out


class TestUnpublishedTimes:
    """A placed start exists so the night window can position the event.

    Printing it as a clock time is the most convincing kind of wrong, so the
    label says which kind of not-knowing it is.
    """

    def test_an_all_day_event_says_so(self):
        out = render_recommendations([_pair("a", title="Open Studio", timing="all_day")])

        assert "all day" in out
        assert "20:00" not in out

    def test_an_unpublished_time_says_so(self):
        out = render_recommendations([_pair("a", title="Drop-In", timing="unknown")])

        assert "time TBC" in out
        assert "20:00" not in out

    def test_the_two_are_not_collapsed(self):
        """A calendar declaring all day and a listing omitting the hour differ."""
        out = render_recommendations(
            [_pair("a", title="Exhibition", timing="all_day"),
             _pair("b", title="Drop-In", rank=2, timing="unknown")]
        )

        assert "all day" in out and "time TBC" in out

    def test_a_stated_time_still_prints_as_a_clock(self):
        out = render_recommendations([_pair("a", title="Gig")])

        assert "20:00" in out

    def test_the_raw_view_agrees(self):
        out = render_raw([_event("a", start=datetime(2025, 6, 21, 20, 0, tzinfo=TZ),
                                 timing="all_day")])

        assert "all day" in out
        assert "20:00" not in out


class TestSourceAttribution:
    """Where an event came from, when the event itself carries no link.

    A listing that publishes no per-event URL renders with nothing to check, so
    a source's own error reads as ours. The site stands in.
    """

    SITES = {"northshorenightout": "https://northshorenightout.com/"}

    def test_an_event_without_a_url_is_attributed_to_its_source_site(self):
        pairs = [_pair(source_type="northshorenightout", url=None)]

        output = render_recommendations(pairs, source_urls=self.SITES)

        assert "source: https://northshorenightout.com/" in output

    def test_the_events_own_url_wins_and_is_not_labelled_a_source(self):
        # A per-event link goes to the event; the site only goes to the site.
        # Labelling the former "source:" would overclaim what it opens.
        pairs = [
            _pair(source_type="northshorenightout", url="https://example.com/event/9")
        ]

        output = render_recommendations(pairs, source_urls=self.SITES)

        assert "https://example.com/event/9" in output
        assert "source:" not in output

    def test_an_unmapped_source_renders_no_attribution_line(self):
        pairs = [_pair(source_type="synthetic", url=None)]

        output = render_recommendations(pairs, source_urls=self.SITES)

        assert "source:" not in output

    def test_attribution_is_absent_when_no_map_is_supplied(self):
        pairs = [_pair(source_type="northshorenightout", url=None)]

        assert "source:" not in render_recommendations(pairs)


class TestTheHandleIsOnScreen:
    """The copyable name of an event, which a rank is not.

    A rank is re-derived every night and, once numbering is view-local, names a
    different event in every view. The handle is derived from `event_id`, which
    is stable across runs, so it is the same string in `--raw`, in a filtered
    listing, and in yesterday's scrollback.
    """

    SITES = {"northshorenightout": "https://northshorenightout.com/"}

    def test_the_handle_leads_the_link_line(self):
        """Position 7 on every row regardless of title length, so it reads as a
        column and survives the wrapping of a long link."""
        pairs = [_pair(event_id="evt-9", source_type="northshorenightout", url=None)]

        output = render_recommendations(pairs, source_urls=self.SITES)

        assert (
            f"      #{short_handle('evt-9')}  source: https://northshorenightout.com/"
            in output
        )

    def test_the_handle_leads_an_events_own_url_too(self):
        """The link is bare in this case and labelled in the other. The handle
        comes first either way, so its column does not move."""
        pairs = [_pair(event_id="evt-9", url="https://example.com/event/9")]

        output = render_recommendations(pairs, source_urls=self.SITES)

        assert f"      #{short_handle('evt-9')}  https://example.com/event/9" in output

    def test_an_event_with_no_link_at_all_still_shows_its_handle(self):
        """There is no link line to lead when the source is unmapped, and the
        handle is not optional — an event nobody can name cannot be explained."""
        pairs = [_pair(event_id="evt-9", source_type="synthetic", url=None)]

        output = render_recommendations(pairs, source_urls=self.SITES)

        assert f"      #{short_handle('evt-9')}" in output
        assert "source:" not in output

    def test_every_event_gets_its_own_handle(self):
        pairs = [_pair(event_id="evt-1"), _pair(event_id="evt-2")]

        output = render_recommendations(pairs)

        assert f"#{short_handle('evt-1')}" in output
        assert f"#{short_handle('evt-2')}" in output

    def test_the_handle_line_is_dimmed_with_the_rest_of_the_block(self):
        """Time and title are what the eye scans; the handle is there when
        something catches it."""
        pairs = [_pair(event_id="evt-9", url="https://example.com/event/9")]

        output = render_recommendations(pairs, color=True)

        assert f"\033[2m      #{short_handle('evt-9')}" in output


class TestStalenessNotice:
    """A ranking older than tonight must never pass as tonight's.

    On 2026-08-12 the batch died before ranking, and the CLI printed today's
    date above the previous night's order — including a scoring bug we believed
    fixed. Nothing said the listing was a day old. The order is the product, so
    serving a stale one silently is the worst failure the view has.
    """

    def test_a_run_from_tonight_says_nothing(self):
        assert staleness_notice(date(2025, 6, 21), TODAY) is None

    def test_a_run_from_the_future_says_nothing(self):
        """A --run-date ahead of tonight is not staleness."""
        assert staleness_notice(date(2025, 6, 22), TODAY) is None

    def test_a_day_old_run_is_announced_with_its_date(self):
        notice = staleness_notice(date(2025, 6, 20), TODAY)

        assert notice is not None
        assert "2025-06-20" in notice

    def test_the_age_is_stated_in_days(self):
        assert "1 day old" in staleness_notice(date(2025, 6, 20), TODAY)
        assert "5 days old" in staleness_notice(date(2025, 6, 16), TODAY)

    def test_it_points_at_the_batch_log(self):
        """The next question is always "why", and the answer is in one place."""
        assert "batch-latest.log" in staleness_notice(date(2025, 6, 20), TODAY)


class TestRenderExplanation:
    """The home for everything the default view deliberately does not show.

    Keeping the list clean keeps producing information with nowhere to go —
    `score_reasons` is 6,327 rows of exactly "why is this ranked here". This is
    where it goes, so the decision stops being re-made every time a field lands.
    """

    def _score(self, **kwargs) -> EventScore:
        fields = dict(
            event_id="evt-1",
            run_date=TODAY,
            base_score=-0.067,
            match="maybe",
            tag_score=-0.064,
            summary_score=-0.012,
            tag_confidence=1.0,
            # Deliberately NOT in strength order — the weakest first. Stored
            # order is `position`, which is the order the scorer emitted, so a
            # fixture that happens to be pre-sorted makes the ordering test
            # vacuous. It did: reverting the sort passed until this was fixed.
            reasons=[
                Reason("dislike_similarity", "dancing", 0.444, -0.004, "negative", "breakfast"),
                Reason("dislike_similarity", "dancing", 0.456, -0.012, "negative", None),
                Reason("dislike_similarity", "pop music", 0.778, -0.461, "negative", "music"),
                Reason("like_similarity", "karaoke", 0.509, 0.024, "positive", "cookout bbq"),
            ],
        )
        fields.update(kwargs)
        return EventScore(**fields)

    def test_the_title_and_rank_lead(self):
        out = render_explanation(
            _event("a", title="Steamboats"), self._score(), _ranking(rank=782)
        )

        assert "782" in out
        assert "Steamboats" in out

    def test_every_reason_appears_with_its_preference_and_similarity(self):
        out = render_explanation(_event("a"), self._score(), _ranking())

        assert "pop music" in out
        assert "0.778" in out
        assert "music" in out

    def test_reasons_are_ordered_by_strength(self):
        """Strongest first, because the question is "why is it here" and one
        contribution usually dominates — 85% of this score is a single tag."""
        out = render_explanation(_event("a"), self._score(), _ranking())

        assert out.index("pop music") < out.index("karaoke") < out.index("breakfast")

    def test_direction_is_visible_per_reason(self):
        out = render_explanation(_event("a"), self._score(), _ranking())

        assert "+" in out and "−" in out

    def test_the_summary_reason_is_labelled_rather_than_blank(self):
        """It carries no tag. An empty column would read as a missing value."""
        out = render_explanation(_event("a"), self._score(), _ranking())

        assert "summary" in out

    def test_the_score_line_shows_the_stored_components(self):
        out = render_explanation(_event("a"), self._score(), _ranking(final=-0.067))

        assert "-0.067" in out
        assert "maybe" in out

    def test_a_negative_base_is_shown_being_divided(self):
        """The multiplier is direction-aware: it divides a negative base so the
        clearest rejections are not rewarded. That looks like a bug from the
        outside, which is exactly why the view says which happened."""
        out = render_explanation(_event("a"), self._score(base_score=-0.067), _ranking())

        assert "÷" in out

    def test_a_positive_base_is_shown_being_multiplied(self):
        out = render_explanation(_event("a"), self._score(base_score=0.4), _ranking())

        assert "×" in out

    def test_tags_appear_with_their_weights(self):
        event = _event("a")
        event.tags = [Tag(text="steamboats", weight=1.0), Tag(text="food", weight=0.7)]

        out = render_explanation(event, self._score(), _ranking())

        assert "steamboats" in out
        assert "1.0" in out
        assert "0.7" in out

    def test_extraction_provenance_appears(self):
        event = _event("a")
        event.extraction_model = "gemma4:e4b"
        event.extraction_prompt_version = "v3"

        out = render_explanation(event, self._score(), _ranking())

        assert "gemma4:e4b" in out

    def test_a_row_with_no_provenance_says_so_rather_than_leaving_a_blank(self):
        """Most stored events predate provenance. A blank would read as a value
        the batch failed to record rather than one it never had."""
        out = render_explanation(_event("a"), self._score(), _ranking())

        assert "not recorded" in out

    def test_a_degraded_extraction_says_how(self):
        event = _event("a")
        event.extraction_degradation = "no_summary"

        out = render_explanation(event, self._score(), _ranking())

        assert "no_summary" in out

    def test_a_non_semantic_reason_is_named_by_its_factor(self):
        """`(summary)` is right for the summary term and wrong for everything
        else that carries no tag. Against the live database the match
        classification rendered as `(summary)`, which reads as a second summary
        contribution that does not exist."""
        score = self._score(
            reasons=[Reason("match_classification", "maybe", 1.0, 0.0, "positive", None)]
        )

        out = render_explanation(_event("a"), score, _ranking())

        assert "match" in out
        assert "(summary)" not in out

    def test_a_whole_weight_keeps_its_decimal(self):
        """`1` in a column of `0.9` and `0.7` reads as a different quantity."""
        event = _event("a")
        event.tags = [Tag(text="steamboats", weight=1.0)]

        out = render_explanation(event, self._score(), _ranking())

        assert "steamboats 1.0" in out

    def test_a_finer_weight_is_not_rounded_away(self):
        event = _event("a")
        event.tags = [Tag(text="trivia", weight=0.85)]

        out = render_explanation(event, self._score(), _ranking())

        assert "0.85" in out

    def test_an_event_with_no_reasons_still_renders(self):
        out = render_explanation(_event("a"), self._score(reasons=[]), _ranking())

        assert "Karaoke Night" in out


class TestExplainingAnUnrankedEvent:
    """A superseded event has no `event_scores` row, so there is no ranking to
    explain — but "this merge looks wrong, why?" is exactly the case `--raw`
    marking exists for, so it must still say what it knows."""

    def test_it_says_there_is_no_ranking_rather_than_failing(self):
        out = render_explanation(_event("a", superseded_by="winner"), None, None)

        assert "not ranked" in out.lower()

    def test_it_still_names_what_absorbed_it(self):
        out = render_explanation(
            _event("a", superseded_by="0bbe43b6", merged_by="semantic",
                   merge_similarity=0.926),
            None,
            None,
        )

        assert "superseded by 0bbe43b6" in out
        assert "0.926" in out

    def test_it_still_shows_tags_and_provenance(self):
        event = _event("a", superseded_by="winner")
        event.tags = [Tag(text="trivia", weight=1.0)]
        event.extraction_model = "gemma4:e4b"

        out = render_explanation(event, None, None)

        assert "trivia" in out
        assert "gemma4:e4b" in out
