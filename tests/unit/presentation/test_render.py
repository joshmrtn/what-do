"""Unit tests for CLI rendering."""

from datetime import date, datetime, timedelta, timezone

from src.models.event import Event
from src.models.event_score import EventScore
from src.models.ranked_event import RankedEvent
from src.models.ranking import Ranking
from src.presentation.render import render_raw, render_recommendations
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

    def test_the_batchs_rank_is_shown_not_a_position_in_the_list(self):
        # Filters run before rendering, so the visible list can start at rank 4.
        out = render_recommendations([_pair("a", rank=4, title="Fourth")])

        assert "4. " in out


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

    def test_rank_is_shown(self):
        out = render_recommendations([_pair("a", rank=4)])

        assert "4." in out


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

    def test_empty_input_renders_a_friendly_message(self):
        assert "No events" in render_raw([])


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
