"""Unit tests for CLI rendering."""

from datetime import date, datetime, timedelta, timezone

from src.models.event import Event
from src.models.recommendation import Recommendation
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
    tier: str = "top_pick",
    rank: int = 1,
    reasons: list[Reason] | None = None,
    start: datetime | None = datetime(2025, 6, 21, 20, 0, tzinfo=TZ),
    **event_kwargs,
) -> tuple[Recommendation, Event]:
    recommendation = Recommendation(
        recommendation_id=f"{TODAY.isoformat()}:{event_id}",
        event_id=event_id,
        run_date=TODAY,
        base_score=0.42,
        weather_adjustment=0.05,
        tag_confidence=1.0,
        final_score=0.68,
        match="yes",
        tier=tier,
        rank=rank,
        reasons=reasons if reasons is not None else [_reason()],
    )
    return recommendation, _event(event_id, start=start, **event_kwargs)


class TestSections:
    def test_top_picks_and_worth_considering_are_separate_sections(self):
        out = render_recommendations(
            [_pair("a", tier="top_pick"), _pair("b", tier="worth_considering", rank=2)]
        )

        assert "TOP PICKS" in out
        assert "WORTH CONSIDERING" in out
        assert out.index("TOP PICKS") < out.index("WORTH CONSIDERING")

    def test_a_section_with_no_events_is_not_printed(self):
        out = render_recommendations([_pair("a", tier="top_pick")])

        assert "WORTH CONSIDERING" not in out

    def test_events_appear_under_their_own_tier(self):
        out = render_recommendations(
            [_pair("a", tier="top_pick", title="Top Thing"),
             _pair("b", tier="worth_considering", rank=2, title="Maybe Thing")]
        )

        assert out.index("Top Thing") < out.index("WORTH CONSIDERING") < out.index("Maybe Thing")

    def test_rank_order_within_a_section_is_preserved(self):
        out = render_recommendations(
            [
                _pair("a", rank=1, title="First"),
                _pair("b", rank=2, title="Second"),
                _pair("c", rank=3, title="Third"),
            ]
        )

        assert out.index("First") < out.index("Second") < out.index("Third")

    def test_heading_is_rendered_when_given(self):
        out = render_recommendations([_pair()], heading="Tonight - Saturday 21 June")

        assert "Tonight - Saturday 21 June" in out

    def test_empty_input_renders_a_friendly_message(self):
        out = render_recommendations([])

        assert "No events" in out
        assert "TOP PICKS" not in out


class TestBottomTierIsFoldedNotHidden:
    def _mixed(self) -> list[tuple[Recommendation, Event]]:
        return [
            _pair("a", tier="top_pick", rank=1, title="Great Thing"),
            _pair("b", tier="everything_else", rank=2, title="Dull Thing"),
            _pair("c", tier="everything_else", rank=3, title="Duller Thing"),
        ]

    def test_bottom_tier_events_are_not_listed_by_default(self):
        out = render_recommendations(self._mixed())

        assert "Dull Thing" not in out

    def test_the_count_of_folded_events_is_always_visible(self):
        """Thresholds are uncalibrated; a low-ranked event may be folded, never invisible."""
        out = render_recommendations(self._mixed())

        assert "2 more" in out
        assert "--all" in out

    def test_show_all_expands_them(self):
        out = render_recommendations(self._mixed(), show_all=True)

        assert "Dull Thing" in out
        assert "Duller Thing" in out
        assert "EVERYTHING ELSE" in out

    def test_show_all_drops_the_folded_count_line(self):
        out = render_recommendations(self._mixed(), show_all=True)

        assert "2 more" not in out

    def test_no_count_line_when_nothing_is_folded(self):
        out = render_recommendations([_pair("a", tier="top_pick")])

        assert "more (--all)" not in out

    def test_folded_events_keep_their_rank_when_expanded(self):
        out = render_recommendations(self._mixed(), show_all=True)

        assert out.index("Dull Thing") < out.index("Duller Thing")


class TestUndatedEvents:
    def test_undated_events_get_their_own_labelled_section(self):
        out = render_recommendations([_pair("a", start=None, title="Sometime Thing")])

        assert "UNDATED" in out
        assert "Sometime Thing" in out

    def test_undated_section_follows_the_timed_sections(self):
        out = render_recommendations(
            [_pair("a", title="Timed Thing"), _pair("b", rank=2, start=None, title="Untimed")]
        )

        assert out.index("Timed Thing") < out.index("UNDATED") < out.index("Untimed")

    def test_undated_section_says_the_timing_is_unconfirmed(self):
        """The caveat belongs on the heading, so the list is not read as tonight's."""
        out = render_recommendations([_pair("a", start=None)])

        assert "timing unconfirmed" in out.lower()

    def test_a_bottom_tier_undated_event_folds_like_any_other(self):
        out = render_recommendations(
            [_pair("a", tier="everything_else", start=None, title="Dull Undated")]
        )

        assert "Dull Undated" not in out
        assert "1 more" in out

    def test_no_undated_section_when_every_event_has_a_time(self):
        out = render_recommendations([_pair("a")])

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

    def test_the_folded_count_is_unaffected(self):
        """The bottom tier stays a count; a URL must not leak into it."""
        out = render_recommendations(
            [_pair("a"), _pair("b", tier="everything_else", rank=2, url=LINK)]
        )

        assert "+ 1 more event ranked lower (--all)" in out
        assert LINK not in out

    def test_the_url_shows_in_the_expanded_bottom_tier(self):
        out = render_recommendations(
            [_pair("b", tier="everything_else", url=LINK)], show_all=True
        )

        assert LINK in out

    def test_the_url_shows_in_the_undated_section(self):
        out = render_recommendations([_pair("a", start=None, url=LINK)])

        assert "UNDATED" in out
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
