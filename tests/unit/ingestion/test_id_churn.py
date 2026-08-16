from __future__ import annotations

from datetime import datetime, timezone

from src.ingestion.id_churn import churn_by_source, content_key
from src.models.event_candidate import EventCandidate

NOW = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
START = datetime(2026, 8, 22, 23, 0, tzinfo=timezone.utc)


def _candidate(cid: str, **overrides: object) -> EventCandidate:
    fields: dict = {
        "id": cid,
        # Deliberately unequal to `source_type`. Held equal — or held constant
        # while only `source_type` varies — no test can tell which field the
        # measurement keys on, and the choice becomes untestable.
        "source": "nsno_ics",
        "source_type": "northshorenightout",
        "title": "Isabel Stover",
        "venue": "The Joy Nest",
        "start_time": START,
        "discovered_at": NOW,
    }
    fields.update(overrides)
    return EventCandidate(**fields)  # type: ignore[arg-type]


def test_a_republished_id_is_not_churn():
    """The publisher reused its identifier, which is what an identifier is for."""
    stored = _candidate("uid-1")

    result = churn_by_source([_candidate("uid-1")], stored=[stored])

    assert result["nsno_ics"].rate == 0.0


def test_a_new_id_for_a_stored_listing_is_churn():
    """Same title, venue and start; a different id. The listing is not new — its
    identifier is, which is exactly the failure this measures."""
    stored = _candidate("uid-old")

    result = churn_by_source([_candidate("uid-new")], stored=[stored])

    assert result["nsno_ics"].rate == 1.0


def test_a_genuinely_new_listing_is_outside_the_rate():
    """It has never been seen, so it says nothing about whether ids are stable.
    Counting it would dilute the rate with every real addition."""
    stored = _candidate("uid-1")
    fresh = [_candidate("uid-1"), _candidate("uid-2", title="Something Else")]

    result = churn_by_source(fresh, stored=stored and [stored])

    assert result["nsno_ics"].rate == 0.0
    assert result["nsno_ics"].seen_before == 1


def test_nothing_seen_before_leaves_the_rate_undefined():
    """A source's first run stores everything and matches nothing. Zero would
    read as 'ids are stable', which is the opposite of what is known."""
    result = churn_by_source([_candidate("uid-1")], stored=[])

    assert result["nsno_ics"].rate is None


def test_feeds_are_measured_separately():
    """Identity policy is a property of one feed; an average over all of them
    would describe none."""
    stored = [_candidate("a-old", source="a"), _candidate("b-1", source="b")]
    fresh = [_candidate("a-new", source="a"), _candidate("b-1", source="b")]

    result = churn_by_source(fresh, stored=stored)

    assert result["a"].rate == 1.0
    assert result["b"].rate == 0.0


def test_two_feeds_under_one_category_are_measured_apart():
    """`source_type` is a category and one category holds several feeds (#34).
    Live: `northshorenightout` covers an ICS feed that re-mints every Google
    UID and a listing page keyed on a content hash. Averaged, 100% and 0%
    report 60% and describe neither.
    """
    stored = [
        _candidate("ics-old", source="nsno_ics"),
        _candidate("html-1", source="nsno_html", title="Other"),
    ]
    fresh = [
        _candidate("ics-new", source="nsno_ics"),
        _candidate("html-1", source="nsno_html", title="Other"),
    ]

    result = churn_by_source(fresh, stored=stored)

    assert result["nsno_ics"].rate == 1.0
    assert result["nsno_html"].rate == 0.0


def test_a_listing_carried_by_two_feeds_is_not_churn_in_either():
    """A calendar feed and a listing page legitimately cover the same event. If
    the content key spanned the category, the second feed's own first
    publication would match the first feed's stored row and its id — correct,
    stable and new to us — would be counted as churn.
    """
    stored = [_candidate("ics-1", source="nsno_ics")]
    fresh = [_candidate("html-1", source="nsno_html")]

    result = churn_by_source(fresh, stored=stored)

    assert result["nsno_html"].rate is None
    assert result["nsno_html"].seen_before == 0


def test_the_key_is_canonical_so_casing_is_not_a_new_listing():
    """A venue written two ways is one venue. Compared raw, a re-cased republish
    reads as a brand new listing and the churn it proves is invisible."""
    stored = _candidate("uid-old", venue="The Joy Nest", title="Isabel Stover")
    fresh = [_candidate("uid-new", venue="THE JOY NEST", title="ISABEL STOVER")]

    result = churn_by_source(fresh, stored=[stored])

    assert result["nsno_ics"].rate == 1.0


def test_a_partly_churning_source_reports_the_share():
    """The rate is what the gate reads, so it must be a proportion, not a flag."""
    stored = [_candidate(f"old-{i}", title=f"Event {i}") for i in range(4)]
    fresh = [
        _candidate("new-0", title="Event 0"),
        _candidate("new-1", title="Event 1"),
        _candidate("old-2", title="Event 2"),
        _candidate("old-3", title="Event 3"),
    ]

    result = churn_by_source(fresh, stored=stored)

    assert result["nsno_ics"].rate == 0.5
    assert result["nsno_ics"].churned == 2
    assert result["nsno_ics"].seen_before == 4


def test_the_key_holds_the_start_so_a_series_is_not_one_listing():
    """Every occurrence of a recurring programme shares a title and venue. On a
    key without the start they collapse, and a whole season reads as churn."""
    first = _candidate("uid-1")
    second = _candidate("uid-2", start_time=START.replace(day=29))

    assert content_key(first) != content_key(second)
