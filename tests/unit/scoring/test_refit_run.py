"""The refit as the batch runs it."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.models.event import Event
from src.models.tag import Tag
from src.scoring.curve_fit import expected_tags
from src.scoring.refit import observations, run_refit

NOW = datetime(2026, 8, 15, 2, 0, tzinfo=timezone.utc)
STATIC = (5.0, 190.0)


def _event(i: int, chars: int, tags: int, *, model: str = "m", prompt: str = "v1",
           source: str | None = "feed-a", superseded: str | None = None) -> Event:
    event = Event(
        event_id=f"evt-{i}",
        source_event_candidates=[],
        source_type="category",
        created_at=NOW - timedelta(days=100) + timedelta(minutes=i),
        updated_at=NOW,
        title="t",
    )
    event.extraction_model = model
    event.extraction_prompt_version = prompt
    event.extraction_input_chars = chars
    event.source = source
    event.superseded_by = superseded
    event.tags = [Tag(text=f"t{n}", weight=1.0) for n in range(tags)]
    return event


def _corpus(n: int, cap: float = 3.7, saturation: float = 125.0) -> list[Event]:
    return [
        _event(i, 30 + (i * 37) % 900,
               max(1, round(expected_tags(30 + (i * 37) % 900, cap, saturation) + 0.3 * ((i % 7) - 3))))
        for i in range(n)
    ]


class TestObservations:
    def test_rows_without_provenance_are_excluded(self):
        events = _corpus(5) + [_event(99, 100, 2, model=None)]  # type: ignore[arg-type]

        assert len(observations(events)) == 5

    def test_rows_without_a_stored_input_are_excluded(self):
        stale = _event(99, 0, 2)
        stale.extraction_input_chars = None

        assert observations(_corpus(5) + [stale]) == observations(_corpus(5))

    def test_superseded_rows_are_excluded(self):
        """A merged-away event is kept for provenance, not for training."""
        assert len(observations(_corpus(5) + [_event(99, 100, 2, superseded="other")])) == 5

    def test_it_is_keyed_on_the_feed_not_the_category(self):
        """`source_type` conflates feeds with different characteristics (#34)."""
        rows = observations([_event(1, 100, 2, source="feed-b")])

        assert rows[0].source_type == "feed-b"

    def test_it_falls_back_to_the_category_when_no_feed_is_recorded(self):
        rows = observations([_event(1, 100, 2, source=None)])

        assert rows[0].source_type == "category"

    def test_rows_are_chronological(self):
        """The change detector reads them as a series."""
        rows = observations(list(reversed(_corpus(20))))

        assert [r.event_id for r in rows] == [f"evt-{i}" for i in range(20)]


class TestRunRefit:
    def test_no_usable_rows_records_nothing(self):
        assert run_refit([], incumbent=STATIC, now=NOW) is None

    def test_an_unarmed_regime_still_records_why(self):
        """A refusal belongs in the record as much as a move."""
        state = run_refit(_corpus(20), incumbent=STATIC, now=NOW)

        assert state is not None
        assert (state.cap, state.saturation) == STATIC
        assert state.provenance["accepted"] is False
        assert "arm" in state.provenance["reason"]

    def test_an_armed_regime_moves_toward_the_fit(self):
        state = run_refit(_corpus(250), incumbent=STATIC, now=NOW)

        assert state is not None
        assert state.provenance["accepted"] is True
        assert state.cap < STATIC[0]
        assert state.cap > state.provenance["fitted"][0]

    def test_the_record_carries_what_a_later_reader_needs(self):
        state = run_refit(_corpus(250), incumbent=STATIC, now=NOW)

        assert state is not None
        for key in ("regime", "rows", "train_rows", "holdout_rows",
                    "incumbent_score", "candidate_score", "fitted", "applied",
                    "source_multipliers", "change_points"):
            assert key in state.provenance, key

    def test_an_unarmed_regime_records_no_multipliers(self):
        """They would describe a curve nothing was fitted to."""
        state = run_refit(_corpus(20), incumbent=STATIC, now=NOW)

        assert state is not None
        assert "source_multipliers" not in state.provenance


class TestCurveStateRoundTrip:
    """The refit writes; the next run reads. A cycle, not a calculation."""

    def test_what_is_written_is_what_comes_back(self):
        from src.storage.memory.curve_state import InMemoryCurveStateRepository

        repo = InMemoryCurveStateRepository()
        state = run_refit(_corpus(250), incumbent=STATIC, now=NOW)
        assert state is not None
        repo.save(state)

        loaded = repo.load()

        assert loaded is not None
        assert (loaded.cap, loaded.saturation) == (state.cap, state.saturation)
        assert loaded.provenance["accepted"] is True

    def test_nothing_written_means_the_defaults_stand(self):
        from src.storage.memory.curve_state import InMemoryCurveStateRepository

        assert InMemoryCurveStateRepository().load() is None

    def test_successive_runs_step_toward_the_fit(self):
        """The EWMA compounds across nights because each run starts from what
        the last one accepted, not from the config file."""
        corpus = _corpus(250)
        incumbent = STATIC
        for _ in range(5):
            state = run_refit(corpus, incumbent=incumbent, now=NOW)
            assert state is not None
            incumbent = (state.cap, state.saturation)

        assert incumbent[0] < 4.4
        assert incumbent[0] > state.provenance["fitted"][0]
