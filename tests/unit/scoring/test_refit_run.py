"""The refit as the batch runs it."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.scoring.curve_fit import expected_tags
from src.scoring.refit import observations, run_refit
from src.storage.extraction_observations import ExtractionObservation

NOW = datetime(2026, 8, 15, 2, 0, tzinfo=timezone.utc)
STATIC = (5.0, 190.0)


def _event(i: int, chars: int, tags: int, *, model: str | None = "m", prompt: str = "v1",
           source: str | None = "feed-a") -> ExtractionObservation:
    """One recorded extraction. Named for what it stands in for."""
    return ExtractionObservation(
        event_id=f"evt-{i}",
        observed_at=NOW - timedelta(days=100) + timedelta(minutes=i),
        chars=chars,
        tags=tags,
        model=model,
        prompt_version=prompt,
        degradation=None,
        source=source,
    )


def _corpus(n: int, cap: float = 3.7, saturation: float = 125.0) -> list[Event]:
    return [
        _event(i, 30 + (i * 37) % 900,
               max(1, round(expected_tags(30 + (i * 37) % 900, cap, saturation) + 0.3 * ((i % 7) - 3))))
        for i in range(n)
    ]


class TestObservations:
    def test_rows_without_provenance_are_excluded(self):
        assert len(observations(_corpus(5) + [_event(99, 100, 2, model=None)])) == 5

    def test_rows_without_a_stored_input_are_excluded(self):
        assert observations(_corpus(5) + [_event(99, 0, 2)]) == observations(_corpus(5))

    def test_it_is_keyed_on_the_feed_not_the_category(self):
        """`source_type` conflates feeds with different characteristics (#34)."""
        rows = observations([_event(1, 100, 2, source="feed-b")])

        assert rows[0].source == "feed-b"

    def test_a_row_with_no_feed_recorded_is_still_usable(self):
        rows = observations([_event(1, 100, 2, source=None)])

        assert rows[0].source == "unknown"

    def test_rows_are_ordered_by_when_they_were_observed(self):
        """The change detector reads them as a series, and this is the only
        honest chronology — `events.created_at` answers a different question."""
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


class TestChangePointsAreActedOn:
    """Not merely recorded. A mutation removing the response passed until this
    existed, because no corpus here contained a change point to respond to."""

    def _shifted(self, n: int) -> list[ExtractionObservation]:
        """A feed that abruptly starts earning far more tags."""
        rows = _corpus(n)
        half = n // 2
        return [
            ExtractionObservation(
                event_id=r.event_id,
                observed_at=r.observed_at,
                chars=r.chars,
                tags=r.tags if i < half else r.tags + 3,
                model=r.model,
                prompt_version=r.prompt_version,
                degradation=None,
                source="shifting-feed",
            )
            for i, r in enumerate(rows)
        ]

    def test_the_change_is_reported(self):
        state = run_refit(self._shifted(300), incumbent=STATIC, now=NOW)

        assert state is not None
        assert state.provenance["change_points"]

    def test_the_pre_change_rows_stop_counting(self):
        """The fit is made from the population that exists now, so the corpus it
        reports is smaller than what it was handed."""
        state = run_refit(self._shifted(300), incumbent=STATIC, now=NOW)

        assert state is not None
        assert state.provenance["rows"] < 300

    def test_an_unchanged_corpus_keeps_every_row(self):
        state = run_refit(_corpus(300), incumbent=STATIC, now=NOW)

        assert state is not None
        assert state.provenance["change_points"] == {}
        assert state.provenance["rows"] == 300
