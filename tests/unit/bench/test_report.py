"""Unit tests for bench output and the recorded baseline."""

from __future__ import annotations

from src.bench.report import format_report, load_run, write_run
from src.bench.runner import Measurement, Sample
from src.models.tag import Tag


def _sample(name="bare-performer-name", note="A name and nothing else.") -> Sample:
    return Sample(name=name, title="Ava Valianti", note=note)


def _measurement(variant="gemma4:e4b", tags=("music",), **kwargs) -> Measurement:
    defaults = dict(
        sample="bare-performer-name",
        variant=variant,
        tags=[Tag(text=t, weight=1.0) for t in tags],
        summary="An evening of live music.",
        seconds=12.3,
    )
    defaults.update(kwargs)
    return Measurement(**defaults)


def test_it_lists_every_variant_under_its_sample():
    report = format_report([_sample()], [_measurement("gemma4:e4b"), _measurement("gemma4:e2b")])

    assert "bare-performer-name" in report
    assert "gemma4:e4b" in report
    assert "gemma4:e2b" in report


def test_it_prints_the_note_that_justifies_the_sample():
    """The reason the sample is in the set is the context for reading the row."""
    report = format_report([_sample(note="Anti-invention nulls everything.")], [_measurement()])

    assert "Anti-invention nulls everything." in report


def test_tags_are_shown_with_their_weights():
    """Weights are the part a model most often gets wrong — every tag at 0.9
    means the model is not discriminating, and only the numbers show it."""
    report = format_report([_sample()], [_measurement(tags=("music", "live"))])

    assert "music" in report and "1.0" in report
    assert "live" in report


def test_a_degradation_is_shown():
    report = format_report(
        [_sample()], [_measurement(tags=(), degradation="tag count 0 is below minimum 1")]
    )

    assert "tag count 0" in report


def test_an_unreachable_model_says_so_in_its_row():
    report = format_report(
        [_sample()], [_measurement(tags=(), error="LLMError: connection refused")]
    )

    assert "connection refused" in report


def test_it_states_no_verdict():
    """The issue's hardest requirement. `jazz` where another model said `music`
    is different, not wrong, and a bench that scores it teaches us to prefer
    whichever model matches a guess we wrote down."""
    report = format_report([_sample()], [_measurement()])

    assert "PASS" not in report and "FAIL" not in report


def test_a_run_round_trips_through_a_file(tmp_path):
    path = tmp_path / "bench-run.json"
    measurements = [_measurement(tags=("music", "live"))]

    write_run(measurements, path)

    assert load_run(path) == measurements


def test_a_baseline_marks_what_changed():
    """The manual step is reading two outputs side by side. The least the bench
    can do is put yesterday's beside today's."""
    baseline = [_measurement(tags=("music",))]

    report = format_report([_sample()], [_measurement(tags=("comedy",))], baseline=baseline)

    assert "-music" in report
    assert "+comedy" in report


def test_an_unchanged_variant_is_not_marked():
    baseline = [_measurement(tags=("music",))]

    report = format_report([_sample()], [_measurement(tags=("music",))], baseline=baseline)

    assert "+music" not in report
    assert "-music" not in report


def test_a_variant_absent_from_the_baseline_is_new_not_changed():
    """A model being benched for the first time has not *changed* its answer,
    and marking every tag as added would bury the comparison that matters."""
    baseline = [_measurement(variant="gemma4:e4b", tags=("music",))]

    report = format_report(
        [_sample()], [_measurement(variant="brand-new-model", tags=("jazz",))], baseline=baseline
    )

    assert "+jazz" not in report
    assert "jazz" in report
