"""Unit tests for the model bench runner.

The bench is run by hand, but it is our code, so it is covered by the default
suite with the model boundary faked — the same rule every stage follows. What
cannot be covered here is whether a model is any *good*, which is the whole
reason the bench exists and the reason it asserts nothing about tag content.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.bench.runner import Sample, Variant, run_variant
from src.processing.extraction_input import extraction_input
from src.utils.chat_client import LLMError


def _reply(tags=None, summary="An evening of live jazz.") -> str:
    return json.dumps(
        {
            "title": "Jazz Night",
            "venue": "The Vault",
            "start_time": None,
            "end_time": None,
            "tags": tags if tags is not None else [{"tag": "jazz", "weight": 1.0}],
            "summary": summary,
            "setting": "indoor",
        }
    )


def _client(reply=None) -> MagicMock:
    client = MagicMock()
    client.chat.return_value = _reply() if reply is None else reply
    return client


def _sample(**kwargs) -> Sample:
    defaults = dict(
        name="bare-performer-name",
        title="Ava Valianti",
        description=None,
        venue="The Blue Room",
        location="Riverport",
        listing_category="Music",
        note="A name and nothing else — the shape that failed ten times.",
    )
    defaults.update(kwargs)
    return Sample(**defaults)


def _variant(client=None, **kwargs) -> Variant:
    defaults = dict(name="gemma4:e4b", model="gemma4:e4b", client=client or _client())
    defaults.update(kwargs)
    return Variant(**defaults)


def test_it_measures_what_the_model_returned():
    measurement = run_variant(_sample(), _variant())

    assert measurement.sample == "bare-performer-name"
    assert measurement.variant == "gemma4:e4b"
    assert [t.text for t in measurement.tags] == ["jazz"]
    assert measurement.summary == "An evening of live jazz."
    assert measurement.error is None


def test_the_prompt_is_built_by_the_production_path():
    """The one requirement the bench cannot compromise on. A bench that builds
    its own prompt measures the bench."""
    client = _client()

    run_variant(_sample(), _variant(client=client))

    prompt = client.chat.call_args.kwargs["messages"][0]["content"]
    expected = extraction_input(_sample().as_event())
    assert expected in prompt
    assert "Event category: Music" in prompt


def test_a_variant_may_supply_its_own_input_builder():
    """The axis the issue did not have and 3c needed: A/B of a *change*, not
    only of a model. `--variant current --variant venue-line` is this."""
    client = _client()

    def venue_line(event):
        return f"{event.title}\nVenue: {event.venue}, {event.location}"

    run_variant(_sample(), _variant(client=client, input_builder=venue_line))

    prompt = client.chat.call_args.kwargs["messages"][0]["content"]
    assert "Venue: The Blue Room, Riverport" in prompt


def test_a_thin_reply_is_recorded_not_raised():
    """Extraction stopped raising on a shortfall, and the bench is exactly where
    a shortfall is the interesting result rather than an error."""
    measurement = run_variant(
        _sample(), _variant(client=_client(_reply(tags=[], summary=None)))
    )

    assert measurement.tags == []
    assert "tag count 0" in measurement.degradation
    assert measurement.error is None


def test_a_clean_reply_records_no_degradation():
    assert run_variant(_sample(), _variant()).degradation is None


def test_an_unreachable_model_is_recorded_and_does_not_end_the_run():
    """One dead model must not cost the measurements of the others — the bench
    exists to compare, and a comparison missing a column is worth less than one
    with a column saying `unreachable`."""
    client = MagicMock()
    client.chat.side_effect = LLMError("connection refused")

    measurement = run_variant(_sample(), _variant(client=client))

    assert measurement.error is not None
    assert "refused" in measurement.error
    assert measurement.tags == []


def test_it_records_how_long_the_model_took():
    """Model selection is partly a speed decision: gemma4:e4b at ~2 min an event
    is the reason the whole batch is bounded."""
    assert run_variant(_sample(), _variant()).seconds >= 0


def test_the_reference_date_reaches_the_model():
    """`this Saturday` is only resolvable against a stated today, and which
    models manage the arithmetic is a real difference between them."""
    client = _client()
    monday = datetime(2026, 8, 3, tzinfo=timezone.utc)

    run_variant(_sample(reference_date=monday), _variant(client=client))

    assert "2026-08-03" in client.chat.call_args.kwargs["messages"][0]["content"]


def test_a_sample_becomes_an_event_the_pipeline_would_recognise():
    """The bench feeds `extraction_input`, which takes an Event, so a sample
    that is not a real Event would quietly diverge from what production sends."""
    event = _sample().as_event()

    assert event.title == "Ava Valianti"
    assert event.venue == "The Blue Room"
    assert event.metadata["listing_category"] == "Music"
