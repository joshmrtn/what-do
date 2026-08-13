"""Unit tests for the disambiguation half of the bench."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.bench.disambiguation import (
    HandleSample,
    HandleVariant,
    format_classifications,
    run_handle_variant,
)
from src.utils.chat_client import LLMError


def _sample(**kwargs) -> HandleSample:
    defaults = dict(
        name="obvious-venue",
        handle="@thevaultlounge",
        context="Come enjoy live jazz at @thevaultlounge this Saturday — doors at 7pm!",
        note="A handle naming a room, with the room's own listing around it.",
    )
    defaults.update(kwargs)
    return HandleSample(**defaults)


def _variant(reply="venue") -> HandleVariant:
    client = MagicMock()
    # The shape the production provider parses. A bare word is exactly what
    # a non-complying model returns, and that is a separate test.
    client.chat.return_value = json.dumps({"classification": reply})
    return HandleVariant(name="gemma4:e2b", model="gemma4:e2b", client=client)


def test_it_records_what_the_model_answered():
    result = run_handle_variant(_sample(), _variant())

    assert result.sample == "obvious-venue"
    assert result.variant == "gemma4:e2b"
    assert result.answer == "venue"
    assert result.error is None


def test_the_prompt_comes_from_the_production_provider():
    """Same rule as extraction: a bench that builds its own prompt measures the
    bench. The handle and context must reach the model the way production sends
    them."""
    variant = _variant()

    run_handle_variant(_sample(), variant)

    prompt = variant.client.chat.call_args.kwargs["messages"][0]["content"]
    assert "@thevaultlounge" in prompt
    assert "live jazz" in prompt


def test_an_unreachable_model_is_recorded_not_raised():
    client = MagicMock()
    client.chat.side_effect = LLMError("connection refused")
    variant = HandleVariant(name="gemma4:e2b", model="gemma4:e2b", client=client)

    result = run_handle_variant(_sample(), variant)

    assert result.answer is None
    assert "refused" in result.error


def test_the_report_shows_the_answer_and_the_note():
    report = format_classifications([_sample()], [run_handle_variant(_sample(), _variant())])

    assert "obvious-venue" in report
    assert "venue" in report
    assert "A handle naming a room" in report


def test_the_report_states_no_verdict():
    """`person` for an ambiguous handle is a judgement, not an error. The
    bench shows what each model said and lets a person decide."""
    report = format_classifications(
        [_sample()], [run_handle_variant(_sample(), _variant(reply="person"))]
    )

    assert "PASS" not in report and "FAIL" not in report


def test_a_model_that_will_not_comply_is_recorded_not_raised():
    """The single most interesting thing a bench can tell you about a candidate
    model: it ignored the output contract. Raising would end the run and lose
    every other variant's answer."""
    client = MagicMock()
    client.chat.return_value = "I think it is probably a venue!"
    variant = HandleVariant(name="chatty-model", model="chatty-model", client=client)

    result = run_handle_variant(_sample(), variant)

    assert result.answer is None
    assert "after 1 retry" in result.error
