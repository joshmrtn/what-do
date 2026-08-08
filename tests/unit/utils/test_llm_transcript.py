"""Unit tests for LLMTranscript."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.utils.llm_transcript import LLMTranscript


def _clock(*, year: int = 2026, month: int = 8, day: int = 8):
    """A fixed, timezone-aware clock."""
    return lambda: datetime(year, month, day, 12, 0, 0, tzinfo=timezone.utc)


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_record_writes_one_json_line_per_call(tmp_path):
    path = tmp_path / "llm.jsonl"
    transcript = LLMTranscript(path, get_now=_clock())

    transcript.record(
        component="extraction",
        model="gemma4:e4b",
        sequence=1,
        request={"model": "gemma4:e4b", "messages": [{"role": "user", "content": "hi"}]},
        response={"message": {"content": "{}"}},
        status=200,
        duration_ms=1234,
    )

    records = _lines(path)
    assert len(records) == 1
    assert records[0]["component"] == "extraction"
    assert records[0]["model"] == "gemma4:e4b"
    assert records[0]["sequence"] == 1
    assert records[0]["status"] == 200
    assert records[0]["duration_ms"] == 1234


def test_record_preserves_the_full_request_and_response_verbatim(tmp_path):
    path = tmp_path / "llm.jsonl"
    transcript = LLMTranscript(path, get_now=_clock())
    request = {
        "model": "gemma4:e4b",
        "messages": [{"role": "user", "content": "the entire prompt text"}],
        "format": "json",
        "options": {"temperature": 0.2},
    }
    response = {"message": {"content": "", "thinking": "long reasoning"}, "done": True}

    transcript.record(
        component="extraction",
        model="gemma4:e4b",
        sequence=1,
        request=request,
        response=response,
        status=200,
        duration_ms=1,
    )

    record = _lines(path)[0]
    assert record["request"] == request
    assert record["response"] == response


def test_record_timestamps_from_the_injected_clock(tmp_path):
    path = tmp_path / "llm.jsonl"
    transcript = LLMTranscript(path, get_now=_clock())

    transcript.record(
        component="extraction",
        model="m",
        sequence=1,
        request={},
        response={},
        status=200,
        duration_ms=0,
    )

    assert _lines(path)[0]["timestamp"] == "2026-08-08T12:00:00+00:00"


def test_record_appends_rather_than_overwriting(tmp_path):
    path = tmp_path / "llm.jsonl"
    transcript = LLMTranscript(path, get_now=_clock())

    for sequence in (1, 2, 3):
        transcript.record(
            component="extraction",
            model="m",
            sequence=sequence,
            request={},
            response={},
            status=200,
            duration_ms=0,
        )

    assert [r["sequence"] for r in _lines(path)] == [1, 2, 3]


def test_record_is_readable_before_the_transcript_is_closed(tmp_path):
    """A run lasts hours; a buffered record helps nobody watching it."""
    path = tmp_path / "llm.jsonl"
    transcript = LLMTranscript(path, get_now=_clock())

    transcript.record(
        component="extraction",
        model="m",
        sequence=1,
        request={},
        response={},
        status=200,
        duration_ms=0,
    )

    assert len(_lines(path)) == 1


def test_record_replaces_image_payloads_with_their_size(tmp_path):
    """Base64 image data would swamp the file and tells us nothing."""
    path = tmp_path / "llm.jsonl"
    transcript = LLMTranscript(path, get_now=_clock())

    transcript.record(
        component="extraction",
        model="gemma4:e4b",
        sequence=1,
        request={
            "messages": [
                {"role": "user", "content": "describe", "images": ["QUJDRA==" * 500]}
            ]
        },
        response={"message": {"content": "ok"}},
        status=200,
        duration_ms=0,
    )

    raw = path.read_text()
    assert "QUJDRA==" not in raw
    images = _lines(path)[0]["request"]["messages"][0]["images"]
    assert images == [{"bytes": 4000}]


def test_record_leaves_messages_without_images_untouched(tmp_path):
    path = tmp_path / "llm.jsonl"
    transcript = LLMTranscript(path, get_now=_clock())

    transcript.record(
        component="extraction",
        model="m",
        sequence=1,
        request={"messages": [{"role": "user", "content": "text only"}]},
        response={},
        status=200,
        duration_ms=0,
    )

    message = _lines(path)[0]["request"]["messages"][0]
    assert message == {"role": "user", "content": "text only"}


def test_record_does_not_mutate_the_caller_payload(tmp_path):
    """The redacted copy must not reach back into the request being sent."""
    path = tmp_path / "llm.jsonl"
    transcript = LLMTranscript(path, get_now=_clock())
    request = {"messages": [{"role": "user", "content": "hi", "images": ["QUJD"]}]}

    transcript.record(
        component="extraction",
        model="m",
        sequence=1,
        request=request,
        response={},
        status=200,
        duration_ms=0,
    )

    assert request["messages"][0]["images"] == ["QUJD"]


def test_record_carries_an_error_when_the_call_failed(tmp_path):
    path = tmp_path / "llm.jsonl"
    transcript = LLMTranscript(path, get_now=_clock())

    transcript.record(
        component="extraction",
        model="m",
        sequence=1,
        request={},
        response=None,
        status=None,
        duration_ms=50,
        error="timed out: too slow",
    )

    record = _lines(path)[0]
    assert record["error"] == "timed out: too slow"
    assert record["response"] is None
    assert record["status"] is None


def test_record_omits_the_error_field_on_success(tmp_path):
    path = tmp_path / "llm.jsonl"
    transcript = LLMTranscript(path, get_now=_clock())

    transcript.record(
        component="extraction",
        model="m",
        sequence=1,
        request={},
        response={},
        status=200,
        duration_ms=0,
    )

    assert "error" not in _lines(path)[0]


def test_record_falls_back_to_a_string_for_unserialisable_values(tmp_path):
    """A transcript must never be the thing that fails a batch."""
    path = tmp_path / "llm.jsonl"
    transcript = LLMTranscript(path, get_now=_clock())

    transcript.record(
        component="extraction",
        model="m",
        sequence=1,
        request={"messages": [{"role": "user", "content": object()}]},
        response={},
        status=200,
        duration_ms=0,
    )

    assert len(_lines(path)) == 1


def test_creates_the_parent_directory_when_missing(tmp_path):
    path = tmp_path / "nested" / "dir" / "llm.jsonl"
    transcript = LLMTranscript(path, get_now=_clock())

    transcript.record(
        component="extraction",
        model="m",
        sequence=1,
        request={},
        response={},
        status=200,
        duration_ms=0,
    )

    assert len(_lines(path)) == 1
