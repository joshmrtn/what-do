"""Unit tests for OllamaClient.

The client goes through the shared request policy like every other caller. Only
the **session** is a double here — the policy, the throttle and the patience are
the real objects, so these exercise the real retry and the real timeout.

Locality is not an exemption from retry. Politeness protects a third party and
there is none at `localhost`; retry protects our own pipeline, and a model that
answers a request in minutes is exactly where a dropped connection costs most.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
import requests

from src.config import ConfigError, Patience
from src.utils.chat_client import GENERATION_PATIENCE
from src.utils.ollama_client import OllamaClient, OllamaError

from tests.support.network import fetcher_policy

HOST = "http://localhost:11434"
NOW = datetime(2026, 8, 20, 2, 0, tzinfo=timezone.utc)

#: What the host's own policy allows, so a test can tell the two apart.
HOST_TIMEOUT = 30.0
#: …and what a generation is given instead.
GENERATION_TIMEOUT = 1200.0

_GENERATION = {
    GENERATION_PATIENCE: Patience(
        timeout_seconds=GENERATION_TIMEOUT,
        max_attempts=2,
        backoff_base_seconds=5.0,
        backoff_max_seconds=30.0,
    )
}


def _response(status_code: int, json_body: dict | None = None, text: str = ""):
    """A real `requests.Response`, so `raise_for_status` and `.text` are real.

    A mock would have to restate what a response does, and the parts that matter
    here — the status a `HTTPError` carries, and the body behind it — are exactly
    what the transient check reads.
    """
    response = requests.Response()
    response.status_code = status_code
    body = json.dumps(json_body) if json_body is not None else text
    response._content = body.encode("utf-8")
    return response


class _FakeSession:
    """Answers with what it was seeded with, and records what arrived.

    A spy at the transport boundary: it performs nothing and claims nothing about
    how a request is made. Answers are consumed in order and the last one repeats,
    so a retry test seeds `(failure, success)` and an exhaustion test seeds one.
    """

    def __init__(self, *answers) -> None:
        self._answers = list(answers) or [_response(200, {"message": {"content": "ok"}})]
        self.posts: list[dict] = []

    def post(self, url, *, json, timeout):  # noqa: A002 - requests' own keyword
        self.posts.append({"url": url, "payload": json, "timeout": timeout})
        answer = self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer

    @property
    def payload(self) -> dict:
        """What the first post carried, which is what most tests ask about."""
        return self.posts[0]["payload"]


class _FakeTranscript:
    """Captures record() calls instead of writing a file."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def record(self, **fields) -> None:
        self.records.append(fields)


def _client(
    session,
    *,
    host: str = HOST,
    sleeps: list[float] | None = None,
    patience: dict | None = None,
    **kwargs,
) -> OllamaClient:
    """The real client over a faked session, with a real policy behind it."""
    return OllamaClient(
        host,
        session=session,
        policy=fetcher_policy(
            urls=host,
            now=NOW,
            cache_ttl=None,
            timeout_seconds=HOST_TIMEOUT,
            max_attempts=3,
            sleeps=sleeps,
            patience=_GENERATION if patience is None else patience,
        ),
        get_now=lambda: NOW,
        **kwargs,
    )


def _chat(client: OllamaClient, model: str = "gemma4:e2b") -> str:
    return client.chat(model=model, messages=[{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# OllamaClient.chat
# ---------------------------------------------------------------------------


def test_chat_returns_content_on_success():
    session = _FakeSession(_response(200, {"message": {"content": "hello world"}}))

    assert _chat(_client(session)) == "hello world"
    assert len(session.posts) == 1


def test_chat_raises_on_http_error():
    session = _FakeSession(_response(500, text="internal error"))

    with pytest.raises(OllamaError, match="500"):
        _chat(_client(session))


def test_chat_raises_on_timeout():
    session = _FakeSession(requests.Timeout("timed out"))

    with pytest.raises(OllamaError, match="timed out"):
        _chat(_client(session))


def test_chat_raises_on_connection_error():
    session = _FakeSession(requests.ConnectionError("refused"))

    with pytest.raises(OllamaError, match="refused"):
        _chat(_client(session))


def test_chat_passes_images_when_provided():
    session = _FakeSession(_response(200, {"message": {"content": "described"}}))

    _client(session).chat(
        model="gemma4:e4b",
        messages=[{"role": "user", "content": "describe this"}],
        images=[b"\x89PNG"],
    )

    message = session.payload["messages"][0]
    assert "images" in message
    assert len(message["images"]) == 1


def test_chat_omits_images_field_when_none():
    session = _FakeSession(_response(200, {"message": {"content": "ok"}}))

    _client(session).chat(
        model="gemma4:e4b",
        messages=[{"role": "user", "content": "text only"}],
        images=None,
    )

    assert "images" not in session.payload["messages"][0]


def test_chat_uses_configured_host():
    session = _FakeSession(_response(200, {"message": {"content": "ok"}}))

    _chat(_client(session, host="http://192.168.1.50:11434"))

    assert "192.168.1.50:11434" in session.posts[0]["url"]


def test_a_host_with_no_hostname_is_refused_when_the_client_is_built():
    """The policy is per host, so a client that cannot name its host has none.

    Caught at construction rather than at the first call: the batch builds its
    clients before it fetches anything, so this is hours earlier than the
    alternative and does not waste a run.
    """
    with pytest.raises(OllamaError, match="host"):
        _client(_FakeSession(), host="localhost:11434")


def test_a_model_host_with_no_policy_is_refused_when_the_client_is_built():
    """Hours earlier than the alternative.

    The batch builds its clients before it ingests anything, so an unassigned
    model host is named at 02:00:01 rather than after the fetching it would
    waste — and rather than in the middle of extraction, where the run is
    already hours in.
    """
    session = _FakeSession()

    with pytest.raises(ConfigError, match="gpu-box.lan"):
        OllamaClient(
            "http://gpu-box.lan:11434",
            session=session,
            policy=fetcher_policy(urls=HOST, now=NOW, patience=_GENERATION),
            get_now=lambda: NOW,
        )


# ---------------------------------------------------------------------------
# Retry, and the patience a generation is given
# ---------------------------------------------------------------------------


class TestRetryIsNotPoliteness:
    """Ollama is exempt from spacing as localhost, and from nothing else.

    #36: the exemption answers politeness — there is no third party to protect —
    and says nothing about a transport dropping a request we care about.
    """

    def test_a_transient_failure_is_tried_again(self):
        session = _FakeSession(
            _response(503, text="model loading"),
            _response(200, {"message": {"content": "hello"}}),
        )

        assert _chat(_client(session, sleeps=[])) == "hello"
        assert len(session.posts) == 2

    def test_a_timeout_is_tried_again(self):
        session = _FakeSession(
            requests.Timeout("too slow"),
            _response(200, {"message": {"content": "hello"}}),
        )

        assert _chat(_client(session, sleeps=[])) == "hello"
        assert len(session.posts) == 2

    def test_a_bad_request_is_not_repeated(self):
        """A 400 fails identically however politely it is asked again."""
        session = _FakeSession(_response(400, text="bad payload"))

        with pytest.raises(OllamaError):
            _chat(_client(session, sleeps=[]))

        assert len(session.posts) == 1

    def test_a_generation_stops_at_the_attempts_its_patience_allows(self):
        """Two, because a local model failing twice is dead rather than busy —
        and because a retried extraction spends the budget twice."""
        session = _FakeSession(_response(503, text="still loading"))

        with pytest.raises(OllamaError):
            _chat(_client(session, sleeps=[]))

        assert len(session.posts) == 2

    def test_an_embedding_falls_back_to_the_hosts_attempts(self):
        """It names no patience, so nothing about it changed."""
        session = _FakeSession(_response(503, text="still loading"))

        with pytest.raises(OllamaError):
            _client(session, sleeps=[]).embed(model="nomic-embed-text", text="jazz")

        assert len(session.posts) == 3


class TestPatienceBelongsToTheRequest:
    """A generation takes minutes; an embedding from the same host takes
    milliseconds. One host, two shapes, so the timeout cannot live with the host."""

    def test_a_chat_is_given_the_generation_timeout(self):
        session = _FakeSession(_response(200, {"message": {"content": "ok"}}))

        _chat(_client(session))

        assert session.posts[0]["timeout"] == GENERATION_TIMEOUT

    def test_an_embedding_is_given_the_hosts_own_timeout(self):
        """The CLI embeds on the read path, and it promises to be snappy."""
        session = _FakeSession(_response(200, {"embeddings": [[0.1]]}))

        _client(session).embed(model="nomic-embed-text", text="jazz")

        assert session.posts[0]["timeout"] == HOST_TIMEOUT

    def test_a_chat_is_refused_when_no_generation_patience_is_declared(self):
        """Config, not code, and it fails before the model is troubled."""
        session = _FakeSession(_response(200, {"message": {"content": "ok"}}))

        with pytest.raises(ConfigError, match=GENERATION_PATIENCE):
            _chat(_client(session, patience={}))

        assert session.posts == []


# ---------------------------------------------------------------------------
# Request parameters
# ---------------------------------------------------------------------------


def test_chat_sends_no_parameters_when_none_are_configured():
    """The bare payload stays the default, so no caller is changed by accident."""
    session = _FakeSession(_response(200, {"message": {"content": "ok"}}))

    _chat(_client(session))

    assert "format" not in session.payload
    assert "think" not in session.payload
    assert "options" not in session.payload


def test_chat_sends_the_response_format_when_configured():
    session = _FakeSession(_response(200, {"message": {"content": "{}"}}))

    _chat(_client(session, response_format="json"), model="gemma4:e4b")

    assert session.payload["format"] == "json"


def test_chat_sends_think_false_when_configured():
    """False is a real instruction here, not an absent value."""
    session = _FakeSession(_response(200, {"message": {"content": "{}"}}))

    _chat(_client(session, think=False), model="gemma4:e4b")

    assert session.payload["think"] is False


def test_chat_sends_sampling_options_together():
    session = _FakeSession(_response(200, {"message": {"content": "{}"}}))
    options = {"temperature": 0.2, "top_p": 0.9, "num_ctx": 32768}

    _chat(_client(session, options=options), model="gemma4:e4b")

    assert session.payload["options"] == options


def test_chat_sends_keep_alive_when_configured():
    """Residency is the caller's to decide; the server default pins forever."""
    session = _FakeSession(_response(200, {"message": {"content": "{}"}}))

    _chat(_client(session, keep_alive="30m"), model="gemma4:e4b")

    assert session.payload["keep_alive"] == "30m"


def test_embed_sends_keep_alive_when_configured():
    """An embedding model left resident squats memory the extraction model wants."""
    session = _FakeSession(_response(200, {"embeddings": [[0.1, 0.2]]}))

    _client(session, keep_alive="30m").embed(model="nomic-embed-text", text="live jazz")

    assert session.payload["keep_alive"] == "30m"


def test_keep_alive_is_omitted_when_not_configured():
    session = _FakeSession(_response(200, {"message": {"content": "{}"}}))

    _chat(_client(session), model="gemma4:e4b")

    assert "keep_alive" not in session.payload


def test_embed_does_not_send_chat_parameters():
    """format and think are chat concerns; an embedding endpoint rejects them."""
    session = _FakeSession(_response(200, {"embeddings": [[0.1, 0.2]]}))
    client = _client(
        session, response_format="json", think=False, options={"temperature": 0.2}
    )

    client.embed(model="nomic-embed-text", text="live jazz")

    assert "format" not in session.payload
    assert "think" not in session.payload
    assert "options" not in session.payload


def test_the_configured_parameters_reach_the_transcript():
    """What we actually sent is the first thing to check when output is wrong."""
    transcript = _FakeTranscript()
    session = _FakeSession(_response(200, {"message": {"content": "{}"}}))
    client = _client(
        session,
        transcript=transcript,
        response_format="json",
        think=False,
        options={"num_ctx": 32768},
    )

    _chat(client, model="gemma4:e4b")

    request = transcript.records[0]["request"]
    assert request["format"] == "json"
    assert request["think"] is False
    assert request["options"] == {"num_ctx": 32768}


# ---------------------------------------------------------------------------
# Transcript recording
# ---------------------------------------------------------------------------


def test_chat_records_the_request_and_response_to_the_transcript():
    transcript = _FakeTranscript()
    body = {"message": {"content": "{}", "thinking": "reasoning"}, "done": True}
    session = _FakeSession(_response(200, body))

    _chat(_client(session, transcript=transcript, component="extraction"), model="gemma4:e4b")

    assert len(transcript.records) == 1
    entry = transcript.records[0]
    assert entry["component"] == "extraction"
    assert entry["model"] == "gemma4:e4b"
    assert entry["status"] == 200
    assert entry["request"]["messages"] == [{"role": "user", "content": "hi"}]
    assert entry["response"] == body


def test_chat_records_thinking_even_when_content_is_empty():
    """The case we are actually hunting: the answer went to the wrong channel."""
    transcript = _FakeTranscript()
    body = {"message": {"content": "", "thinking": "I should reply with JSON..."}}
    session = _FakeSession(_response(200, body))

    _chat(_client(session, transcript=transcript), model="gemma4:e4b")

    assert transcript.records[0]["response"]["message"]["thinking"]


def test_chat_numbers_successive_calls():
    transcript = _FakeTranscript()
    session = _FakeSession(_response(200, {"message": {"content": "ok"}}))
    client = _client(session, transcript=transcript)

    for _ in range(3):
        _chat(client)

    assert [record["sequence"] for record in transcript.records] == [1, 2, 3]


def test_every_attempt_is_recorded_separately():
    """A call that failed and was repeated is two exchanges, and the first is
    the one worth reading — the transcript is where a stall is diagnosed."""
    transcript = _FakeTranscript()
    session = _FakeSession(
        _response(503, text="model loading"),
        _response(200, {"message": {"content": "hello"}}),
    )

    _chat(_client(session, transcript=transcript, sleeps=[]))

    assert [record["sequence"] for record in transcript.records] == [1, 2]
    assert transcript.records[0]["status"] == 503
    assert transcript.records[1]["status"] == 200


def test_chat_records_before_raising_on_http_error():
    transcript = _FakeTranscript()
    session = _FakeSession(_response(500, text="boom"))

    with pytest.raises(OllamaError):
        _chat(_client(session, transcript=transcript, sleeps=[]))

    assert transcript.records[0]["status"] == 500
    assert "500" in transcript.records[0]["error"]


def test_chat_records_before_raising_on_timeout():
    transcript = _FakeTranscript()
    session = _FakeSession(requests.Timeout("too slow"))

    with pytest.raises(OllamaError):
        _chat(_client(session, transcript=transcript, sleeps=[]))

    assert transcript.records[0]["status"] is None
    assert "timed out" in transcript.records[0]["error"]


def test_chat_records_before_raising_on_connection_error():
    transcript = _FakeTranscript()
    session = _FakeSession(requests.ConnectionError("nope"))

    with pytest.raises(OllamaError):
        _chat(_client(session, transcript=transcript, sleeps=[]))

    assert transcript.records[0]["status"] is None
    assert "refused" in transcript.records[0]["error"]


def test_chat_without_a_transcript_behaves_exactly_as_before():
    session = _FakeSession(_response(200, {"message": {"content": "hello"}}))

    assert _chat(_client(session)) == "hello"


def test_embed_records_input_length_not_the_vector():
    """A 768-float vector per event would drown the transcript."""
    transcript = _FakeTranscript()
    session = _FakeSession(_response(200, {"embeddings": [[0.1] * 768]}))

    _client(session, transcript=transcript).embed(model="nomic-embed-text", text="live jazz")

    entry = transcript.records[0]
    assert entry["request"]["input_length"] == len("live jazz")
    assert "0.1" not in json.dumps(entry, default=str)


# ---------------------------------------------------------------------------
# OllamaClient.embed
# ---------------------------------------------------------------------------


def test_embed_returns_vector_on_success():
    session = _FakeSession(_response(200, {"embeddings": [[0.1, 0.2, 0.3]]}))

    result = _client(session).embed(model="nomic-embed-text", text="karaoke")

    assert result == [0.1, 0.2, 0.3]


def test_embed_posts_model_and_input_to_embed_endpoint():
    session = _FakeSession(_response(200, {"embeddings": [[0.1]]}))

    _client(session).embed(model="nomic-embed-text", text="karaoke")

    assert session.posts[0]["url"] == "http://localhost:11434/api/embed"
    assert session.payload["model"] == "nomic-embed-text"
    assert session.payload["input"] == "karaoke"


def test_embed_strips_trailing_slash_from_host():
    session = _FakeSession(_response(200, {"embeddings": [[0.1]]}))

    _client(session, host="http://localhost:11434/").embed(
        model="nomic-embed-text", text="karaoke"
    )

    assert session.posts[0]["url"] == "http://localhost:11434/api/embed"


def test_embed_raises_on_http_error():
    session = _FakeSession(_response(500, text="boom"))

    with pytest.raises(OllamaError, match="500"):
        _client(session, sleeps=[]).embed(model="nomic-embed-text", text="karaoke")


def test_embed_raises_on_timeout():
    session = _FakeSession(requests.Timeout("slow"))

    with pytest.raises(OllamaError, match="timed out"):
        _client(session, sleeps=[]).embed(model="nomic-embed-text", text="karaoke")


def test_embed_raises_on_connection_error():
    session = _FakeSession(requests.ConnectionError("refused"))

    with pytest.raises(OllamaError, match="refused"):
        _client(session, sleeps=[]).embed(model="nomic-embed-text", text="karaoke")


def test_embed_raises_on_malformed_response():
    session = _FakeSession(_response(200, {"unexpected": "shape"}))

    with pytest.raises(OllamaError, match="unexpected response"):
        _client(session).embed(model="nomic-embed-text", text="karaoke")


def test_embed_raises_on_empty_embeddings_list():
    session = _FakeSession(_response(200, {"embeddings": []}))

    with pytest.raises(OllamaError, match="unexpected response"):
        _client(session).embed(model="nomic-embed-text", text="karaoke")


def test_a_malformed_response_is_not_retried():
    """The model answered; asking again gets the same answer more slowly."""
    session = _FakeSession(_response(200, {"unexpected": "shape"}))

    with pytest.raises(OllamaError):
        _client(session, sleeps=[]).embed(model="nomic-embed-text", text="karaoke")

    assert len(session.posts) == 1


def test_a_cache_is_declared_rather_than_forgotten():
    """Two identical prompts are two calls, on purpose.

    Extraction skips on `extraction_input_hash` one layer up, which avoids the
    call rather than replaying its answer, and a tag vector is stored by the
    caller that knows what identifies it.
    """
    session = _FakeSession(_response(200, {"message": {"content": "ok"}}))
    client = _client(session)

    _chat(client)
    _chat(client)

    assert len(session.posts) == 2
