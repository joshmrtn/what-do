"""Unit tests for OllamaClient."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests
import requests as req

from src.utils.ollama_client import OllamaClient, OllamaError


def _make_response(status_code: int, json_body: dict | None = None, text: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_body is not None:
        resp.json.return_value = json_body
    return resp


# ---------------------------------------------------------------------------
# OllamaClient.chat
# ---------------------------------------------------------------------------


def test_chat_returns_content_on_success():

    client = OllamaClient(host="http://localhost:11434", timeout=30)
    mock_resp = _make_response(200, {"message": {"content": "hello world"}})

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = client.chat(model="gemma4:e2b", messages=[{"role": "user", "content": "hi"}])

    assert result == "hello world"
    mock_post.assert_called_once()


def test_chat_raises_on_http_error():

    client = OllamaClient(host="http://localhost:11434", timeout=30)
    mock_resp = _make_response(500, text="internal error")

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(OllamaError, match="500"):
            client.chat(model="gemma4:e2b", messages=[{"role": "user", "content": "hi"}])


def test_chat_raises_on_timeout():

    client = OllamaClient(host="http://localhost:11434", timeout=30)

    with patch("requests.post", side_effect=req.Timeout("timed out")):
        with pytest.raises(OllamaError, match="timed out"):
            client.chat(model="gemma4:e2b", messages=[{"role": "user", "content": "hi"}])


def test_chat_raises_on_connection_error():

    client = OllamaClient(host="http://localhost:11434", timeout=30)

    with patch("requests.post", side_effect=req.ConnectionError("refused")):
        with pytest.raises(OllamaError, match="refused"):
            client.chat(model="gemma4:e2b", messages=[{"role": "user", "content": "hi"}])


def test_chat_passes_images_when_provided():

    client = OllamaClient(host="http://localhost:11434", timeout=30)
    mock_resp = _make_response(200, {"message": {"content": "described"}})

    with patch("requests.post", return_value=mock_resp) as mock_post:
        client.chat(
            model="gemma4:e4b",
            messages=[{"role": "user", "content": "describe this"}],
            images=[b"\x89PNG"],
        )

    payload = mock_post.call_args[1]["json"]
    msg = payload["messages"][0]
    assert "images" in msg
    assert len(msg["images"]) == 1


def test_chat_omits_images_field_when_none():

    client = OllamaClient(host="http://localhost:11434", timeout=30)
    mock_resp = _make_response(200, {"message": {"content": "ok"}})

    with patch("requests.post", return_value=mock_resp) as mock_post:
        client.chat(
            model="gemma4:e4b",
            messages=[{"role": "user", "content": "text only"}],
            images=None,
        )

    payload = mock_post.call_args[1]["json"]
    msg = payload["messages"][0]
    assert "images" not in msg


def test_chat_uses_configured_host():

    client = OllamaClient(host="http://192.168.1.50:11434", timeout=30)
    mock_resp = _make_response(200, {"message": {"content": "ok"}})

    with patch("requests.post", return_value=mock_resp) as mock_post:
        client.chat(model="gemma4:e2b", messages=[{"role": "user", "content": "hi"}])

    url = mock_post.call_args[0][0]
    assert "192.168.1.50:11434" in url


# ---------------------------------------------------------------------------
# Request parameters
# ---------------------------------------------------------------------------


def test_chat_sends_no_parameters_when_none_are_configured():
    """The bare payload stays the default, so no caller is changed by accident."""
    client = OllamaClient(host="http://localhost:11434", timeout=30)
    mock_resp = _make_response(200, {"message": {"content": "ok"}})

    with patch("requests.post", return_value=mock_resp) as mock_post:
        client.chat(model="gemma4:e2b", messages=[{"role": "user", "content": "hi"}])

    payload = mock_post.call_args[1]["json"]
    assert "format" not in payload
    assert "think" not in payload
    assert "options" not in payload


def test_chat_sends_the_response_format_when_configured():

    client = OllamaClient(host="http://localhost:11434", timeout=30, response_format="json")
    mock_resp = _make_response(200, {"message": {"content": "{}"}})

    with patch("requests.post", return_value=mock_resp) as mock_post:
        client.chat(model="gemma4:e4b", messages=[{"role": "user", "content": "hi"}])

    assert mock_post.call_args[1]["json"]["format"] == "json"


def test_chat_sends_think_false_when_configured():
    """False is a real instruction here, not an absent value."""
    client = OllamaClient(host="http://localhost:11434", timeout=30, think=False)
    mock_resp = _make_response(200, {"message": {"content": "{}"}})

    with patch("requests.post", return_value=mock_resp) as mock_post:
        client.chat(model="gemma4:e4b", messages=[{"role": "user", "content": "hi"}])

    assert mock_post.call_args[1]["json"]["think"] is False


def test_chat_sends_sampling_options_together():

    client = OllamaClient(
        host="http://localhost:11434",
        timeout=30,
        options={"temperature": 0.2, "top_p": 0.9, "num_ctx": 32768},
    )
    mock_resp = _make_response(200, {"message": {"content": "{}"}})

    with patch("requests.post", return_value=mock_resp) as mock_post:
        client.chat(model="gemma4:e4b", messages=[{"role": "user", "content": "hi"}])

    assert mock_post.call_args[1]["json"]["options"] == {
        "temperature": 0.2,
        "top_p": 0.9,
        "num_ctx": 32768,
    }


def test_chat_sends_keep_alive_when_configured():
    """Residency is the caller's to decide; the server default pins forever."""
    client = OllamaClient(host="http://localhost:11434", timeout=30, keep_alive="30m")
    mock_resp = _make_response(200, {"message": {"content": "{}"}})

    with patch("requests.post", return_value=mock_resp) as mock_post:
        client.chat(model="gemma4:e4b", messages=[{"role": "user", "content": "hi"}])

    assert mock_post.call_args[1]["json"]["keep_alive"] == "30m"


def test_embed_sends_keep_alive_when_configured():
    """An embedding model left resident squats memory the extraction model wants."""
    client = OllamaClient(host="http://localhost:11434", timeout=30, keep_alive="30m")
    mock_resp = _make_response(200, {"embeddings": [[0.1, 0.2]]})

    with patch("requests.post", return_value=mock_resp) as mock_post:
        client.embed(model="nomic-embed-text", text="live jazz")

    assert mock_post.call_args[1]["json"]["keep_alive"] == "30m"


def test_keep_alive_is_omitted_when_not_configured():

    client = OllamaClient(host="http://localhost:11434", timeout=30)
    mock_resp = _make_response(200, {"message": {"content": "{}"}})

    with patch("requests.post", return_value=mock_resp) as mock_post:
        client.chat(model="gemma4:e4b", messages=[{"role": "user", "content": "hi"}])

    assert "keep_alive" not in mock_post.call_args[1]["json"]


def test_embed_does_not_send_chat_parameters():
    """format and think are chat concerns; an embedding endpoint rejects them."""
    client = OllamaClient(
        host="http://localhost:11434",
        timeout=30,
        response_format="json",
        think=False,
        options={"temperature": 0.2},
    )
    mock_resp = _make_response(200, {"embeddings": [[0.1, 0.2]]})

    with patch("requests.post", return_value=mock_resp) as mock_post:
        client.embed(model="nomic-embed-text", text="live jazz")

    payload = mock_post.call_args[1]["json"]
    assert "format" not in payload
    assert "think" not in payload
    assert "options" not in payload


def test_the_configured_parameters_reach_the_transcript():
    """What we actually sent is the first thing to check when output is wrong."""
    transcript = _FakeTranscript()
    client = OllamaClient(
        host="http://localhost:11434",
        timeout=30,
        transcript=transcript,
        response_format="json",
        think=False,
        options={"num_ctx": 32768},
    )
    mock_resp = _make_response(200, {"message": {"content": "{}"}})

    with patch("requests.post", return_value=mock_resp):
        client.chat(model="gemma4:e4b", messages=[{"role": "user", "content": "hi"}])

    request = transcript.records[0]["request"]
    assert request["format"] == "json"
    assert request["think"] is False
    assert request["options"] == {"num_ctx": 32768}


# ---------------------------------------------------------------------------
# Transcript recording
# ---------------------------------------------------------------------------


class _FakeTranscript:
    """Captures record() calls instead of writing a file."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def record(self, **fields) -> None:
        self.records.append(fields)


def test_chat_records_the_request_and_response_to_the_transcript():

    transcript = _FakeTranscript()
    client = OllamaClient(
        host="http://localhost:11434", timeout=30, transcript=transcript, component="extraction"
    )
    body = {"message": {"content": "{}", "thinking": "reasoning"}, "done": True}

    with patch("requests.post", return_value=_make_response(200, body)):
        client.chat(model="gemma4:e4b", messages=[{"role": "user", "content": "hi"}])

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
    client = OllamaClient(host="http://localhost:11434", timeout=30, transcript=transcript)
    body = {"message": {"content": "", "thinking": "I should reply with JSON..."}}

    with patch("requests.post", return_value=_make_response(200, body)):
        client.chat(model="gemma4:e4b", messages=[{"role": "user", "content": "hi"}])

    assert transcript.records[0]["response"]["message"]["thinking"]


def test_chat_numbers_successive_calls():

    transcript = _FakeTranscript()
    client = OllamaClient(host="http://localhost:11434", timeout=30, transcript=transcript)
    mock_resp = _make_response(200, {"message": {"content": "ok"}})

    with patch("requests.post", return_value=mock_resp):
        for _ in range(3):
            client.chat(model="gemma4:e2b", messages=[{"role": "user", "content": "hi"}])

    assert [r["sequence"] for r in transcript.records] == [1, 2, 3]


def test_chat_records_before_raising_on_http_error():

    transcript = _FakeTranscript()
    client = OllamaClient(host="http://localhost:11434", timeout=30, transcript=transcript)

    with patch("requests.post", return_value=_make_response(500, text="boom")):
        with pytest.raises(OllamaError):
            client.chat(model="gemma4:e2b", messages=[{"role": "user", "content": "hi"}])

    assert transcript.records[0]["status"] == 500
    assert "500" in transcript.records[0]["error"]


def test_chat_records_before_raising_on_timeout():

    transcript = _FakeTranscript()
    client = OllamaClient(host="http://localhost:11434", timeout=30, transcript=transcript)

    with patch("requests.post", side_effect=req.Timeout("too slow")):
        with pytest.raises(OllamaError):
            client.chat(model="gemma4:e2b", messages=[{"role": "user", "content": "hi"}])

    assert transcript.records[0]["status"] is None
    assert "timed out" in transcript.records[0]["error"]


def test_chat_records_before_raising_on_connection_error():

    transcript = _FakeTranscript()
    client = OllamaClient(host="http://localhost:11434", timeout=30, transcript=transcript)

    with patch("requests.post", side_effect=req.ConnectionError("nope")):
        with pytest.raises(OllamaError):
            client.chat(model="gemma4:e2b", messages=[{"role": "user", "content": "hi"}])

    assert transcript.records[0]["status"] is None
    assert "refused" in transcript.records[0]["error"]


def test_chat_without_a_transcript_behaves_exactly_as_before():

    client = OllamaClient(host="http://localhost:11434", timeout=30)
    mock_resp = _make_response(200, {"message": {"content": "hello"}})

    with patch("requests.post", return_value=mock_resp):
        assert client.chat(model="gemma4:e2b", messages=[{"role": "user", "content": "hi"}]) == "hello"


def test_embed_records_input_length_not_the_vector():
    """A 768-float vector per event would drown the transcript."""
    transcript = _FakeTranscript()
    client = OllamaClient(host="http://localhost:11434", timeout=30, transcript=transcript)
    mock_resp = _make_response(200, {"embeddings": [[0.1] * 768]})

    with patch("requests.post", return_value=mock_resp):
        client.embed(model="nomic-embed-text", text="live jazz")

    entry = transcript.records[0]
    assert entry["request"]["input_length"] == len("live jazz")
    assert "0.1" not in json.dumps(entry, default=str)


# ---------------------------------------------------------------------------
# OllamaClient.embed
# ---------------------------------------------------------------------------


def test_embed_returns_vector_on_success():

    client = OllamaClient(host="http://localhost:11434", timeout=30)
    mock_resp = _make_response(200, {"embeddings": [[0.1, 0.2, 0.3]]})

    with patch("requests.post", return_value=mock_resp):
        result = client.embed(model="nomic-embed-text", text="karaoke")

    assert result == [0.1, 0.2, 0.3]


def test_embed_posts_model_and_input_to_embed_endpoint():

    client = OllamaClient(host="http://localhost:11434", timeout=30)
    mock_resp = _make_response(200, {"embeddings": [[0.1]]})

    with patch("requests.post", return_value=mock_resp) as mock_post:
        client.embed(model="nomic-embed-text", text="karaoke")

    assert mock_post.call_args[0][0] == "http://localhost:11434/api/embed"
    payload = mock_post.call_args.kwargs["json"]
    assert payload["model"] == "nomic-embed-text"
    assert payload["input"] == "karaoke"


def test_embed_strips_trailing_slash_from_host():

    client = OllamaClient(host="http://localhost:11434/", timeout=30)
    mock_resp = _make_response(200, {"embeddings": [[0.1]]})

    with patch("requests.post", return_value=mock_resp) as mock_post:
        client.embed(model="nomic-embed-text", text="karaoke")

    assert mock_post.call_args[0][0] == "http://localhost:11434/api/embed"


def test_embed_raises_on_http_error():

    client = OllamaClient(host="http://localhost:11434", timeout=30)
    mock_resp = _make_response(500, text="boom")

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(OllamaError, match="500"):
            client.embed(model="nomic-embed-text", text="karaoke")


def test_embed_raises_on_timeout():


    client = OllamaClient(host="http://localhost:11434", timeout=30)

    with patch("requests.post", side_effect=requests.Timeout("slow")):
        with pytest.raises(OllamaError, match="timed out"):
            client.embed(model="nomic-embed-text", text="karaoke")


def test_embed_raises_on_connection_error():


    client = OllamaClient(host="http://localhost:11434", timeout=30)

    with patch("requests.post", side_effect=requests.ConnectionError("refused")):
        with pytest.raises(OllamaError, match="refused"):
            client.embed(model="nomic-embed-text", text="karaoke")


def test_embed_raises_on_malformed_response():

    client = OllamaClient(host="http://localhost:11434", timeout=30)
    mock_resp = _make_response(200, {"unexpected": "shape"})

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(OllamaError, match="unexpected response"):
            client.embed(model="nomic-embed-text", text="karaoke")


def test_embed_raises_on_empty_embeddings_list():

    client = OllamaClient(host="http://localhost:11434", timeout=30)
    mock_resp = _make_response(200, {"embeddings": []})

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(OllamaError, match="unexpected response"):
            client.embed(model="nomic-embed-text", text="karaoke")
