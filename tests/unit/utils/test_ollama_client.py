"""Unit tests for OllamaClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


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
    from src.utils.ollama_client import OllamaClient

    client = OllamaClient(host="http://localhost:11434", timeout=30)
    mock_resp = _make_response(200, {"message": {"content": "hello world"}})

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = client.chat(model="gemma4:e2b", messages=[{"role": "user", "content": "hi"}])

    assert result == "hello world"
    mock_post.assert_called_once()


def test_chat_raises_on_http_error():
    from src.utils.ollama_client import OllamaClient, OllamaError

    client = OllamaClient(host="http://localhost:11434", timeout=30)
    mock_resp = _make_response(500, text="internal error")

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(OllamaError, match="500"):
            client.chat(model="gemma4:e2b", messages=[{"role": "user", "content": "hi"}])


def test_chat_raises_on_timeout():
    from src.utils.ollama_client import OllamaClient, OllamaError
    import requests as req

    client = OllamaClient(host="http://localhost:11434", timeout=30)

    with patch("requests.post", side_effect=req.Timeout("timed out")):
        with pytest.raises(OllamaError, match="timed out"):
            client.chat(model="gemma4:e2b", messages=[{"role": "user", "content": "hi"}])


def test_chat_raises_on_connection_error():
    from src.utils.ollama_client import OllamaClient, OllamaError
    import requests as req

    client = OllamaClient(host="http://localhost:11434", timeout=30)

    with patch("requests.post", side_effect=req.ConnectionError("refused")):
        with pytest.raises(OllamaError, match="refused"):
            client.chat(model="gemma4:e2b", messages=[{"role": "user", "content": "hi"}])


def test_chat_passes_images_when_provided():
    from src.utils.ollama_client import OllamaClient

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
    from src.utils.ollama_client import OllamaClient

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
    from src.utils.ollama_client import OllamaClient

    client = OllamaClient(host="http://192.168.1.50:11434", timeout=30)
    mock_resp = _make_response(200, {"message": {"content": "ok"}})

    with patch("requests.post", return_value=mock_resp) as mock_post:
        client.chat(model="gemma4:e2b", messages=[{"role": "user", "content": "hi"}])

    url = mock_post.call_args[0][0]
    assert "192.168.1.50:11434" in url


# ---------------------------------------------------------------------------
# OllamaClient.embed
# ---------------------------------------------------------------------------


def test_embed_returns_vector_on_success():
    from src.utils.ollama_client import OllamaClient

    client = OllamaClient(host="http://localhost:11434", timeout=30)
    mock_resp = _make_response(200, {"embeddings": [[0.1, 0.2, 0.3]]})

    with patch("requests.post", return_value=mock_resp):
        result = client.embed(model="nomic-embed-text", text="karaoke")

    assert result == [0.1, 0.2, 0.3]


def test_embed_posts_model_and_input_to_embed_endpoint():
    from src.utils.ollama_client import OllamaClient

    client = OllamaClient(host="http://localhost:11434", timeout=30)
    mock_resp = _make_response(200, {"embeddings": [[0.1]]})

    with patch("requests.post", return_value=mock_resp) as mock_post:
        client.embed(model="nomic-embed-text", text="karaoke")

    assert mock_post.call_args[0][0] == "http://localhost:11434/api/embed"
    payload = mock_post.call_args.kwargs["json"]
    assert payload["model"] == "nomic-embed-text"
    assert payload["input"] == "karaoke"


def test_embed_strips_trailing_slash_from_host():
    from src.utils.ollama_client import OllamaClient

    client = OllamaClient(host="http://localhost:11434/", timeout=30)
    mock_resp = _make_response(200, {"embeddings": [[0.1]]})

    with patch("requests.post", return_value=mock_resp) as mock_post:
        client.embed(model="nomic-embed-text", text="karaoke")

    assert mock_post.call_args[0][0] == "http://localhost:11434/api/embed"


def test_embed_raises_on_http_error():
    from src.utils.ollama_client import OllamaClient, OllamaError

    client = OllamaClient(host="http://localhost:11434", timeout=30)
    mock_resp = _make_response(500, text="boom")

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(OllamaError, match="500"):
            client.embed(model="nomic-embed-text", text="karaoke")


def test_embed_raises_on_timeout():
    import requests

    from src.utils.ollama_client import OllamaClient, OllamaError

    client = OllamaClient(host="http://localhost:11434", timeout=30)

    with patch("requests.post", side_effect=requests.Timeout("slow")):
        with pytest.raises(OllamaError, match="timed out"):
            client.embed(model="nomic-embed-text", text="karaoke")


def test_embed_raises_on_connection_error():
    import requests

    from src.utils.ollama_client import OllamaClient, OllamaError

    client = OllamaClient(host="http://localhost:11434", timeout=30)

    with patch("requests.post", side_effect=requests.ConnectionError("refused")):
        with pytest.raises(OllamaError, match="refused"):
            client.embed(model="nomic-embed-text", text="karaoke")


def test_embed_raises_on_malformed_response():
    from src.utils.ollama_client import OllamaClient, OllamaError

    client = OllamaClient(host="http://localhost:11434", timeout=30)
    mock_resp = _make_response(200, {"unexpected": "shape"})

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(OllamaError, match="unexpected response"):
            client.embed(model="nomic-embed-text", text="karaoke")


def test_embed_raises_on_empty_embeddings_list():
    from src.utils.ollama_client import OllamaClient, OllamaError

    client = OllamaClient(host="http://localhost:11434", timeout=30)
    mock_resp = _make_response(200, {"embeddings": []})

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(OllamaError, match="unexpected response"):
            client.embed(model="nomic-embed-text", text="karaoke")
