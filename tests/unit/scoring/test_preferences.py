"""Unit tests for preference parsing and the embedding cache."""

from __future__ import annotations

import io
import sqlite3

import pytest

from src.storage.db import init_db
from src.utils.logging import get_logger


class _CountingProvider:
    """Embedding provider that records every call it receives."""

    def __init__(self, dim: int = 4):
        self._dim = dim
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        # Deterministic per-text vector so cached and fresh values are comparable.
        seed = float(sum(ord(c) for c in text) % 97) / 97.0
        return [seed + i for i in range(self._dim)]


def _repo(tmp_path, provider=None, logger=None):
    from src.scoring.preferences import PreferenceRepository

    db_path = tmp_path / "test.db"
    init_db(db_path)
    return PreferenceRepository(
        provider=provider or _CountingProvider(),
        db_path=db_path,
        logger=logger or get_logger("test", stream=io.StringIO()),
    ), db_path


def _write(tmp_path, name: str, content: str):
    path = tmp_path / name
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_lines_before_first_header_are_general():
    from src.scoring.preferences import parse_preferences

    prefs = parse_preferences("karaoke\nlive music\n", preference_type="like")

    assert [p.domain for p in prefs] == ["general", "general"]
    assert [p.text for p in prefs] == ["karaoke", "live music"]


def test_section_headers_set_domain():
    from src.scoring.preferences import parse_preferences

    prefs = parse_preferences(
        "karaoke\n\n[movies]\nhorror films\n\n[restaurants]\nsushi\n",
        preference_type="like",
    )

    assert [(p.domain, p.text) for p in prefs] == [
        ("general", "karaoke"),
        ("movies", "horror films"),
        ("restaurants", "sushi"),
    ]


def test_explicit_general_header_supported():
    from src.scoring.preferences import parse_preferences

    prefs = parse_preferences("[general]\nkaraoke\n", preference_type="like")

    assert prefs[0].domain == "general"


def test_header_case_and_whitespace_normalised():
    from src.scoring.preferences import parse_preferences

    prefs = parse_preferences("[ Movies ]\nhorror films\n", preference_type="like")

    assert prefs[0].domain == "movies"


def test_blank_lines_and_whitespace_skipped():
    from src.scoring.preferences import parse_preferences

    prefs = parse_preferences("karaoke\n\n   \n\nlive music\n", preference_type="like")

    assert [p.text for p in prefs] == ["karaoke", "live music"]


def test_surrounding_whitespace_stripped_from_entries():
    from src.scoring.preferences import parse_preferences

    prefs = parse_preferences("   karaoke   \n", preference_type="like")

    assert prefs[0].text == "karaoke"


def test_comment_lines_skipped():
    """A user will write a comment; embedding it would pollute scoring."""
    from src.scoring.preferences import parse_preferences

    prefs = parse_preferences("# things I like\nkaraoke\n", preference_type="like")

    assert [p.text for p in prefs] == ["karaoke"]


def test_preference_type_recorded():
    from src.scoring.preferences import parse_preferences

    assert parse_preferences("bars\n", preference_type="dislike")[0].preference_type == "dislike"


def test_empty_file_yields_no_preferences():
    from src.scoring.preferences import parse_preferences

    assert parse_preferences("", preference_type="like") == []


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------


def test_cold_cache_embeds_every_line_once(tmp_path):
    provider = _CountingProvider()
    repo, _ = _repo(tmp_path, provider)
    likes = _write(tmp_path, "likes.txt", "karaoke\npunk music\n")
    dislikes = _write(tmp_path, "dislikes.txt", "bars\n")

    repo.load(likes, dislikes)

    assert sorted(provider.calls) == ["bars", "karaoke", "punk music"]


def test_embeddings_persisted_as_blobs(tmp_path):
    repo, db_path = _repo(tmp_path)
    likes = _write(tmp_path, "likes.txt", "karaoke\n")
    dislikes = _write(tmp_path, "dislikes.txt", "")

    repo.load(likes, dislikes)

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT line_text, domain, preference_type, embedding FROM preference_embeddings_cache"
    ).fetchone()
    conn.close()

    assert row[0] == "karaoke"
    assert row[1] == "general"
    assert row[2] == "like"
    assert isinstance(row[3], bytes)


def test_unchanged_files_do_not_call_provider(tmp_path):
    provider = _CountingProvider()
    repo, _ = _repo(tmp_path, provider)
    likes = _write(tmp_path, "likes.txt", "karaoke\npunk music\n")
    dislikes = _write(tmp_path, "dislikes.txt", "bars\n")

    repo.load(likes, dislikes)
    provider.calls.clear()
    repo.load(likes, dislikes)

    assert provider.calls == []


def test_cached_embeddings_match_freshly_generated(tmp_path):
    provider = _CountingProvider()
    repo, _ = _repo(tmp_path, provider)
    likes = _write(tmp_path, "likes.txt", "karaoke\n")
    dislikes = _write(tmp_path, "dislikes.txt", "")

    first = repo.load(likes, dislikes)
    second = repo.load(likes, dislikes)

    assert first.likes[0].embedding == second.likes[0].embedding


def test_only_edited_line_is_re_embedded(tmp_path):
    provider = _CountingProvider()
    repo, _ = _repo(tmp_path, provider)
    likes = _write(tmp_path, "likes.txt", "karaoke\npunk music\n")
    dislikes = _write(tmp_path, "dislikes.txt", "")

    repo.load(likes, dislikes)
    provider.calls.clear()
    likes.write_text("karaoke\nemo music\n")
    repo.load(likes, dislikes)

    assert provider.calls == ["emo music"]


def test_editing_likes_leaves_dislikes_cached(tmp_path):
    provider = _CountingProvider()
    repo, _ = _repo(tmp_path, provider)
    likes = _write(tmp_path, "likes.txt", "karaoke\n")
    dislikes = _write(tmp_path, "dislikes.txt", "bars\nnightclubs\n")

    repo.load(likes, dislikes)
    provider.calls.clear()
    likes.write_text("karaoke\nemo music\n")
    repo.load(likes, dislikes)

    assert "bars" not in provider.calls
    assert "nightclubs" not in provider.calls


def test_deleted_line_is_pruned_from_cache(tmp_path):
    repo, db_path = _repo(tmp_path)
    likes = _write(tmp_path, "likes.txt", "karaoke\npunk music\n")
    dislikes = _write(tmp_path, "dislikes.txt", "")

    repo.load(likes, dislikes)
    likes.write_text("karaoke\n")
    result = repo.load(likes, dislikes)

    assert [p.text for p in result.likes] == ["karaoke"]
    conn = sqlite3.connect(db_path)
    texts = [r[0] for r in conn.execute("SELECT line_text FROM preference_embeddings_cache")]
    conn.close()
    assert texts == ["karaoke"]


def test_moving_a_line_to_another_domain_updates_without_re_embedding(tmp_path):
    """The embedding depends on text alone, so a domain move must not re-embed."""
    provider = _CountingProvider()
    repo, _ = _repo(tmp_path, provider)
    likes = _write(tmp_path, "likes.txt", "horror films\n")
    dislikes = _write(tmp_path, "dislikes.txt", "")

    repo.load(likes, dislikes)
    provider.calls.clear()
    likes.write_text("[movies]\nhorror films\n")
    result = repo.load(likes, dislikes)

    assert provider.calls == []
    assert result.likes[0].domain == "movies"


def test_duplicate_lines_embedded_once(tmp_path):
    provider = _CountingProvider()
    repo, _ = _repo(tmp_path, provider)
    likes = _write(tmp_path, "likes.txt", "karaoke\nkaraoke\n")
    dislikes = _write(tmp_path, "dislikes.txt", "")

    repo.load(likes, dislikes)

    assert provider.calls == ["karaoke"]


def test_same_text_in_likes_and_dislikes_kept_separate(tmp_path):
    """Cache rows are scoped per file, so identical text in both files coexists."""
    repo, _ = _repo(tmp_path)
    likes = _write(tmp_path, "likes.txt", "live music\n")
    dislikes = _write(tmp_path, "dislikes.txt", "live music\n")

    result = repo.load(likes, dislikes)

    assert [p.text for p in result.likes] == ["live music"]
    assert [p.text for p in result.dislikes] == ["live music"]


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_missing_file_yields_empty_list_and_warning(tmp_path):
    stream = io.StringIO()
    repo, _ = _repo(tmp_path, logger=get_logger("test", stream=stream))
    dislikes = _write(tmp_path, "dislikes.txt", "bars\n")

    result = repo.load(tmp_path / "nope.txt", dislikes)

    assert result.likes == []
    assert [p.text for p in result.dislikes] == ["bars"]
    assert "nope.txt" in stream.getvalue()


class _Failing:
    """Provider that fails on one specific line."""

    def __init__(self, failing_text: str = "punk music"):
        self.calls: list[str] = []
        self._failing_text = failing_text

    def embed(self, text):
        from src.scoring.embeddings import EmbeddingError

        self.calls.append(text)
        if text == self._failing_text:
            raise EmbeddingError("model unavailable")
        return [1.0, 2.0]


def test_embedding_failure_is_fatal(tmp_path):
    """A partial preference set silently re-scores the whole batch, so refuse it."""
    from src.scoring.embeddings import EmbeddingError

    repo, _ = _repo(tmp_path, _Failing())
    likes = _write(tmp_path, "likes.txt", "karaoke\npunk music\n")
    dislikes = _write(tmp_path, "dislikes.txt", "")

    with pytest.raises(EmbeddingError, match="punk music"):
        repo.load(likes, dislikes)


def test_embedding_failure_names_the_file_and_logs(tmp_path):
    from src.scoring.embeddings import EmbeddingError

    stream = io.StringIO()
    repo, _ = _repo(tmp_path, _Failing(), logger=get_logger("test", stream=stream))
    likes = _write(tmp_path, "likes.txt", "karaoke\npunk music\n")
    dislikes = _write(tmp_path, "dislikes.txt", "")

    with pytest.raises(EmbeddingError, match="likes.txt"):
        repo.load(likes, dislikes)

    assert "punk music" in stream.getvalue()


def test_successful_lines_before_a_failure_are_still_cached(tmp_path):
    """A retry after a transient failure must not re-embed what already worked."""
    from src.scoring.embeddings import EmbeddingError

    provider = _Failing()
    repo, _ = _repo(tmp_path, provider)
    likes = _write(tmp_path, "likes.txt", "karaoke\npunk music\n")
    dislikes = _write(tmp_path, "dislikes.txt", "")

    with pytest.raises(EmbeddingError):
        repo.load(likes, dislikes)

    provider.calls.clear()
    likes.write_text("karaoke\n")
    repo.load(likes, dislikes)

    assert provider.calls == []


def test_undecodable_file_is_fatal(tmp_path):
    """A present-but-unreadable file means real preferences are being ignored."""
    from src.scoring.preferences import PreferenceError

    repo, _ = _repo(tmp_path)
    likes = tmp_path / "likes.txt"
    likes.write_bytes(b"karaoke\n\xff\xfe not utf8\n")
    dislikes = _write(tmp_path, "dislikes.txt", "bars\n")

    with pytest.raises(PreferenceError, match="likes.txt"):
        repo.load(likes, dislikes)


def test_absent_file_is_tolerated_but_unreadable_is_not(tmp_path):
    """Absent means 'none configured'; unreadable means 'configured but broken'."""
    repo, _ = _repo(tmp_path)
    dislikes = _write(tmp_path, "dislikes.txt", "bars\n")

    result = repo.load(tmp_path / "absent.txt", dislikes)

    assert result.likes == []


def test_overlong_line_is_rejected_before_embedding(tmp_path):
    """A pasted document would otherwise time out the embedder on every run."""
    from src.scoring.preferences import PreferenceError

    provider = _CountingProvider()
    repo, _ = _repo(tmp_path, provider)
    likes = _write(tmp_path, "likes.txt", "karaoke\n" + ("x" * 100_000) + "\n")
    dislikes = _write(tmp_path, "dislikes.txt", "")

    with pytest.raises(PreferenceError, match="too long"):
        repo.load(likes, dislikes)

    assert "x" * 100_000 not in provider.calls


def test_overlong_line_error_identifies_the_line(tmp_path):
    from src.scoring.preferences import PreferenceError

    repo, _ = _repo(tmp_path)
    likes = _write(tmp_path, "likes.txt", "sushi " * 200 + "\n")
    dislikes = _write(tmp_path, "dislikes.txt", "")

    with pytest.raises(PreferenceError, match="likes.txt"):
        repo.load(likes, dislikes)


def test_line_at_the_length_limit_is_accepted(tmp_path):
    from src.scoring.preferences import MAX_PREFERENCE_LENGTH, parse_preferences

    prefs = parse_preferences("a" * MAX_PREFERENCE_LENGTH, preference_type="like")

    assert len(prefs) == 1


def test_zero_width_characters_stripped(tmp_path):
    """Invisible characters would embed as a real but meaningless preference."""
    from src.scoring.preferences import parse_preferences

    prefs = parse_preferences("kara​ok﻿e\n", preference_type="like")

    assert [p.text for p in prefs] == ["karaoke"]


def test_line_of_only_invisible_characters_skipped(tmp_path):
    from src.scoring.preferences import parse_preferences

    prefs = parse_preferences("​﻿\n‍\n", preference_type="like")

    assert prefs == []


# ---------------------------------------------------------------------------
# Domain scoping helper
# ---------------------------------------------------------------------------


def test_domain_lookup_returns_general_plus_domain(tmp_path):
    repo, _ = _repo(tmp_path)
    likes = _write(tmp_path, "likes.txt", "karaoke\n\n[movies]\nhorror films\n")
    dislikes = _write(tmp_path, "dislikes.txt", "")

    result = repo.load(likes, dislikes)

    assert sorted(p.text for p in result.likes_for("movies")) == ["horror films", "karaoke"]


def test_domain_lookup_excludes_other_domains(tmp_path):
    repo, _ = _repo(tmp_path)
    likes = _write(tmp_path, "likes.txt", "karaoke\n\n[movies]\nhorror films\n")
    dislikes = _write(tmp_path, "dislikes.txt", "")

    result = repo.load(likes, dislikes)

    assert [p.text for p in result.likes_for("general")] == ["karaoke"]


def test_domain_lookup_for_unknown_domain_returns_general_only(tmp_path):
    repo, _ = _repo(tmp_path)
    likes = _write(tmp_path, "likes.txt", "karaoke\n\n[movies]\nhorror films\n")
    dislikes = _write(tmp_path, "dislikes.txt", "bars\n")

    result = repo.load(likes, dislikes)

    assert [p.text for p in result.likes_for("restaurants")] == ["karaoke"]
    assert [p.text for p in result.dislikes_for("restaurants")] == ["bars"]


def test_emoji_stripped_from_preference_line():
    from src.scoring.preferences import parse_preferences

    prefs = parse_preferences("🍻 bars\nnightclubs 💃\n", preference_type="dislike")

    assert [p.text for p in prefs] == ["bars", "nightclubs"]


def test_emoji_only_preference_line_falls_back_to_its_name():
    from src.scoring.preferences import parse_preferences

    prefs = parse_preferences("🍺\n", preference_type="dislike")

    assert [p.text for p in prefs] == ["beer mug"]


def test_preference_line_of_only_invisible_characters_skipped():
    from src.scoring.preferences import parse_preferences

    assert parse_preferences("​﻿\n", preference_type="dislike") == []
