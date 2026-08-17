"""Contract every PreferenceRevisionRepository implementation must satisfy.

Run against both implementations, on the same terms as the other repositories:
the in-memory one is the single official fake, and a hand-written double drifts
from the contract silently.

The behaviour that matters most is that an unchanged preference file reuses its
revision. Recording a fresh row every night would grow a table forever to say
"still the same", and would make "which revision produced that ranking?"
answerable only by comparing hashes anyway.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.models.preference_revision import PreferenceLine, PreferenceRevision
from src.storage.memory.preference_revisions import InMemoryPreferenceRevisionRepository
from src.storage.sqlite.connection import init_db
from src.storage.sqlite.preference_revisions import SqlitePreferenceRevisionRepository

_CAPTURED = datetime(2026, 8, 16, 2, 0, tzinfo=timezone.utc)


@pytest.fixture(params=["sqlite", "memory"])
def repo(request, tmp_path):
    """One repository per implementation, so every test below runs twice."""
    if request.param == "sqlite":
        path = tmp_path / "revisions.db"
        init_db(path)
        return SqlitePreferenceRevisionRepository(path)
    return InMemoryPreferenceRevisionRepository()


def _revision(
    content_hash: str = "hash-a",
    captured_at: datetime = _CAPTURED,
    lines: list[PreferenceLine] | None = None,
) -> PreferenceRevision:
    return PreferenceRevision(
        captured_at=captured_at,
        content_hash=content_hash,
        lines=lines
        if lines is not None
        else [
            PreferenceLine(
                file_name="likes.txt",
                position=0,
                domain="movies",
                preference_type="like",
                line_text="subtitled films",
                line_hash="line-hash-a",
            )
        ],
    )


class TestRecord:
    def test_recording_returns_an_id(self, repo):
        assert repo.record(_revision())

    def test_the_same_content_reuses_its_revision(self, repo):
        """An unedited preference file must not mint a row every night."""
        first = repo.record(_revision())
        second = repo.record(_revision(captured_at=_CAPTURED + timedelta(days=1)))

        assert first == second

    def test_different_content_is_a_different_revision(self, repo):
        assert repo.record(_revision("hash-a")) != repo.record(_revision("hash-b"))

    def test_reusing_a_revision_keeps_the_original_capture_time(self, repo):
        """The revision is when the preferences *became* this, not when last seen.

        `last_seen` is a different question and nothing asks it; recording the
        later stamp would silently answer the first question with the second.
        """
        revision_id = repo.record(_revision())
        repo.record(_revision(captured_at=_CAPTURED + timedelta(days=1)))

        assert repo.get(revision_id).captured_at == _CAPTURED


class TestGet:
    def test_an_unknown_revision_is_none(self, repo):
        assert repo.get("no-such-revision") is None

    def test_every_field_of_every_line_round_trips(self, repo):
        """Asserted at non-default values, per the round-trip footgun.

        A field left at its default passes against a column that does not exist.
        """
        line = PreferenceLine(
            file_name="dislikes.txt",
            position=7,
            domain="restaurants",
            preference_type="dislike",
            line_text="all-you-can-eat buffets",
            line_hash="line-hash-z",
        )
        revision_id = repo.record(_revision(lines=[line]))

        assert repo.get(revision_id).lines == [line]

    def test_lines_come_back_in_file_order(self, repo):
        """Position is recorded, so it must survive the round trip."""
        lines = [
            PreferenceLine("likes.txt", position, "general", "like", f"line {position}", f"h{position}")
            for position in (0, 1, 2)
        ]
        revision_id = repo.record(_revision(lines=list(reversed(lines))))

        assert [line.position for line in repo.get(revision_id).lines] == [0, 1, 2]

    def test_a_revision_with_no_lines_round_trips(self, repo):
        """An empty preference file is a real state, not a missing revision."""
        revision_id = repo.record(_revision(lines=[]))

        assert repo.get(revision_id).lines == []


class TestLatest:
    """What the read path compares against.

    Only a batch or a rescore writes a revision, so the newest row is what
    produced the newest ranking — which is the question the CLI asks without
    needing to join a ranking's run date to a `run_history` row.
    """

    def test_there_is_no_latest_before_anything_is_recorded(self, repo):
        assert repo.latest() is None

    def test_the_latest_is_the_most_recently_captured(self, repo):
        repo.record(_revision("hash-old", captured_at=_CAPTURED))
        repo.record(_revision("hash-new", captured_at=_CAPTURED + timedelta(days=1)))

        assert repo.latest().content_hash == "hash-new"

    def test_re_recording_an_older_revision_does_not_make_it_latest(self, repo):
        """Reverting a file is a new capture only if the content is new.

        Re-recording an existing hash reuses the original row, so the ordering
        must come from `captured_at` on that row rather than from write order.
        """
        repo.record(_revision("hash-old", captured_at=_CAPTURED))
        repo.record(_revision("hash-new", captured_at=_CAPTURED + timedelta(days=1)))
        repo.record(_revision("hash-old", captured_at=_CAPTURED + timedelta(days=2)))

        assert repo.latest().content_hash == "hash-new"
