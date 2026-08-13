"""Unit tests for bench sample loading."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.bench.samples import SampleError, load_samples, samples_from_db
from src.models.event import Event
from src.storage.sqlite.connection import init_db
from src.storage.sqlite.events import SqliteEventRepository


def _write(tmp_path, text: str):
    path = tmp_path / "bench-samples.yaml"
    path.write_text(text)
    return path


def test_it_loads_a_sample(tmp_path):
    path = _write(
        tmp_path,
        """
        - name: bare-performer-name
          title: Ava Valianti
          venue: The Blue Room
          location: Riverport
          listing_category: Music
          note: A name and nothing else.
        """,
    )

    samples = load_samples(path)

    assert len(samples) == 1
    assert samples[0].title == "Ava Valianti"
    assert samples[0].listing_category == "Music"


def test_a_sample_must_say_what_makes_it_tricky(tmp_path):
    """A bench sample earns its place by being hard. Without the note the next
    reader cannot tell a case that once broke the extractor from filler, and
    the set silently rots into a list of ordinary events."""
    path = _write(tmp_path, "- {name: plain, title: Jazz Night}")

    with pytest.raises(SampleError, match="note"):
        load_samples(path)


def test_a_sample_must_be_named(tmp_path):
    path = _write(tmp_path, "- {title: Jazz Night, note: something}")

    with pytest.raises(SampleError, match="name"):
        load_samples(path)


def test_an_unknown_field_is_rejected(tmp_path):
    """A typo that is silently ignored makes the sample quietly stop testing
    what it was written to test."""
    path = _write(
        tmp_path,
        "- {name: x, title: Jazz Night, note: n, catgeory: Music}",
    )

    with pytest.raises(SampleError, match="catgeory"):
        load_samples(path)


def test_a_reference_date_is_parsed(tmp_path):
    """`this Saturday` is only resolvable against a stated today."""
    path = _write(
        tmp_path,
        """
        - name: relative-date
          title: Live music this Saturday at 8pm
          note: Resolving a weekday against the reference date.
          reference_date: 2026-08-03
        """,
    )

    assert load_samples(path)[0].reference_date == datetime(2026, 8, 3, tzinfo=timezone.utc)


def test_the_shipped_example_file_loads():
    """It is the only sample set in the repository, so a typo in it is a broken
    tool for anyone who has not written their own."""
    samples = load_samples("data/bench-samples.example.yaml")

    assert len(samples) >= 5
    assert all(s.note for s in samples)


def test_samples_can_be_drawn_from_the_database(tmp_path):
    """The question worth asking most often is "why did *this* event tag
    badly", and that event is in the database, not in a fixture."""
    db = tmp_path / "bench.db"
    init_db(db)
    now = datetime(2026, 8, 13, tzinfo=timezone.utc)
    SqliteEventRepository(db).save(
        [
            Event(
                event_id="evt-1",
                source_event_candidates=[],
                source_type="northshorenightout",
                created_at=now,
                updated_at=now,
                title="Ava Valianti",
                venue="The Blue Room",
                location="Riverport",
                metadata={"listing_category": "Music"},
            )
        ]
    )

    samples = samples_from_db(SqliteEventRepository(db), ["evt-1"])

    assert samples[0].title == "Ava Valianti"
    assert samples[0].venue == "The Blue Room"
    assert samples[0].listing_category == "Music"
    assert "evt-1" in samples[0].note


def test_an_unknown_event_id_is_an_error(tmp_path):
    """Silently returning fewer samples than asked for would look like the
    model handled a case it never saw."""
    db = tmp_path / "bench.db"
    init_db(db)

    with pytest.raises(SampleError, match="missing-id"):
        samples_from_db(SqliteEventRepository(db), ["missing-id"])
