"""Karaoke stems writes against the migrated test database (rolled back afterwards)."""

import os
import uuid

import psycopg
import pytest

from core.library import store as library
from core.stems import store


@pytest.fixture
def db():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL must point to a disposable migrated database")
    with psycopg.connect(url) as connection:
        yield connection
        connection.rollback()


def _song(db, *, lyrics: str | None):
    root = "https://music.test/" + uuid.uuid4().hex
    lib = library.library_id(db, root, uuid.uuid4())
    track = uuid.uuid4()
    song = {"title": "S", "artist": "A", "album": "B", "year": None, "duration_seconds": 1}
    library.add_track(db, track, uuid.uuid4().hex, genres=[], metadata={}, **song)
    library.bind_source(db, lib, track, None, "s1", {})
    if lyrics is not None:
        db.execute(
            "INSERT INTO lyrics (track_id, source, text) VALUES (%s, 'embedded', %s)",
            (track, lyrics),
        )
    return lib, track


def _planned(db, lib, recipe="r1"):
    return [row["external_id"] for row in store.songs_without_stems(db, lib, ["s1"], recipe)]


def test_only_songs_with_lyrics_are_planned(db):
    lib, _ = _song(db, lyrics="  ")
    assert _planned(db, lib) == []
    lib, _ = _song(db, lyrics="la la")
    assert _planned(db, lib) == ["s1"]


def test_stored_stems_leave_the_plan_until_the_recipe_changes(db):
    lib, track = _song(db, lyrics="la la")
    assert store.save_stems(db, track, "r1", path="a.webm", file_bytes=10) is None
    assert _planned(db, lib) == []
    assert _planned(db, lib, "r2") == ["s1"]
    # New stems replace the old file, which the caller deletes.
    assert store.save_stems(db, track, "r2", path="b.webm", file_bytes=12) == "a.webm"
    assert store.stems_counts(db)["complete"] >= 1


def test_a_failure_keeps_older_stems_and_is_not_planned_again(db):
    lib, track = _song(db, lyrics="la la")
    store.save_stems(db, track, "r1", path="a.webm", file_bytes=10)
    assert store.save_stems(db, track, "r2", error="RuntimeError") is None
    row = db.execute(
        "SELECT recipe, status, path FROM karaoke_stems WHERE track_id=%s", (track,)
    ).fetchone()
    assert row == ("r1", "complete", "a.webm")
    lib2, other = _song(db, lyrics="la")
    store.save_stems(db, other, "r2", error="RuntimeError")
    assert _planned(db, lib2, "r2") == []
    assert _planned(db, lib2, "r3") == ["s1"]


def test_stems_are_visible_only_through_the_users_library(db):
    _, track = _song(db, lyrics="la la")
    store.save_stems(db, track, "r1", path="a.webm", file_bytes=10, duration_seconds=1.5)
    stems_id = db.execute("SELECT id FROM karaoke_stems WHERE track_id=%s", (track,)).fetchone()[0]
    stranger = uuid.uuid4()
    assert store.track_stems(db, stranger, track) is None
    assert store.visible_stems_path(db, stems_id, stranger) is None
