"""Voice classification writes against the migrated test database (rolled back afterwards)."""

import os
import uuid

import psycopg
import pytest

from core.library import store as library
from core.voice import store


@pytest.fixture
def db():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL must point to a disposable migrated database")
    with psycopg.connect(url) as connection:
        yield connection
        connection.rollback()


def test_a_classified_song_leaves_the_plan(db):
    root = "https://music.test/" + uuid.uuid4().hex
    lib = library.library_id(db, root, uuid.uuid4())
    track = uuid.uuid4()
    song = {"title": "S", "artist": "A", "album": "B", "year": None, "duration_seconds": 1}
    library.add_track(db, track, uuid.uuid4().hex, genres=[], metadata={}, **song)
    library.bind_source(db, lib, track, None, "s1", {})
    run = store.voice_run(db, "voice-test-" + uuid.uuid4().hex, "h", {}, {})
    assert [
        row["external_id"]
        for row in store.songs_without_voice(db, lib, None, "voice-gender", run, 5)
    ] == ["s1"]
    assert store.songs_without_voice(db, lib, ["other"], "voice-gender", run, 5) == []
    for _ in range(2):
        store.save_voice(db, track, run, "voice-gender", "[0.5,0.5]", 2, {"windows": []})
    assert store.songs_without_voice(db, lib, ["s1"], "voice-gender", run, 5) == []
    count = db.execute(
        "SELECT count(*) FROM embeddings WHERE track_id=%s AND run_id=%s", (track, run)
    ).fetchone()[0]
    assert count == 1
