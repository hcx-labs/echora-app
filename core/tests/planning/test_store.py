"""Planning queries against the migrated test database (rolled back afterwards)."""

import os
import uuid

import psycopg
import pytest

from core.library import store as library
from core.planning import store


@pytest.fixture
def db():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL must point to a disposable migrated database")
    with psycopg.connect(url) as connection:
        yield connection
        connection.rollback()


def test_a_new_song_needs_every_analysis(db):
    root = "https://music.test/" + uuid.uuid4().hex
    namespace = uuid.uuid4()
    lib = library.library_id(db, root, namespace)
    track = uuid.uuid4()
    song = {"title": "S", "artist": "A", "album": "B", "year": None, "duration_seconds": 1}
    library.add_track(db, track, uuid.uuid4().hex, genres=[], metadata={}, **song)
    library.bind_source(db, lib, track, None, "s1", {})
    assert store.libraries_at(db, root.upper() + "/") == [lib]
    assert store.library_with_namespace(db, namespace) == lib
    assert store.known_sources(db, lib, ["s1", "s2"]) == {"s1"}
    assert store.songs_needing_lyrics(db, ["s1"], lib, "r") == ["s1"]
    assert store.songs_needing_lyrics(db, [], lib, "r") == []
    assert store.songs_needing_karaoke(db, ["s1"], lib, "p", None) == []  # No lyrics text.
    revisions = dict.fromkeys(
        ("muq", "mert", "melody", "descriptors", "waveform", "visual_features"), "r"
    )
    for resolved in (None, {"s1": track}):
        [row] = store.audio_artifacts(db, lib, ["s1"], resolved, revisions)
        assert row["track_id"] == track and not row["has_muq"] and not row["has_waveform"]
        assert store.songs_needing_recording_fingerprints(db, lib, ["s1"], resolved, "x") == ["s1"]
    assert store.songs_needing_backfill(db, lib, ["s1"]) == {"s1"}
    assert store.songs_needing_transcription(db, lib, ["s1"]) == {"s1"}
    [fresh] = store.source_freshness(db, 60, lib, ["s1"])
    assert fresh["stale"] is True
