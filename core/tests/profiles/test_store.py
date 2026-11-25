"""Audio profile writes against the migrated test database (rolled back after each test)."""

import os
import uuid

import psycopg
import pytest

from core.embeddings import store as embeddings
from core.profiles import store


@pytest.fixture
def db():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL must point to a disposable migrated database")
    with psycopg.connect(url) as connection:
        yield connection
        connection.rollback()


def test_a_profile_replaces_its_segments_and_modes(db):
    track = uuid.uuid4()
    db.execute(
        "INSERT INTO tracks (id, audio_hash, title, duration_seconds) VALUES (%s, %s, 'S', 20)",
        (track, uuid.uuid4().hex),
    )
    source = embeddings.audio_embedding_run(db, "mert", uuid.uuid4().hex, "h", {}, {}, "cpu")
    embeddings.save_audio_embeddings(
        db,
        track,
        source,
        {"dimension": 2, "embedding": "[1,0]", "inference_ms": 1, "peak_vram_bytes": None},
        [(0, 0.0, 10.0, 2, "[1,0]"), (1, 5.0, 15.0, 2, "[0,1]")],
    )
    assert [row["window_index"] for row in store.window_embeddings(db, track, source)] == [0, 1]
    run = store.profile_run(db, "mert", uuid.uuid4().hex, "h", {}, {})
    profile = (track, run, source, "mert", 2, "[1,0]", "[1,0]", 1, 1, 1, 0, 0, True, 1)
    segment = (0, 0.0, 15.0, 2, "[1,0]", 1.0, 0)
    mode = ((0, 2, "[1,0]", 1.0, 1.0, 0), [(0.0, 15.0)])
    store.save_profile(db, profile, [segment, (1, *segment[1:])], [mode])
    store.save_profile(db, profile, [segment], [mode])
    counts = db.execute(
        """SELECT (SELECT count(*) FROM track_audio_profiles WHERE track_id=%s),
                  (SELECT count(*) FROM audio_temporal_segments s
                     JOIN track_audio_profiles p ON p.id=s.profile_id WHERE p.track_id=%s),
                  (SELECT count(*) FROM audio_mode_intervals i JOIN audio_modes m ON m.id=i.mode_id
                     JOIN track_audio_profiles p ON p.id=m.profile_id WHERE p.track_id=%s)""",
        (track, track, track),
    ).fetchone()
    assert counts == (1, 1, 1)
