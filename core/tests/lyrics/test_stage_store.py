"""The lyrics stage's writes against the migrated test database (rolled back after each test)."""

import os
import uuid

import psycopg
import pytest

from core.analysis import store as runs
from core.lyrics import store


@pytest.fixture
def db():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL must point to a disposable migrated database")
    with psycopg.connect(url) as connection:
        yield connection
        connection.rollback()


def song(db):
    track = uuid.uuid4()
    db.execute(
        "INSERT INTO tracks (id, audio_hash, title, artist, duration_seconds)"
        " VALUES (%s, %s, 'Song', 'Artist', 200)",
        (track, uuid.uuid4().hex),
    )
    return track


def test_a_transcript_never_replaces_lyrics_and_refines_only_its_draft(db):
    track = song(db)
    assert store.track_artist(db, track) == "Artist"
    first = store.save_lyrics(
        db, track, "transcribed", "draft", "en", {"ai_generated": True}, "available"
    )
    assert first is not None
    assert store.refine_transcript(db, track, "other", "x", "en", {}) == 0
    assert store.refine_transcript(db, track, "draft", "better", "en", {"refined": True}) == 1
    assert store.stored_lyrics(db, track) == {
        "text": "better",
        "provenance": {"refined": True},
        "availability_status": "available",
        "source": "transcribed",
    }
    # Embedded lyrics replace a transcript; a later transcript does not replace them.
    assert store.save_lyrics(db, track, "embedded", "real", "en", {}, "available") is not None
    assert store.save_lyrics(db, track, "transcribed", "again", "en", {}, "available") is None
    assert store.kept_lyrics(db, track)["text"] == "real"
    assert store.lyrics_source(db, track) == {"text": "real", "provenance": {}}


def test_embeddings_are_replaced_per_run(db):
    track = song(db)
    run = store.lyrics_embedding_run(db, "test-model", uuid.uuid4().hex, "h", {}, {}, "cpu")
    assert store.lyrics_embedding_run(db, "test-model", "x", "h", {}, {}, "cpu") != run
    whole = {"dimension": 2, "embedding": "[1,0]", "inference_ms": 1, "peak_vram_bytes": None}
    for _ in range(2):
        store.replace_lyrics_embeddings(db, track, run, whole, [(0, 2, "tokens:0-4", "[0,1]")])
    count = db.execute(
        "SELECT count(*) FROM embeddings WHERE track_id=%s AND run_id=%s", (track, run)
    ).fetchone()[0]
    assert count == 2
    runs.complete_run(db, run)
    status = db.execute("SELECT status FROM analysis_runs WHERE id=%s", (run,)).fetchone()[0]
    assert status == "complete"


def test_themes_are_stored_and_replaced(db):
    track = song(db)
    store.save_lyrics(db, track, "embedded", "words", "en", {}, "available")
    common = {"description": None, "themes": [], "embedding": None, "embedding_model": None}
    store.save_themes(db, track, "c", 1, "m", status="failed", error_code="x", **common)
    store.save_themes(
        db,
        track,
        "c",
        1,
        "m",
        status="ready",
        error_code=None,
        **{
            **common,
            "themes": ["love"],
            "description": "Love.",
            "embedding": "[" + ",".join(["0.5"] * 1024) + "]",
            "embedding_model": "bge_m3",
        },
    )
    row = db.execute(
        "SELECT status, error_code, themes FROM track_lyric_themes WHERE track_id=%s", (track,)
    ).fetchone()
    assert row == ("ready", None, ["love"])


def test_karaoke_writes_replace_the_unbounded_variant(db):
    track = song(db)
    library = uuid.uuid4()
    db.execute(
        "INSERT INTO libraries (id, name, root_path, namespace) VALUES (%s, 'N', %s, %s)",
        (library, "https://music.test/" + uuid.uuid4().hex, uuid.uuid4()),
    )
    db.execute(
        "INSERT INTO track_sources (library_id, track_id, source_type, external_id, source_data)"
        " VALUES (%s, %s, 'subsonic', 's1', '{}')",
        (library, track),
    )
    store.save_lyrics(
        db,
        track,
        "embedded",
        "Hello",
        "en",
        {"word_timing": [1], "word_timing_digest": "d"},
        "available",
    )
    [row] = store.karaoke_sources(db, ["s1"], library)
    assert (row["text"], row["word_timing"], row["word_timing_digest"]) == ("Hello", [1], "d")
    store.save_supplied_karaoke(db, track, [{"text": "Hello"}], {"alignment_mode": "supplied"})
    store.save_aligned_karaoke(
        db, track, [{"text": "Hi"}], "rev", {"alignment_mode": "ctc"}, {}, {}
    )
    variant = db.execute(
        "SELECT model, lines FROM karaoke_lyrics_variants WHERE track_id=%s", (track,)
    ).fetchall()
    assert variant == [("fa_kara", [{"text": "Hi"}])]
    lyrics = db.execute(
        "SELECT karaoke_model, provenance->'karaoke' FROM lyrics WHERE track_id=%s", (track,)
    ).fetchone()
    assert lyrics == ("fa_kara", {"alignment_mode": "ctc"})
