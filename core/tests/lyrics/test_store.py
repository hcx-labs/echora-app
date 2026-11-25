"""Run the lyrics store's upserts against PostgreSQL, including type inference."""

import os
from uuid import uuid4

import psycopg
import pytest

from core.lyrics import store


@pytest.fixture
def db():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL required")
    with psycopg.connect(url) as connection:
        connection.execute("""CREATE TEMP TABLE lyrics (
            track_id uuid PRIMARY KEY, source text NOT NULL, text text,
            language text, provenance jsonb, availability_status text NOT NULL,
            created_at timestamptz DEFAULT now())""")
        yield connection
        connection.rollback()


def test_instrumental_and_missing_insert_and_update(db):
    for first, second in [("instrumental", "missing"), ("missing", "instrumental")]:
        track = uuid4()
        for status in [first, second]:
            store.set_manual_status(db, track, status)
            row = db.execute(
                "SELECT availability_status,provenance FROM lyrics WHERE track_id=%s", (track,)
            ).fetchone()
            assert row[0] == status
            assert row[1]["manual_status"] == status


def test_forced_transcription_insert_and_update(db):
    track = uuid4()
    for language in ["en", "ja"]:
        store.force_transcription(db, track, language)
        row = db.execute(
            "SELECT availability_status,provenance FROM lyrics WHERE track_id=%s", (track,)
        ).fetchone()
        assert row[0] == "missing"
        assert row[1]["transcription_language"] == language
        assert row[1]["forced_transcription"] is True


def test_manual_lyrics_keep_line_timing_only_while_the_text_is_unchanged(db):
    track = uuid4()
    db.execute(
        "INSERT INTO lyrics VALUES (%s,'navidrome','old','en','{\"lines\": [1]}','available')",
        (track,),
    )
    store.save_manual_lyrics(db, track, "old", "en")
    assert "lines" in store_provenance(db, track)
    store.save_manual_lyrics(db, track, "new", None)
    provenance = store_provenance(db, track)
    assert "lines" not in provenance and provenance["manual"] is True
    assert store.lyrics_text(db, track) == {"text": "new", "language": None}


def store_provenance(db, track):
    return db.execute("SELECT provenance FROM lyrics WHERE track_id=%s", (track,)).fetchone()[0]
