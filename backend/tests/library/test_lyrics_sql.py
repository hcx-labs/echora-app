"""Run the lyrics endpoints' upserts against PostgreSQL, including type inference."""

import os
from uuid import uuid4

import psycopg
import pytest

from backend.api.v1.library import track_lyrics_status, track_transcription_language


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
            db.execute(track_lyrics_status.UPSERT_SQL, (track, status, status))
            row = db.execute(
                "SELECT availability_status,provenance FROM lyrics WHERE track_id=%s", (track,)
            ).fetchone()
            assert row[0] == status
            assert row[1]["manual_status"] == status


def test_forced_transcription_insert_and_update(db):
    track = uuid4()
    for language in ["en", "ja"]:
        db.execute(track_transcription_language.UPSERT_SQL, (track, language, language))
        row = db.execute(
            "SELECT availability_status,provenance FROM lyrics WHERE track_id=%s", (track,)
        ).fetchone()
        assert row[0] == "missing"
        assert row[1]["transcription_language"] == language
        assert row[1]["forced_transcription"] is True
