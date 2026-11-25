"""The concept corpus digest against the migrated test database (rolled back afterwards)."""

import os
import uuid

import psycopg
import pytest
from psycopg.types.json import Jsonb

from core.embeddings import store


@pytest.fixture
def db():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL must point to a disposable migrated database")
    with psycopg.connect(url) as connection:
        yield connection
        connection.rollback()


def _embed(db, track, run, vector):
    db.execute(
        "DELETE FROM embeddings WHERE track_id=%s AND run_id=%s AND embedding_type='lyrics'",
        (track, run),
    )
    db.execute(
        """INSERT INTO embeddings (track_id, run_id, embedding_type, dimension, embedding)
           VALUES (%s, %s, 'lyrics', 2, %s)""",
        (track, run, vector),
    )


def test_the_digest_changes_when_a_song_is_re_embedded_or_retagged(db):
    user, library, track = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db.execute(
        "INSERT INTO users (id, username, display_name) VALUES (%s, %s, 'U')", (user, user.hex)
    )
    db.execute(
        "INSERT INTO libraries (id, name, root_path, namespace) VALUES (%s, 'L', %s, %s)",
        (library, library.hex, uuid.uuid4()),
    )
    db.execute(
        "INSERT INTO tracks (id, audio_hash, title, duration_seconds) VALUES (%s, %s, 'Old', 1)",
        (track, uuid.uuid4().hex),
    )
    db.execute(
        "INSERT INTO user_track_links (user_id, library_id, track_id, external_id) VALUES (%s,%s,%s,'x')",
        (user, library, track),
    )
    db.execute(
        """INSERT INTO active_representation_specs
             (kind, model_name, model_revision, config_hash, dimension, config)
           VALUES ('lyrics_embedding', 'bge_m3', 'r', 'h', 2, %s)
           ON CONFLICT (kind, model_name) DO UPDATE SET model_revision='r', config_hash='h',
             dimension=2""",
        (Jsonb({}),),
    )
    run = store_run(db)
    assert store.corpus_digest(db, "lyrics", user) == ""

    _embed(db, track, run, "[1,0]")
    first = store.corpus_digest(db, "lyrics", user)
    assert first
    assert [row["id"] for row in store.user_corpus(db, "lyrics", user)] == [track]
    assert store.corpus_digest(db, "lyrics", user) == first

    # The lyrics worker reuses its run and replaces the row: same (song, run), new vector.
    _embed(db, track, run, "[0,1]")
    second = store.corpus_digest(db, "lyrics", user)
    assert second != first

    db.execute("UPDATE tracks SET title='New' WHERE id=%s", (track,))
    assert store.corpus_digest(db, "lyrics", user) != second


def store_run(db):
    return db.execute(
        """INSERT INTO analysis_runs
             (kind, model_name, model_revision, config_hash, config, environment, device,
              precision, status, started_at, finished_at)
           VALUES ('lyrics_embedding', 'bge_m3', 'r', 'h', %s, %s, 'cpu', 'float32', 'complete',
                   now(), now())
           RETURNING id""",
        (Jsonb({}), Jsonb({})),
    ).fetchone()[0]
