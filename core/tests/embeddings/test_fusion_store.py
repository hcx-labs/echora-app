"""Semantic fusion writes against the migrated test database (rolled back afterwards)."""

import os
import uuid

import psycopg
import pytest

from core.embeddings import store


@pytest.fixture
def db():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL must point to a disposable migrated database")
    with psycopg.connect(url) as connection:
        yield connection
        connection.rollback()


def test_a_fusion_rebuild_replaces_vectors_runs_and_the_active_spec(db):
    track = uuid.uuid4()
    db.execute(
        "INSERT INTO tracks (id, audio_hash, title, duration_seconds) VALUES (%s, %s, 'S', 1)",
        (track, uuid.uuid4().hex),
    )
    store.fusion_pairs(db)  # Readable on the real schema.
    for revision in ("old", "new"):
        run = store.replace_fusion_run(db, "semantic_fusion", "fusion", revision, "h", {}, {})
        store.save_fusion(
            db,
            [(track, run, "semantic_fusion", 2, "zscore", "[1,0]")],
            "fusion",
            revision,
            "h",
            2,
            {},
        )
    assert db.execute(
        "SELECT count(*) FROM analysis_runs WHERE kind='semantic_fusion'"
    ).fetchone() == (1,)
    assert db.execute(
        "SELECT run_id FROM embeddings WHERE embedding_type='semantic_fusion'"
    ).fetchall() == [(run,)]
    assert db.execute(
        "SELECT model_revision FROM active_representation_specs WHERE kind='semantic_fusion'"
    ).fetchone() == ("new",)
