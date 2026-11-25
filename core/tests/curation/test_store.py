"""The curation refresh and publication queries against PostgreSQL (temporary tables)."""

import os
import uuid

import psycopg
import pytest
from psycopg.rows import dict_row

from core.curation import store


@pytest.fixture
def db():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL required")
    with psycopg.connect(url) as connection:
        connection.execute("""
            CREATE TEMP TABLE tracks (id uuid PRIMARY KEY, title text, artist text);
            CREATE TEMP TABLE curations (
                id uuid PRIMARY KEY, user_id uuid, status text, last_error text,
                refresh_enabled boolean, refresh_interval_hours int, next_refresh_at timestamptz,
                last_refreshed_at timestamptz, navidrome_playlist_id text,
                updated_at timestamptz);
            CREATE TEMP TABLE curation_publications (
                id uuid PRIMARY KEY, curation_id uuid, job_id uuid, state text, plan jsonb,
                playlist_id text, updated_at timestamptz);
            CREATE TEMP TABLE curation_revisions (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(), curation_id uuid,
                revision_number int, recipe jsonb);
            CREATE TEMP TABLE curation_revision_tracks (
                revision_id uuid, position int, track_id uuid, score real, evidence jsonb,
                source_id text)""")
        yield connection
        connection.rollback()


def test_a_refresh_publishes_a_revision_and_schedules_the_next(db):
    user, curation, job, song = uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db.execute("INSERT INTO tracks VALUES (%s, 'Song', 'Artist')", (song,))
    db.execute(
        "INSERT INTO curations (id, user_id, status, refresh_enabled, refresh_interval_hours)"
        " VALUES (%s, %s, 'draft', true, 24)",
        (curation, user),
    )
    assert store.due_curation_ids(db, 10) == [curation]
    assert store.curation_row(db, curation, uuid.uuid4()) is None
    assert store.try_lock_curation(db, curation)
    store.unlock_curation(db, curation)

    store.mark_queued(db, curation)
    assert store.curation_row(db, curation, user)["status"] == "pending"
    assert store.due_curation(db, curation) is None

    publication = store.add_publication(db, curation, job, {"name": "Mix"}, None)
    assert publication["state"] == "prepared" and publication["plan"] == {"name": "Mix"}
    assert store.unresolved_publication(db, curation)["id"] == publication["id"]
    store.mark_publishing(db, publication["id"], curation)
    store.record_playlist(db, publication["id"], curation, "p1")
    assert store.next_revision_number(db, curation) == 1
    store.add_revision(
        db,
        curation,
        1,
        {"model": "m"},
        [{"id": song, "score": 0.5, "source_id": "s1", "evidence": {"percentile": 90}}],
    )
    store.complete_publication(db, publication["id"], curation)

    row = store.curation_row(db, curation, user)
    assert (row["status"], row["navidrome_playlist_id"], row["last_error"]) == ("ready", "p1", None)
    assert row["next_refresh_at"] > row["last_refreshed_at"]
    assert store.unresolved_publication(db, curation) is None
    assert store.job_publication(db, curation, job)["playlist_id"] == "p1"
    assert store.latest_revision_track_ids(db, curation, 10) == [song]
    assert store.next_revision_number(db, curation) == 2
    assert store.track_names(db, [song]) == [{"id": str(song), "title": "Song", "artist": "Artist"}]
    with db.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT evidence, source_id FROM curation_revision_tracks")
        assert cursor.fetchone() == {"evidence": {"percentile": 90}, "source_id": "s1"}

    store.mark_failed(db, curation, "boom")
    assert store.curation_row(db, curation, user)["last_error"] == "boom"
