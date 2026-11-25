"""Melody contour and hum corpus writes against the migrated test database (rolled back)."""

import os
import uuid

import psycopg
import pytest

from core.melody import store


@pytest.fixture
def db():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL must point to a disposable migrated database")
    with psycopg.connect(url) as connection:
        yield connection
        connection.rollback()


def test_contours_are_replaced_and_listed_for_their_owner(db):
    user, track, library = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db.execute(
        "INSERT INTO users (id, username, display_name) VALUES (%s, %s, 'U')",
        (user, uuid.uuid4().hex),
    )
    db.execute(
        "INSERT INTO tracks (id, audio_hash, title, duration_seconds) VALUES (%s, %s, 'S', 1)",
        (track, uuid.uuid4().hex),
    )
    db.execute(
        "INSERT INTO libraries (id, name, root_path, namespace) VALUES (%s, 'N', %s, %s)",
        (library, "https://music.test/" + uuid.uuid4().hex, uuid.uuid4()),
    )
    db.execute(
        "INSERT INTO user_track_links (user_id, library_id, track_id, external_id)"
        " VALUES (%s, %s, %s, 's1')",
        (user, library, track),
    )
    revision = "test-" + uuid.uuid4().hex
    run = store.contour_run(db, revision, "h", {}, {})
    assert not store.has_contours(db, track, run)
    store.save_contour(db, track, run, "full-mix", 0.01, [1.0], [True])
    store.save_contour(db, track, run, "full-mix", 0.01, [2.0], [False])
    assert store.has_contours(db, track, run)
    assert store.contour_digest(db, revision, user)["count"] == 1
    [contour] = store.user_contours(db, revision, user)
    assert (contour["pitch"], contour["voiced"]) == ([2.0], [False])
    assert store.user_contours(db, revision, uuid.uuid4()) == []
    assert [row["source_id"] for row in store.matched_songs(db, user, [track])] == ["s1"]
    corpus = uuid.uuid4()
    store.add_corpus(db, corpus, user, 5)
    store.add_corpus(db, corpus, uuid.uuid4(), 9)  # The first creation stands.
    assert store.corpus_owner(db, corpus) == user
    assert store.owns_corpus(db, corpus, user) and not store.owns_corpus(db, corpus, uuid.uuid4())
    store.set_corpus_run(db, corpus, user, run)
    store.add_corpus_track(db, corpus, track)
    store.add_corpus_track(db, corpus, track)
    assert db.execute(
        "SELECT c.run_id, count(t.track_id) FROM hum_corpora c"
        " JOIN hum_corpus_tracks t ON t.corpus_id=c.id WHERE c.id=%s GROUP BY c.run_id",
        (corpus,),
    ).fetchone() == (run, 1)
