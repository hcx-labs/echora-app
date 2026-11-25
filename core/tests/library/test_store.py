"""Library ingest writes against the migrated test database (rolled back after each test)."""

import os
import uuid

import psycopg
import pytest

from core.library import store


@pytest.fixture
def db():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL must point to a disposable migrated database")
    with psycopg.connect(url) as connection:
        yield connection
        connection.rollback()


SONG = {"title": "Song", "artist": "A", "album": "B", "year": 2001, "duration_seconds": 200.0}


def add_song(db, library, external_id, digest):
    track = uuid.uuid4()
    assert store.add_track(db, track, digest, genres=[], metadata={"x": 1}, **SONG)
    assert store.track_with_hash(db, digest) == track
    store.bind_source(db, library, track, "a.flac", external_id, {"coverArt": "c"})
    return track


def test_sources_bind_refresh_and_verify(db):
    root = "https://music.test/" + uuid.uuid4().hex
    library = store.library_id(db, root, uuid.uuid4())
    track = add_song(db, library, "s1", uuid.uuid4().hex)
    assert not store.add_track(
        db,
        uuid.uuid4(),
        store.known_source(db, library, "s1")["audio_hash"],
        genres=[],
        metadata={},
        **SONG,
    )
    assert store.source_track_id(db, library, "s1") == track
    store.refresh_source(
        db, library, "s1", {"coverArt": "d"}, track, genres=["Pop"], **{**SONG, "title": "New"}
    )
    store.mark_source_verified(db, library, "s1", track)
    row = db.execute(
        """SELECT t.title, t.genres, ts.source_data, ts.audio_verified_at IS NOT NULL
           FROM track_sources ts JOIN tracks t ON t.id=ts.track_id WHERE ts.library_id=%s""",
        (library,),
    ).fetchone()
    assert row == ("New", ["Pop"], {"coverArt": "d"}, True)


def test_fingerprints_group_matching_songs(db):
    library = store.library_id(db, "https://music.test/" + uuid.uuid4().hex, uuid.uuid4())
    first = add_song(db, library, "s1", uuid.uuid4().hex)
    second = add_song(db, library, "s2", uuid.uuid4().hex)
    assert store.unfingerprinted_duration(db, first) == {"duration_seconds": 200.0}
    store.save_fingerprint(db, first, "1", b"fp", 200.0)
    assert store.has_fingerprint(db, first) and store.unfingerprinted_duration(db, first) is None
    candidates = store.fingerprint_candidates(db, second, 199.0, 201.0)
    assert {"track_id": first, "fingerprint": b"fp", "duration_seconds": 200.0} in candidates
    assert store.embedding_similarity(db, first, second, "mert") is None
    left, right = sorted((first, second))
    evidence = store.save_match_evidence(db, left, right, 0.9, 0.5, None, None, 99, {"t": 1})
    assert store.save_match_evidence(db, left, right, 0.95, 0.5, None, None, 99, {}) == evidence
    assert store.group_of(db, first) is None
    group = store.add_recording_group(db, first)
    store.add_group_member(db, group, first, 1)
    store.add_group_member(db, group, second, 0.9)
    store.add_group_member(db, group, second, 0.1)  # already a member: kept
    assert store.group_of(db, second) == group


def test_per_song_results_are_replaced(db):
    library = store.library_id(db, "https://music.test/" + uuid.uuid4().hex, uuid.uuid4())
    track = add_song(db, library, "s1", uuid.uuid4().hex)
    store.save_waveform(db, track, "1", 200.0, [0.1])
    store.save_waveform(db, track, "2", 200.0, [0.2])
    store.save_audio_descriptors(db, track, "r", "partial", {"a": 1})
    store.save_audio_descriptors(db, track, "r", "complete", {"a": 2})
    store.save_unsupported_visual_features(db, track, "v", 900.0, 0.05)
    store.save_visual_features(db, track, "v", 200.0, 0.05, {"bands": []})
    assert db.execute(
        "SELECT revision, peaks FROM track_waveforms WHERE track_id=%s", (track,)
    ).fetchone() == ("2", [0.2])
    assert db.execute(
        "SELECT status, descriptors FROM track_audio_descriptors WHERE track_id=%s", (track,)
    ).fetchall() == [("complete", {"a": 2})]
    assert db.execute(
        "SELECT status, features FROM track_visual_features WHERE track_id=%s", (track,)
    ).fetchone() == ("complete", {"bands": []})
