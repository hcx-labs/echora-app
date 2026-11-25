"""Source replacement must refresh identity before planning new analysis."""

import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock
from uuid import uuid4

from psycopg.rows import tuple_row

import worker.library.ingest as ingest
from core.config import reset_config_cache
from worker.planning.audio import AudioProcessingPlan


def song(identifier="source"):
    return SimpleNamespace(
        id=identifier,
        title="Song",
        artist="Artist",
        album="Album",
        year=2020,
        duration=180,
        genre=None,
        path="song.flac",
        raw={},
    )


def test_upsert_refreshes_only_matching_source_links_and_reuses_identity(database):
    library, elsewhere, owner, other, old, existing = [uuid4() for _ in range(6)]
    digest = hashlib.sha256(b"changed bytes").hexdigest()
    with database() as db:
        db.execute("""CREATE TABLE tracks(id uuid PRIMARY KEY,audio_hash text UNIQUE,title text,
            artist text,album text,year integer,duration_seconds float,genres text[],metadata jsonb);
            CREATE TABLE library_tracks(library_id uuid,track_id uuid,relative_path text,
                PRIMARY KEY(library_id,track_id));
            CREATE TABLE track_sources(library_id uuid,track_id uuid,source_type text,
                external_id text,source_data jsonb,UNIQUE(library_id,source_type,external_id));
            CREATE TABLE user_track_links(user_id uuid,library_id uuid,track_id uuid,external_id text,
                PRIMARY KEY(user_id,library_id,track_id),
                UNIQUE(user_id,library_id,external_id));
            CREATE TABLE user_source_memberships(user_id uuid,library_id uuid,external_id text,
                PRIMARY KEY(user_id,library_id,external_id));""")
        db.execute(
            "INSERT INTO tracks(id,audio_hash) VALUES (%s,'old'),(%s,%s)", (old, existing, digest)
        )
        db.execute(
            "INSERT INTO track_sources VALUES (%s,%s,'subsonic','source','{}'),(%s,%s,'subsonic','alias','{}')",
            (library, old, library, old),
        )
        for row in [
            (owner, library, old, "source"),
            (owner, library, existing, "duplicate"),
            (other, library, old, "alias"),
            (owner, elsewhere, old, "source"),
        ]:
            db.execute("INSERT INTO user_track_links VALUES (%s,%s,%s,%s)", row)
        db.execute(
            "INSERT INTO track_sources VALUES (%s,%s,'subsonic','duplicate','{}')",
            (library, existing),
        )
        db.execute(
            "INSERT INTO user_source_memberships SELECT user_id,library_id,external_id FROM user_track_links"
        )
        # The owner also proved access to the old recording through an alias.
        db.execute("INSERT INTO user_source_memberships VALUES (%s,%s,'alias')", (owner, library))
        db.row_factory = tuple_row
        assert ingest._upsert_track(db, library, song(), digest) == (existing, False)
        assert ingest._source_track_id(db, library, "source") == existing
        assert ingest._source_track_id(db, library, "alias") == old
        links = set(db.execute("SELECT * FROM user_track_links").fetchall())
        assert links == {
            (owner, library, existing, "duplicate"),
            (owner, library, old, "alias"),
            (other, library, old, "alias"),
            (owner, elsewhere, old, "source"),
        }
        assert db.execute("SELECT audio_hash FROM tracks WHERE id=%s", (old,)).fetchone() == (
            "old",
        )
        assert ingest._upsert_track(db, library, song(), digest) == (existing, False)
        assert set(db.execute("SELECT * FROM user_track_links").fetchall()) == links
        new_id, inserted = ingest._upsert_track(db, library, song(), "another-hash")
        assert inserted and new_id != existing
        assert set(
            db.execute(
                "SELECT track_id,external_id FROM user_track_links WHERE user_id=%s AND library_id=%s",
                (owner, library),
            ).fetchall()
        ) == {(new_id, "source"), (existing, "duplicate"), (old, "alias")}
        assert set(
            db.execute(
                "SELECT track_id,external_id FROM user_track_links WHERE user_id=%s", (other,)
            ).fetchall()
        ) == {(old, "alias")}


def test_explicit_source_hashes_before_single_bound_plan_even_when_complete(monkeypatch):
    empty = AudioProcessingPlan(frozenset(), frozenset(), frozenset(), frozenset())
    connection = MagicMock()
    connection.__enter__.return_value = connection
    client = MagicMock()
    client.__enter__.return_value = client
    client.tracks.return_value = [song()]
    client.audio_bytes.return_value = b"new bytes"
    monkeypatch.setattr(ingest, "connect", lambda **_: connection)
    monkeypatch.setenv("DATABASE_URL", "unused")
    reset_config_cache()
    monkeypatch.setattr(ingest, "NavidromeClient", lambda *a: client)
    monkeypatch.setattr(ingest, "configure_representations", Mock())
    monkeypatch.setattr(ingest, "_library", lambda *a: "library")
    monkeypatch.setattr(ingest, "_source_track_id", lambda *a: "old-track")
    upsert = Mock(return_value=("current-track", True))
    monkeypatch.setattr(ingest, "_upsert_track", upsert)
    calls = []

    def plan(*args, **kwargs):
        calls.append(args)
        upsert.assert_called_once_with(
            connection,
            "library",
            client.tracks.return_value[0],
            hashlib.sha256(b"new bytes").hexdigest(),
        )
        assert kwargs["resolved_track_ids"] == {"source": "current-track"}
        # Explicit ingestion refreshes bytes even when all artifacts exist.
        return empty

    monkeypatch.setattr(ingest, "plan_audio", plan)
    monkeypatch.setattr(ingest, "active_cache", lambda: None)
    summary = ingest.ingest_navidrome("https://music", "user", "password", ["source"])
    assert len(calls) == 1
    assert summary.inserted == summary.downloaded == 1
    assert summary.recording_fingerprinted == summary.embedded_mert == 0
    client.audio_bytes.assert_called_once_with("source")
