"""Source rechecks are independent of canonical analysis completeness."""

from core.config import reset_config_cache

import os
from types import SimpleNamespace
from uuid import uuid4

import psycopg
import pytest

from worker.planning.freshness import sources_needing_refresh


@pytest.fixture
def source(monkeypatch):
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL required")
    monkeypatch.setenv("ECHORA_SOURCE_RECHECK_SECONDS", "604800")
    reset_config_cache()
    db = psycopg.connect(url)
    library, namespace, track = [uuid4() for _ in range(3)]
    try:
        db.execute(
            "INSERT INTO libraries(id,namespace,name,root_path) VALUES (%s,%s,'freshness test','https://test')",
            (library, namespace),
        )
        db.execute(
            "INSERT INTO tracks(id,audio_hash,title,duration_seconds) VALUES (%s,%s,'test',1)",
            (track, str(track).replace("-", "") * 2),
        )
        db.execute(
            "INSERT INTO track_sources(library_id,source_type,external_id,track_id,source_data,audio_verified_at) VALUES (%s,'subsonic','source',%s,'{\"size\":100,\"title\":\"Song\"}',now())",
            (library, track),
        )
        yield db, library
    finally:
        db.rollback()
        db.close()


def catalog(**raw):
    return [SimpleNamespace(id="source", raw=raw)]


def test_changed_metadata_rechecks_recent_complete_sources(source):
    db, library = source
    assert sources_needing_refresh(db, library, ["source"], catalog(size=100)) == set()
    assert sources_needing_refresh(db, library, ["source"], catalog(size=101)) == {"source"}
    assert sources_needing_refresh(db, library, ["source"], catalog(title="Edited")) == {"source"}
    assert (
        sources_needing_refresh(db, library, ["source"], catalog(playCount=99, starred="today"))
        == set()
    )


@pytest.mark.parametrize("age", [None, "8 days"])
def test_unknown_or_expired_verification_rechecks_without_metadata_changes(source, age):
    db, library = source
    db.execute(
        "UPDATE track_sources SET audio_verified_at=now()-%s::interval WHERE library_id=%s",
        (age, library),
    )
    assert sources_needing_refresh(db, library, ["source"], catalog(size=100)) == {"source"}


def test_strict_mode_rechecks_every_full_sync(source, monkeypatch):
    db, library = source
    monkeypatch.setenv("ECHORA_SOURCE_RECHECK_SECONDS", "0")
    reset_config_cache()
    assert sources_needing_refresh(db, library, ["source"], catalog(size=100)) == {"source"}


def test_freshness_is_scoped_to_library_and_requested_sources(source):
    db, library = source
    db.execute("UPDATE track_sources SET audio_verified_at=NULL WHERE library_id=%s", (library,))
    assert sources_needing_refresh(db, uuid4(), ["source"]) == set()
    assert sources_needing_refresh(db, library, ["other"]) == set()
