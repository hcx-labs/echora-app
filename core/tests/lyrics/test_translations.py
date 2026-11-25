"""Opt-in isolated-schema PostgreSQL persistence/migration tests, as in test_jobs."""

import os
import runpy
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

import core.lyrics.translations as storage
from core.integrations.external_ai import ExternalAISettings, LanguagePair


@pytest.fixture
def database(monkeypatch):
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL required for isolated PostgreSQL tests")
    schema = "test_translation_" + uuid4().hex
    with psycopg.connect(url, autocommit=True) as db:
        db.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    scoped = make_conninfo(url, options=f"-c search_path={schema}")
    monkeypatch.setattr(storage, "require_database_url", lambda: scoped)
    migration = runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "src/db/migrations/versions/0051_external_ai_translations.py"
        )
    )
    try:
        with psycopg.connect(scoped) as db:
            db.execute(
                "CREATE TABLE tracks(id uuid PRIMARY KEY); CREATE TABLE lyrics(track_id uuid PRIMARY KEY, text text)"
            )
            with patch("alembic.op.execute", side_effect=db.execute):
                migration["upgrade"]()
        yield scoped, migration
    finally:
        with psycopg.connect(url, autocommit=True) as db:
            db.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_settings_storage(database):
    assert storage.load_settings() == (ExternalAISettings(), None)
    settings = ExternalAISettings(url="http://localhost:8000/v1")
    assert storage.save_settings(settings, b"ciphertext", replace_key=True)
    assert storage.load_settings() == (settings, b"ciphertext")
    assert storage.save_settings(settings, None, replace_key=False)
    assert storage.load_settings()[1] == b"ciphertext"
    assert not storage.save_settings(settings, None, replace_key=True)


def test_translation_staleness_provenance_and_rollback(database):
    url, migration = database
    track = uuid4()
    pair = LanguagePair(source="ja", target="en")
    with psycopg.connect(url) as db:
        db.execute("INSERT INTO tracks VALUES (%s);", (track,))
        db.execute("INSERT INTO lyrics VALUES (%s, 'original')", (track,))
    for provenance in ["ai", "provider"]:
        storage.save_translation(
            track,
            pair,
            ["original"],
            [{"id": 0, "text": "translation"}],
            model="local",
            prompt="prompt",
            provenance=provenance,
        )
    rows = storage.load_translations(track, ["original"])
    assert len(rows) == 2 and all(row["status"] == "ready" for row in rows)
    assert all(row["status"] == "stale" for row in storage.load_translations(track, ["changed"]))
    storage.save_translation(
        track, pair, ["changed"], [{"id": 0, "text": "new"}], model="local", prompt="prompt"
    )
    rows = storage.load_translations(track, ["changed"])
    assert {r["provenance"]: r["status"] for r in rows} == {"ai": "ready", "provider": "stale"}
    with psycopg.connect(url) as db:
        with patch("alembic.op.execute", side_effect=db.execute):
            migration["downgrade"]()
        assert db.execute("SELECT text FROM lyrics").fetchone() == ("original",)
        with patch("alembic.op.execute", side_effect=db.execute):
            migration["upgrade"]()
        assert db.execute("SELECT count(*) FROM lyric_translations").fetchone() == (0,)
