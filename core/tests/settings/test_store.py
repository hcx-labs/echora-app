"""The analysis settings store against PostgreSQL (a temporary stand-in for the singleton row)."""

import os

import psycopg
import pytest

from core.settings import store


@pytest.fixture
def db():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL required")
    with psycopg.connect(url) as connection:
        connection.execute("""CREATE TEMP TABLE analysis_settings (
            singleton boolean PRIMARY KEY,
            karaoke_processing_enabled boolean NOT NULL DEFAULT true,
            karaoke_modal_enabled boolean NOT NULL DEFAULT true,
            hum_processing_enabled boolean NOT NULL DEFAULT true,
            hum_modal_enabled boolean NOT NULL DEFAULT true,
            transcription_processing_enabled boolean NOT NULL DEFAULT false,
            transcription_modal_enabled boolean NOT NULL DEFAULT false,
            karaoke_stems_processing_enabled boolean NOT NULL DEFAULT false,
            karaoke_stems_modal_enabled boolean NOT NULL DEFAULT false,
            karaoke_bound_to_synced_lines boolean NOT NULL DEFAULT true,
            hf_token_account text, hf_token_encrypted bytea,
            updated_at timestamptz)""")
        yield connection
        connection.rollback()


def test_switches_start_unset_and_save_one_column(db):
    assert store.model_switches(db) is None
    assert store.transcription_enabled(db) is False
    store.set_switch(db, "transcription_modal_enabled", True, unbind_karaoke=False)
    switches = store.model_switches(db)
    assert switches["transcription_modal_enabled"] is True
    assert switches["has_huggingface_token"] is False
    assert switches["karaoke_stems_processing_enabled"] is False
    assert switches["karaoke_stems_processing_enabled"] is False
    assert store.transcription_enabled(db) is True
    bound = db.execute("SELECT karaoke_bound_to_synced_lines FROM analysis_settings").fetchone()
    assert bound == (True,)


def test_karaoke_and_hum_switches_unbind_karaoke_from_synced_lines(db):
    store.set_switch(db, "transcription_processing_enabled", False, unbind_karaoke=False)
    store.set_switch(db, "hum_modal_enabled", False, unbind_karaoke=True)
    row = db.execute(
        "SELECT hum_modal_enabled, karaoke_bound_to_synced_lines FROM analysis_settings"
    ).fetchone()
    assert row == (False, False)


def test_unknown_switch_is_refused(db):
    with pytest.raises(ValueError, match="Unknown feature switch"):
        store.set_switch(db, "singleton=false; --", True, unbind_karaoke=False)


def test_huggingface_token_and_external_processing_round_trip(db):
    assert store.huggingface_token(db) is None and not store.has_huggingface_token(db)
    store.save_huggingface_token(db, b"secret", "hcX02")
    assert store.has_huggingface_token(db)
    assert store.huggingface_token(db) == {
        "hf_token_encrypted": b"secret",
        "hf_token_account": "hcX02",
    }
    store.save_huggingface_token(db, None, None)
    assert not store.has_huggingface_token(db)

    db.execute("""CREATE TEMP TABLE external_processing_settings (
        singleton boolean PRIMARY KEY, status text, gpu text DEFAULT 'L40S')""")
    assert store.external_processing(db) is None
    store.save_external_processing(db, status="ready")
    store.save_external_processing(db, gpu="H100")
    assert store.external_processing(db) == {"singleton": True, "status": "ready", "gpu": "H100"}

    assert store.try_advisory_lock(db, 0x5445535421)
    store.advisory_unlock(db, 0x5445535421)

    db.execute("""CREATE TEMP TABLE jobs (id int, kind text, status text, progress jsonb,
        error text, created_at timestamptz)""")
    db.execute("""INSERT INTO jobs VALUES (1, 'modal_setup', 'complete', NULL, NULL, now() - interval '1 hour'),
        (2, 'modal_setup', 'failed', NULL, 'x', now()), (3, 'other', 'queued', NULL, NULL, now())""")
    assert store.latest_job(db, "modal_setup")["id"] == 2
