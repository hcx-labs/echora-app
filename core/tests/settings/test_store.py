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
