"""Per-song analysis endpoints against a migrated database (TEST_DATABASE_URL)."""

import os
import uuid
from contextlib import contextmanager

import psycopg
import pytest
from psycopg.types.json import Jsonb

from backend.api.v1.library.track_audio_descriptors import TrackAudioDescriptors
from backend.api.v1.library.track_visual_features import TrackVisualFeatures
from backend.api.v1.library.track_waveform import TrackWaveform
from core.embeddings.representations import (
    config_hash,
    configure_representations,
    embedding_config,
)
from core.errors import NotFoundError
from worker.library.descriptors import DESCRIPTOR_REVISION
from worker.library.visual_features import VISUAL_FEATURE_REVISION
from worker.library.waveforms import WAVEFORM_REVISION


@pytest.fixture
def db():
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL must point to a disposable migrated database")
    with psycopg.connect(url) as connection:
        try:
            configure_representations(connection)
            yield connection
        finally:
            connection.rollback()


@pytest.fixture
def visible(db, monkeypatch):
    """A song in a fresh user's library; the endpoints read through ``db``."""
    config = embedding_config("muq_mulan")
    run, track, user, library = uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db.execute(
        """INSERT INTO analysis_runs (id, kind, model_name, model_revision, config_hash, config, status)
           VALUES (%s,'audio_embedding','muq_mulan',%s,%s,%s,'complete')""",
        (run, config["revision"], config_hash(config), Jsonb(config)),
    )
    db.execute(
        "INSERT INTO tracks (id, audio_hash, title, artist, duration_seconds) VALUES (%s,%s,'Test','Test',30)",
        (track, str(track)),
    )
    db.execute(
        "INSERT INTO users (id, username, display_name) VALUES (%s,%s,'Test')", (user, str(user))
    )
    db.execute(
        "INSERT INTO libraries (id, name, root_path, namespace) VALUES (%s,'Test','http://test',%s)",
        (library, library),
    )
    db.execute(
        "INSERT INTO track_sources (library_id, track_id, source_type, external_id) VALUES (%s,%s,'subsonic','song')",
        (library, track),
    )
    db.execute(
        "INSERT INTO user_track_links (user_id, library_id, track_id, external_id) VALUES (%s,%s,%s,'song')",
        (user, library, track),
    )

    @contextmanager
    def same_connection(*args, **kwargs):
        previous = db.row_factory
        db.row_factory = kwargs.get("row_factory", previous)
        try:
            yield db
        finally:
            db.row_factory = previous

    monkeypatch.setattr(psycopg, "connect", same_connection)
    return {"id": user}, track


def test_descriptors_enforce_visibility_and_report_pending(visible):
    user, track = visible
    assert TrackAudioDescriptors().handle(track, user)["status"] == "pending"
    with pytest.raises(NotFoundError):
        TrackAudioDescriptors().handle(uuid.uuid4(), user)


def test_waveform_visibility_pending_and_complete(visible, db):
    user, track = visible
    assert TrackWaveform().handle(track, user)["status"] == "pending"
    with pytest.raises(NotFoundError):
        TrackWaveform().handle(uuid.uuid4(), user)
    db.execute(
        "INSERT INTO track_waveforms (track_id, revision, duration_seconds, peaks) VALUES (%s,%s,10,'[0,1]')",
        (track, WAVEFORM_REVISION),
    )
    result = TrackWaveform().handle(track, user)
    assert result["status"] == "complete"
    assert result["waveform"]["peaks"] == [0, 1]


def test_visual_features_follow_the_cache_and_enrich_at_read_time(visible, db):
    user, track = visible
    endpoint = TrackVisualFeatures()
    assert endpoint.handle(track, user)["status"] == "pending"
    with pytest.raises(NotFoundError):
        endpoint.handle(uuid.uuid4(), user)
    db.execute(
        """INSERT INTO track_visual_features (track_id, revision, duration_seconds, hop_seconds, features)
           VALUES (%s,%s,10,.1,%s)""",
        (track, VISUAL_FEATURE_REVISION, '{"bands":[[0.1]],"level":[0.1],"onset":[0]}'),
    )
    response = endpoint.handle(track, user)
    assert response["visual_features"]["features"]["bands"] == [[0.1]]
    assert response["enrichment"] == {"descriptors": None, "vocal_activity": None, "melody": None}
    db.execute(
        "INSERT INTO track_audio_descriptors (track_id, revision, status, descriptors) VALUES (%s,%s,'complete',%s)",
        (track, DESCRIPTOR_REVISION, Jsonb({"rhythm": {"bpm": 120}})),
    )
    enrichment = endpoint.handle(track, user)["enrichment"]
    assert enrichment["descriptors"]["descriptors"]["rhythm"]["bpm"] == 120
    db.execute(
        "UPDATE track_visual_features SET status='unsupported', features='{}' WHERE track_id=%s",
        (track,),
    )
    response = endpoint.handle(track, user)
    assert response["status"] == "unsupported"
    assert response["visual_features"] is None
    db.execute("UPDATE track_visual_features SET revision='1' WHERE track_id=%s", (track,))
    assert endpoint.handle(track, user)["status"] == "pending"
