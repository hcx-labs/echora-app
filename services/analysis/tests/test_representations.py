import os
import uuid

import psycopg
import pytest
from core.config import reset_config_cache
from core.curation.preview import curation_corpus
from core.embeddings.representations import (
    config_hash,
    configure_representations,
    embedding_config,
    model_settings,
)
from core.library.membership import reconcile_user_tracks
from psycopg.types.json import Jsonb
from worker.library.attempts import finish_attempt, record_track, start_attempt


def test_default_contract_preserves_existing_audio_configuration():
    config = embedding_config("mert")
    assert config["coverage"] == "full-track"
    assert config["window_seconds"] == 10
    assert config["stride_seconds"] == 5
    assert config["store_window_embeddings"] is True
    assert config_hash(config) != config_hash({**config, "stride_seconds": 10})


def test_custom_repository_and_revision_cannot_reuse_default_contract(monkeypatch):
    original = config_hash(embedding_config("muq_mulan"))
    monkeypatch.setenv("MUQ_MODEL_ID", "example/custom")
    reset_config_cache()
    assert config_hash(embedding_config("muq_mulan")) != original
    monkeypatch.setenv("MUQ_REVISION", "another-revision")
    reset_config_cache()
    assert model_settings("muq_mulan") == ("example/custom", "another-revision")
    assert embedding_config("muq_mulan")["revision"] == "another-revision"


@pytest.fixture()
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


def add_run(db, revision=None, stride=5, status="complete"):
    config = embedding_config("muq_mulan", revision)
    config["stride_seconds"] = stride
    run_id = uuid.uuid4()
    db.execute(
        """INSERT INTO analysis_runs
           (id, kind, model_name, model_revision, config_hash, config, status)
           VALUES (%s,'audio_embedding','muq_mulan',%s,%s,%s,%s)""",
        (run_id, config["revision"], config_hash(config), Jsonb(config), status),
    )
    return run_id


def add_embedding(db, run_id, track_id=None, dimension=512):
    if track_id is None:
        track_id = uuid.uuid4()
        db.execute(
            "INSERT INTO tracks (id, audio_hash, title, artist, duration_seconds) VALUES (%s,%s,'Test','Test',30)",
            (track_id, str(track_id)),
        )
    vector = "[" + ",".join(["1"] + ["0"] * (dimension - 1)) + "]"
    db.execute(
        """INSERT INTO embeddings (track_id, run_id, embedding_type, dimension, embedding)
           VALUES (%s,%s,'audio-track',%s,%s::vector)""",
        (track_id, run_id, dimension, vector),
    )
    return track_id


def test_current_view_excludes_newer_incompatible_runs_and_wrong_dimensions(db):
    active = add_run(db)
    older_revision = add_run(db, revision="different-space")
    other_preprocessing = add_run(db, stride=10)
    track = add_embedding(db, active)
    add_embedding(db, older_revision, track)
    add_embedding(db, other_preprocessing, track)
    add_embedding(db, active, dimension=768)
    rows = db.execute("SELECT track_id, run_id FROM current_embeddings").fetchall()
    assert rows == [(track, active)]


def test_committed_tracks_remain_visible_while_batch_is_running(db):
    run = add_run(db, status="running")
    track = add_embedding(db, run)
    assert db.execute(
        "SELECT track_id FROM current_embeddings WHERE run_id=%s", (run,)
    ).fetchall() == [(track,)]
    attempt = start_attempt(db, run, 2)
    record_track(db, attempt, "one", track)
    record_track(db, attempt, "two", error="Failed")
    finish_attempt(db, attempt)
    assert db.execute(
        "SELECT status, succeeded, failed FROM analysis_attempts WHERE id=%s", (attempt,)
    ).fetchone() == ("partial", 1, 1)
    assert (
        db.execute("SELECT count(*) FROM current_embeddings WHERE run_id=%s", (run,)).fetchone()[0]
        == 1
    )


def test_revision_switch_reduces_coverage_instead_of_mixing_spaces(db, monkeypatch):
    old = add_run(db)
    add_embedding(db, old)
    monkeypatch.setenv("MUQ_REVISION", "replacement")
    reset_config_cache()
    configure_representations(db)
    assert db.execute("SELECT count(*) FROM current_embeddings").fetchone()[0] == 0
    new = add_run(db)
    track = add_embedding(db, new)
    assert db.execute("SELECT track_id FROM current_embeddings").fetchall() == [(track,)]


@pytest.fixture()
def visible_track(db, monkeypatch):
    from contextlib import contextmanager

    import echora_analysis.main as main

    run = add_run(db)
    track = add_embedding(db, run)
    user, library, connection_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
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
    db.execute(
        "INSERT INTO navidrome_connections (id, owner_user_id, url, username, encrypted_password) VALUES (%s,%s,'http://test','test',%s)",
        (connection_id, user, b"test"),
    )

    @contextmanager
    def same_connection(*args, **kwargs):
        previous = db.row_factory
        db.row_factory = kwargs.get("row_factory", previous)
        try:
            yield db
        finally:
            db.row_factory = previous

    monkeypatch.setattr(main.psycopg, "connect", same_connection)
    monkeypatch.setattr(main, "_session_user", lambda _: {"id": user})
    return main, user, library, connection_id, track, run


def test_curation_corpus_reports_exact_representation_and_ignores_incompatible(db, visible_track):
    main, user, _, connection_id, track, run = visible_track
    other = add_run(db, revision="incompatible")
    add_embedding(db, other, track)
    rows, matrix, *_ = curation_corpus(user, connection_id)
    assert matrix.shape == (1, 512)
    assert rows[0]["representation_runs"] == {"muq_mulan": str(run)}


def test_descriptor_endpoint_enforces_visibility_and_reports_pending(visible_track):
    from fastapi import HTTPException

    main, _, _, _, track, _ = visible_track
    assert main.track_audio_descriptors(track, "test")["status"] == "pending"
    with pytest.raises(HTTPException) as error:
        main.track_audio_descriptors(uuid.uuid4(), "test")
    assert error.value.status_code == 404


def test_planner_retries_partial_descriptors_and_keeps_current_audio(db, visible_track):
    from worker.library.descriptors import DESCRIPTOR_REVISION
    from worker.planning.audio import plan_audio

    _, _, library, _, track, _ = visible_track
    plan = plan_audio(db, library, ["song"])
    assert not plan.needs_muq
    assert plan.descriptor_external_ids == frozenset({"song"})
    db.execute(
        "INSERT INTO track_audio_descriptors (track_id, revision, status, descriptors) VALUES (%s,%s,'partial','{}')",
        (track, DESCRIPTOR_REVISION),
    )
    assert plan_audio(db, library, ["song"]).descriptor_external_ids == frozenset({"song"})
    db.execute("UPDATE track_audio_descriptors SET status='complete' WHERE track_id=%s", (track,))
    assert not plan_audio(db, library, ["song"]).descriptor_external_ids


def test_waveform_endpoint_visibility_and_pending(visible_track, db):
    from fastapi import HTTPException
    from worker.library.waveforms import WAVEFORM_REVISION

    main, _, _, _, track, _ = visible_track
    assert main.track_waveform(track, "test")["status"] == "pending"
    with pytest.raises(HTTPException) as error:
        main.track_waveform(uuid.uuid4(), "test")
    assert error.value.status_code == 404
    db.execute(
        "INSERT INTO track_waveforms (track_id, revision, duration_seconds, peaks) VALUES (%s,%s,10,'[0,1]')",
        (track, WAVEFORM_REVISION),
    )
    result = main.track_waveform(track, "test")
    assert result["status"] == "complete"
    assert result["waveform"]["peaks"] == [0, 1]


def test_waveform_planner_skips_current_and_retries_old_revision(db, visible_track):
    from worker.library.waveforms import WAVEFORM_REVISION
    from worker.planning.audio import plan_audio

    _, _, library, _, track, _ = visible_track
    assert plan_audio(db, library, ["song"]).waveform_external_ids == frozenset({"song"})
    db.execute(
        "INSERT INTO track_waveforms (track_id, revision, duration_seconds, peaks) VALUES (%s,%s,10,'[0,1]')",
        (track, WAVEFORM_REVISION),
    )
    assert not plan_audio(db, library, ["song"]).waveform_external_ids
    db.execute("UPDATE track_waveforms SET revision='old' WHERE track_id=%s", (track,))
    assert plan_audio(db, library, ["song"]).waveform_external_ids == frozenset({"song"})


def test_visual_feature_endpoint_visibility_and_planner(db, visible_track):
    from fastapi import HTTPException
    from worker.library.visual_features import VISUAL_FEATURE_REVISION
    from worker.planning.audio import plan_audio

    from echora_analysis.main import track_visual_features

    _, _, library, _, track, _ = visible_track
    assert plan_audio(db, library, ["song"]).visual_feature_external_ids == frozenset({"song"})
    assert track_visual_features(track, "test")["status"] == "pending"
    with pytest.raises(HTTPException) as error:
        track_visual_features(uuid.uuid4(), "test")
    assert error.value.status_code == 404
    db.execute(
        """INSERT INTO track_visual_features (track_id, revision, duration_seconds, hop_seconds, features)
           VALUES (%s,%s,10,.1,%s)""",
        (track, VISUAL_FEATURE_REVISION, '{"bands":[[0.1]],"level":[0.1],"onset":[0]}'),
    )
    assert not plan_audio(db, library, ["song"]).visual_feature_external_ids
    response = track_visual_features(track, "test")
    assert response["visual_features"]["features"]["bands"] == [[0.1]]
    assert response["enrichment"] == {"descriptors": None, "vocal_activity": None, "melody": None}
    # Read-time enrichment appears without a visual-cache rerun.
    from worker.library.descriptors import DESCRIPTOR_REVISION

    db.execute(
        "INSERT INTO track_audio_descriptors (track_id, revision, status, descriptors) VALUES (%s,%s,'complete',%s)",
        (track, DESCRIPTOR_REVISION, Jsonb({"rhythm": {"bpm": 120}})),
    )
    assert (
        track_visual_features(track, "test")["enrichment"]["descriptors"]["descriptors"]["rhythm"][
            "bpm"
        ]
        == 120
    )
    assert not plan_audio(db, library, ["song"]).visual_feature_external_ids
    db.execute(
        "UPDATE track_visual_features SET status='unsupported', features='{}' WHERE track_id=%s",
        (track,),
    )
    assert not plan_audio(db, library, ["song"]).visual_feature_external_ids
    response = track_visual_features(track, "test")
    assert response["status"] == "unsupported"
    assert response["visual_features"] is None
    db.execute("UPDATE track_visual_features SET revision='1' WHERE track_id=%s", (track,))
    assert plan_audio(db, library, ["song"]).visual_feature_external_ids == frozenset({"song"})
    assert track_visual_features(track, "test")["status"] == "pending"


def test_job_transition_interrupts_only_its_unfinished_attempts(db, monkeypatch):
    job_id = uuid.uuid4()
    db.execute(
        "INSERT INTO jobs(id,kind,worker_type,user_id,status) VALUES (%s,'analysis_batch','analysis',%s,'running')",
        (job_id, uuid.uuid4()),
    )
    run = add_run(db)
    unrelated = start_attempt(db, run, 1)
    monkeypatch.setenv("ECHORA_JOB_ID", str(job_id))
    reset_config_cache()
    owned = start_attempt(db, run, 1)
    db.execute("UPDATE jobs SET status='queued' WHERE id=%s", (job_id,))
    assert db.execute("SELECT status FROM analysis_attempts WHERE id=%s", (owned,)).fetchone() == (
        "interrupted",
    )
    assert db.execute(
        "SELECT status FROM analysis_attempts WHERE id=%s", (unrelated,)
    ).fetchone() == ("running",)


def test_catalog_reconciliation_deduplicates_identical_source_tracks(db, visible_track):
    main, user, library, _, track, _ = visible_track
    db.execute(
        "UPDATE libraries SET namespace=%s WHERE id=%s",
        (uuid.uuid5(uuid.NAMESPACE_URL, "http://test"), library),
    )
    db.execute(
        "INSERT INTO track_sources(library_id,track_id,source_type,external_id) VALUES (%s,%s,'subsonic','alias')",
        (library, track),
    )
    result = reconcile_user_tracks(user, "http://test", ["song", "alias"])
    assert result["linked"] == 1
    assert db.execute(
        "SELECT external_id FROM user_track_links WHERE user_id=%s", (user,)
    ).fetchall() == [("alias",)]


def test_sync_selection_respects_mode_and_library_scope(db, visible_track):
    from worker.planning.sync import select_sync_tracks

    _, _, library, _, _, _ = visible_track
    db.execute(
        "UPDATE libraries SET namespace=%s WHERE id=%s",
        (uuid.uuid5(uuid.NAMESPACE_URL, "http://test"), library),
    )
    assert select_sync_tracks(db, "http://test", ["song", "new"], "missing") == ["new"]
    # Existing source has only MuQ, so entire-library repair includes it too.
    assert select_sync_tracks(db, "http://test", ["song", "new"], "all") == ["song", "new"]
    assert select_sync_tracks(db, "http://another-library", ["song"], "missing") == ["song"]
