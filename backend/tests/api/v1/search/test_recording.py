"""Recording API boundaries and isolated-schema lifecycle integration tests."""

import io
import json
import wave
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import numpy as np
import pytest
from fastapi import HTTPException

import core.jobs.queue as jobs
import worker.recordings.search as search
from core.config import reset_config_cache
from worker.recordings.config import RecordingConfig


@pytest.fixture
def api(monkeypatch, search_app):
    user_id = uuid4()
    monkeypatch.setattr(
        search,
        "search_configuration",
        lambda: (SimpleNamespace(representation_id="model"), object(), "policy"),
    )
    return search_app(lambda: {"id": user_id}), user_id


def test_api_does_not_infer_and_returns_job(api, monkeypatch):
    client, user_id = api
    enqueue = Mock(return_value={"job_id": str(uuid4())})
    monkeypatch.setattr(search, "enqueue", enqueue)
    response = client.post("/api/v1/search/recording", content=b"audio")
    assert response.status_code == 202
    enqueue.assert_called_once_with(user_id, b"audio", "model", "policy")


def test_api_limits_and_configuration(api, monkeypatch):
    client, _ = api
    enqueue = Mock(side_effect=search.QueueFull)
    monkeypatch.setattr(search, "enqueue", enqueue)
    assert client.post("/api/v1/search/recording", content=b"").status_code == 422
    assert (
        client.post(
            "/api/v1/search/recording",
            content=b"x" * (RecordingConfig.load().max_upload_bytes + 1),
        ).status_code
        == 413
    )
    enqueue.assert_not_called()
    response = client.post("/api/v1/search/recording", content=b"audio")
    assert response.status_code == 429
    assert response.headers["retry-after"] == "15"
    monkeypatch.setattr(search, "search_configuration", Mock(side_effect=ValueError))
    assert client.post("/api/v1/search/recording", content=b"audio").status_code == 503


def test_routes_require_authentication(search_app):
    def unauthorized():
        raise HTTPException(401, "Authentication required")

    client = search_app(unauthorized)
    assert client.get("/api/v1/search/recording/status").status_code == 401
    assert client.post("/api/v1/search/recording", content=b"audio").status_code == 401
    assert client.get(f"/api/v1/search/recording/{uuid4()}").status_code == 401


def _wav(seconds):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(np.zeros(int(seconds * 8000), dtype="<i2").tobytes())
    return stream.getvalue()


def test_bounded_decode():
    assert search.decode_query(_wav(3)).shape == (24000,)
    assert search.decode_query(_wav(20.25)).shape == (160000,)
    with pytest.raises(ValueError, match="20 seconds"):
        search.decode_query(_wav(22))
    with pytest.raises(ValueError):
        search.decode_query(b"not audio")


def test_policy_requires_calibration_and_matching_representation(monkeypatch, tmp_path):
    import worker.recordings.encoder as recording_encoder
    import worker.recordings.matcher as recording_matcher

    monkeypatch.setattr(
        recording_encoder, "config_from_env", lambda: SimpleNamespace(representation_id="one")
    )
    path = tmp_path / "policy.json"
    monkeypatch.setenv("ECHORA_RECORDING_MATCH_POLICY", str(path))
    reset_config_cache()
    data = {
        "representation_id": "one",
        "matcher_revision": recording_matcher.MATCHER_REVISION,
        "calibrated": False,
        "validation_dataset": "held-out-phone-recordings",
        "thresholds": {},
    }
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="calibrated"):
        search.search_configuration()
    data.update(calibrated=True, representation_id="other")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        search.search_configuration()


def test_queue_admission_fifo_and_terminal_audio_erasure(recording_db):
    owner = uuid4()
    background = jobs.enqueue("analysis", "analysis", owner)
    queued = search.enqueue(owner, b"private audio", "model", "policy")
    with pytest.raises(search.QueueFull):
        search.enqueue(owner, b"more", "model", "policy")
    first = jobs.claim("analysis", "worker")
    assert str(first["id"]) == background["id"]
    jobs.finish(first["id"], first["token"], {})
    job = jobs.claim("analysis", "worker")
    assert str(job["id"]) == queued["job_id"]
    assert job["payload"] == {}
    assert search.result(job["id"], uuid4()) is None
    assert "private audio" not in str(jobs.get_job(job["id"], owner))
    jobs.finish(job["id"], job["token"], {"state": "no_match"})
    with recording_db() as db:
        assert (
            db.execute(
                "SELECT audio FROM recording_searches WHERE job_id=%s", (job["id"],)
            ).fetchone()["audio"]
            is None
        )
    assert jobs.claim("analysis", "worker") is None


def test_cancellation_and_expiry_clear_private_audio(recording_db):
    owner = uuid4()
    queued = search.enqueue(owner, b"private", "model", "policy")
    jobs.cancel(queued["job_id"], owner)
    other = search.enqueue(owner, b"private again", "model", "policy")
    with recording_db() as db:
        assert (
            db.execute(
                "SELECT audio FROM recording_searches WHERE job_id=%s", (queued["job_id"],)
            ).fetchone()["audio"]
            is None
        )
        db.execute(
            "UPDATE recording_searches SET audio_expires_at=now()-interval '1 second' WHERE job_id=%s",
            (other["job_id"],),
        )
    search.cleanup()
    with recording_db() as db:
        assert (
            db.execute(
                "SELECT audio FROM recording_searches WHERE job_id=%s", (other["job_id"],)
            ).fetchone()["audio"]
            is None
        )


def test_references_scope_before_search_and_results_recheck_access(recording_db):
    owner, other, track, hidden, library = [uuid4() for _ in range(5)]
    matrix = np.eye(128, dtype="<f4")[:8]
    with recording_db() as db:
        for track_id, user_id in ((track, owner), (hidden, other)):
            db.execute("INSERT INTO tracks(id,title) VALUES (%s,'Song')", (track_id,))
            db.execute(
                "INSERT INTO user_track_links VALUES (%s,%s,'source',%s)",
                (user_id, track_id, library),
            )
            db.execute(
                "INSERT INTO recording_fingerprints(track_id,representation_id,segment_count,fingerprints) VALUES (%s,'model',8,%s)",
                (track_id, matrix.tobytes()),
            )
        refs = list(search._references(db, owner, "model", lambda: None))
        assert [ref["track_id"] for ref in refs] == [str(track)]
    queued = search.enqueue(owner, b"audio", "model", "policy")
    job = jobs.claim("analysis", "worker")
    from psycopg.types.json import Jsonb

    with recording_db() as db:
        db.execute(
            "UPDATE recording_searches SET result=%s WHERE job_id=%s",
            (
                Jsonb(
                    {
                        "state": "identified",
                        "matches": [
                            {"track_id": str(track), "score": 0.9, "offset_seconds": 1.5},
                            {"track_id": str(hidden), "score": 0.8, "offset_seconds": 1.5},
                        ],
                    }
                ),
                job["id"],
            ),
        )
    jobs.finish(job["id"], job["token"], {"state": "identified"})
    value = search.result(queued["job_id"], owner)["result"]
    assert value["state"] == "identified"
    assert [item["track_id"] for item in value["matches"]] == [str(track)]
    with recording_db() as db:
        db.execute("DELETE FROM user_track_links WHERE track_id=%s", (track,))
    assert search.result(queued["job_id"], owner)["result"] == {"state": "no_match", "matches": []}


def test_worker_roundtrip_and_configuration_fence(recording_db, monkeypatch):
    import worker.recordings.encoder as recording_encoder
    from worker.recordings.matcher import MatchPolicy

    owner, track, library = uuid4(), uuid4(), uuid4()
    matrix = np.eye(128, dtype="<f4")[:30]
    config = SimpleNamespace(representation_id="model")
    policy = MatchPolicy(
        window_similarity=0.8,
        min_score=0.8,
        min_support=6,
        min_coverage=0.8,
        min_temporal_spread=0.6,
        min_margin=0.08,
    )
    monkeypatch.setattr(search, "search_configuration", lambda: (config, policy, "policy"))
    monkeypatch.setattr(
        search, "decode_query", lambda audio, **kwargs: np.ones(80000, dtype=np.float32)
    )
    monkeypatch.setattr(recording_encoder, "encode", lambda *a, **kw: matrix[5:25])
    with recording_db() as db:
        db.execute("INSERT INTO tracks(id,title) VALUES (%s,'Matched recording')", (track,))
        db.execute(
            "INSERT INTO user_track_links VALUES (%s,%s,'source',%s)", (owner, track, library)
        )
        db.execute(
            "INSERT INTO recording_fingerprints(track_id,representation_id,segment_count,fingerprints) VALUES (%s,'model',30,%s)",
            (track, matrix.tobytes()),
        )
    queued = search.enqueue(owner, b"audio", "model", "policy")
    job = jobs.claim("analysis", "test-worker")
    summary = search.execute(job, jobs.JobContext(job))
    assert summary == {"state": "identified"}
    # A computed but uncommitted job result is not public yet.
    assert "result" not in search.result(queued["job_id"], owner)
    jobs.finish(job["id"], job["token"], summary)
    value = search.result(queued["job_id"], owner)["result"]
    assert value["matches"][0]["track_id"] == str(track)
    assert value["matches"][0]["offset_seconds"] == 2.5
    search.enqueue(owner, b"audio", "model", "old-policy")
    next_job = jobs.claim("analysis", "test-worker")
    with pytest.raises(ValueError, match="configuration changed"):
        search.execute(next_job, jobs.JobContext(next_job))


def test_cancel_api_is_owner_and_kind_scoped(api, monkeypatch):
    client, _ = api
    cancel = Mock()
    monkeypatch.setattr(jobs, "cancel", cancel)
    for value in (None, {"kind": "navidrome_sync"}):
        monkeypatch.setattr(jobs, "get_job", lambda *args, value=value: value)
        assert client.delete(f"/api/v1/search/recording/{uuid4()}").status_code == 404
    cancel.assert_not_called()
    monkeypatch.setattr(jobs, "get_job", lambda *args: {"kind": "recording_search"})
    assert client.delete(f"/api/v1/search/recording/{uuid4()}").status_code == 200
    cancel.assert_called_once()


def test_sync_index_status_scopes_catalog_owner_and_representation(recording_db, monkeypatch):
    from psycopg.rows import tuple_row

    import worker.recordings.encoder as recording_encoder

    owner, other, library, elsewhere, namespace = [uuid4() for _ in range(5)]
    monkeypatch.setattr(
        recording_encoder, "config_from_env", lambda: SimpleNamespace(representation_id="current")
    )
    monkeypatch.delenv("ECHORA_RECORDING_MATCH_POLICY", raising=False)
    reset_config_cache()
    with recording_db() as db:
        db.execute(
            "INSERT INTO libraries(id,namespace) VALUES (%s,%s),(%s,%s)",
            (library, namespace, elsewhere, uuid4()),
        )
        for external, who, lib, version, windows in [
            ("ready", owner, library, "current", 1),
            ("short", owner, library, "current", 0),
            ("stale", owner, library, "old", 1),
            ("hidden", other, library, "current", 1),
            ("elsewhere", owner, elsewhere, "current", 1),
            ("removed", owner, library, "current", 1),
        ]:
            track = uuid4()
            db.execute("INSERT INTO tracks(id,title) VALUES (%s,'Song')", (track,))
            db.execute(
                "INSERT INTO track_sources(track_id,library_id,external_id,source_type) VALUES (%s,%s,%s,'subsonic')",
                (track, lib, external),
            )
            db.execute(
                "INSERT INTO user_track_links VALUES (%s,%s,%s,%s)", (who, track, external, lib)
            )
            db.execute(
                "INSERT INTO recording_fingerprints(track_id,representation_id,segment_count,fingerprints) VALUES (%s,%s,%s,%s)",
                (track, version, windows, bytes(windows * 128 * 4)),
            )
        db.row_factory = tuple_row
        result = search.index_status(
            db, namespace, ["ready", "short", "stale", "hidden", "elsewhere", "new", "ready"], owner
        )
        assert result == {"enabled": True, "total": 6, "indexed": 1, "unsupported": 1, "missing": 4}
        assert search.index_status(db, namespace, [], owner)["total"] == 0


def test_sync_index_status_disabled_and_invalid_model(monkeypatch):
    from unittest.mock import MagicMock

    import worker.recordings.encoder as recording_encoder

    connection = MagicMock()
    monkeypatch.setattr(recording_encoder, "config_from_env", lambda: None)
    result = search.index_status(connection, uuid4(), ["song"], uuid4())
    assert not result["enabled"] and result["missing"] == 1
    monkeypatch.setattr(
        recording_encoder, "config_from_env", Mock(side_effect=ValueError("private path"))
    )
    result = search.index_status(connection, uuid4(), ["song"], uuid4())
    assert not result["enabled"] and "private path" not in result["reason"]
    connection.cursor.assert_not_called()


def test_query_and_library_decoders_agree_on_stereo_resampling():
    from worker.media.audio import decode_audio

    rate = 48000
    t = np.arange(rate * 2) / rate
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), 0.5 * np.sin(2 * np.pi * 997 * t)], axis=1)
    stream = io.BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes((stereo * 12000).astype("<i2").tobytes())
    audio = stream.getvalue()
    reference = decode_audio(audio, search.SAMPLE_RATE)
    query = search.decode_query(audio)
    assert query.shape == reference.shape == (16000,)
    np.testing.assert_array_equal(query, reference)


def test_match_playback_uses_matching_owned_connection_not_selected_other_server(recording_db):
    owner, other, track, library, unrelated, right, wrong, foreign = [uuid4() for _ in range(8)]
    with recording_db() as db:
        db.execute("INSERT INTO tracks(id,title) VALUES (%s,'Song')", (track,))
        db.execute(
            "INSERT INTO libraries(id,root_path) VALUES (%s,'https://server-a/'),(%s,'https://server-b')",
            (library, unrelated),
        )
        db.execute(
            "INSERT INTO navidrome_connections VALUES (%s,%s,'https://server-a'),(%s,%s,'https://server-b'),(%s,%s,'https://server-a')",
            (right, owner, wrong, owner, foreign, other),
        )
        db.execute("INSERT INTO user_preferences VALUES (%s,%s)", (owner, wrong))
        db.execute(
            "INSERT INTO user_track_links VALUES (%s,%s,'source-a',%s)", (owner, track, library)
        )
        db.execute(
            "INSERT INTO track_sources VALUES (%s,%s,'source-a','subsonic','{\"coverArt\":\"art-a\"}')",
            (track, library),
        )
        matches = [{"track_id": str(track), "score": 0.8}]
        result = search._visible_matches(db, matches, owner)[0]
        assert result["source_id"] == "source-a"
        assert result["connection_id"] == str(right)
        assert result["cover_art"] == "art-a"
        db.execute("DELETE FROM navidrome_connections WHERE id=%s", (right,))
        result = search._visible_matches(db, matches, owner)[0]
        assert result["connection_id"] is None and result["source_id"] is None
        assert result["cover_art"] is None
        assert search._visible_matches(db, matches, other) == []


def test_match_prefers_selected_connection_when_track_exists_on_both_servers(recording_db):
    owner, track, first, second, first_connection, second_connection = [uuid4() for _ in range(6)]
    with recording_db() as db:
        db.execute("INSERT INTO tracks(id,title) VALUES (%s,'Song')", (track,))
        for library, connection, url, source in [
            (first, first_connection, "https://a", "a-source"),
            (second, second_connection, "https://b", "b-source"),
        ]:
            db.execute("INSERT INTO libraries(id,root_path) VALUES (%s,%s)", (library, url))
            db.execute(
                "INSERT INTO navidrome_connections VALUES (%s,%s,%s)", (connection, owner, url)
            )
            db.execute(
                "INSERT INTO user_track_links VALUES (%s,%s,%s,%s)", (owner, track, source, library)
            )
        db.execute("INSERT INTO user_preferences VALUES (%s,%s)", (owner, second_connection))
        match = search._visible_matches(db, [{"track_id": str(track), "score": 0.8}], owner)[0]
        assert match["connection_id"] == str(second_connection) and match["source_id"] == "b-source"


def test_decoder_cancellation_kills_child_and_drains_pipes(monkeypatch):
    import subprocess
    from unittest.mock import MagicMock

    process = MagicMock()
    process.__enter__.return_value = process
    process.communicate.side_effect = [subprocess.TimeoutExpired("ffmpeg", 0.25), (b"", None)]
    monkeypatch.setattr(search.subprocess, "Popen", Mock(return_value=process))
    check = Mock(side_effect=[None, None, jobs.JobCancelled()])
    with pytest.raises(jobs.JobCancelled):
        search.decode_query(b"audio", check=check)
    process.kill.assert_called_once()
    assert process.communicate.call_count == 2


def test_decoder_resumes_communication_without_resending_input(monkeypatch):
    import subprocess
    from unittest.mock import MagicMock

    process = MagicMock()
    process.__enter__.return_value = process
    process.returncode = 0
    process.communicate.side_effect = [
        subprocess.TimeoutExpired("ffmpeg", 0.25),
        (bytes(8000 * 4), None),
    ]
    monkeypatch.setattr(search.subprocess, "Popen", Mock(return_value=process))
    assert search.decode_query(b"audio").shape == (8000,)
    assert process.communicate.call_args_list[0].kwargs["input"] == b"audio"
    assert process.communicate.call_args_list[1].kwargs["input"] is None
    process.kill.assert_not_called()


def test_decoder_timeout_terminates_child(monkeypatch):
    from unittest.mock import MagicMock

    process = MagicMock()
    process.__enter__.return_value = process
    process.communicate.return_value = (b"", None)
    monkeypatch.setattr(search.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(search.time, "monotonic", Mock(side_effect=[0, 31]))
    with pytest.raises(ValueError, match="decode timed out"):
        search.decode_query(b"audio")
    process.kill.assert_called_once()


def test_execution_budget_starts_before_configuration_and_preparation(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(search.time, "monotonic", lambda: clock[0])

    def configuration():
        clock[0] = 121
        return SimpleNamespace(), SimpleNamespace(), "policy"

    monkeypatch.setattr(search, "search_configuration", configuration)
    context = SimpleNamespace(check=Mock())
    with pytest.raises(ValueError, match="time budget"):
        search.execute({}, context)
    context.check.assert_called_once()
