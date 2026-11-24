"""Private disk retention, consent, owner scope and HTTP calibration contracts."""

from core.settings import get_settings

from datetime import datetime, timezone
import io
import os
from pathlib import Path
import runpy
from unittest.mock import Mock, patch
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest
import wave

from core.testing import apply_migrations
import worker.recordings.calibration as calibration
import worker.recordings.encoder as recording_encoder
import worker.recordings.search as recording_search
from echora_analysis.recording_routes import create_router


def _wav(seconds):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(np.zeros(int(seconds * 8000), dtype="<i2").tobytes())
    return stream.getvalue()


@pytest.fixture
def calibration_db(recording_db, monkeypatch, tmp_path):  # noqa: F811 -- fixture injection
    monkeypatch.setenv("ECHORA_RECORDING_CALIBRATION_ENABLED", "true")
    get_settings.cache_clear()
    monkeypatch.setenv("ECHORA_RECORDING_CALIBRATION_DIR", str(tmp_path / "private-clips"))
    get_settings.cache_clear()
    with recording_db() as db:
        db.execute("CREATE TABLE IF NOT EXISTS users(id uuid PRIMARY KEY)")
        apply_migrations(db, "0045_recording_calibration")
    return recording_db


def owner(db_factory):
    user_id = uuid4()
    with db_factory() as db:
        db.execute("INSERT INTO users(id) VALUES (%s)", (user_id,))
    return user_id


def capture(user_id, **kw):
    return calibration.save(user_id, _wav(2), not_in_library=True,
                            consent=calibration.CONSENT_VERSION, **kw)


def client(user_id):
    app = FastAPI()
    app.include_router(create_router(lambda: {"id": user_id}))
    return TestClient(app)


def test_calibration_enables_capture_without_bypassing_recognition_policy(monkeypatch):
    monkeypatch.setenv("ECHORA_RECORDING_CALIBRATION_ENABLED", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(recording_encoder, "config_from_env", lambda: None)
    status = recording_search.status(uuid4())
    assert status["enabled"] and status["calibration_enabled"]
    assert not status["recognition_enabled"]
    response = client(uuid4()).post("/library/recording/search", content=_wav(2))
    assert response.status_code == 503


def test_api_requires_consent_label_and_size_before_saving(monkeypatch):
    monkeypatch.setenv("ECHORA_RECORDING_CALIBRATION_ENABLED", "true")
    get_settings.cache_clear()
    save = Mock(return_value={"id": str(uuid4())})
    monkeypatch.setattr(calibration, "save", save)
    api = client(uuid4())
    url = "/library/recording/calibration/samples"
    headers = {"X-Echora-Calibration-Consent": calibration.CONSENT_VERSION}
    assert api.post(url + "?not_in_library=true", content=b"audio").status_code == 422
    assert api.post(url, content=b"audio", headers=headers).status_code == 422
    assert api.post(url + f"?not_in_library=true&expected_track_id={uuid4()}", content=b"audio", headers=headers).status_code == 422
    assert api.post(url + "?not_in_library=true", content=b"x" * (get_settings().recording_max_upload_bytes + 1), headers=headers).status_code == 413
    save.assert_not_called()
    assert api.post(url + "?not_in_library=true", content=b"audio", headers=headers).status_code == 201
    assert save.call_args.kwargs["consent"] == calibration.CONSENT_VERSION
    encoded_headers = {**headers, "X-Echora-Calibration-Notes": "%E9%9F%B3%E6%A5%BD%20speaker"}
    assert api.post(url + "?not_in_library=true", content=b"audio", headers=encoded_headers).status_code == 201
    assert save.call_args.kwargs["notes"] == "音楽 speaker"
    assert api.post(url + "?not_in_library=true", content=b"audio",
                    headers={**headers, "X-Echora-Calibration-Notes": "x" * 301}).status_code == 422
    monkeypatch.setenv("ECHORA_RECORDING_CALIBRATION_ENABLED", "false")
    get_settings.cache_clear()
    assert api.post(url + "?not_in_library=true", content=b"audio", headers=headers).status_code == 503


def test_private_file_and_owner_only_list_delete(calibration_db):
    user, other = owner(calibration_db), owner(calibration_db)
    sample = capture(user, notes="Phone beside speaker")
    path = calibration.directory() / f"{sample['id']}.audio"
    assert path.read_bytes() == _wav(2)
    assert path.stat().st_mode & 0o777 == 0o600
    assert calibration.directory().stat().st_mode & 0o777 == 0o700
    assert sample["duration_seconds"] == 2
    assert 6.9 < (sample["expires_at"] - datetime.now(timezone.utc)).total_seconds() / 86400 <= 7
    assert "user_id" not in sample and "audio_sha256" not in sample and "path" not in sample
    assert calibration.list_samples(other) == {"samples": []}
    assert not calibration.delete(sample["id"], other)
    assert path.exists()
    assert calibration.list_samples(user)["samples"][0]["id"] == sample["id"]
    # Owners can erase retained clips even after the operator disables collection.
    with patch.dict(os.environ, {"ECHORA_RECORDING_CALIBRATION_ENABLED": "false"}):
        assert calibration.delete(sample["id"], user)
    assert not path.exists() and not calibration.list_samples(user)["samples"]


def test_known_track_label_requires_current_visibility(calibration_db):
    user, other = owner(calibration_db), owner(calibration_db)
    track = uuid4()
    with calibration_db() as db:
        db.execute("INSERT INTO tracks(id,title,artist) VALUES (%s,'Actual song','Artist')", (track,))
        db.execute("INSERT INTO user_track_links VALUES (%s,%s,'source',%s)", (user, track, uuid4()))
    with pytest.raises(calibration.TrackUnavailable):
        calibration.save(other, _wav(2), expected_track_id=track, consent=calibration.CONSENT_VERSION)
    sample = calibration.save(user, _wav(2), expected_track_id=track, consent=calibration.CONSENT_VERSION)
    assert sample["expected_track_id"] == str(track)
    assert calibration.list_samples(user)["samples"][0]["expected_title"] == "Actual song"
    with calibration_db() as db:
        db.execute("DELETE FROM user_track_links WHERE user_id=%s", (user,))
    assert "expected_title" not in calibration.list_samples(user)["samples"][0]


def test_expiry_and_orphan_cleanup_without_touching_other_files(calibration_db):
    user = owner(calibration_db)
    expired, retained = capture(user), capture(user)
    root = calibration.directory()
    old_orphan = root / f"{uuid4()}.audio"
    old_orphan.write_bytes(b"orphan")
    os.utime(old_orphan, (1, 1))
    fresh_orphan = root / f"{uuid4()}.audio"
    fresh_orphan.write_bytes(b"pending")
    unrelated = root / "notes.txt"
    unrelated.write_text("keep")
    with calibration_db() as db:
        db.execute("UPDATE recording_calibration_samples SET expires_at=now()-interval '1 second' WHERE id=%s", (expired["id"],))
    calibration.cleanup()
    assert not (root / f"{expired['id']}.audio").exists()
    assert not old_orphan.exists()
    assert fresh_orphan.exists() and unrelated.exists()
    assert (root / f"{retained['id']}.audio").exists()


def test_quota_and_disk_failure_do_not_publish_sample_rows(calibration_db, monkeypatch):
    user = owner(calibration_db)
    monkeypatch.setenv("ECHORA_RECORDING_CALIBRATION_MAX_USER_SAMPLES", "1")
    get_settings.cache_clear()
    sample = capture(user)
    with pytest.raises(recording_search.QueueFull):
        capture(user)
    calibration.delete(sample["id"], user)
    monkeypatch.setattr(calibration.os, "open", Mock(side_effect=OSError("disk unavailable")))
    with pytest.raises(OSError):
        capture(user)
    assert not calibration.list_samples(user)["samples"]
    assert not list(calibration.directory().glob("*.audio"))


def test_http_delete_is_owner_scoped(calibration_db):
    user, other = owner(calibration_db), owner(calibration_db)
    sample = capture(user)
    url = f"/library/recording/calibration/samples/{sample['id']}"
    assert client(other).delete(url).status_code == 404
    assert client(user).delete(url).status_code == 204
    assert client(user).delete(url).status_code == 404
