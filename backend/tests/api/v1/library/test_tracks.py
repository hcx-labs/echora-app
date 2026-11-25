import uuid

import backend.api.v1.library.track_audio_quality as quality
import backend.api.v1.library.track_motion_artwork as motion
import backend.api.v1.library.track_visual_features as visual
from backend.library.melody_preview import melody_preview

from .fakes import install

TRACK = uuid.UUID(int=9)
BASE = f"/api/v1/library/tracks/{TRACK}"


def test_unknown_sort_is_422(client):
    response = client.get("/api/v1/library/tracks", params={"sort_by": "mood"})
    assert response.status_code == 422
    assert response.json() == {"detail": "Unknown track sort order"}
    assert client.get("/api/v1/library/tracks", params={"sort_direction": "up"}).json() == {
        "detail": "Unknown sort direction"
    }


def test_invalid_browse_filter_is_422(client):
    response = client.get("/api/v1/library/tracks", params={"vocals": "bad"})
    assert response.json() == {"detail": "Unknown vocals filter"}


def test_audio_quality_reports_lossless(client, monkeypatch):
    install(
        monkeypatch,
        quality,
        [
            {
                "codec": "FLAC",
                "content_type": "audio/flac",
                "bit_rate_kbps": 900,
                "bit_depth": 24,
                "sample_rate_hz": 48000,
                "channels": 2,
            }
        ],
    )
    body = client.get(f"{BASE}/audio-quality").json()
    assert body["lossless"] is True and body["bit_depth"] == 24
    install(monkeypatch, quality, [None])
    assert client.get(f"{BASE}/audio-quality").status_code == 404


def test_motion_artwork_points_at_the_media_route(client, monkeypatch):
    artwork = {"id": uuid.UUID(int=3), "width": 512, "height": 512, "frames": 96, "fps": 24}
    db = install(monkeypatch, motion, [{"x": 1}])
    monkeypatch.setattr(motion, "track_artwork", lambda db, user_id, track_id: artwork)
    body = client.get(f"{BASE}/motion-artwork").json()
    assert body["url"] == f"/media/motion-artwork/{artwork['id']}.mp4"
    assert body["available"] is True and body["frames"] == 96
    assert db.executed
    monkeypatch.setattr(motion, "track_artwork", lambda db, user_id, track_id: None)
    install(monkeypatch, motion, [{"x": 1}])
    assert client.get(f"{BASE}/motion-artwork").json() == {
        "track_id": str(TRACK),
        "available": False,
    }


# Visual features: visibility first, then cache, descriptors, vocals, melody (reads only).


def test_visual_features_check_visibility_before_any_read(client, user, monkeypatch):
    db = install(monkeypatch, visual, [None])
    response = client.get(f"{BASE}/visual-features")
    assert response.status_code == 404
    assert db.executed == [(db.executed[0][0], (user["id"], TRACK))]


def test_visual_features_return_current_enrichment_without_writes(client, monkeypatch):
    descriptor = {"revision": "1", "status": "complete", "descriptors": {"rhythm": {"bpm": 120}}}
    vocal = {"activity": {"windows": []}, "run_id": "voice-run"}
    contour = {"source": "full-mix", "pitch": [69, 69], "voiced": [True, True], "hop_seconds": 0.01}
    db = install(
        monkeypatch,
        visual,
        [{"x": 1}, {"status": "complete", "features": {"bands": []}}, descriptor, vocal, contour],
    )
    body = client.get(f"{BASE}/visual-features").json()
    assert body["status"] == "complete"
    assert body["enrichment"]["descriptors"] == descriptor
    assert body["enrichment"]["vocal_activity"] == vocal
    assert body["enrichment"]["melody"] == melody_preview("full-mix", [69, 69], [True, True], 0.01)
    assert all(sql.lstrip().startswith("SELECT") for sql, _ in db.executed)
    assert db.executed[1][1] == (TRACK, visual.VISUAL_FEATURE_REVISION)
    assert "current_analysis_runs" in db.executed[3][0]
    assert db.executed[4][1] == (TRACK, visual.MELODY_CONTOUR_REVISION)


def test_pending_or_unsupported_cache_serves_no_features(client, monkeypatch):
    install(monkeypatch, visual, [{"x": 1}, None, {"status": "complete"}, None, None])
    body = client.get(f"{BASE}/visual-features").json()
    assert body["status"] == "pending" and body["visual_features"] is None
    assert body["enrichment"]["descriptors"] == {"status": "complete"}
    install(monkeypatch, visual, [{"x": 1}, {"status": "unsupported", "features": {}}])
    body = client.get(f"{BASE}/visual-features").json()
    assert body["status"] == "unsupported" and body["visual_features"] is None


def test_library_needs_a_session(anonymous):
    assert anonymous.get(f"{BASE}/visual-features").status_code == 401
