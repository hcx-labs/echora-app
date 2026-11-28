import uuid

import backend.api.v1.library.track_processing_resync as resync_endpoint

from .fakes import install

TRACK = uuid.UUID(int=5)
BASE = f"/api/v1/library/tracks/{TRACK}/processing"


def test_marking_a_pipeline_records_a_request_for_the_next_sync(client, user, monkeypatch):
    db = install(monkeypatch, resync_endpoint, [{"x": 1}])
    response = client.put(f"{BASE}/karaoke", json={})
    assert response.json() == {"track_id": str(TRACK), "pipeline": "karaoke", "requested": True}
    sql, params = db.executed[1]
    assert "INSERT INTO track_resync_requests" in sql
    assert params[:2] == (TRACK, "karaoke") and params[3] == user["id"]


def test_a_transcription_resync_needs_a_language_and_ai_lyrics(client, monkeypatch):
    install(monkeypatch, resync_endpoint, [{"x": 1}])
    assert client.put(f"{BASE}/transcription", json={}).status_code == 409
    install(monkeypatch, resync_endpoint, [{"x": 1}, {"enabled": False}])
    response = client.put(f"{BASE}/transcription", json={"language": "ja"})
    assert response.json()["detail"].startswith("Enable AI lyric generation")
    db = install(monkeypatch, resync_endpoint, [{"x": 1}, {"enabled": True}])
    assert client.put(f"{BASE}/transcription", json={"language": "pt-BR"}).status_code == 200
    assert db.executed[2][1][2].obj == {"language": "pt-BR"}


def test_unknown_pipelines_and_foreign_tracks_are_refused(client, monkeypatch):
    install(monkeypatch, resync_endpoint, [{"x": 1}])
    assert client.put(f"{BASE}/audio", json={}).status_code == 422
    install(monkeypatch, resync_endpoint, [None])
    assert client.put(f"{BASE}/karaoke", json={}).status_code == 404


def test_withdrawing_a_resync(client, monkeypatch):
    db = install(monkeypatch, resync_endpoint, [{"x": 1}])
    assert client.delete(f"{BASE}/themes").json()["requested"] is False
    assert "DELETE FROM track_resync_requests" in db.executed[1][0]
