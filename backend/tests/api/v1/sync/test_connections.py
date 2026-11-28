from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

import backend.api.v1.sync.discover as discover
import backend.api.v1.sync.sample as sample
import backend.api.v1.sync.start as start
import backend.sync.connections as connections
import core.jobs.queue as jobs

from ._fakes import snapshot

CREDENTIALS = ("https://music", "user", "secret")


@pytest.fixture
def enqueue(monkeypatch):
    fake = Mock(side_effect=lambda kind, *a, **k: snapshot(kind))
    monkeypatch.setattr(jobs, "enqueue", fake)
    return fake


@pytest.fixture
def owned(monkeypatch):
    load = Mock(return_value=CREDENTIALS)
    monkeypatch.setattr(connections, "load_connection", load)
    return load


def test_sync_returns_before_catalog_scan_and_never_queues_secrets(
    client, user, monkeypatch, enqueue, owned
):
    connection_id = uuid4()
    scan = Mock(side_effect=AssertionError("API must not scan Navidrome"))
    for module in (discover, sample):
        monkeypatch.setattr(module, "NavidromeClient", scan)
    compute = Mock(return_value="local")
    monkeypatch.setattr(start, "compute_for", compute)
    response = client.post(f"/api/v1/sync/connections/{connection_id}", json={"mode": "missing"})
    assert response.status_code == 202
    assert response.json()["kind"] == "navidrome_sync"
    assert enqueue.call_args.args == ("navidrome_sync", "analysis", user["id"])
    compute.assert_called_once_with(user["id"], None)
    assert enqueue.call_args.kwargs["payload"] == {
        "mode": "missing",
        "verify_audio_hashes": True,
        "compute": "local",
    }
    assert enqueue.call_args.kwargs["dedupe_key"] == f"navidrome_sync:{user['id']}:{connection_id}"
    assert "secret" not in repr(enqueue.call_args)
    scan.assert_not_called()


def test_sync_rejects_foreign_connection(client, user, monkeypatch, enqueue):
    connection_id = uuid4()
    load = Mock(return_value=None)
    monkeypatch.setattr(connections, "load_connection", load)
    response = client.post(f"/api/v1/sync/connections/{connection_id}", json={})
    assert response.status_code == 404
    assert response.json() == {"detail": "Connection not found"}
    load.assert_called_once_with(str(connection_id), user["id"])
    enqueue.assert_not_called()


def test_sync_refuses_modal_for_users_without_it(client, monkeypatch, enqueue, owned):
    def refuse(user_id, requested):
        raise PermissionError("Modal processing is not available")

    monkeypatch.setattr(start, "compute_for", refuse)
    response = client.post(f"/api/v1/sync/connections/{uuid4()}", json={"compute": "modal"})
    assert response.status_code == 403
    assert response.json() == {"detail": "Modal processing is not available to you"}
    enqueue.assert_not_called()


@pytest.mark.parametrize("body", [{"mode": "everything"}, {"compute": "gpu"}, {"other": 1}])
def test_sync_validates_its_body(client, owned, enqueue, body):
    assert client.post(f"/api/v1/sync/connections/{uuid4()}", json=body).status_code == 422


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("recordings", "recordings_backfill"),
        ("lyrics", "lyrics_backfill"),
        ("voice", "voice_backfill"),
        ("karaoke", "karaoke_backfill"),
    ],
)
def test_backfills_queue_one_job_per_connection(client, user, enqueue, owned, name, kind):
    user["is_admin"] = True
    connection_id = str(uuid4())
    response = client.post(f"/api/v1/sync/connections/{connection_id}/backfills/{name}")
    assert response.status_code == 202
    assert enqueue.call_args.args == (kind, "analysis", user["id"])
    assert enqueue.call_args.kwargs == {
        "connection_id": connection_id,
        "payload": {},
        "dedupe_key": f"{kind}:{user['id']}:{connection_id}",
    }


@pytest.mark.parametrize("name", ["recordings", "lyrics", "voice", "karaoke"])
def test_backfills_are_for_administrators_only(client, enqueue, owned, name):
    response = client.post(f"/api/v1/sync/connections/{uuid4()}/backfills/{name}")
    assert response.status_code == 403
    assert response.json() == {"detail": "Administrator access is required"}
    enqueue.assert_not_called()


def test_import_dedupes_the_same_selection(client, user, enqueue, owned):
    connection_id = str(uuid4())
    url = f"/api/v1/sync/connections/{connection_id}/import"
    assert client.post(url, json={"track_ids": ["b", "a", "b"]}).status_code == 202
    first = enqueue.call_args.kwargs
    assert first["payload"] == {"track_ids": ["a", "b"]}
    client.post(url, json={"track_ids": ["a", "b"]})
    assert enqueue.call_args.kwargs["dedupe_key"] == first["dedupe_key"]
    assert client.post(url, json={"track_ids": []}).status_code == 422


def _track(identifier):
    return SimpleNamespace(
        id=identifier,
        title="Song",
        artist="Artist",
        album=None,
        duration=200.0,
        genre="Pop",
        raw={"coverArt": "al-1"},
    )


class FakeNavidrome:
    def __init__(self, *credentials, fail=False):
        self.credentials, self.fail = credentials, fail

    def __enter__(self):
        if self.fail:
            raise OSError("refused")
        return self

    def __exit__(self, *exc):
        return False

    def ping(self):
        return "0.55.0"

    def random_tracks(self, limit):
        return [_track(str(i)) for i in range(min(limit, 2))]


def test_discover_saves_the_connection_and_previews(client, user, monkeypatch):
    save = Mock(return_value="c1")
    monkeypatch.setattr(discover, "save_connection", save)
    monkeypatch.setattr(discover, "NavidromeClient", FakeNavidrome)
    body = {"url": "https://music.example.test/", "username": "u", "password": " p w ", "limit": 5}
    response = client.post("/api/v1/sync/discover", json=body)
    assert response.status_code == 200
    assert response.json() == {
        "connection_id": "c1",
        "server": {"version": "0.55.0", "url": "https://music.example.test"},
        "tracks": [
            {
                "id": str(i),
                "title": "Song",
                "artist": "Artist",
                "album": None,
                "duration": 200.0,
                "genre": "Pop",
                "cover_art": "al-1",
            }
            for i in range(2)
        ],
    }
    # Passwords are kept exactly as typed.
    save.assert_called_once_with(("https://music.example.test", "u", " p w "), user["id"])


def test_discover_failure_is_400_and_saves_nothing(client, monkeypatch):
    save = Mock()
    monkeypatch.setattr(discover, "save_connection", save)
    monkeypatch.setattr(discover, "NavidromeClient", lambda *c: FakeNavidrome(fail=True))
    body = {"url": "https://music.example.test", "username": "u", "password": "p"}
    response = client.post("/api/v1/sync/discover", json=body)
    assert response.status_code == 400
    assert response.json() == {"detail": "Could not connect to the Navidrome server"}
    save.assert_not_called()


def test_sample_failure_is_502(client, owned, monkeypatch):
    monkeypatch.setattr(sample, "NavidromeClient", lambda *c: FakeNavidrome(fail=True))
    response = client.get(f"/api/v1/sync/connections/{uuid4()}/sample")
    assert response.status_code == 502
    assert response.json() == {"detail": "Could not read tracks from the Navidrome server"}


def test_ingest_saves_then_imports(client, user, monkeypatch, enqueue):
    import backend.api.v1.sync.ingest as ingest

    monkeypatch.setattr(ingest, "save_connection", Mock(return_value="c1"))
    body = {"url": "https://m.test", "username": "u", "password": "p", "track_ids": ["2", "1"]}
    assert client.post("/api/v1/sync/ingest", json=body).status_code == 202
    assert enqueue.call_args.args == ("import", "analysis", user["id"])
    assert enqueue.call_args.kwargs["connection_id"] == "c1"
    assert enqueue.call_args.kwargs["payload"] == {"track_ids": ["1", "2"]}


def test_sync_endpoints_need_a_session(anonymous):
    assert anonymous.get(f"/api/v1/sync/connections/{uuid4()}/status").status_code == 401
    assert anonymous.get("/api/v1/sync/semantic-fusion").status_code == 401
    assert anonymous.post("/api/v1/sync/audio-profiles").status_code == 401


def test_a_full_sync_can_redo_pipelines_for_every_song(client, monkeypatch, enqueue, owned):
    monkeypatch.setattr(start, "compute_for", Mock(return_value="local"))
    url = f"/api/v1/sync/connections/{uuid4()}"
    body = {"mode": "all", "resync": ["themes", "karaoke", "karaoke"]}
    assert client.post(url, json=body).status_code == 202
    assert enqueue.call_args.kwargs["payload"]["resync"] == ["karaoke", "themes"]
    # A transcription needs a language, so it is redone per song only.
    assert client.post(url, json={"resync": ["transcription"]}).status_code == 422
