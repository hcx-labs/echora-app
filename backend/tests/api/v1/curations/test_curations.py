"""Curation endpoints: locking, ownership, status codes and the queued refresh."""

import contextlib
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import IntegrityError

import worker.curation.jobs as curation_jobs
from backend.api.v1.curations import create, delete, update

CURATION = uuid.UUID(int=42)
LOCK = object()


@pytest.fixture
def lock(monkeypatch):
    """Record entering and leaving ``curation_jobs.locked`` and every mutability check."""
    events = []

    @contextlib.contextmanager
    def locked(curation_id):
        events.append(("enter", curation_id))
        yield LOCK
        events.append(("exit", curation_id))

    monkeypatch.setattr(curation_jobs, "locked", locked)
    monkeypatch.setattr(
        curation_jobs,
        "assert_mutable",
        lambda connection, curation_id, **kw: events.append(("mutable", connection, kw)),
    )
    return events


def sessions(monkeypatch, module, curation):
    """``session_scope`` of ``module`` yields a session holding ``curation`` (or none)."""
    deleted = []

    @contextlib.contextmanager
    def session_scope():
        yield SimpleNamespace(
            scalar=lambda query: curation, delete=deleted.append, add=lambda v: None
        )

    monkeypatch.setattr(module, "session_scope", session_scope)
    return deleted


def test_update_runs_inside_the_curation_lock(client, user, lock, monkeypatch):
    curation = SimpleNamespace(refresh_interval_hours=24, next_refresh_at=None)
    sessions(monkeypatch, update, curation)
    response = client.patch(
        f"/api/v1/curations/{CURATION}", json={"refresh_enabled": True, "track_limit": 40}
    )
    assert response.status_code == 200
    assert response.json() == {"id": str(CURATION), "refresh_enabled": True, "track_limit": 40}
    assert lock == [("enter", CURATION), ("mutable", LOCK, {}), ("exit", CURATION)]
    assert curation.track_limit == 40 and curation.next_refresh_at is not None


def test_update_of_a_missing_curation_is_404_inside_the_lock(client, lock, monkeypatch):
    sessions(monkeypatch, update, None)
    response = client.patch(f"/api/v1/curations/{CURATION}", json={"refresh_enabled": False})
    assert response.status_code == 404
    assert response.json() == {"detail": "Curation not found"}
    assert lock == [("enter", CURATION)]


def test_update_rejects_unknown_fields(client, lock):
    response = client.patch(f"/api/v1/curations/{CURATION}", json={"name": "renamed"})
    assert response.status_code == 422
    assert lock == []


def test_delete_runs_inside_the_curation_lock(client, lock, monkeypatch):
    curation = SimpleNamespace(navidrome_connection_id=None, navidrome_playlist_id=None)
    deleted = sessions(monkeypatch, delete, curation)
    response = client.delete(f"/api/v1/curations/{CURATION}?delete_navidrome=false")
    assert response.status_code == 204
    assert deleted == [curation]
    assert lock == [
        ("enter", CURATION),
        ("mutable", LOCK, {"deleting": True, "delete_remote": False}),
        ("exit", CURATION),
    ]


def test_delete_keeps_the_curation_when_navidrome_is_unreachable(client, lock, monkeypatch):
    curation = SimpleNamespace(navidrome_connection_id=uuid.uuid4(), navidrome_playlist_id="p1")
    deleted = sessions(monkeypatch, delete, curation)
    monkeypatch.setattr(delete, "load_connection", lambda connection_id: None)
    response = client.delete(f"/api/v1/curations/{CURATION}")
    assert response.status_code == 409
    assert deleted == []


def test_refresh_queues_a_job_and_is_202(client, user, monkeypatch):
    calls = []
    queued = {"job_id": "j1", "status": "queued", "curation_id": str(CURATION)}
    monkeypatch.setattr(curation_jobs, "enqueue", lambda *a: calls.append(a) or queued)
    response = client.post(f"/api/v1/curations/{CURATION}/refresh")
    assert response.status_code == 202
    assert response.json() == queued
    assert calls == [(CURATION, user["id"])]


def test_create_needs_a_connection(client):
    response = client.post("/api/v1/curations", json={"name": "Late night"})
    assert response.status_code == 409
    assert response.json() == {"detail": "No Navidrome connection is configured"}


def test_create_saves_and_queues_the_first_refresh(client, user, monkeypatch):
    user["navidrome_connection_id"] = uuid.uuid4()
    added = []

    @contextlib.contextmanager
    def session_scope():
        def add(curation):
            curation.id = CURATION
            added.append(curation)

        yield SimpleNamespace(add=add, flush=lambda: None)

    monkeypatch.setattr(create, "session_scope", session_scope)
    queued = {"job_id": "j1", "status": "queued", "curation_id": str(CURATION)}
    monkeypatch.setattr(curation_jobs, "enqueue", lambda *a: queued)
    response = client.post(
        "/api/v1/curations", json={"name": "  Late night ", "positive_prompt": " neon "}
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["id"] == str(CURATION) and body["tracks"] == [] and body["job_id"] == "j1"
    assert body["name"] == "  Late night "  # the response echoes the request, as before
    assert added[0].name == "Late night" and added[0].positive_prompt == "neon"


def test_duplicate_names_are_409(client, user, monkeypatch):
    user["navidrome_connection_id"] = uuid.uuid4()

    @contextlib.contextmanager
    def session_scope():
        yield SimpleNamespace(add=lambda curation: None, flush=lambda: None)
        raise IntegrityError("INSERT", {}, Exception("curations_user_name_idx"))

    monkeypatch.setattr(create, "session_scope", session_scope)
    response = client.post("/api/v1/curations", json={"name": "Taken"})
    assert response.status_code == 409
    assert response.json() == {"detail": "A curation with this name already exists"}


def test_preview_needs_a_connection(client):
    assert client.post("/api/v1/curations/preview", json={}).status_code == 409


def test_journey_endpoints_must_differ(client):
    track = str(uuid.uuid4())
    response = client.post(
        "/api/v1/curations/journeys/preview",
        json={"start_track_id": track, "end_track_id": track},
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "Journey endpoints must be different tracks"}
