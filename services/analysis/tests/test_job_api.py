"""API contract tests without inference, Navidrome, or live database access."""

import os
from uuid import uuid4
from unittest.mock import Mock

import pytest


@pytest.fixture
def api(monkeypatch):
    # SQLAlchemy creates its engine on import; these tests never connect to it.
    monkeypatch.setenv(
        "DATABASE_URL", os.getenv("TEST_DATABASE_URL", "postgresql://localhost/unused")
    )
    from echora_analysis import main
    from fastapi.testclient import TestClient

    user_id = uuid4()
    monkeypatch.setattr(main, "_session_user", lambda _: {"id": user_id})
    return main, TestClient(main.app), user_id


def test_job_get_and_cancel_are_owner_scoped(api, monkeypatch):
    main, client, user_id = api
    job_id = uuid4()
    get = Mock(return_value=None)
    cancel = Mock(return_value=None)
    monkeypatch.setattr(main.jobs, "get_job", get)
    monkeypatch.setattr(main.jobs, "cancel", cancel)
    assert client.get(f"/jobs/{job_id}").status_code == 404
    assert client.post(f"/jobs/{job_id}/cancel").status_code == 404
    get.assert_called_once_with(job_id, user_id)
    cancel.assert_called_once_with(job_id, user_id)
    assert client.get("/jobs/not-a-uuid").status_code == 422


def test_list_is_owner_scoped_and_bounded(api, monkeypatch):
    main, client, user_id = api
    connection_id = uuid4()
    listing = Mock(return_value=[])
    monkeypatch.setattr(main.jobs, "list_jobs", listing)
    response = client.get(f"/jobs?connection_id={connection_id}&active_only=true&limit=10000")
    assert response.json() == {"jobs": []}
    listing.assert_called_once_with(
        user_id, connection_id=connection_id, active_only=True, limit=100, library_only=False
    )


def test_sync_returns_before_catalog_scan_and_never_queues_secrets(api, monkeypatch):
    main, client, user_id = api
    connection_id = uuid4()
    monkeypatch.setattr(
        main, "_load_connection", Mock(return_value=("https://music", "user", "secret"))
    )
    client_factory = Mock(side_effect=AssertionError("API must not scan Navidrome"))
    monkeypatch.setattr(main, "NavidromeClient", client_factory)
    enqueue = Mock(return_value={"job_id": str(uuid4()), "status": "queued"})
    monkeypatch.setattr(main.jobs, "enqueue", enqueue)
    compute = Mock(return_value="local")
    monkeypatch.setattr(main, "compute_for", compute)
    response = client.post(f"/navidrome/connections/{connection_id}/sync", json={"mode": "missing"})
    assert response.status_code == 202
    assert enqueue.call_args.args == ("navidrome_sync", "analysis", user_id)
    compute.assert_called_once_with(user_id, None)
    assert enqueue.call_args.kwargs["payload"] == {
        "mode": "missing",
        "verify_audio_hashes": True,
        "compute": "local",
    }
    assert "secret" not in repr(enqueue.call_args)
    client_factory.assert_not_called()


def test_sync_rejects_foreign_connection(api, monkeypatch):
    main, client, user_id = api
    connection_id = uuid4()
    load = Mock(return_value=None)
    enqueue = Mock()
    monkeypatch.setattr(main, "_load_connection", load)
    monkeypatch.setattr(main.jobs, "enqueue", enqueue)
    assert client.post(f"/navidrome/connections/{connection_id}/sync", json={}).status_code == 404
    load.assert_called_once_with(str(connection_id), user_id)
    enqueue.assert_not_called()


def test_batch_endpoint_enforces_parent_ownership(api, monkeypatch):
    main, client, owner = api
    identifier = uuid4()
    listing = Mock(return_value=None)
    monkeypatch.setattr(main.jobs, "list_batches", listing)
    assert client.get(f"/jobs/{identifier}/batches?limit=10&offset=20").status_code == 404
    listing.assert_called_once_with(identifier, owner, limit=10, offset=20)
    listing.return_value = {"batches": [], "total": 0}
    assert client.get(f"/jobs/{identifier}/batches").json() == {"batches": [], "total": 0}
