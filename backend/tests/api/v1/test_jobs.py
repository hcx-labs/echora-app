import uuid

import core.jobs.queue as jobs

JOB = {
    "id": "j1",
    "job_id": "j1",
    "kind": "navidrome_sync",
    "worker_type": "cpu",
    "connection_id": None,
    "parent_id": None,
    "status": "running",
    "cancel_requested": False,
    "progress": {"phase": "lyrics"},
    "summary": None,
    "error": None,
    "created_at": "2026-11-25T00:00:00+00:00",
    "updated_at": None,
    "finished_at": None,
    "dismissed_at": None,
    "phase": "lyrics",
    "completed": 0,
    "total": 0,
    "existing": False,
}
JOB_ID = uuid.UUID(int=7)


def test_list_clamps_the_limit(client, user, monkeypatch):
    calls = []
    monkeypatch.setattr(jobs, "list_jobs", lambda *a, **k: calls.append((a, k)) or [JOB])
    response = client.get("/api/v1/jobs", params={"limit": 500, "active_only": True})
    assert response.json() == {"jobs": [JOB]}
    assert calls == [
        (
            (user["id"],),
            {"connection_id": None, "active_only": True, "limit": 100, "library_only": False},
        )
    ]


def test_get_missing_job_is_404(client, monkeypatch):
    monkeypatch.setattr(jobs, "get_job", lambda job_id, user_id: None)
    response = client.get(f"/api/v1/jobs/{JOB_ID}")
    assert response.status_code == 404
    assert response.json() == {"detail": "Job not found"}


def test_get_job(client, monkeypatch):
    monkeypatch.setattr(jobs, "get_job", lambda job_id, user_id: JOB)
    assert client.get(f"/api/v1/jobs/{JOB_ID}").json() == JOB


def test_cancel(client, monkeypatch):
    monkeypatch.setattr(jobs, "cancel", lambda job_id, user_id: {**JOB, "cancel_requested": True})
    assert client.post(f"/api/v1/jobs/{JOB_ID}/cancel").json()["cancel_requested"] is True


def test_dismiss_active_job_conflicts(client, monkeypatch):
    def refuse(job_id, user_id):
        raise ValueError("Active jobs cannot be dismissed")

    monkeypatch.setattr(jobs, "dismiss", refuse)
    response = client.post(f"/api/v1/jobs/{JOB_ID}/dismiss")
    assert response.status_code == 409
    assert response.json() == {"detail": "Active jobs cannot be dismissed"}


def test_batches(client, monkeypatch):
    page = {"batches": [{**JOB, "batch_number": 1, "track_count": 3}], "total": 1}
    monkeypatch.setattr(jobs, "list_batches", lambda *a, **k: page)
    assert client.get(f"/api/v1/jobs/{JOB_ID}/batches").json() == page


def test_jobs_need_a_session(anonymous):
    response = anonymous.get("/api/v1/jobs")
    assert response.status_code == 401
