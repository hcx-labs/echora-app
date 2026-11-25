"""A job snapshot as core.jobs.queue returns it."""


def snapshot(kind="navidrome_sync", **changes):
    job = {
        "id": "j1",
        "job_id": "j1",
        "kind": kind,
        "worker_type": "analysis",
        "connection_id": None,
        "parent_id": None,
        "status": "queued",
        "cancel_requested": False,
        "progress": {},
        "summary": None,
        "error": None,
        "created_at": "2026-11-25T00:00:00+00:00",
        "updated_at": None,
        "finished_at": None,
        "dismissed_at": None,
        "phase": "queued",
        "completed": 0,
        "total": 0,
        "existing": False,
    }
    return {**job, **changes}
