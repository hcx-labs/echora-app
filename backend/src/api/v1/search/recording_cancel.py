"""``DELETE /search/recording/{job_id}``."""

from __future__ import annotations

import uuid

from fastapi import Depends

import core.jobs.queue as jobs
from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.search._recording import CancelRequested
from core.errors import NotFoundError


class RecordingCancel(Endpoint):
    """Stop one of the user's recording searches."""

    method = "DELETE"
    path = "/recording/{job_id}"
    response = CancelRequested

    def handle(self, job_id: uuid.UUID, user: SessionUser = Depends(current_user)) -> dict:
        job = jobs.get_job(job_id, user["id"])
        if job is None or job["kind"] != "recording_search":
            raise NotFoundError("Recording search not found")
        jobs.cancel(job_id, user["id"])
        return {"cancel_requested": True}
