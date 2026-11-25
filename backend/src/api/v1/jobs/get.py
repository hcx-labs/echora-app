"""``GET /jobs/{job_id}``."""

from __future__ import annotations

import uuid

from fastapi import Depends

import core.jobs.queue as jobs
from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import JobSnapshot
from core.errors import NotFoundError


class GetJob(Endpoint):
    """One of the user's jobs."""

    method = "GET"
    path = "/{job_id}"
    response = JobSnapshot

    def handle(self, job_id: uuid.UUID, user: SessionUser = Depends(current_user)) -> dict:
        value = jobs.get_job(job_id, user["id"])
        if value is None:
            raise NotFoundError("Job not found")
        return value
