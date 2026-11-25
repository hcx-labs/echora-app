"""``POST /jobs/{job_id}/dismiss``."""

from __future__ import annotations

import uuid

from fastapi import Depends

import core.jobs.queue as jobs
from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import JobSnapshot
from core.errors import ConflictError, NotFoundError


class DismissJob(Endpoint):
    """Acknowledge a finished job, keeping its history."""

    method = "POST"
    path = "/{job_id}/dismiss"
    response = JobSnapshot

    def handle(self, job_id: uuid.UUID, user: SessionUser = Depends(current_user)) -> dict:
        try:
            value = jobs.dismiss(job_id, user["id"])
        except ValueError as error:
            raise ConflictError(str(error)) from error
        if value is None:
            raise NotFoundError("Job not found")
        return value
