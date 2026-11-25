"""``POST /jobs/{job_id}/cancel``."""

from __future__ import annotations

import uuid

from fastapi import Depends

import core.jobs.queue as jobs
from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import JobSnapshot
from core.errors import NotFoundError


class CancelJob(Endpoint):
    """Ask a job (and its batches) to stop."""

    method = "POST"
    path = "/{job_id}/cancel"
    response = JobSnapshot

    def handle(self, job_id: uuid.UUID, user: SessionUser = Depends(current_user)) -> dict:
        value = jobs.cancel(job_id, user["id"])
        if value is None:
            raise NotFoundError("Job not found")
        return value
