"""``GET /jobs/{job_id}/batches``."""

from __future__ import annotations

import uuid

from fastapi import Depends

import core.jobs.queue as jobs
from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.jobs._models import BatchPage
from core.errors import NotFoundError


class JobBatches(Endpoint):
    """A page of a job's batches."""

    method = "GET"
    path = "/{job_id}/batches"
    response = BatchPage

    def handle(
        self,
        job_id: uuid.UUID,
        limit: int = 25,
        offset: int = 0,
        user: SessionUser = Depends(current_user),
    ) -> dict:
        result = jobs.list_batches(job_id, user["id"], limit=limit, offset=offset)
        if result is None:
            raise NotFoundError("Job not found")
        return result
