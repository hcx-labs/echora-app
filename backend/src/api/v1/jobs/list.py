"""``GET /jobs``."""

from __future__ import annotations

import uuid

from fastapi import Depends

import core.jobs.queue as jobs
from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.jobs._models import JobList


class ListJobs(Endpoint):
    """List the user's jobs, optionally for one connection or only active ones."""

    method = "GET"
    path = ""
    response = JobList

    def handle(
        self,
        connection_id: uuid.UUID | None = None,
        active_only: bool = False,
        limit: int = 20,
        library_only: bool = False,
        user: SessionUser = Depends(current_user),
    ) -> dict[str, object]:
        return {
            "jobs": jobs.list_jobs(
                user["id"],
                connection_id=connection_id,
                active_only=active_only,
                limit=min(max(limit, 1), 100),
                library_only=library_only,
            )
        }
