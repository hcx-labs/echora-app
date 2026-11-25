"""``POST /settings/external-processing/prepare``."""

from __future__ import annotations

from fastapi import Depends

import core.jobs.queue as jobs
from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.responses import JobSnapshot
from backend.api.v1.settings._deps import admin_required
from core.errors import ConflictError
from worker.remote.settings import SETUP_JOB, config


class PrepareModal(Endpoint):
    """Queue the Modal deployment (administrators only; one at a time)."""

    method = "POST"
    path = "/external-processing/prepare"
    response = JobSnapshot
    status_code = 202

    def handle(self, user: SessionUser = Depends(admin_required)) -> dict:
        if config() is None:
            raise ConflictError("Enable Modal and save a token first")
        return jobs.enqueue(SETUP_JOB, "analysis", user["id"], payload={}, dedupe_key=SETUP_JOB)
