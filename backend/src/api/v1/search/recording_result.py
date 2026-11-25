"""``GET /search/recording/{job_id}``."""

from __future__ import annotations

import uuid

from fastapi import Depends

import worker.recordings.search as recording_search
from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.search._recording import RecordingResult as RecordingResultModel
from core.errors import NotFoundError


class RecordingResult(Endpoint):
    """A recording search's state, and its matches once complete (rechecked for access)."""

    method = "GET"
    path = "/recording/{job_id}"
    response = RecordingResultModel

    def handle(self, job_id: uuid.UUID, user: SessionUser = Depends(current_user)) -> dict:
        value = recording_search.result(job_id, user["id"])
        if value is None:
            raise NotFoundError("Recording search not found")
        return value
