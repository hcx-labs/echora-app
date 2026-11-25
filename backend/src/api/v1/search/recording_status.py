"""``GET /search/recording/status``."""

from __future__ import annotations

from fastapi import Depends

import worker.recordings.search as recording_search
from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.search._recording import RecordingAvailability


class RecordingStatus(Endpoint):
    """Whether recording search works here and how many of the user's songs it can find."""

    method = "GET"
    path = "/recording/status"
    response = RecordingAvailability

    def handle(self, user: SessionUser = Depends(current_user)) -> dict:
        return recording_search.status(user["id"])
