"""``GET /search/recording/calibration/samples``."""

from __future__ import annotations

from fastapi import Depends

import worker.recordings.calibration as recording_calibration
from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.search._recording import CalibrationSamples as CalibrationSampleList


class CalibrationSamples(Endpoint):
    """The user's saved calibration clips."""

    method = "GET"
    path = "/recording/calibration/samples"
    response = CalibrationSampleList

    def handle(self, user: SessionUser = Depends(current_user)) -> dict:
        return recording_calibration.list_samples(user["id"])
