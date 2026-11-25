"""``DELETE /search/recording/calibration/samples/{sample_id}``."""

from __future__ import annotations

import uuid

from fastapi import Depends, Response

import worker.recordings.calibration as recording_calibration
from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from core.errors import NotFoundError


class CalibrationSampleDelete(Endpoint):
    """Delete a saved calibration clip and its audio."""

    method = "DELETE"
    path = "/recording/calibration/samples/{sample_id}"
    status_code = 204

    def handle(self, sample_id: uuid.UUID, user: SessionUser = Depends(current_user)) -> Response:
        if not recording_calibration.delete(sample_id, user["id"]):
            raise NotFoundError("Calibration clip not found")
        return Response(status_code=204)
