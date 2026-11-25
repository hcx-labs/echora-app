"""``POST /search/recording/calibration/samples``."""

from __future__ import annotations

import uuid
from urllib.parse import unquote

from fastapi import Depends, Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool

import worker.recordings.calibration as recording_calibration
from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.search._recording import CalibrationSample, audio_body
from worker.recordings.errors import CalibrationDisabled, QueueFull, TrackUnavailable


class CalibrationSampleSave(Endpoint):
    """Save a clip (the body) for calibrating recognition, with explicit consent.

    The clip names the song it is (``expected_track_id``) or is marked ``not_in_library``. Notes
    come URL-encoded in ``X-Echora-Calibration-Notes`` (at most 300 characters).
    """

    method = "POST"
    path = "/recording/calibration/samples"
    response = CalibrationSample
    status_code = 201

    async def handle(
        self,
        request: Request,
        expected_track_id: uuid.UUID | None = None,
        not_in_library: bool = False,
        encoded_notes: str = Header(default="", alias="X-Echora-Calibration-Notes"),
        consent: str | None = Header(default=None, alias="X-Echora-Calibration-Consent"),
        user: SessionUser = Depends(current_user),
    ) -> dict:
        if not recording_calibration.enabled():
            raise HTTPException(503, "Calibration collection is disabled")
        if consent != recording_calibration.CONSENT_VERSION:
            raise HTTPException(422, "Explicit consent to save this calibration clip is required")
        if (expected_track_id is not None) == not_in_library:
            raise HTTPException(
                422, "Choose the actual song or mark the clip as not in your library"
            )
        if len(encoded_notes) > 3600:
            raise HTTPException(422, "Calibration notes must be at most 300 characters")
        try:
            notes = unquote(encoded_notes, errors="strict")
        except UnicodeError:
            raise HTTPException(422, "Invalid calibration notes") from None
        if len(notes) > 300:
            raise HTTPException(422, "Calibration notes must be at most 300 characters")
        content = await audio_body(request)
        try:
            return await run_in_threadpool(
                recording_calibration.save,
                user["id"],
                content,
                expected_track_id=expected_track_id,
                not_in_library=not_in_library,
                notes=notes,
                consent=consent,
            )
        except CalibrationDisabled:
            raise HTTPException(503, "Calibration collection is disabled") from None
        except TrackUnavailable:
            raise HTTPException(404, "This song is not available in your library") from None
        except QueueFull:
            raise HTTPException(
                429, "Calibration storage is full or busy. Delete a saved clip or try later."
            ) from None
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        except OSError:
            raise HTTPException(503, "Calibration storage is unavailable") from None
