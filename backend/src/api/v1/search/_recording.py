"""Shared by the recording search endpoints: the streamed upload and response models."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from fastapi import HTTPException, Request

from backend.api.responses import ResponseModel
from worker.recordings.config import RecordingConfig


async def audio_body(request: Request) -> bytes:
    """The uploaded clip, read with the configured size limit (413) and timeout (408)."""
    content = bytearray()
    try:
        async with asyncio.timeout(RecordingConfig.load().upload_read_timeout_seconds):
            async for chunk in request.stream():
                if len(content) + len(chunk) > RecordingConfig.load().max_upload_bytes:
                    raise HTTPException(413, "The recording exceeds the configured upload limit")
                content.extend(chunk)
    except TimeoutError:
        raise HTTPException(408, "Recording upload timed out") from None
    if not content:
        raise HTTPException(422, "The recording is empty")
    return bytes(content)


class RecordingAvailability(ResponseModel):
    """Whether recognition and calibration collection are available, and why not."""

    enabled: bool
    recognition_enabled: bool
    calibration_enabled: bool
    indexed_tracks: int
    reason: str | None = None


class RecordingQueued(ResponseModel):
    """The search job; poll ``GET /search/recording/{job_id}``."""

    job_id: str


class RecordingMatches(ResponseModel):
    """``state`` is the matcher's verdict; matches carry the track and its playback source."""

    state: str
    matches: list[dict[str, Any]]


class RecordingResult(ResponseModel):
    """A search job's status, its error, or (once complete) its result."""

    status: str
    error: str | None = None
    result: RecordingMatches | None = None


class CancelRequested(ResponseModel):
    """The search was asked to stop."""

    cancel_requested: bool


class CalibrationSample(ResponseModel):
    """A saved calibration clip (fields without a value are left out)."""

    id: str
    expected_track_id: str | None = None
    not_in_library: bool | None = None
    notes: str | None = None
    duration_seconds: float | None = None
    created_at: datetime | None = None
    expires_at: datetime | None = None
    expected_title: str | None = None
    expected_artist: str | None = None


class CalibrationSamples(ResponseModel):
    """The user's saved calibration clips."""

    samples: list[CalibrationSample]
