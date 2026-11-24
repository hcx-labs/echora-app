"""Thin authenticated HTTP routes; all audio inference runs in the analysis worker."""

from __future__ import annotations

import asyncio
from uuid import UUID
from urllib.parse import unquote

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool

import worker.recordings.calibration as recording_calibration
import worker.recordings.search as recording_search
from core.settings import get_settings


async def _audio_body(request: Request) -> bytes:
    content = bytearray()
    try:
        async with asyncio.timeout(get_settings().recording_upload_read_timeout_seconds):
            async for chunk in request.stream():
                if len(content) + len(chunk) > get_settings().recording_max_upload_bytes:
                    raise HTTPException(413, "The recording exceeds the configured upload limit")
                content.extend(chunk)
    except TimeoutError:
        raise HTTPException(408, "Recording upload timed out") from None
    if not content:
        raise HTTPException(422, "The recording is empty")
    return bytes(content)


def create_router(user_dependency) -> APIRouter:
    router = APIRouter(prefix="/library/recording", tags=["recording search"])

    @router.get("/status")
    def status(user=Depends(user_dependency)):
        return recording_search.status(user["id"])

    @router.post("/search", status_code=202)
    async def search(request: Request, user=Depends(user_dependency)):
        try:
            config, _, policy_id = await run_in_threadpool(recording_search.search_configuration)
        except (ValueError, OSError):
            raise HTTPException(503, "Recording search is not configured") from None
        content = await _audio_body(request)
        try:
            return await run_in_threadpool(
                recording_search.enqueue,
                user["id"],
                bytes(content),
                config.representation_id,
                policy_id,
            )
        except recording_search.QueueFull:
            raise HTTPException(
                429, "Recording search is busy. Try again shortly.", headers={"Retry-After": "15"}
            ) from None

    @router.get("/search/{job_id}")
    def result(job_id: UUID, user=Depends(user_dependency)):
        value = recording_search.result(job_id, user["id"])
        if value is None:
            raise HTTPException(404, "Recording search not found")
        return value

    @router.delete("/search/{job_id}")
    def cancel(job_id: UUID, user=Depends(user_dependency)):
        import core.jobs.queue as jobs

        job = jobs.get_job(job_id, user["id"])
        if job is None or job["kind"] != "recording_search":
            raise HTTPException(404, "Recording search not found")
        jobs.cancel(job_id, user["id"])
        return {"cancel_requested": True}

    @router.post("/calibration/samples", status_code=201)
    async def save_calibration(
        request: Request,
        expected_track_id: UUID | None = None,
        not_in_library: bool = False,
        encoded_notes: str = Header(default="", alias="X-Echora-Calibration-Notes"),
        consent: str | None = Header(default=None, alias="X-Echora-Calibration-Consent"),
        user=Depends(user_dependency),
    ):
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
        content = await _audio_body(request)
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
        except recording_calibration.CalibrationDisabled:
            raise HTTPException(503, "Calibration collection is disabled") from None
        except recording_calibration.TrackUnavailable:
            raise HTTPException(404, "This song is not available in your library") from None
        except recording_search.QueueFull:
            raise HTTPException(
                429, "Calibration storage is full or busy. Delete a saved clip or try later."
            ) from None
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        except OSError:
            raise HTTPException(503, "Calibration storage is unavailable") from None

    @router.get("/calibration/samples")
    def calibration_samples(user=Depends(user_dependency)):
        return recording_calibration.list_samples(user["id"])

    @router.delete("/calibration/samples/{sample_id}", status_code=204)
    def delete_calibration(sample_id: UUID, user=Depends(user_dependency)):
        if not recording_calibration.delete(sample_id, user["id"]):
            raise HTTPException(404, "Calibration clip not found")
        return Response(status_code=204)

    return router
