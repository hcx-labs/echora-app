"""``POST /search/recording``."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

import worker.recordings.search as recording_search
from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.search._recording import RecordingQueued, audio_body
from worker.recordings.errors import QueueFull


class RecordingSearch(Endpoint):
    """Queue a search for a recorded clip (the body); the worker runs it, nothing here infers."""

    method = "POST"
    path = "/recording"
    response = RecordingQueued
    status_code = 202

    async def handle(self, request: Request, user: SessionUser = Depends(current_user)) -> dict:
        try:
            config, _, policy_id = await run_in_threadpool(recording_search.search_configuration)
        except (ValueError, OSError):
            raise HTTPException(503, "Recording search is not configured") from None
        content = await audio_body(request)
        try:
            return await run_in_threadpool(
                recording_search.enqueue,
                user["id"],
                bytes(content),
                config.representation_id,
                policy_id,
            )
        except QueueFull:
            raise HTTPException(
                429, "Recording search is busy. Try again shortly.", headers={"Retry-After": "15"}
            ) from None
