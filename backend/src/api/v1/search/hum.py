"""``POST /search/hum``."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import Depends, HTTPException, Request

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.search.hum import HumQuery, HumSearch
from backend.search.service import run_search
from core.errors import InvalidQueryError

logger = logging.getLogger(__name__)

MAX_RECORDING_BYTES = 8 * 1024 * 1024


class HumMatch(ResponseModel):
    """A song whose melody matches, and where."""

    id: uuid.UUID
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    duration_seconds: float | None = None
    source_id: str | None = None
    cover_art: str | None = None
    similarity: float
    matched_at_seconds: float
    match_cost: float
    matched_source: str


class HumMatches(ResponseModel):
    """Best matches first."""

    tracks: list[HumMatch]
    query_seconds: float
    matcher: str | None = None
    candidate_contours: int | None = None
    query_windows: int | None = None
    diagnostic_id: str | None = None


class HumSearchEndpoint(Endpoint):
    """Find songs by humming: the body is the recording (any format ffmpeg reads, up to 8 MB)."""

    method = "POST"
    path = "/hum"
    response = HumMatches
    summary = "Find songs by humming"

    async def handle(
        self, request: Request, limit: int = 10, user: SessionUser = Depends(current_user)
    ) -> Any:
        audio = await request.body()
        if not audio:
            raise InvalidQueryError("The recording is empty")
        if len(audio) > MAX_RECORDING_BYTES:
            raise HTTPException(status_code=413, detail="The recording exceeds 8 MB")
        query = HumQuery(audio=audio, limit=min(max(limit, 1), 25))
        try:
            return await run_search(HumSearch, user["id"], query)
        except ValueError as error:
            raise InvalidQueryError(str(error)) from error
        except Exception as error:
            logger.exception("Hum search failed")
            raise HTTPException(
                status_code=500, detail="Hum search could not process this recording"
            ) from error
