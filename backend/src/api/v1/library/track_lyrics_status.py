"""``PUT /library/tracks/{track_id}/lyrics/status``."""

from __future__ import annotations

import uuid

from fastapi import Depends
from pydantic import Field

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.requests import PutRequest
from backend.api.responses import ResponseModel
from core.db.connection import connect
from core.library import store as library
from core.lyrics import store as lyrics


class LyricsStatusUpdate(PutRequest):
    """Mark the song instrumental, or as missing lyrics."""

    status: str = Field(pattern="^(instrumental|missing)$")


class LyricsStatusUpdated(ResponseModel):
    """The status now stored."""

    track_id: str
    status: str


class TrackLyricsStatus(Endpoint):
    """Clear a song's lyrics and set its lyrics status."""

    method = "PUT"
    path = "/tracks/{track_id}/lyrics/status"
    request = LyricsStatusUpdate
    response = LyricsStatusUpdated

    def handle(
        self,
        track_id: uuid.UUID,
        body: LyricsStatusUpdate,
        user: SessionUser = Depends(current_user),
    ) -> dict:
        with connect() as connection:
            library.require_visible(
                connection, user["id"], track_id, "Track is not in your library"
            )
            lyrics.set_manual_status(connection, track_id, body.status)
            lyrics.discard_derived(connection, track_id)
        return {"track_id": str(track_id), "status": body.status}
