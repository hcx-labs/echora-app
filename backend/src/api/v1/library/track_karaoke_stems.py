"""``GET /library/tracks/{track_id}/karaoke-stems``."""

from __future__ import annotations

import uuid

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from core.db.connection import connect
from core.library import store as library
from core.stems import store


class KaraokeStems(ResponseModel):
    """The song's separated vocals and instrumental, when they have been stored."""

    track_id: str
    available: bool
    url: str | None = None
    """Relative to ``/api/v1``: ``/media/karaoke-stems/<id>.webm``. Four-channel Opus: vocals
    left and right, then instrumental left and right."""
    duration_seconds: float | None = None


class TrackKaraokeStems(Endpoint):
    """The karaoke stems of one of the user's songs."""

    method = "GET"
    path = "/tracks/{track_id}/karaoke-stems"
    response = KaraokeStems

    def handle(self, track_id: uuid.UUID, user: SessionUser = Depends(current_user)) -> dict:
        with connect() as connection:
            library.require_visible(connection, user["id"], track_id, "Track not found")
            stems = store.track_stems(connection, user["id"], track_id)
        if stems is None:
            return {"track_id": str(track_id), "available": False}
        return {
            "track_id": str(track_id),
            "available": True,
            "url": f"/media/karaoke-stems/{stems['id']}.webm",
            "duration_seconds": stems["duration_seconds"],
        }
