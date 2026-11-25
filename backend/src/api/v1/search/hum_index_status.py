"""``GET /search/hum/index``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from core.db.connection import connect
from core.library import store


class HumIndexState(ResponseModel):
    """How many of the user's songs have melody contours."""

    status: str
    """``complete`` when any song is indexed, else ``missing``."""
    indexed_tracks: int
    track_limit: int


class HumIndexStatus(Endpoint):
    """Report whether the user's hum index exists."""

    method = "GET"
    path = "/hum/index"
    response = HumIndexState

    def handle(self, user: SessionUser = Depends(current_user)) -> dict:
        with connect() as connection:
            indexed = store.songs_with_contours(connection, user["id"])
        return {
            "status": "complete" if indexed else "missing",
            "indexed_tracks": indexed,
            "track_limit": indexed,
        }
