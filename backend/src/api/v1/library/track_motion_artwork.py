"""``GET /library/tracks/{track_id}/motion-artwork``."""

from __future__ import annotations

import uuid

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from core.db.connection import connect
from core.library import store
from worker.artwork.settings import track_artwork


class MotionArtwork(ResponseModel):
    """The song cover's looping video, when one has been generated."""

    track_id: str
    available: bool
    id: str | None = None
    url: str | None = None
    """The video, relative to ``/api/v1``: ``/media/motion-artwork/<id>.mp4``."""
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    fps: float | None = None


class TrackMotionArtwork(Endpoint):
    """The motion artwork of one of the user's songs."""

    method = "GET"
    path = "/tracks/{track_id}/motion-artwork"
    response = MotionArtwork

    def handle(self, track_id: uuid.UUID, user: SessionUser = Depends(current_user)) -> dict:
        with connect(dict_rows=True) as connection, connection.cursor() as cursor:
            store.require_visible(connection, user["id"], track_id, "Track not found")
            artwork = track_artwork(cursor, user["id"], track_id)
        if artwork is None:
            return {"track_id": str(track_id), "available": False}
        return {
            "track_id": str(track_id),
            "available": True,
            "id": str(artwork["id"]),
            "url": f"/media/motion-artwork/{artwork['id']}.mp4",
            "width": artwork["width"],
            "height": artwork["height"],
            "frames": artwork["frames"],
            "fps": artwork["fps"],
        }
