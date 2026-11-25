"""``GET /media/motion-artwork/{artwork_id}.mp4``."""

from __future__ import annotations

import uuid

from fastapi import Depends
from fastapi.responses import FileResponse

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from core.artwork import store
from core.db.connection import connect
from core.errors import NotFoundError
from worker.artwork.settings import artwork_path


class MotionArtworkVideo(Endpoint):
    """A motion artwork loop, served only for covers in the caller's own library."""

    method = "GET"
    path = "/motion-artwork/{artwork_id}.mp4"

    def handle(
        self, artwork_id: uuid.UUID, user: SessionUser = Depends(current_user)
    ) -> FileResponse:
        with connect() as connection:
            stored = store.visible_loop_path(connection, artwork_id, user["id"])
        if stored is None:
            raise NotFoundError("Motion artwork not found")
        try:
            path = artwork_path(stored)
        except ValueError:
            raise NotFoundError("Motion artwork not found") from None
        if not path.is_file():
            raise NotFoundError("Motion artwork not found")
        # The file for an id never changes, so browsers may keep it.
        return FileResponse(
            path,
            media_type="video/mp4",
            headers={"Cache-Control": "private, max-age=604800, immutable"},
        )
