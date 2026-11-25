"""``GET /media/motion-artwork/{artwork_id}.mp4``."""

from __future__ import annotations

import uuid

from fastapi import Depends
from fastapi.responses import FileResponse

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from core.errors import NotFoundError
from worker.artwork.settings import _connect, artwork_path


class MotionArtworkVideo(Endpoint):
    """A motion artwork loop, served only for covers in the caller's own library."""

    method = "GET"
    path = "/motion-artwork/{artwork_id}.mp4"

    def handle(
        self, artwork_id: uuid.UUID, user: SessionUser = Depends(current_user)
    ) -> FileResponse:
        with _connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT ma.path FROM motion_artworks ma
                   WHERE ma.id = %s AND ma.status = 'complete' AND EXISTS (
                     SELECT 1 FROM track_motion_artwork tma
                     JOIN user_track_links utl ON utl.track_id = tma.track_id
                     WHERE tma.cover_sha256 = ma.cover_sha256 AND utl.user_id = %s)""",
                (artwork_id, user["id"]),
            )
            row = cursor.fetchone()
        if row is None:
            raise NotFoundError("Motion artwork not found")
        try:
            path = artwork_path(row["path"])
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
