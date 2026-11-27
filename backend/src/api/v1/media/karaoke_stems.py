"""``GET /media/karaoke-stems/{stems_id}.webm``."""

from __future__ import annotations

import uuid

from fastapi import Depends
from fastapi.responses import FileResponse

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from core.db.connection import connect
from core.errors import NotFoundError
from core.stems import store
from core.stems.files import stems_path


class KaraokeStemsAudio(Endpoint):
    """A song's karaoke stems, served only for songs in the caller's own library.

    Four-channel Opus in WebM: vocals left and right, then instrumental left and right. Range
    requests are answered, so the player can seek.
    """

    method = "GET"
    path = "/karaoke-stems/{stems_id}.webm"

    def handle(
        self, stems_id: uuid.UUID, user: SessionUser = Depends(current_user)
    ) -> FileResponse:
        with connect() as connection:
            stored = store.visible_stems_path(connection, stems_id, user["id"])
        if stored is None:
            raise NotFoundError("Karaoke stems not found")
        try:
            path = stems_path(stored)
        except ValueError:
            raise NotFoundError("Karaoke stems not found") from None
        if not path.is_file():
            raise NotFoundError("Karaoke stems not found")
        # The file for an id never changes, so browsers may keep it.
        return FileResponse(
            path,
            media_type="audio/webm",
            headers={"Cache-Control": "private, max-age=604800, immutable"},
        )
