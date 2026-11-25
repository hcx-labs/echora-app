"""``GET /library/tracks/{track_id}/audio-quality``."""

from __future__ import annotations

import uuid

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.library.track_info import LOSSLESS_CODECS
from core.db.connection import connect
from core.errors import NotFoundError
from core.library import store


class AudioQuality(ResponseModel):
    """The song file's format, from Navidrome."""

    codec: str | None = None
    content_type: str | None = None
    bit_rate_kbps: int | None = None
    bit_depth: int | None = None
    sample_rate_hz: int | None = None
    channels: int | None = None
    lossless: bool


class TrackAudioQuality(Endpoint):
    """Codec, bit rate, depth, sample rate and channels of one of the user's songs."""

    method = "GET"
    path = "/tracks/{track_id}/audio-quality"
    response = AudioQuality

    def handle(self, track_id: uuid.UUID, user: SessionUser = Depends(current_user)) -> dict:
        with connect() as connection:
            row = store.audio_quality(connection, user["id"], track_id)
        if row is None:
            raise NotFoundError("Track is not in your library")
        codec = str(row.get("codec") or "").lower()
        return {**row, "lossless": codec in LOSSLESS_CODECS}
