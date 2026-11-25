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
        with connect(dict_rows=True) as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT ts.source_data->>'suffix' AS codec,
                          ts.source_data->>'contentType' AS content_type,
                          (ts.source_data->>'bitRate')::integer AS bit_rate_kbps,
                          (ts.source_data->>'bitDepth')::integer AS bit_depth,
                          (ts.source_data->>'samplingRate')::integer AS sample_rate_hz,
                          (ts.source_data->>'channelCount')::integer AS channels
                   FROM user_track_links link
                   JOIN track_sources ts ON ts.track_id=link.track_id AND ts.library_id=link.library_id
                                        AND ts.external_id=link.external_id
                   WHERE link.user_id=%s AND link.track_id=%s
                   ORDER BY ts.id LIMIT 1""",
                (user["id"], track_id),
            )
            row = cursor.fetchone()
        if row is None:
            raise NotFoundError("Track is not in your library")
        codec = str(row.get("codec") or "").lower()
        return {**row, "lossless": codec in LOSSLESS_CODECS}
