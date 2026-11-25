"""``PUT /library/tracks/{track_id}/lyrics/status``."""

from __future__ import annotations

import uuid

from fastapi import Depends
from pydantic import Field

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.requests import PutRequest
from backend.api.responses import ResponseModel
from backend.library.db import connect, require_visible

UPSERT_SQL = """INSERT INTO lyrics (track_id, source, text, provenance, availability_status)
   VALUES (%s,'none',NULL,jsonb_build_object('manual_status',%s::text,'edited_at',now()::text),%s)
   ON CONFLICT (track_id) DO UPDATE SET source='none', text=NULL, language=NULL,
     provenance=(coalesce(lyrics.provenance,'{}'::jsonb)
       - 'manual' - 'lines' - 'synced' - 'forced_transcription' - 'transcription_language')
       || jsonb_build_object('manual_status',EXCLUDED.availability_status,'edited_at',now()::text),
     availability_status=EXCLUDED.availability_status, created_at=now()"""


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
        with connect() as connection, connection.cursor() as cursor:
            require_visible(cursor, user["id"], track_id, "Track is not in your library")
            cursor.execute(UPSERT_SQL, (track_id, body.status, body.status))
            cursor.execute("DELETE FROM karaoke_lyrics_variants WHERE track_id=%s", (track_id,))
            cursor.execute(
                "DELETE FROM embeddings WHERE track_id=%s AND embedding_type='lyrics'", (track_id,)
            )
        return {"track_id": str(track_id), "status": body.status}
