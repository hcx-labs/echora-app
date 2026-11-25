"""``PUT /library/tracks/{track_id}/lyrics/transcription-language``."""

from __future__ import annotations

import uuid

from fastapi import Depends
from pydantic import Field

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.requests import PutRequest
from backend.api.responses import ResponseModel
from backend.library.db import require_visible
from core.db.connection import connect
from core.errors import ConflictError

UPSERT_SQL = """INSERT INTO lyrics (track_id, source, text, provenance, availability_status)
   VALUES (%s,'none',NULL,jsonb_build_object('transcription_language',%s::text,'forced_transcription',true),'missing')
   ON CONFLICT (track_id) DO UPDATE SET source='none', text=NULL, language=NULL,
     provenance=(coalesce(lyrics.provenance,'{}'::jsonb) - 'manual' - 'lines' - 'synced')
       || jsonb_build_object('transcription_language',%s::text,'forced_transcription',true),
     availability_status='missing', created_at=now()"""


class TranscriptionLanguage(PutRequest):
    """The language to transcribe the song in (``en``, ``ja``, ``pt-BR``...)."""

    language: str = Field(pattern="^[a-z]{2,3}(?:-[A-Z]{2})?$")


class TranscriptionQueued(ResponseModel):
    """The song is transcribed by the next sync where AI lyrics are on."""

    track_id: str
    language: str
    transcription_pending: bool


class TrackTranscriptionLanguage(Endpoint):
    """Discard a song's lyrics and have them transcribed in the given language."""

    method = "PUT"
    path = "/tracks/{track_id}/lyrics/transcription-language"
    request = TranscriptionLanguage
    response = TranscriptionQueued

    def handle(
        self,
        track_id: uuid.UUID,
        body: TranscriptionLanguage,
        user: SessionUser = Depends(current_user),
    ) -> dict:
        language = body.language.strip()
        with connect(dict_rows=True) as connection, connection.cursor() as cursor:
            require_visible(cursor, user["id"], track_id, "Track is not in your library")
            # The track is transcribed by the next sync that runs where generation is on.
            cursor.execute(
                """SELECT transcription_processing_enabled OR transcription_modal_enabled AS enabled
                   FROM analysis_settings WHERE singleton=true"""
            )
            setting = cursor.fetchone()
            if not setting or not setting["enabled"]:
                raise ConflictError(
                    "Enable AI lyric generation in Settings before forcing transcription"
                )
            cursor.execute(UPSERT_SQL, (track_id, language, language))
            cursor.execute("DELETE FROM karaoke_lyrics_variants WHERE track_id=%s", (track_id,))
            cursor.execute(
                "DELETE FROM embeddings WHERE track_id=%s AND embedding_type='lyrics'", (track_id,)
            )
        return {"track_id": str(track_id), "language": language, "transcription_pending": True}
