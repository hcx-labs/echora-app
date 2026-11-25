"""``PUT /library/tracks/{track_id}/lyrics/transcription-language``."""

from __future__ import annotations

import uuid

from fastapi import Depends
from pydantic import Field

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.requests import PutRequest
from backend.api.responses import ResponseModel
from core.db.connection import connect
from core.errors import ConflictError
from core.library import store as library
from core.lyrics import store as lyrics
from core.settings import store as settings


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
        with connect() as connection:
            library.require_visible(
                connection, user["id"], track_id, "Track is not in your library"
            )
            # The track is transcribed by the next sync that runs where generation is on.
            if not settings.transcription_enabled(connection):
                raise ConflictError(
                    "Enable AI lyric generation in Settings before forcing transcription"
                )
            lyrics.force_transcription(connection, track_id, language)
            lyrics.discard_derived(connection, track_id)
        return {"track_id": str(track_id), "language": language, "transcription_pending": True}
