"""``PUT`` / ``DELETE /library/tracks/{track_id}/processing/{pipeline}``."""

from __future__ import annotations

import uuid
from typing import Literal

from fastapi import Depends
from pydantic import Field

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.requests import PutRequest
from backend.api.responses import ResponseModel
from core.db.connection import connect
from core.errors import ConflictError
from core.library import store as library
from core.resync import store as resync
from core.settings import store as settings

Pipeline = Literal["transcription", "karaoke", "karaoke_stems", "translation", "themes"]


class ResyncBody(PutRequest):
    """Options of the resync."""

    language: str | None = Field(default=None, pattern="^[a-z]{2,3}(?:-[A-Z]{2})?$")
    """For ``transcription``: the language to transcribe the song in (``en``, ``pt-BR``)."""


class ResyncState(ResponseModel):
    """Whether the pipeline is now asked for."""

    track_id: str
    pipeline: str
    requested: bool


class TrackResync(Endpoint):
    """Have the next sync, partial or full, redo one pipeline of the song.

    Nothing changes until then: a transcription replaces the lyrics when the sync reaches it.
    """

    method = "PUT"
    path = "/tracks/{track_id}/processing/{pipeline}"
    request = ResyncBody
    response = ResyncState

    def handle(
        self,
        track_id: uuid.UUID,
        pipeline: Pipeline,
        body: ResyncBody,
        user: SessionUser = Depends(current_user),
    ) -> dict:
        options = {}
        with connect() as connection:
            library.require_visible(
                connection, user["id"], track_id, "Track is not in your library"
            )
            if pipeline == "transcription":
                if not body.language:
                    raise ConflictError("Choose the language to transcribe the song in")
                if not settings.transcription_enabled(connection):
                    raise ConflictError(
                        "Enable AI lyric generation in Settings before transcribing a song"
                    )
                options = {"language": body.language}
            resync.request(connection, track_id, pipeline, options, requested_by=user["id"])
        return {"track_id": str(track_id), "pipeline": pipeline, "requested": True}


class TrackResyncCancel(Endpoint):
    """Withdraw a song's resync request before a sync takes it."""

    method = "DELETE"
    path = "/tracks/{track_id}/processing/{pipeline}"
    response = ResyncState

    def handle(
        self, track_id: uuid.UUID, pipeline: Pipeline, user: SessionUser = Depends(current_user)
    ) -> dict:
        with connect() as connection:
            library.require_visible(
                connection, user["id"], track_id, "Track is not in your library"
            )
            resync.cancel(connection, track_id, pipeline)
        return {"track_id": str(track_id), "pipeline": pipeline, "requested": False}
