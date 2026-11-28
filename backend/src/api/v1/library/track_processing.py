"""``GET /library/tracks/{track_id}/processing``."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.api.v1.library._processing import processing
from core.db.connection import connect
from core.library import store as library


class TranslationState(ResponseModel):
    """One stored translation of the song."""

    source: str
    target: str
    status: str
    revision: str
    updated_at: datetime | None = None


class ResyncRequest(ResponseModel):
    """An open resync request and its options (the transcription language)."""

    options: dict[str, Any]
    requested_at: datetime


class PipelineState(ResponseModel):
    """One pipeline's output for the song.

    ``status``: ``current``, ``outdated`` (another revision than this code makes), ``missing``,
    ``failed``, ``stale`` (lyrics changed), ``pending`` or ``not_applicable``.
    """

    key: str
    label: str
    status: str
    stored_revision: str | None = None
    current_revision: str | None = None
    detail: str | None = None
    resyncable: bool
    requested: ResyncRequest | None = None
    """The open resync request: the next sync, partial or full, redoes the pipeline."""
    updated_at: datetime | None = None
    language: str | None = None
    translations: list[TranslationState] | None = None


class TrackProcessing(ResponseModel):
    """Every pipeline of the song."""

    track_id: str
    pipelines: list[PipelineState]


class TrackProcessingEndpoint(Endpoint):
    """Each pipeline's revision for a song, whether it is out of date, and open resyncs."""

    method = "GET"
    path = "/tracks/{track_id}/processing"
    response = TrackProcessing

    def handle(self, track_id: uuid.UUID, user: SessionUser = Depends(current_user)) -> dict:
        with connect() as connection:
            library.require_visible(
                connection, user["id"], track_id, "Track is not in your library"
            )
            return {"track_id": str(track_id), "pipelines": processing(connection, track_id)}
