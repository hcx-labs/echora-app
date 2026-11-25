"""``GET /library/tracks/{track_id}/audio-profile``."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import Depends
from fastapi.encoders import jsonable_encoder

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from core.db.connection import connect
from core.errors import InvalidQueryError, NotFoundError
from core.library import store
from core.revisions import SUPPORTED_PROFILE_MODELS


class AudioProfile(ResponseModel):
    """A song's temporal segments and recurring modes, from one embedding model."""

    track_id: str
    profile_run_id: str
    source_run_id: str
    profile_revision: str | None = None
    profile_created_at: str
    profile_config: Any = None
    source: dict[str, Any]
    """``model``, ``revision`` and ``config`` of the embeddings it was built from."""
    timestamps_exact: bool | None = None
    diagnostics: dict[str, Any]
    segments: list[dict[str, Any]]
    modes: list[dict[str, Any]]


class TrackAudioProfile(Endpoint):
    """The audio profile of one of the user's songs (``model``: muq_mulan or mert)."""

    method = "GET"
    path = "/tracks/{track_id}/audio-profile"
    response = AudioProfile

    def handle(
        self,
        track_id: uuid.UUID,
        model: str = "muq_mulan",
        user: SessionUser = Depends(current_user),
    ) -> dict:
        if model not in SUPPORTED_PROFILE_MODELS:
            raise InvalidQueryError("Audio profiles support muq_mulan or mert")
        with connect() as connection:
            profile = store.audio_profile(connection, user["id"], track_id, model)
            if profile is None:
                raise NotFoundError("Audio profile not found")
            segments = store.profile_segments(connection, profile["id"])
            modes = store.profile_modes(connection, profile["id"])
        return jsonable_encoder(
            {
                "track_id": track_id,
                "profile_run_id": profile["profile_run_id"],
                "source_run_id": profile["source_run_id"],
                "profile_revision": profile["profile_revision"],
                "profile_created_at": profile["profile_created_at"],
                "profile_config": profile["profile_config"],
                "source": {
                    "model": profile["model_name"],
                    "revision": profile["source_model_revision"],
                    "config": profile["source_config"],
                },
                "timestamps_exact": profile["timestamps_exact"],
                "diagnostics": {
                    "resultant_length": profile["resultant_length"],
                    "mean_global_similarity": profile["mean_global_similarity"],
                    "p05_global_similarity": profile["p05_global_similarity"],
                    "adjacent_change_mean": profile["adjacent_change_mean"],
                    "adjacent_change_p95": profile["adjacent_change_p95"],
                },
                "segments": segments,
                "modes": modes,
            }
        )
