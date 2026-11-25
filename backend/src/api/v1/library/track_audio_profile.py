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
        with connect(dict_rows=True) as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT tap.*, profile_run.model_revision AS profile_revision,
                          profile_run.config AS profile_config, profile_run.created_at AS profile_created_at,
                          source_run.model_revision AS source_model_revision,
                          source_run.config AS source_config
                   FROM current_audio_profiles tap
                   JOIN analysis_runs profile_run ON profile_run.id=tap.profile_run_id
                   JOIN analysis_runs source_run ON source_run.id=tap.source_run_id
                   WHERE tap.track_id=%s AND tap.model_name=%s
                     AND EXISTS (SELECT 1 FROM user_track_links utl
                                 WHERE utl.user_id=%s AND utl.track_id=tap.track_id)
                   ORDER BY profile_run.created_at DESC, tap.created_at DESC LIMIT 1""",
                (track_id, model, user["id"]),
            )
            profile = cursor.fetchone()
            if profile is None:
                raise NotFoundError("Audio profile not found")
            cursor.execute(
                """SELECT segment_index, start_seconds, end_seconds, cohesion,
                          representative_window_index
                   FROM audio_temporal_segments
                   WHERE profile_id=%s ORDER BY segment_index""",
                (profile["id"],),
            )
            segments = cursor.fetchall()
            cursor.execute(
                """SELECT am.id, am.mode_index, am.duration_weight, am.cohesion,
                          am.representative_window_index
                   FROM audio_modes am
                   WHERE am.profile_id=%s ORDER BY am.mode_index""",
                (profile["id"],),
            )
            modes = cursor.fetchall()
            for mode in modes:
                cursor.execute(
                    """SELECT start_seconds, end_seconds FROM audio_mode_intervals
                       WHERE mode_id=%s ORDER BY interval_index""",
                    (mode["id"],),
                )
                mode["intervals"] = cursor.fetchall()
                mode.pop("id")
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
