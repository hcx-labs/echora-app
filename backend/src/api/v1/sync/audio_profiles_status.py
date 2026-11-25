"""``GET /sync/audio-profiles``."""

from __future__ import annotations

import psycopg
from fastapi import Depends
from psycopg.rows import dict_row

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.sync._models import AudioProfileCoverage
from core.db.config import DatabaseConfig
from core.revisions import AUDIO_PROFILE_REVISION, SUPPORTED_PROFILE_MODELS


class AudioProfilesStatus(Endpoint):
    """How many of the user's songs have a current audio profile, per model."""

    method = "GET"
    path = "/audio-profiles"
    response = AudioProfileCoverage

    def handle(self, user: SessionUser = Depends(current_user)) -> dict:
        models: dict[str, dict[str, int]] = {}
        with (
            psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
            connection.cursor() as cursor,
        ):
            for model_name in SUPPORTED_PROFILE_MODELS:
                cursor.execute(
                    """WITH latest_source AS (
                     SELECT DISTINCT ON (e.track_id) e.track_id, e.run_id
                     FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                     WHERE e.embedding_type='audio-track' AND e.window_index IS NULL
                       AND ar.model_name=%s
                       AND EXISTS (SELECT 1 FROM user_track_links utl
                                   WHERE utl.user_id=%s AND utl.track_id=e.track_id)
                     ORDER BY e.track_id, ar.created_at DESC
                   )
                   SELECT count(*) AS total,
                          count(*) FILTER (WHERE EXISTS (
                            SELECT 1 FROM current_audio_profiles tap
                            JOIN analysis_runs profile_run ON profile_run.id=tap.profile_run_id
                            WHERE tap.track_id=latest_source.track_id
                              AND tap.source_run_id=latest_source.run_id
                              AND tap.model_name=%s
                              AND profile_run.kind='audio_profile'
                              AND profile_run.model_revision=%s

                          )) AS current
                   FROM latest_source""",
                    (model_name, user["id"], model_name, AUDIO_PROFILE_REVISION),
                )
                value = cursor.fetchone()
                total, current = int(value["total"]), int(value["current"])
                models[model_name] = {
                    "total": total,
                    "current": current,
                    "missing": total - current,
                }
        return {
            "revision": AUDIO_PROFILE_REVISION,
            "models": models,
            "total": sum(value["total"] for value in models.values()),
            "current": sum(value["current"] for value in models.values()),
            "missing": sum(value["missing"] for value in models.values()),
        }
