"""``POST /sync/audio-profiles``."""

from __future__ import annotations

import psycopg
from fastapi import Depends

import core.jobs.queue as jobs
from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import JobSnapshot
from core.db.config import DatabaseConfig
from core.revisions import SUPPORTED_PROFILE_MODELS


class AudioProfilesRebuild(Endpoint):
    """Rebuild the audio profiles of every user song with a source embedding."""

    method = "POST"
    path = "/audio-profiles"
    response = JobSnapshot
    status_code = 202

    def handle(self, user: SessionUser = Depends(current_user)) -> dict:
        with (
            psycopg.connect(DatabaseConfig.load().database_url) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(
                """SELECT DISTINCT e.track_id, ar.model_name
                   FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                   WHERE e.embedding_type='audio-track' AND e.window_index IS NULL
                     AND ar.model_name=ANY(%s)
                     AND EXISTS (SELECT 1 FROM user_track_links utl
                                 WHERE utl.user_id=%s AND utl.track_id=e.track_id)
                   ORDER BY e.track_id, ar.model_name""",
                (list(SUPPORTED_PROFILE_MODELS), user["id"]),
            )
            track_ids = sorted({row[0] for row in cursor.fetchall()})
        return jobs.enqueue(
            "audio_profiles",
            "analysis",
            user["id"],
            payload={"track_ids": [str(track_id) for track_id in track_ids]},
            dedupe_key=f"audio_profiles:{user['id']}",
        )
