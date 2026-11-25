"""``POST /sync/audio-profiles``."""

from __future__ import annotations

from fastapi import Depends

import core.jobs.queue as jobs
from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import JobSnapshot
from core.db.connection import connect
from core.embeddings import store
from core.revisions import SUPPORTED_PROFILE_MODELS


class AudioProfilesRebuild(Endpoint):
    """Rebuild the audio profiles of every user song with a source embedding."""

    method = "POST"
    path = "/audio-profiles"
    response = JobSnapshot
    status_code = 202

    def handle(self, user: SessionUser = Depends(current_user)) -> dict:
        with connect() as connection:
            track_ids = store.songs_with_audio_embeddings(
                connection, user["id"], list(SUPPORTED_PROFILE_MODELS)
            )
        return jobs.enqueue(
            "audio_profiles",
            "analysis",
            user["id"],
            payload={"track_ids": [str(track_id) for track_id in track_ids]},
            dedupe_key=f"audio_profiles:{user['id']}",
        )
