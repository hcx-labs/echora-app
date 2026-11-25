"""``POST /search/hum/index``."""

from __future__ import annotations

from fastapi import Depends

import core.jobs.queue as jobs
from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import JobSnapshot
from core.errors import ConflictError
from core.library.connections import load_connection
from worker.melody.index import DEFAULT_CORPUS_SIZE


class HumIndexBuild(Endpoint):
    """Queue a hum index build over the user's library (one at a time per user)."""

    method = "POST"
    path = "/hum/index"
    response = JobSnapshot
    status_code = 202

    def handle(
        self, track_limit: int = DEFAULT_CORPUS_SIZE, user: SessionUser = Depends(current_user)
    ) -> dict:
        track_limit = min(max(track_limit, 1), 500)
        connection_id = user.get("navidrome_connection_id")
        credentials = load_connection(str(connection_id), user["id"]) if connection_id else None
        if credentials is None:
            raise ConflictError("Connect Navidrome before building a hum index")
        return jobs.enqueue(
            "hum_corpus",
            "analysis",
            user["id"],
            connection_id=connection_id,
            payload={"track_limit": track_limit},
            dedupe_key=f"hum_corpus:{user['id']}",
        )
