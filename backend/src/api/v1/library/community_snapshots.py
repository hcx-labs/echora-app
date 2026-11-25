"""``GET /library/community-snapshots``."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from core.db.connection import connect
from core.library import store


class CommunitySnapshotSummary(ResponseModel):
    """A stored map: its representation, corpus, clustering parameters and quality."""

    id: uuid.UUID
    model_name: str
    semantic_weight: Any
    corpus_hash: str
    algorithm_revision: int
    track_count: int
    parameters: dict[str, Any]
    metrics: dict[str, Any]
    created_at: datetime


class CommunitySnapshotList(ResponseModel):
    """Stored maps, newest first."""

    snapshots: list[CommunitySnapshotSummary]


class CommunitySnapshots(Endpoint):
    """The stored library maps."""

    method = "GET"
    path = "/community-snapshots"
    response = CommunitySnapshotList

    def handle(self, limit: int = 20, user: SessionUser = Depends(current_user)) -> dict:
        limit = min(max(limit, 1), 100)
        with connect() as connection:
            snapshots = store.community_snapshots(connection, limit)
        return {"snapshots": snapshots}
