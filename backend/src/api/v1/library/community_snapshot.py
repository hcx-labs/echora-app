"""``GET /library/community-snapshots/{snapshot_id}``."""

from __future__ import annotations

import uuid

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.library._models import MapPayload
from core.db.connection import connect
from core.errors import NotFoundError
from core.library import store


class CommunitySnapshot(Endpoint):
    """One stored library map."""

    method = "GET"
    path = "/community-snapshots/{snapshot_id}"
    response = MapPayload

    def handle(self, snapshot_id: uuid.UUID, user: SessionUser = Depends(current_user)) -> dict:
        with connect() as connection:
            snapshot = store.community_snapshot(connection, snapshot_id)
        if snapshot is None:
            raise NotFoundError("Community snapshot not found")
        return {
            **snapshot["payload"],
            "snapshot_id": str(snapshot_id),
            "snapshot_created_at": snapshot["created_at"],
            "cache_hit": True,
        }
