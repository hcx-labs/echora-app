"""``GET /library/community-snapshots/{snapshot_id}``."""

from __future__ import annotations

import uuid

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.library._models import MapPayload
from core.db.connection import connect
from core.errors import NotFoundError


class CommunitySnapshot(Endpoint):
    """One stored library map."""

    method = "GET"
    path = "/community-snapshots/{snapshot_id}"
    response = MapPayload

    def handle(self, snapshot_id: uuid.UUID, user: SessionUser = Depends(current_user)) -> dict:
        with connect(dict_rows=True) as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT payload, created_at FROM community_snapshots WHERE id=%s", (snapshot_id,)
            )
            snapshot = cursor.fetchone()
        if snapshot is None:
            raise NotFoundError("Community snapshot not found")
        return {
            **snapshot["payload"],
            "snapshot_id": str(snapshot_id),
            "snapshot_created_at": snapshot["created_at"],
            "cache_hit": True,
        }
