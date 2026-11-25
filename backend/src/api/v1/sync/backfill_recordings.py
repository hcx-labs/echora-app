"""``POST /sync/connections/{connection_id}/backfills/recordings``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import JobSnapshot
from backend.sync.connections import enqueue_connection_job


class RecordingsBackfill(Endpoint):
    """Run the recording fingerprint stage over the connection's songs."""

    method = "POST"
    path = "/connections/{connection_id}/backfills/recordings"
    response = JobSnapshot
    status_code = 202

    def handle(self, connection_id: str, user: SessionUser = Depends(current_user)) -> dict:
        return enqueue_connection_job("recordings_backfill", connection_id, user)
