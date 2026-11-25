"""``POST /sync/connections/{connection_id}/backfills/voice``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser, admin_user
from backend.api.endpoint import Endpoint
from backend.api.responses import JobSnapshot
from backend.sync.connections import enqueue_connection_job


class VoiceBackfill(Endpoint):
    """Run the voice stage over the connection's songs."""

    method = "POST"
    path = "/connections/{connection_id}/backfills/voice"
    response = JobSnapshot
    status_code = 202

    def handle(self, connection_id: str, user: SessionUser = Depends(admin_user)) -> dict:
        return enqueue_connection_job("voice_backfill", connection_id, user)
