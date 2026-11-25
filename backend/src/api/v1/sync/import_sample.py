"""``POST /sync/connections/{connection_id}/import``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import JobSnapshot
from backend.api.v1.sync._models import SampleImportRequest
from backend.sync.connections import enqueue_import, owned_connection


class ImportSample(Endpoint):
    """Analyze chosen songs of a saved connection: the onboarding sample."""

    method = "POST"
    path = "/connections/{connection_id}/import"
    request = SampleImportRequest
    response = JobSnapshot
    status_code = 202

    def handle(
        self,
        connection_id: str,
        request: SampleImportRequest,
        user: SessionUser = Depends(current_user),
    ) -> dict:
        owned_connection(connection_id, user["id"])
        return enqueue_import(connection_id, user["id"], request.track_ids)
