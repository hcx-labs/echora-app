"""``POST /sync/ingest``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import JobSnapshot
from backend.api.v1.sync._models import IngestRequest
from backend.sync.connections import enqueue_import
from core.library.connections import save_connection


class Ingest(Endpoint):
    """Save a connection and analyze the given songs on it."""

    method = "POST"
    path = "/ingest"
    request = IngestRequest
    response = JobSnapshot
    status_code = 202

    def handle(self, request: IngestRequest, user: SessionUser = Depends(current_user)) -> dict:
        connection_id = save_connection(request.as_tuple(), user["id"])
        return enqueue_import(connection_id, user["id"], request.track_ids)
