"""``POST /sync/connections/{connection_id}``."""

from __future__ import annotations

from fastapi import Depends, HTTPException

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import JobSnapshot
from backend.api.v1.sync._models import SyncRequest
from backend.sync.connections import enqueue_connection_job
from worker.remote.settings import compute_for


class StartSync(Endpoint):
    """Sync a connection: analyze new and changed songs (or only missing ones)."""

    method = "POST"
    path = "/connections/{connection_id}"
    request = SyncRequest
    response = JobSnapshot
    status_code = 202

    def handle(
        self, connection_id: str, request: SyncRequest, user: SessionUser = Depends(current_user)
    ) -> dict:
        try:
            compute = compute_for(user["id"], request.compute)
        except PermissionError:
            raise HTTPException(
                status_code=403, detail="Modal processing is not available to you"
            ) from None
        return enqueue_connection_job(
            "navidrome_sync",
            connection_id,
            user,
            payload={
                "mode": request.mode,
                "verify_audio_hashes": request.verify_audio_hashes,
                "compute": compute,
                **({"resync": sorted(set(request.resync))} if request.resync else {}),
            },
        )
