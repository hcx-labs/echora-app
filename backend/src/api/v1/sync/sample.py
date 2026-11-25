"""``GET /sync/connections/{connection_id}/sample``."""

from __future__ import annotations

from fastapi import Depends, Query

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.sync._models import Sample
from backend.sync.connections import owned_connection, track_summary
from core.errors import IntegrationError
from core.integrations.navidrome import NavidromeClient


class ConnectionSample(Endpoint):
    """Random songs of a saved connection, so onboarding needs no password after connecting."""

    method = "GET"
    path = "/connections/{connection_id}/sample"
    response = Sample

    def handle(
        self,
        connection_id: str,
        limit: int = Query(default=100, ge=1, le=500),
        user: SessionUser = Depends(current_user),
    ) -> dict:
        credentials = owned_connection(connection_id, user["id"])
        try:
            with NavidromeClient(*credentials) as client:
                tracks = client.random_tracks(limit)
        except Exception as error:
            raise IntegrationError("Could not read tracks from the Navidrome server") from error
        return {"tracks": [track_summary(track) for track in tracks]}
