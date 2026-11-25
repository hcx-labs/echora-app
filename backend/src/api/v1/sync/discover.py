"""``POST /sync/discover``."""

from __future__ import annotations

import logging

from fastapi import Depends, HTTPException

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.sync._models import Discovered, DiscoverRequest
from backend.sync.connections import track_summary
from core.integrations.navidrome import NavidromeClient
from core.library.connections import save_connection

logger = logging.getLogger(__name__)


class Discover(Endpoint):
    """Connect a Navidrome server: check it answers, save the connection, preview songs."""

    method = "POST"
    path = "/discover"
    request = DiscoverRequest
    response = Discovered

    def handle(self, request: DiscoverRequest, user: SessionUser = Depends(current_user)) -> dict:
        credentials = request.as_tuple()
        try:
            with NavidromeClient(*credentials) as client:
                version = client.ping()
                tracks = client.random_tracks(request.limit)
        except Exception as error:
            logger.warning("Navidrome discovery failed for %s: %s", request.url, error)
            raise HTTPException(
                status_code=400, detail="Could not connect to the Navidrome server"
            ) from error
        connection_id = save_connection(credentials, user["id"])
        return {
            "connection_id": connection_id,
            "server": {"version": version, "url": credentials[0]},
            "tracks": [track_summary(track) for track in tracks],
        }
