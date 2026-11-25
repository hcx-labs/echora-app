"""``POST /media/scrobble``."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Response
from pydantic import Field

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.requests import PostRequest
from backend.sync.connections import owned_connection
from core.integrations.navidrome import media_navidrome_client


class ScrobbleRequest(PostRequest):
    """A song the player started (``submission=false``) or finished playing."""

    connection_id: str
    song_id: str = Field(min_length=1, max_length=200)
    submission: bool = True


class Scrobble(Endpoint):
    """Report playback to the media server (Subsonic scrobble).

    Fire-and-forget from the player: failures must never interrupt listening, hence an empty
    204 on success and a bare 502 the client ignores on failure.
    """

    method = "POST"
    path = "/scrobble"
    request = ScrobbleRequest
    status_code = 204
    response_class = Response

    def handle(
        self, request: ScrobbleRequest, user: SessionUser = Depends(current_user)
    ) -> Response:
        credentials = owned_connection(request.connection_id, user["id"])
        try:
            with media_navidrome_client(*credentials) as client:
                client.scrobble(request.song_id, request.submission)
        except Exception as error:
            raise HTTPException(status_code=502, detail="Scrobble failed") from error
        return Response(status_code=204)
