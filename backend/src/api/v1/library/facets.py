"""``GET /library/facets``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.api.v1.library._models import NameCount
from core.db.connection import connect
from core.library import browse


class Facets(ResponseModel):
    """Artists and albums matching the queries, alphabetically."""

    artists: list[NameCount]
    albums: list[NameCount]


class FacetsEndpoint(Endpoint):
    """Artists and albums of the user's library, by name (albums optionally of one artist)."""

    method = "GET"
    path = "/facets"
    response = Facets

    def handle(
        self,
        artist_query: str = "",
        album_query: str = "",
        artist: str = "",
        limit: int = 20,
        user: SessionUser = Depends(current_user),
    ) -> dict:
        limit = min(max(limit, 1), 50)
        with connect() as connection:
            artists, albums = browse.facets(
                connection, user["id"], artist_query, album_query, artist, limit
            )
        return {"artists": artists, "albums": albums}
