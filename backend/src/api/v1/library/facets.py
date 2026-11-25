"""``GET /library/facets``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.api.v1.library._models import NameCount
from backend.library.db import connect


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
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT artist AS name, count(*) AS tracks FROM tracks t
                   WHERE artist IS NOT NULL AND artist ILIKE %s
                     AND EXISTS (SELECT 1 FROM user_track_links utl
                                 WHERE utl.track_id=t.id AND utl.user_id=%s)
                   GROUP BY artist ORDER BY artist LIMIT %s""",
                (f"%{artist_query.strip()}%", user["id"], limit),
            )
            artists = cursor.fetchall()
            album_clauses = [
                "album IS NOT NULL",
                "album ILIKE %s",
                "EXISTS (SELECT 1 FROM user_track_links utl WHERE utl.track_id=t.id AND utl.user_id=%s)",
            ]
            album_parameters: list[object] = [f"%{album_query.strip()}%", user["id"]]
            if artist:
                album_clauses.append("artist = %s")
                album_parameters.append(artist)
            cursor.execute(
                f"SELECT album AS name, count(*) AS tracks FROM tracks t WHERE {' AND '.join(album_clauses)} GROUP BY album ORDER BY album LIMIT %s",
                [*album_parameters, limit],
            )
            albums = cursor.fetchall()
        return {"artists": artists, "albums": albums}
