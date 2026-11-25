"""``GET /search/hum/index``."""

from __future__ import annotations

import psycopg
from fastapi import Depends
from psycopg.rows import dict_row

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from core.db.config import DatabaseConfig


class HumIndexState(ResponseModel):
    """How many of the user's songs have melody contours."""

    status: str
    """``complete`` when any song is indexed, else ``missing``."""
    indexed_tracks: int
    track_limit: int


class HumIndexStatus(Endpoint):
    """Report whether the user's hum index exists."""

    method = "GET"
    path = "/hum/index"
    response = HumIndexState

    def handle(self, user: SessionUser = Depends(current_user)) -> dict:
        with (
            psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(
                """SELECT count(DISTINCT mc.track_id) AS indexed_tracks
                   FROM melody_contours mc
                   WHERE EXISTS (SELECT 1 FROM user_track_links utl
                                 WHERE utl.user_id=%s AND utl.track_id=mc.track_id)""",
                (user["id"],),
            )
            indexed = int(cursor.fetchone()["indexed_tracks"])
        return {
            "status": "complete" if indexed else "missing",
            "indexed_tracks": indexed,
            "track_limit": indexed,
        }
