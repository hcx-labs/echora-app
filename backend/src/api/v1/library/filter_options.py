"""``GET /library/filter-options``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.api.v1.library._models import NameCount
from core.db.connection import connect


class FilterOptions(ResponseModel):
    """Values the browse filters offer, with song counts."""

    language: list[NameCount]
    """Lyrics languages (``unknown`` for none)."""
    genre: list[NameCount]
    translation: list[NameCount]
    """Target languages of ready translations."""


class FilterOptionsEndpoint(Endpoint):
    """The languages, genres and translations in the user's library."""

    method = "GET"
    path = "/filter-options"
    response = FilterOptions

    def handle(self, user: SessionUser = Depends(current_user)) -> dict:
        with connect(dict_rows=True) as db:
            rows = db.execute(
                """WITH visible AS MATERIALIZED (
                SELECT DISTINCT track_id FROM user_track_links WHERE user_id=%s
            ), options AS (
                SELECT 'language' AS kind, coalesce(nullif(l.language,''),'unknown') AS value, v.track_id
                FROM visible v LEFT JOIN lyrics l ON l.track_id=v.track_id
                UNION ALL
                SELECT 'genre', g, v.track_id FROM visible v JOIN tracks t ON t.id=v.track_id
                CROSS JOIN LATERAL unnest(t.genres) g WHERE g<>''
                UNION ALL
                SELECT 'translation', lt.target_language, v.track_id FROM visible v
                JOIN lyric_translations lt ON lt.track_id=v.track_id WHERE lt.status='ready'
            ) SELECT kind,value,count(DISTINCT track_id) AS tracks FROM options
              GROUP BY kind,value ORDER BY kind,lower(value)""",
                (user["id"],),
            ).fetchall()
        return {
            kind: [
                {"name": row["value"], "tracks": row["tracks"]}
                for row in rows
                if row["kind"] == kind
            ]
            for kind in ("language", "genre", "translation")
        }
