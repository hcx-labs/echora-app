"""``GET /library/lyrics/status``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from core.db.connection import connect


class LyricsCoverage(ResponseModel):
    """Songs by lyrics status, across the instance, and how many have a lyrics embedding."""

    total: int
    available: int
    missing: int
    unavailable: int
    instrumental: int
    embedded: int


class LyricsStatus(Endpoint):
    """Lyrics coverage of the instance's songs."""

    method = "GET"
    path = "/lyrics/status"
    response = LyricsCoverage

    def handle(self, user: SessionUser = Depends(current_user)) -> dict:
        with connect(dict_rows=True) as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT count(*) AS total,
                          count(*) FILTER (WHERE l.availability_status='available') AS available,
                          count(*) FILTER (WHERE l.availability_status='missing') AS missing,
                          count(*) FILTER (WHERE l.availability_status='unavailable') AS unavailable,
                          count(*) FILTER (WHERE l.availability_status='instrumental') AS instrumental,
                          count(*) FILTER (WHERE EXISTS (
                            SELECT 1 FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                            WHERE e.track_id=t.id AND e.embedding_type='lyrics' AND e.window_index IS NULL
                              AND ar.model_name='bge_m3'
                          )) AS embedded
                   FROM tracks t LEFT JOIN lyrics l ON l.track_id=t.id"""
            )
            return cursor.fetchone()
