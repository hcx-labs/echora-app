"""``GET /library/tracks/{track_id}/similar``."""

from __future__ import annotations

import uuid

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from core.db.connection import connect
from core.errors import NotFoundError


class SimilarTrack(ResponseModel):
    """A song and its cosine similarity to the seed."""

    id: uuid.UUID
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    similarity: float


class SimilarTracks(ResponseModel):
    """The nearest songs by semantic fusion vector."""

    track_id: str
    mode: str
    results: list[SimilarTrack]


class TrackSimilar(Endpoint):
    """The songs nearest to one song by its semantic fusion vector."""

    method = "GET"
    path = "/tracks/{track_id}/similar"
    response = SimilarTracks

    def handle(
        self, track_id: uuid.UUID, limit: int = 20, user: SessionUser = Depends(current_user)
    ) -> dict:
        with connect(dict_rows=True) as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT e.embedding FROM current_embeddings e
                   WHERE e.track_id=%s AND e.embedding_type='semantic_fusion'""",
                (track_id,),
            )
            seed = cursor.fetchone()
            if seed is None:
                raise NotFoundError("Track has no semantic fusion vector")
            cursor.execute(
                """SELECT t.id, t.title, t.artist, t.album,
                          1 - (e.embedding <=> %s::vector) AS similarity
                   FROM current_embeddings e
                   JOIN tracks t ON t.id=e.track_id
                   WHERE e.embedding_type='semantic_fusion' AND e.track_id != %s
                   ORDER BY e.embedding <=> %s::vector LIMIT %s""",
                (seed["embedding"], track_id, seed["embedding"], max(1, min(limit, 100))),
            )
            return {
                "track_id": str(track_id),
                "mode": "semantic_fusion",
                "results": cursor.fetchall(),
            }
