"""``GET /library/artists/profile``."""

from __future__ import annotations

from datetime import datetime

from fastapi import Depends
from fastapi.encoders import jsonable_encoder
from psycopg.types.json import Jsonb

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.api.v1.library._artist_models import ArtistFacet
from backend.library.artist_corpus import artist_embedding_corpus, artist_payload
from backend.library.db import connect
from core.errors import NotFoundError


class ArtistProfileResult(ResponseModel):
    """An artist's facets, stored for reuse."""

    artist: str
    track_count: int
    component_count: int
    facets: list[ArtistFacet]
    model: str
    corpus_hash: str
    profile_id: str
    created_at: datetime


class ArtistProfile(Endpoint):
    """Fit (and store) an artist's profile: a few weighted facets of their songs' sound."""

    method = "GET"
    path = "/artists/profile"
    response = ArtistProfileResult

    def handle(
        self, artist: str, model: str = "muq_mulan", user: SessionUser = Depends(current_user)
    ) -> dict:
        rows, matrix, corpus_hash = artist_embedding_corpus(model)
        indices = [
            index
            for index, row in enumerate(rows)
            if str(row["artist"]).casefold() == artist.casefold()
        ]
        if not indices:
            raise NotFoundError("Artist not found")
        display_name = str(rows[indices[0]]["artist"])
        payload, profile = artist_payload(display_name, rows, matrix, indices)
        stored = jsonable_encoder({**payload, "model": model, "corpus_hash": corpus_hash})
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO artist_profiles
                     (artist_key, artist_name, model_name, corpus_hash, track_count, component_count, payload)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (artist_key, model_name, corpus_hash) DO UPDATE SET payload=EXCLUDED.payload
                   RETURNING id, created_at""",
                (
                    display_name.casefold(),
                    display_name,
                    model,
                    corpus_hash,
                    len(indices),
                    len(profile.weights),
                    Jsonb(stored),
                ),
            )
            saved = cursor.fetchone()
        return {**stored, "profile_id": str(saved["id"]), "created_at": saved["created_at"]}
