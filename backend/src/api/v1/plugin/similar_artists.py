"""``POST /plugin/similar-artists``."""

from __future__ import annotations

from typing import Any

from fastapi import Depends

import backend.search.plugin as ranking
from backend.api.endpoint import Endpoint
from backend.api.v1.plugin._models import ArtistRequest, SimilarArtistList
from backend.settings.plugin_integration import authenticate


class SimilarArtists(Endpoint):
    """Artists whose songs resemble an artist's, by their song-embedding profiles."""

    method = "POST"
    path = "/similar-artists"
    request = ArtistRequest
    response = SimilarArtistList

    def handle(
        self, body: ArtistRequest, principal: dict[str, Any] = Depends(authenticate)
    ) -> dict:
        return ranking.similar_artists(
            ranking.load_corpus(principal), body.id, body.limit, principal["profile"]
        )
