"""``POST /plugin/similar-tracks``."""

from __future__ import annotations

from typing import Any

from fastapi import Depends

import backend.search.plugin as ranking
from backend.api.endpoint import Endpoint
from backend.api.v1.plugin._models import SimilarRequest, SonicMatches
from backend.settings.plugin_integration import authenticate


class SimilarTracks(Endpoint):
    """Songs that sound (and read) like a song, at most max_per_artist per artist."""

    method = "POST"
    path = "/similar-tracks"
    request = SimilarRequest
    response = SonicMatches

    def handle(
        self, body: SimilarRequest, principal: dict[str, Any] = Depends(authenticate)
    ) -> dict:
        return ranking.similar_tracks(
            ranking.load_corpus(principal), body.song.id, body.count, principal["profile"]
        )
