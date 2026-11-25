"""``POST /plugin/artist-radio``."""

from __future__ import annotations

from typing import Any

from fastapi import Depends

import backend.search.plugin as ranking
from backend.api.endpoint import Endpoint
from backend.api.v1.plugin._models import ArtistRequest, SongList
from backend.settings.plugin_integration import authenticate


class ArtistRadio(Endpoint):
    """Representative songs of an artist mixed with songs of related artists."""

    method = "POST"
    path = "/artist-radio"
    request = ArtistRequest
    response = SongList

    def handle(
        self, body: ArtistRequest, principal: dict[str, Any] = Depends(authenticate)
    ) -> dict:
        return ranking.artist_radio(
            ranking.load_corpus(principal), body.id, body.count, principal["profile"]
        )
