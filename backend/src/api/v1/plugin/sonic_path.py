"""``POST /plugin/sonic-path``."""

from __future__ import annotations

from typing import Any

from fastapi import Depends

import backend.search.plugin as ranking
from backend.api.endpoint import Endpoint
from backend.api.v1.plugin._models import PathRequest, SonicMatches
from backend.settings.plugin_integration import authenticate


class SonicPath(Endpoint):
    """A path of songs that moves gradually from one song to another."""

    method = "POST"
    path = "/sonic-path"
    request = PathRequest
    response = SonicMatches

    def handle(self, body: PathRequest, principal: dict[str, Any] = Depends(authenticate)) -> dict:
        return ranking.sonic_path(
            ranking.load_corpus(principal),
            body.startSong.id,
            body.endSong.id,
            body.count,
            principal["profile"],
        )
