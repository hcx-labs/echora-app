"""``POST /plugin/lyrics``."""

from __future__ import annotations

from typing import Any

from fastapi import Depends

from backend.api.endpoint import Endpoint
from backend.api.v1.plugin._models import LyricsList, LyricsRequest
from backend.plugin.lyrics import load_lyrics
from backend.settings.plugin_integration import authenticate


class Lyrics(Endpoint):
    """A song's lyrics as TTML (with translations) or LRC, per the integration settings."""

    method = "POST"
    path = "/lyrics"
    request = LyricsRequest
    response = LyricsList

    def handle(
        self, body: LyricsRequest, principal: dict[str, Any] = Depends(authenticate)
    ) -> dict:
        return load_lyrics(principal, body.track.id)
