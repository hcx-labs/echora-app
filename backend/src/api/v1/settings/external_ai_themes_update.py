"""``PUT /settings/external-ai/themes``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.requests import PutRequest
from backend.api.v1.settings._deps import admin_required
from backend.api.v1.settings._external_ai import ThemesView, themes_view


class ThemesUpdate(PutRequest):
    """Whether syncs summarise each song's lyric themes."""

    lyric_themes: bool = False


class ExternalAIThemesUpdate(Endpoint):
    """Turn lyric themes on or off (administrators only)."""

    method = "PUT"
    path = "/external-ai/themes"
    request = ThemesUpdate
    response = ThemesView

    def handle(self, body: ThemesUpdate, user: SessionUser = Depends(admin_required)) -> dict:
        import core.lyrics.translations as storage

        return themes_view(storage.save_section(body.model_dump()))
