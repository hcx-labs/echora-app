"""``GET /settings/external-ai/themes``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.v1.settings._deps import admin_required
from backend.api.v1.settings._external_ai import ThemesView, current, themes_view


class ExternalAIThemes(Endpoint):
    """The External AI themes settings (administrators only)."""

    method = "GET"
    path = "/external-ai/themes"
    response = ThemesView

    def handle(self, user: SessionUser = Depends(admin_required)) -> dict:
        return themes_view(current())
