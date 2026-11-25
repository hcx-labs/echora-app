"""``GET /settings/external-ai/translation``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.v1.settings._deps import admin_required
from backend.api.v1.settings._external_ai import TranslationView, current, translation_view


class ExternalAITranslation(Endpoint):
    """The External AI translation settings (administrators only)."""

    method = "GET"
    path = "/external-ai/translation"
    response = TranslationView

    def handle(self, user: SessionUser = Depends(admin_required)) -> dict:
        return translation_view(current())
