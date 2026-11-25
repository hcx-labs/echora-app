"""``GET /settings/external-ai``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.v1.settings._deps import admin_required
from backend.api.v1.settings._external_ai import ExternalAIView, current


class ExternalAISettingsRead(Endpoint):
    """Every External AI setting (administrators only)."""

    method = "GET"
    path = "/external-ai"
    response = ExternalAIView

    def handle(self, user: SessionUser = Depends(admin_required)) -> dict:
        return current()
