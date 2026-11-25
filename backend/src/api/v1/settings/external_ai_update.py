"""``PUT /settings/external-ai``."""

from __future__ import annotations

from fastapi import Depends
from pydantic import Field, SecretStr

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.requests import PutRequest
from backend.api.v1.settings._deps import admin_required
from backend.api.v1.settings._external_ai import ExternalAIView
from core.integrations.external_ai import ExternalAISettings
from core.library.connections import credential_cipher


class ExternalAIUpdate(ExternalAISettings, PutRequest):
    """Every External AI setting. ``api_key``: omitted or null keeps it, "" clears it."""

    api_key: SecretStr | None = Field(default=None, max_length=8192, exclude=True)


class ExternalAISettingsUpdate(Endpoint):
    """Replace every External AI setting (administrators only)."""

    method = "PUT"
    path = "/external-ai"
    request = ExternalAIUpdate
    response = ExternalAIView

    def handle(self, body: ExternalAIUpdate, user: SessionUser = Depends(admin_required)) -> dict:
        import core.lyrics.translations as storage

        encrypted = None
        replace = body.api_key is not None
        if replace:
            secret = body.api_key.get_secret_value()
            encrypted = credential_cipher().encrypt(secret.encode()) if secret else None
        settings = ExternalAISettings.model_validate(body.model_dump())
        has_key = storage.save_settings(settings, encrypted, replace_key=replace)
        return {**settings.model_dump(), "has_key": has_key}
