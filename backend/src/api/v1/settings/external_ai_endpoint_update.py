"""``PUT /settings/external-ai/endpoint``."""

from __future__ import annotations

from fastapi import Depends
from pydantic import Field, SecretStr, field_validator, model_validator

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.requests import PutRequest
from backend.api.v1.settings._deps import admin_required
from backend.api.v1.settings._external_ai import EndpointView, endpoint_view
from core.integrations.external_ai import ExternalAISettings, Provider
from core.library.connections import credential_cipher


class EndpointUpdate(PutRequest):
    """The provider, URL and model. ``api_key``: omitted or null keeps it, "" clears it."""

    enabled: bool = False
    provider: Provider = "openai"
    url: str = Field(default="", max_length=2048)
    # Shared by every External AI feature, so it lives with the endpoint.
    model: str = Field(default="", max_length=200)
    api_key: SecretStr | None = Field(default=None, max_length=8192, exclude=True)

    @field_validator("url")
    @classmethod
    def endpoint(cls, value):
        return ExternalAISettings.endpoint(value)

    @field_validator("model")
    @classmethod
    def trim(cls, value):
        return value.strip()

    @model_validator(mode="after")
    def coherent(self):
        if self.enabled and self.provider == "openai" and not self.url:
            raise ValueError("Enabled API requires a URL")
        return self


class ExternalAIEndpointUpdate(Endpoint):
    """Change the provider and its key (administrators only)."""

    method = "PUT"
    path = "/external-ai/endpoint"
    request = EndpointUpdate
    response = EndpointView

    def handle(self, body: EndpointUpdate, user: SessionUser = Depends(admin_required)) -> dict:
        import core.lyrics.translations as storage

        secret = body.api_key.get_secret_value() if body.api_key is not None else None
        encrypted = credential_cipher().encrypt(secret.encode()) if secret else None
        return endpoint_view(
            storage.save_section(body.model_dump(), encrypted, replace_key=secret is not None)
        )
