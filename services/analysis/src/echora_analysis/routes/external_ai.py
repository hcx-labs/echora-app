"""External AI settings endpoints (admin): the endpoint, translation, transcription and themes."""

from __future__ import annotations

from core.integrations.external_ai import (
    DEFAULT_PROMPT,
    DEFAULT_TRANSCRIPTION_PROMPT,
    ExternalAISettings,
    LanguagePair,
    Provider,
)
from fastapi import APIRouter, Depends, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


class SettingsUpdate(ExternalAISettings):
    # Omitted/null preserves; empty string clears; otherwise replaces.
    api_key: SecretStr | None = Field(default=None, max_length=8192, exclude=True)


class EndpointUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
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


class TranslationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(default=DEFAULT_PROMPT, min_length=1, max_length=8000)
    language_pairs: list[LanguagePair] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def coherent(self):
        validated = ExternalAISettings(**self.model_dump())
        self.prompt = validated.prompt
        return self


class TranscriptionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transcription_refinement: bool = True
    transcription_prompt: str = Field(
        default=DEFAULT_TRANSCRIPTION_PROMPT, min_length=1, max_length=8000
    )
    transcription_model: str = Field(default="", max_length=200)

    @model_validator(mode="after")
    def coherent(self):
        validated = ExternalAISettings(**self.model_dump())
        self.transcription_prompt = validated.transcription_prompt
        self.transcription_model = validated.transcription_model
        return self


class ThemesUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lyric_themes: bool = False


class RedactedSettingsRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def redacted(request):
            try:
                return await handler(request)
            except RequestValidationError:
                # Pydantic model-level errors can echo the entire body, including keys.
                raise HTTPException(422, "Invalid External AI settings") from None

        return redacted


def router(require_user, cipher):
    import core.lyrics.translations as storage

    api = APIRouter(
        prefix="/settings/external-ai", tags=["settings"], route_class=RedactedSettingsRoute
    )

    def admin(user=Depends(require_user)):
        if not user.get("is_admin"):
            raise HTTPException(403, "Administrator access required")
        return user

    @api.get("")
    def read_settings(user=Depends(admin)):
        settings, encrypted = storage.load_settings()
        return {**settings.model_dump(), "has_key": bool(encrypted)}

    @api.put("")
    def write_settings(body: SettingsUpdate, user=Depends(admin)):
        encrypted = None
        replace = body.api_key is not None
        if replace:
            secret = body.api_key.get_secret_value()
            encrypted = cipher().encrypt(secret.encode()) if secret else None
        settings = ExternalAISettings.model_validate(body.model_dump())
        has_key = storage.save_settings(settings, encrypted, replace_key=replace)
        return {**settings.model_dump(), "has_key": has_key}

    def endpoint_response(value):
        return {key: value[key] for key in ("enabled", "provider", "url", "model", "has_key")}

    def translation_response(value):
        return {key: value[key] for key in ("prompt", "language_pairs")}

    def transcription_response(value):
        # `enabled` is read-only here: the section only acts while External AI is on.
        return {
            **{
                key: value[key]
                for key in (
                    "enabled",
                    "transcription_refinement",
                    "transcription_prompt",
                    "transcription_model",
                    "model",
                )
            },
            "default_transcription_prompt": DEFAULT_TRANSCRIPTION_PROMPT,
        }

    def themes_response(value):
        # `enabled` is read-only here, as for transcription.
        return {key: value[key] for key in ("enabled", "lyric_themes")}

    @api.get("/endpoint")
    def read_endpoint(user=Depends(admin)):
        return endpoint_response(read_settings(user))

    @api.get("/translation")
    def read_translation(user=Depends(admin)):
        return translation_response(read_settings(user))

    @api.get("/transcription")
    def read_transcription(user=Depends(admin)):
        return transcription_response(read_settings(user))

    @api.get("/themes")
    def read_themes(user=Depends(admin)):
        return themes_response(read_settings(user))

    @api.put("/themes")
    def write_themes(body: ThemesUpdate, user=Depends(admin)):
        return themes_response(storage.save_section(body.model_dump()))

    @api.put("/transcription")
    def write_transcription(body: TranscriptionUpdate, user=Depends(admin)):
        return transcription_response(storage.save_section(body.model_dump()))

    @api.put("/endpoint")
    def write_endpoint(body: EndpointUpdate, user=Depends(admin)):
        secret = body.api_key.get_secret_value() if body.api_key is not None else None
        encrypted = cipher().encrypt(secret.encode()) if secret else None
        return endpoint_response(
            storage.save_section(body.model_dump(), encrypted, replace_key=secret is not None)
        )

    @api.put("/translation")
    def write_translation(body: TranslationUpdate, user=Depends(admin)):
        return translation_response(storage.save_section(body.model_dump()))

    @api.delete("/translations")
    def clear_translations(user=Depends(admin)):
        try:
            count = storage.clear_translations()
        except RuntimeError:
            raise HTTPException(
                409, "Wait for active sync and lyrics jobs to stop before clearing translations"
            ) from None
        return {"deleted": count}

    return api
