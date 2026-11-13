"""Instance-wide, admin-trusted OpenAI-compatible configuration.

Local/private HTTP destinations are intentional: admins control network egress.
Use network policy to restrict it where needed. HTTP exposes keys/lyrics on the
network; prefer HTTPS off-host. Never log request bodies or authorization headers.
URL is the API base (e.g. http://localhost:8000/v1), not a completion URL.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

DEFAULT_PROMPT = "Translate the song into natural, expressive lyrics that feel at home in the target language. Use the whole song for context. Aim to carry its meaning, emotion, imagery, voice, and energy rather than translating word for word. You may reshape phrasing, rhythm, rhyme, slang, and punctuation to make each line flow and sing naturally. Preserve recurring hooks in a recognizable way, but adapt them when the context calls for it.\n\nThe one structural constraint: each source line must produce exactly one translated line, in the same order. Do not merge, split, skip, or add lines. Preserve any supplied line IDs and follow the requested output format. Return only the translation."


class LanguagePair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    target: str

    @field_validator("source", "target")
    @classmethod
    def normalize(cls, value: str) -> str:
        value = value.strip().replace("_", "-").lower()
        if not re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*", value):
            raise ValueError("Expected a language tag")
        return value

    @model_validator(mode="after")
    def different(self):
        if self.source == self.target:
            raise ValueError("Source and target must differ")
        return self


class ExternalAISettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    url: str = Field(default="", max_length=2048)
    model: str = Field(default="", max_length=200)
    prompt: str = Field(default=DEFAULT_PROMPT, min_length=1, max_length=8000)
    language_pairs: list[LanguagePair] = Field(default_factory=list, max_length=64)

    @field_validator("url")
    @classmethod
    def endpoint(cls, value: str) -> str:
        if not value:
            return value
        try:
            parsed = urlsplit(value)
            if (
                any(c.isspace() or ord(c) < 32 for c in value)
                or "\\" in value
                or parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or "?" in value
                or "#" in value
            ):
                raise ValueError()
            _ = parsed.port
        except ValueError:
            raise ValueError(
                "Expected an HTTP(S) base URL without userinfo, query or fragment"
            ) from None
        return value.rstrip("/")

    @field_validator("model", "prompt")
    @classmethod
    def trim(cls, value):
        return value.strip()

    @model_validator(mode="after")
    def coherent(self):
        pairs = [(p.source, p.target) for p in self.language_pairs]
        if len(set(pairs)) != len(pairs):
            raise ValueError("Duplicate language pair")
        if not self.prompt or (self.enabled and not self.url):
            raise ValueError("Enabled API requires a URL; prompt cannot be blank")
        return self


class SettingsUpdate(ExternalAISettings):
    # Omitted/null preserves; empty string clears; otherwise replaces.
    api_key: SecretStr | None = Field(default=None, max_length=8192, exclude=True)


class EndpointUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    url: str = Field(default="", max_length=2048)
    api_key: SecretStr | None = Field(default=None, max_length=8192, exclude=True)

    @field_validator("url")
    @classmethod
    def endpoint(cls, value):
        return ExternalAISettings.endpoint(value)

    @model_validator(mode="after")
    def coherent(self):
        if self.enabled and not self.url:
            raise ValueError("Enabled API requires a URL")
        return self


class TranslationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(default="", max_length=200)
    prompt: str = Field(default=DEFAULT_PROMPT, min_length=1, max_length=8000)
    language_pairs: list[LanguagePair] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def coherent(self):
        validated = ExternalAISettings(**self.model_dump())
        self.model, self.prompt = validated.model, validated.prompt
        return self


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
    from . import translation_storage as storage

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
        return {key: value[key] for key in ("enabled", "url", "has_key")}

    def translation_response(value):
        return {key: value[key] for key in ("model", "prompt", "language_pairs")}

    @api.get("/endpoint")
    def read_endpoint(user=Depends(admin)):
        return endpoint_response(read_settings(user))

    @api.get("/translation")
    def read_translation(user=Depends(admin)):
        return translation_response(read_settings(user))

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
