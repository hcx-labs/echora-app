"""``PUT /settings/external-ai/translation``."""

from __future__ import annotations

from fastapi import Depends
from pydantic import Field, model_validator

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.requests import PutRequest
from backend.api.v1.settings._deps import admin_required
from backend.api.v1.settings._external_ai import TranslationView, translation_view
from core.integrations.external_ai import DEFAULT_PROMPT, ExternalAISettings, LanguagePair


class TranslationUpdate(PutRequest):
    """The translation prompt and the language pairs to translate."""

    prompt: str = Field(default=DEFAULT_PROMPT, min_length=1, max_length=8000)
    language_pairs: list[LanguagePair] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def coherent(self):
        validated = ExternalAISettings(**self.model_dump())
        self.prompt = validated.prompt
        return self


class ExternalAITranslationUpdate(Endpoint):
    """Change the translation settings (administrators only)."""

    method = "PUT"
    path = "/external-ai/translation"
    request = TranslationUpdate
    response = TranslationView

    def handle(self, body: TranslationUpdate, user: SessionUser = Depends(admin_required)) -> dict:
        import core.lyrics.translations as storage

        return translation_view(storage.save_section(body.model_dump()))
