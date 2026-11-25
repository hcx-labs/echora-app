"""``PUT /settings/external-ai/transcription``."""

from __future__ import annotations

from fastapi import Depends
from pydantic import Field, model_validator

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.requests import PutRequest
from backend.api.v1.settings._deps import admin_required
from backend.api.v1.settings._external_ai import TranscriptionView, transcription_view
from core.integrations.external_ai import DEFAULT_TRANSCRIPTION_PROMPT, ExternalAISettings


class TranscriptionUpdate(PutRequest):
    """Transcript refinement: on or off, its prompt and its own model ("" uses the shared one)."""

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


class ExternalAITranscriptionUpdate(Endpoint):
    """Change the transcript refinement settings (administrators only)."""

    method = "PUT"
    path = "/external-ai/transcription"
    request = TranscriptionUpdate
    response = TranscriptionView

    def handle(
        self, body: TranscriptionUpdate, user: SessionUser = Depends(admin_required)
    ) -> dict:
        import core.lyrics.translations as storage

        return transcription_view(storage.save_section(body.model_dump()))
