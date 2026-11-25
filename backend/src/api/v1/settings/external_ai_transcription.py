"""``GET /settings/external-ai/transcription``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.v1.settings._deps import admin_required
from backend.api.v1.settings._external_ai import TranscriptionView, current, transcription_view


class ExternalAITranscription(Endpoint):
    """The External AI transcription settings (administrators only)."""

    method = "GET"
    path = "/external-ai/transcription"
    response = TranscriptionView

    def handle(self, user: SessionUser = Depends(admin_required)) -> dict:
        return transcription_view(current())
