"""``PUT /settings/models/transcription``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.v1.settings._deps import admin_required
from backend.api.v1.settings._models import ModelSwitch, SwitchState
from core.db.connection import connect
from core.settings import store as settings


class TranscriptionSwitch(Endpoint):
    """Turn lyrics transcription on or off for syncs here or on Modal."""

    method = "PUT"
    path = "/models/transcription"
    request = ModelSwitch
    response = SwitchState

    def handle(
        self, request: ModelSwitch, user: SessionUser = Depends(admin_required)
    ) -> dict[str, object]:
        column = (
            "transcription_modal_enabled"
            if request.location == "modal"
            else "transcription_processing_enabled"
        )
        with connect() as connection:
            settings.set_switch(connection, column, request.enabled, unbind_karaoke=False)
        return {"enabled": request.enabled, "location": request.location}
