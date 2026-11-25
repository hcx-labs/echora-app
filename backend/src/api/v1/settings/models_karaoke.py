"""``PUT /settings/models/karaoke``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.v1.settings._deps import admin_required
from backend.api.v1.settings._models import ModelSwitch, SwitchWithBacklog
from core.db.connection import connect
from core.lyrics import store as lyrics
from core.settings import store as settings
from worker.kara.pipeline import KARAOKE_PIPELINE_REVISION


class KaraokeSwitch(Endpoint):
    """Turn karaoke alignment on or off for syncs here or on Modal."""

    method = "PUT"
    path = "/models/karaoke"
    request = ModelSwitch
    response = SwitchWithBacklog

    def handle(
        self, request: ModelSwitch, user: SessionUser = Depends(admin_required)
    ) -> dict[str, object]:
        column = (
            "karaoke_modal_enabled" if request.location == "modal" else "karaoke_processing_enabled"
        )
        with connect() as connection:
            settings.set_switch(connection, column, request.enabled, unbind_karaoke=True)
            pending = (
                lyrics.synced_without_karaoke(connection, KARAOKE_PIPELINE_REVISION)
                if request.enabled
                else 0
            )
        return {"enabled": request.enabled, "location": request.location, "pending": pending}
