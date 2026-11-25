"""``PUT /settings/models/hum``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.v1.settings._deps import admin_required
from backend.api.v1.settings._models import ModelSwitch, SwitchWithBacklog
from core.db.connection import connect
from core.library import store as library
from core.revisions import MELODY_CONTOUR_REVISION
from core.settings import store as settings


class HumSwitch(Endpoint):
    """Turn melody contour extraction (hum search) on or off for syncs here or on Modal."""

    method = "PUT"
    path = "/models/hum"
    request = ModelSwitch
    response = SwitchWithBacklog

    def handle(
        self, request: ModelSwitch, user: SessionUser = Depends(admin_required)
    ) -> dict[str, object]:
        column = "hum_modal_enabled" if request.location == "modal" else "hum_processing_enabled"
        with connect() as connection:
            settings.set_switch(connection, column, request.enabled, unbind_karaoke=True)
            pending = (
                library.songs_without_contour(connection, MELODY_CONTOUR_REVISION)
                if request.enabled
                else 0
            )
        return {"enabled": request.enabled, "location": request.location, "pending": pending}
