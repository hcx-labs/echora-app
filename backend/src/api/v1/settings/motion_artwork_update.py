"""``PUT /settings/motion-artwork``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.v1.settings._deps import admin_required
from backend.api.v1.settings._motion_artwork import MotionArtworkUpdate, MotionArtworkView
from core.db.connection import connect
from worker.artwork.settings import MotionArtworkSettings, save_settings


class MotionArtworkSettingsUpdate(Endpoint):
    """Replace the motion artwork settings (administrators only)."""

    method = "PUT"
    path = "/motion-artwork"
    request = MotionArtworkUpdate
    response = MotionArtworkView

    def handle(
        self, update: MotionArtworkUpdate, user: SessionUser = Depends(admin_required)
    ) -> dict[str, object]:
        settings = MotionArtworkSettings.model_validate(update.model_dump())
        with connect() as connection:
            save_settings(connection, settings)
        return settings.model_dump()
