"""``PUT /settings/motion-artwork/location``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.v1.settings._deps import admin_required
from backend.api.v1.settings._models import ModelSwitch, SwitchState
from worker.artwork.settings import _connect, load_settings, save_settings


class LocationState(SwitchState):
    """The stored switch, and whether the feature itself is on."""

    feature_enabled: bool


class MotionArtworkLocation(Endpoint):
    """Switch generation on or off for syncs on this server or on Modal (administrators only)."""

    method = "PUT"
    path = "/motion-artwork/location"
    request = ModelSwitch
    response = LocationState

    def handle(
        self, update: ModelSwitch, user: SessionUser = Depends(admin_required)
    ) -> dict[str, object]:
        column = "generate_on_modal" if update.location == "modal" else "generate_during_sync"
        with _connect() as connection, connection.cursor() as cursor:
            settings = load_settings(cursor).model_copy(update={column: update.enabled})
            save_settings(cursor, settings)
        return {
            "enabled": update.enabled,
            "location": update.location,
            "feature_enabled": settings.enabled,
        }
