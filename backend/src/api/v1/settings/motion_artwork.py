"""``GET /settings/motion-artwork``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.api.v1.settings._deps import admin_required
from backend.api.v1.settings._motion_artwork import MotionArtworkView
from core.db.connection import connect
from worker.artwork.config import ComfyUIConfig
from worker.artwork.render import DEFAULT_INSTRUCTIONS
from worker.artwork.settings import load_settings


class MotionArtworkDefaults(ResponseModel):
    """What blank settings fall back to."""

    comfyui_url: str
    instructions: str


class MotionArtworkWithDefaults(MotionArtworkView):
    """The stored settings and their fallbacks."""

    defaults: MotionArtworkDefaults


class MotionArtworkSettingsRead(Endpoint):
    """The motion artwork settings (administrators only)."""

    method = "GET"
    path = "/motion-artwork"
    response = MotionArtworkWithDefaults

    def handle(self, user: SessionUser = Depends(admin_required)) -> dict[str, object]:
        with connect() as connection:
            settings = load_settings(connection)
        return {
            **settings.model_dump(),
            "defaults": {
                "comfyui_url": ComfyUIConfig.load().url,
                "instructions": DEFAULT_INSTRUCTIONS,
            },
        }
