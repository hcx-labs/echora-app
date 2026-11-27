"""``PUT /settings/models/karaoke-stems`` and ``GET /settings/models/karaoke-stems``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.api.v1.settings._deps import admin_required
from backend.api.v1.settings._models import ModelSwitch, SwitchState
from core.db.connection import connect
from core.settings import store as settings
from core.stems import store as stems


class KaraokeStemsSwitch(Endpoint):
    """Turn karaoke stems on or off for syncs here or on Modal.

    Syncs then separate and store each song's vocals and instrumental, which the Stage player
    remixes. The files live in ``ECHORA_KARAOKE_STEMS_DIR``, which must be persistent storage.
    """

    method = "PUT"
    path = "/models/karaoke-stems"
    request = ModelSwitch
    response = SwitchState

    def handle(
        self, request: ModelSwitch, user: SessionUser = Depends(admin_required)
    ) -> dict[str, object]:
        column = (
            "karaoke_stems_modal_enabled"
            if request.location == "modal"
            else "karaoke_stems_processing_enabled"
        )
        with connect() as connection:
            settings.set_switch(connection, column, request.enabled, unbind_karaoke=False)
        return {"enabled": request.enabled, "location": request.location}


class KaraokeStemsUsage(ResponseModel):
    """What karaoke stems are stored."""

    complete: int
    failed: int
    bytes: int
    """The complete stems' total size on disk."""


class KaraokeStemsStatus(Endpoint):
    """How many songs have karaoke stems, and the space they take."""

    method = "GET"
    path = "/models/karaoke-stems"
    response = KaraokeStemsUsage

    def handle(self, user: SessionUser = Depends(admin_required)) -> dict[str, int]:
        with connect() as connection:
            counts = stems.stems_counts(connection)
        return {key: int(counts.get(key) or 0) for key in ("complete", "failed", "bytes")}
