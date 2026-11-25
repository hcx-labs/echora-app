"""``GET /library/tracks/{track_id}/waveform``."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.library.melody_preview import melody_preview
from core.db.connection import connect
from core.library import store
from worker.library.waveforms import WAVEFORM_REVISION


class Waveform(ResponseModel):
    """The song's waveform peaks and a melody preview, when computed."""

    track_id: str
    status: str
    """``complete`` or ``pending``."""
    waveform: dict[str, Any] | None
    """``revision``, ``duration_seconds`` and ``peaks``."""
    melody: dict[str, Any] | None


class TrackWaveform(Endpoint):
    """Waveform peaks and melody contour preview of one of the user's songs."""

    method = "GET"
    path = "/tracks/{track_id}/waveform"
    response = Waveform

    def handle(self, track_id: uuid.UUID, user: SessionUser = Depends(current_user)) -> dict:
        with connect() as connection:
            store.require_visible(connection, user["id"], track_id, "Track not found")
            waveform = store.waveform(connection, track_id, WAVEFORM_REVISION)
            contour = store.preview_contour(connection, track_id)
        melody = (
            melody_preview(
                contour["source"],
                contour["pitch"],
                contour["voiced"],
                float(contour["hop_seconds"]),
            )
            if contour
            else None
        )
        return {
            "track_id": str(track_id),
            "status": "complete" if waveform else "pending",
            "waveform": waveform,
            "melody": melody,
        }
