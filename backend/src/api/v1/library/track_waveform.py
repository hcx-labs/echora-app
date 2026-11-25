"""``GET /library/tracks/{track_id}/waveform``."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.library.db import connect, require_visible
from backend.library.melody_preview import melody_preview
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
        with connect() as connection, connection.cursor() as cursor:
            require_visible(cursor, user["id"], track_id, "Track not found")
            cursor.execute(
                "SELECT revision, duration_seconds, peaks FROM track_waveforms WHERE track_id=%s AND revision=%s",
                (track_id, WAVEFORM_REVISION),
            )
            waveform = cursor.fetchone()
            cursor.execute(
                """SELECT mc.source, mc.pitch, mc.voiced, mc.hop_seconds
                   FROM melody_contours mc JOIN analysis_runs ar ON ar.id=mc.run_id
                   WHERE mc.track_id=%s AND ar.status IN ('complete','running')
                   ORDER BY CASE mc.source WHEN 'full-mix' THEN 0 WHEN 'vocals' THEN 1 ELSE 2 END,
                            ar.created_at DESC LIMIT 1""",
                (track_id,),
            )
            contour = cursor.fetchone()
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
