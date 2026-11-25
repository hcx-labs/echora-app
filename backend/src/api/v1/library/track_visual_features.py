"""``GET /library/tracks/{track_id}/visual-features``."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.api.v1.library.track_audio_descriptors import DESCRIPTORS_SQL, VOCAL_ACTIVITY_SQL
from backend.library.db import connect, require_visible
from backend.library.melody_preview import melody_preview
from core.revisions import MELODY_CONTOUR_REVISION
from worker.library.descriptors import DESCRIPTOR_REVISION
from worker.library.visual_features import VISUAL_FEATURE_REVISION


class VisualEnrichment(ResponseModel):
    """Other analysis the visualizers use, read at request time."""

    descriptors: dict[str, Any] | None
    vocal_activity: dict[str, Any] | None
    melody: dict[str, Any] | None


class VisualFeatures(ResponseModel):
    """The player visualizers' per-frame features."""

    track_id: str
    status: str
    """The cache's status (``complete``, ``unsupported``...), ``pending`` before it exists."""
    visual_features: dict[str, Any] | None
    """Only when complete."""
    enrichment: VisualEnrichment


class TrackVisualFeatures(Endpoint):
    """Visualizer features of one of the user's songs, with descriptors, vocals and melody."""

    method = "GET"
    path = "/tracks/{track_id}/visual-features"
    response = VisualFeatures

    def handle(self, track_id: uuid.UUID, user: SessionUser = Depends(current_user)) -> dict:
        with connect() as connection, connection.cursor() as cursor:
            require_visible(cursor, user["id"], track_id, "Track not found")
            cursor.execute(
                """SELECT revision, status, duration_seconds, hop_seconds, features, created_at
                   FROM track_visual_features WHERE track_id=%s AND revision=%s""",
                (track_id, VISUAL_FEATURE_REVISION),
            )
            cache = cursor.fetchone()
            # Read enrichment at request time: ingest ordering never requires a DSP
            # rerun, and this authenticated endpoint never schedules model work.
            cursor.execute(DESCRIPTORS_SQL, (track_id, DESCRIPTOR_REVISION))
            descriptors = cursor.fetchone()
            cursor.execute(VOCAL_ACTIVITY_SQL, (track_id,))
            vocal = cursor.fetchone()
            cursor.execute(
                """SELECT mc.source, mc.pitch, mc.voiced, mc.hop_seconds
                   FROM melody_contours mc JOIN analysis_runs ar ON ar.id=mc.run_id
                   WHERE mc.track_id=%s AND ar.model_revision=%s AND ar.status IN ('complete','running')
                   ORDER BY CASE mc.source WHEN 'full-mix' THEN 0 WHEN 'vocals' THEN 1 ELSE 2 END,
                            ar.created_at DESC LIMIT 1""",
                (track_id, MELODY_CONTOUR_REVISION),
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
        status = cache["status"] if cache else "pending"
        return {
            "track_id": str(track_id),
            "status": status,
            "visual_features": cache if status == "complete" else None,
            "enrichment": {"descriptors": descriptors, "vocal_activity": vocal, "melody": melody},
        }
