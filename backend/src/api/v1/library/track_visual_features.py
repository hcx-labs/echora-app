"""``GET /library/tracks/{track_id}/visual-features``."""

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
        with connect() as connection:
            store.require_visible(connection, user["id"], track_id, "Track not found")
            cache = store.visual_features(connection, track_id, VISUAL_FEATURE_REVISION)
            # Read enrichment at request time: ingest ordering never requires a DSP
            # rerun, and this authenticated endpoint never schedules model work.
            descriptors = store.audio_descriptors(connection, track_id, DESCRIPTOR_REVISION)
            vocal = store.vocal_activity(connection, track_id)
            contour = store.preview_contour(connection, track_id, MELODY_CONTOUR_REVISION)
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
