"""``GET /library/tracks/{track_id}/audio-descriptors``."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from core.db.connection import connect
from core.library import store
from worker.library.descriptors import DESCRIPTOR_REVISION


class AudioDescriptors(ResponseModel):
    """The song's current audio descriptors and vocal activity, when computed."""

    track_id: str
    status: str
    """The descriptors' status, ``pending`` before they are computed."""
    descriptors: dict[str, Any] | None
    """``revision``, ``status``, ``descriptors`` and ``created_at``."""
    vocal_activity: dict[str, Any] | None
    used_in_curation: dict[str, bool]


class TrackAudioDescriptors(Endpoint):
    """Audio descriptors (tempo, key, loudness...) and vocal activity of one of the user's songs."""

    method = "GET"
    path = "/tracks/{track_id}/audio-descriptors"
    response = AudioDescriptors

    def handle(self, track_id: uuid.UUID, user: SessionUser = Depends(current_user)) -> dict:
        with connect() as connection:
            store.require_visible(connection, user["id"], track_id, "Track not found")
            descriptors = store.audio_descriptors(connection, track_id, DESCRIPTOR_REVISION)
            vocal = store.vocal_activity(connection, track_id)
        return {
            "track_id": str(track_id),
            "status": descriptors["status"] if descriptors else "pending",
            "descriptors": descriptors,
            "vocal_activity": vocal,
            "used_in_curation": {"sound_profile": True},
        }
