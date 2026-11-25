"""``GET /library/tracks/{track_id}/audio-descriptors``."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.library.db import connect, require_visible
from worker.library.descriptors import DESCRIPTOR_REVISION

VOCAL_ACTIVITY_SQL = """SELECT activity.activity, activity.run_id
   FROM track_vocal_activity activity
   JOIN current_analysis_runs ar ON ar.id=activity.run_id
   WHERE activity.track_id=%s AND ar.kind='voice_classification'
   ORDER BY activity.created_at DESC LIMIT 1"""

DESCRIPTORS_SQL = """SELECT revision, status, descriptors, created_at FROM track_audio_descriptors
   WHERE track_id=%s AND revision=%s"""


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
        with connect() as connection, connection.cursor() as cursor:
            require_visible(cursor, user["id"], track_id, "Track not found")
            cursor.execute(DESCRIPTORS_SQL, (track_id, DESCRIPTOR_REVISION))
            descriptors = cursor.fetchone()
            cursor.execute(VOCAL_ACTIVITY_SQL, (track_id,))
            vocal = cursor.fetchone()
        return {
            "track_id": str(track_id),
            "status": descriptors["status"] if descriptors else "pending",
            "descriptors": descriptors,
            "vocal_activity": vocal,
            "used_in_curation": {"sound_profile": True},
        }
