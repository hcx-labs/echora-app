"""``GET /library/tracks/{track_id}/recording-group``."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from core.db.connection import connect
from core.library import store


class RecordingGroup(ResponseModel):
    """Songs fingerprinted as the same recording, and the evidence that joined them."""

    id: uuid.UUID
    status: str
    canonical_track_id: uuid.UUID | None = None
    created_at: datetime
    updated_at: datetime | None = None
    members: list[dict[str, Any]]
    evidence: list[dict[str, Any]]


class TrackRecordingGroupState(ResponseModel):
    """The song's group, or none (and whether it has a fingerprint yet)."""

    group: RecordingGroup | None
    fingerprinted: bool


class TrackRecordingGroup(Endpoint):
    """The recording group a song belongs to."""

    method = "GET"
    path = "/tracks/{track_id}/recording-group"
    response = TrackRecordingGroupState

    def handle(self, track_id: uuid.UUID, user: SessionUser = Depends(current_user)) -> dict:
        with connect() as connection:
            group = store.recording_group(connection, track_id)
            if group is None:
                return {
                    "group": None,
                    "fingerprinted": store.is_fingerprinted(connection, track_id),
                }
            members = store.recording_group_members(connection, group["id"])
            evidence = store.recording_match_evidence(
                connection, [member["id"] for member in members]
            )
        return {"group": {**group, "members": members, "evidence": evidence}, "fingerprinted": True}
