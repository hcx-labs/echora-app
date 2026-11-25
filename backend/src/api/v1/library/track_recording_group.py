"""``GET /library/tracks/{track_id}/recording-group``."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.library.db import connect


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
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT rg.id, rg.status, rg.canonical_track_id, rg.created_at, rg.updated_at
                   FROM recording_group_members member JOIN recording_groups rg ON rg.id=member.group_id
                   WHERE member.track_id=%s""",
                (track_id,),
            )
            group = cursor.fetchone()
            if group is None:
                cursor.execute("SELECT 1 FROM track_fingerprints WHERE track_id=%s", (track_id,))
                return {"group": None, "fingerprinted": cursor.fetchone() is not None}
            cursor.execute(
                """SELECT t.id, t.title, t.artist, t.album, t.duration_seconds, member.confidence,
                          member.membership_status, array_agg(DISTINCT ts.external_id) AS source_ids
                   FROM recording_group_members member JOIN tracks t ON t.id=member.track_id
                   LEFT JOIN track_sources ts ON ts.track_id=t.id WHERE member.group_id=%s
                   GROUP BY t.id, member.confidence, member.membership_status ORDER BY t.title""",
                (group["id"],),
            )
            members = cursor.fetchall()
            cursor.execute(
                """SELECT id, left_track_id, right_track_id, decision, chromaprint_score,
                          duration_delta_seconds, semantic_similarity, acoustic_similarity,
                          matcher_revision, created_at FROM recording_match_evidence
                   WHERE left_track_id=ANY(%s) AND right_track_id=ANY(%s) ORDER BY created_at DESC""",
                ([member["id"] for member in members], [member["id"] for member in members]),
            )
            evidence = cursor.fetchall()
        return {"group": {**group, "members": members, "evidence": evidence}, "fingerprinted": True}
