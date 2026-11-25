"""``POST /curations/journeys/preview``."""

from __future__ import annotations

import numpy as np
from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.curations._models import JourneyPreview as JourneyPreviewModel, JourneyRequest
from core.curation.journeys import (
    normalize_rows as normalize_journey_rows,
    select_journey,
    spherical_targets,
)
from core.db.connection import connect
from core.embeddings import store
from core.errors import InvalidQueryError, NotFoundError


class JourneyPreview(Endpoint):
    """A path of songs between two songs through MuQ-MuLan, MERT or blended space."""

    method = "POST"
    path = "/journeys/preview"
    request = JourneyRequest
    response = JourneyPreviewModel

    def handle(self, request: JourneyRequest, user: SessionUser = Depends(current_user)) -> dict:
        if request.start_track_id == request.end_track_id:
            raise InvalidQueryError("Journey endpoints must be different tracks")
        with connect() as connection:
            rows = store.journey_corpus(connection)
        identifiers = {str(row["id"]): index for index, row in enumerate(rows)}
        start_index, end_index = (
            identifiers.get(str(request.start_track_id)),
            identifiers.get(str(request.end_track_id)),
        )
        if start_index is None or end_index is None:
            raise NotFoundError("One or both journey endpoints lack required embeddings")
        start_group = rows[start_index].get("recording_group_id")
        if start_group and start_group == rows[end_index].get("recording_group_id"):
            raise InvalidQueryError("Journey endpoints must be different recordings")
        semantic = normalize_journey_rows(
            np.stack([store.parse_vector(row.pop("semantic_embedding")) for row in rows])
        )
        acoustic = normalize_journey_rows(
            np.stack([store.parse_vector(row.pop("acoustic_embedding")) for row in rows])
        )
        if request.mode == "semantic":
            matrix = semantic
            semantic_weight = 1.0
        elif request.mode == "acoustic":
            matrix = acoustic
            semantic_weight = 0.0
        else:
            semantic_weight = request.semantic_weight
            matrix = np.concatenate(
                [np.sqrt(semantic_weight) * semantic, np.sqrt(1 - semantic_weight) * acoustic],
                axis=1,
            )
        targets = spherical_targets(matrix[start_index], matrix[end_index], request.length)
        selected = select_journey(
            matrix,
            targets,
            start_index,
            end_index,
            [row.get("artist") for row in rows],
            [row.get("recording_group_id") for row in rows],
        )
        steps = []
        for position, (index, target_similarity, transition_similarity) in enumerate(selected):
            row = rows[index]
            steps.append(
                {
                    **row,
                    "position": position,
                    "target_progress": position / max(len(selected) - 1, 1),
                    "target_similarity": target_similarity,
                    "transition_similarity": transition_similarity,
                }
            )
        return {
            "mode": request.mode,
            "semantic_weight": semantic_weight,
            "requested_length": request.length,
            "steps": steps,
        }
