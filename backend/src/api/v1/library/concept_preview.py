"""``POST /library/concepts/preview``."""

from __future__ import annotations

import uuid

import numpy as np
from fastapi import Depends
from pydantic import Field

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.api.v1.library._concept_models import ConceptExamples
from backend.search.corpora import semantic_corpus
from core.curation.concepts import score_concept
from core.errors import InvalidQueryError


class ConceptPreview(ConceptExamples):
    """A concept to try before saving it, and how many songs to return."""

    limit: int = Field(default=50, ge=1, le=200)


class ScoredTrack(ResponseModel):
    """A song, its raw concept score and its percentile in the library."""

    id: uuid.UUID
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    raw_score: float
    percentile: float


class ConceptPreviewResult(ResponseModel):
    """The library's best matches for the concept, scored on MuQ-MuLan embeddings."""

    run_id: str
    corpus_size: int
    calibration: str
    tracks: list[ScoredTrack]


class ConceptPreviewEndpoint(Endpoint):
    """Score the user's songs against an unsaved concept."""

    method = "POST"
    path = "/concepts/preview"
    request = ConceptPreview
    response = ConceptPreviewResult

    def handle(self, body: ConceptPreview, user: SessionUser = Depends(current_user)) -> dict:
        if not body.positive_prompts and not body.positive_track_ids:
            raise InvalidQueryError("A concept needs a positive prompt or positive track")
        rows, matrix, run_id = semantic_corpus(user["id"])
        row_index = {str(row["id"]): index for index, row in enumerate(rows)}

        def examples(identifiers: list[uuid.UUID]) -> np.ndarray | None:
            indices = [
                row_index[str(identifier)]
                for identifier in identifiers
                if str(identifier) in row_index
            ]
            return matrix[indices] if indices else None

        raw, percentiles = score_concept(
            matrix,
            body.positive_prompts,
            body.negative_prompts,
            examples(body.positive_track_ids),
            examples(body.negative_track_ids),
        )
        ranked = np.argsort(raw)[::-1][: body.limit]
        return {
            "run_id": run_id,
            "corpus_size": len(rows),
            "calibration": "empirical-library-percentile",
            "tracks": [
                {
                    **rows[index],
                    "raw_score": float(raw[index]),
                    "percentile": float(percentiles[index]),
                }
                for index in ranked
            ],
        }
