"""``POST /search/concepts``."""

from __future__ import annotations

from fastapi import Depends
from pydantic import Field

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.requests import PostRequest
from backend.api.responses import ResponseModel
from backend.search.concepts import ConceptLens, LensQuery
from backend.search.service import run_search


class ConceptLensRequest(PostRequest):
    """Up to six concept names, by predefined or personal concept name."""

    concepts: list[str] = Field(min_length=1, max_length=6)
    minimum_percentile: float = Field(default=0.5, ge=0, le=1)
    representation: str = Field(default="hybrid", pattern="^(semantic|lyrics|hybrid)$")


class ConceptScore(ResponseModel):
    """How one song scores on one concept."""

    name: str
    raw_score: float
    percentile: float
    semantic_percentile: float | None
    lyrics_percentile: float | None
    lyrics_available: bool


class ConceptScores(ResponseModel):
    """Per song id, the concepts it reaches ``minimum_percentile`` on."""

    run_id: str
    corpus_size: int
    calibration: str
    minimum_percentile: float
    representation: str
    scores: dict[str, list[ConceptScore]]


class ConceptSearch(Endpoint):
    """Score the user's songs against named concepts, calibrated within the library."""

    method = "POST"
    path = "/concepts"
    request = ConceptLensRequest
    response = ConceptScores

    async def handle(
        self, request: ConceptLensRequest, user: SessionUser = Depends(current_user)
    ) -> dict:
        query = LensQuery(
            concepts=request.concepts,
            minimum_percentile=request.minimum_percentile,
            representation=request.representation,
        )
        return await run_search(ConceptLens, user["id"], query)
