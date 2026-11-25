"""``POST /library/concepts``."""

from __future__ import annotations

from fastapi import Depends
from pydantic import Field

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.library._concept_models import Concept, ConceptExamples
from core.curation import store
from core.db.connection import connect
from core.errors import InvalidQueryError


class ConceptCreate(ConceptExamples):
    """A new concept: a name, a description, and prompts or songs for and against it."""

    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)


class ConceptCreateEndpoint(Endpoint):
    """Save a concept of the user's own."""

    method = "POST"
    path = "/concepts"
    request = ConceptCreate
    response = Concept
    status_code = 201

    def handle(self, body: ConceptCreate, user: SessionUser = Depends(current_user)) -> dict:
        if not body.positive_prompts and not body.positive_track_ids:
            raise InvalidQueryError("A concept needs a positive prompt or positive track")
        with connect() as connection:
            return store.create_concept(
                connection,
                user["id"],
                body.name.strip(),
                body.description.strip(),
                body.positive_prompts,
                body.negative_prompts,
                body.positive_track_ids,
                body.negative_track_ids,
            )
