"""``GET /library/concepts``."""

from __future__ import annotations

from typing import Any

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.api.v1.library._concept_models import Concept
from core.curation import store
from core.curation.concepts import predefined_concepts
from core.db.connection import connect


class ConceptList(ResponseModel):
    """Echora's predefined concepts and the user's own."""

    predefined: list[dict[str, Any]]
    personal: list[Concept]


class Concepts(Endpoint):
    """The concepts the user can score songs by."""

    method = "GET"
    path = "/concepts"
    response = ConceptList

    def handle(self, user: SessionUser = Depends(current_user)) -> dict:
        with connect() as connection:
            personal = store.concepts(connection, user["id"])
        return {"predefined": predefined_concepts(), "personal": personal}
