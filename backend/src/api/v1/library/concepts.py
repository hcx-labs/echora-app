"""``GET /library/concepts``."""

from __future__ import annotations

from typing import Any

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.api.v1.library._concept_models import Concept
from backend.library.db import connect
from core.curation.concepts import predefined_concepts


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
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT id, name, description, positive_prompts, negative_prompts,
                          positive_track_ids, negative_track_ids, enabled, created_at, updated_at
                   FROM concepts WHERE user_id=%s ORDER BY lower(name)""",
                (user["id"],),
            )
            personal = cursor.fetchall()
        return {"predefined": predefined_concepts(), "personal": personal}
