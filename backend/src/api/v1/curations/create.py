"""``POST /curations``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.curations._models import CurationCreateRequest, CurationSaved
from backend.api.v1.curations._shared import recipe_columns, refresh, saved, unique_name
from core.db.models import Curation
from core.db.session import session_scope
from core.errors import ConflictError


class CreateCuration(Endpoint):
    """Save a curation and queue its first refresh (which builds the Navidrome playlist)."""

    method = "POST"
    path = ""
    request = CurationCreateRequest
    response = CurationSaved
    status_code = 201

    def handle(
        self, request: CurationCreateRequest, user: SessionUser = Depends(current_user)
    ) -> dict:
        connection_id = user.get("navidrome_connection_id")
        if connection_id is None:
            raise ConflictError("No Navidrome connection is configured")
        with unique_name(), session_scope() as session:
            curation = Curation(
                user_id=user["id"], navidrome_connection_id=connection_id, **recipe_columns(request)
            )
            session.add(curation)
            session.flush()
            curation_id = curation.id
        return saved(request, curation_id, refresh(curation_id, user["id"]))
