"""``GET /curations``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.curations._models import CurationList
from core.curation import store
from core.db.connection import connect


class ListCurations(Endpoint):
    """The user's curations with their latest revision's songs."""

    method = "GET"
    path = ""
    response = CurationList

    def handle(self, user: SessionUser = Depends(current_user)) -> dict:
        with connect() as connection:
            return {"curations": store.curations(connection, user["id"])}
