"""``GET /library/filter-options``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.api.v1.library._models import NameCount
from core.db.connection import connect
from core.library import browse


class FilterOptions(ResponseModel):
    """Values the browse filters offer, with song counts."""

    language: list[NameCount]
    """Lyrics languages (``unknown`` for none)."""
    genre: list[NameCount]
    translation: list[NameCount]
    """Target languages of ready translations."""


class FilterOptionsEndpoint(Endpoint):
    """The languages, genres and translations in the user's library."""

    method = "GET"
    path = "/filter-options"
    response = FilterOptions

    def handle(self, user: SessionUser = Depends(current_user)) -> dict:
        with connect() as connection:
            rows = browse.filter_options(connection, user["id"])
        return {
            kind: [
                {"name": row["value"], "tracks": row["tracks"]}
                for row in rows
                if row["kind"] == kind
            ]
            for kind in ("language", "genre", "translation")
        }
