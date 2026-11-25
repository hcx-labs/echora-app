"""``GET /library/lyrics/status``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from core.db.connection import connect
from core.lyrics import store


class LyricsCoverage(ResponseModel):
    """Songs by lyrics status, across the instance, and how many have a lyrics embedding."""

    total: int
    available: int
    missing: int
    unavailable: int
    instrumental: int
    embedded: int


class LyricsStatus(Endpoint):
    """Lyrics coverage of the instance's songs."""

    method = "GET"
    path = "/lyrics/status"
    response = LyricsCoverage

    def handle(self, user: SessionUser = Depends(current_user)) -> dict:
        with connect() as connection:
            return store.coverage(connection)
