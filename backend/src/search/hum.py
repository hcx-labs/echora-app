"""Hum search: a hummed or sung recording against the user's melody contours."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from backend.search.service import SearchService
from core.db.connection import connect
from worker.melody.index import contour_index_key, load_contours, match_hum


@dataclass(frozen=True)
class HumQuery:
    """A recording and how many matches to return."""

    audio: bytes
    limit: int


class HumSearch(SearchService[HumQuery, dict]):
    """Keeps the user's contours in memory until a contour is added or the recipe changes."""

    name = "hum"
    persistent = True
    max_instances = 4

    scope: uuid.UUID

    def index_key(self) -> tuple[str, int, str]:
        with connect() as connection:
            return contour_index_key(connection, self.scope)

    def open(self) -> None:
        with connect() as connection:
            self.contours = load_contours(connection, self.scope)

    def search(self, query: HumQuery) -> dict:
        return match_hum(self.scope, self.contours, query.audio, query.limit)

    def close(self) -> None:
        self.contours = []
