"""``GET /library/map``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.library._models import MapPayload
from backend.library.community_map import library_map


class LibraryMap(Endpoint):
    """The library map for a representation: muq_mulan, mert, lyrics or blend.

    ``semantic_weight`` (0 to 1, default 0.5) mixes MuQ-MuLan and MERT for ``blend``.
    """

    method = "GET"
    path = "/map"
    response = MapPayload

    def handle(
        self,
        model: str = "muq_mulan",
        limit: int = 1500,
        semantic_weight: float | None = None,
        user: SessionUser = Depends(current_user),
    ) -> dict:
        return library_map(user["id"], model, limit, semantic_weight)
