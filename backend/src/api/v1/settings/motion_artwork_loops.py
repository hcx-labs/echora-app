"""``DELETE /settings/motion-artwork/loops``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.api.v1.settings._deps import admin_required
from core.errors import ConflictError
from worker.artwork.settings import clear_loops


class LoopsCleared(ResponseModel):
    """Loop rows deleted and files removed."""

    deleted: int
    files_removed: int


class ClearMotionArtwork(Endpoint):
    """Delete every rendered loop (administrators only; not while syncs or imports run)."""

    method = "DELETE"
    path = "/motion-artwork/loops"
    response = LoopsCleared

    def handle(self, user: SessionUser = Depends(admin_required)) -> dict[str, int]:
        try:
            return clear_loops()
        except RuntimeError:
            raise ConflictError(
                "Wait for active syncs and imports to finish before clearing loops"
            ) from None
