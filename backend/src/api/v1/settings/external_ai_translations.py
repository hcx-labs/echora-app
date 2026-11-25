"""``DELETE /settings/external-ai/translations``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.api.v1.settings._deps import admin_required
from core.errors import ConflictError


class Deleted(ResponseModel):
    """How many translations were deleted."""

    deleted: int


class ClearTranslations(Endpoint):
    """Delete every stored lyric translation (administrators only; not while syncs run)."""

    method = "DELETE"
    path = "/external-ai/translations"
    response = Deleted

    def handle(self, user: SessionUser = Depends(admin_required)) -> dict[str, int]:
        import core.lyrics.translations as storage

        try:
            count = storage.clear_translations()
        except RuntimeError:
            raise ConflictError(
                "Wait for active sync and lyrics jobs to stop before clearing translations"
            ) from None
        return {"deleted": count}
