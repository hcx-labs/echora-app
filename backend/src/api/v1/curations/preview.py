"""``POST /curations/preview``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.curations._models import CurationPreview, CurationPreviewRequest
from core.curation.preview import preview_curation
from core.errors import ConflictError


class PreviewCuration(Endpoint):
    """The songs a recipe would select now, without saving anything."""

    method = "POST"
    path = "/preview"
    request = CurationPreviewRequest
    response = CurationPreview

    def handle(
        self, request: CurationPreviewRequest, user: SessionUser = Depends(current_user)
    ) -> dict:
        connection_id = user.get("navidrome_connection_id")
        if connection_id is None:
            raise ConflictError("No Navidrome connection is configured")
        return preview_curation(user["id"], connection_id, request)
