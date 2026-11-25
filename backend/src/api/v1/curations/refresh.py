"""``POST /curations/{curation_id}/refresh``."""

from __future__ import annotations

import uuid

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.curations._models import RefreshQueued
from backend.api.v1.curations._shared import refresh


class RefreshCuration(Endpoint):
    """Queue a new revision of a curation now."""

    method = "POST"
    path = "/{curation_id}/refresh"
    response = RefreshQueued
    status_code = 202

    def handle(self, curation_id: uuid.UUID, user: SessionUser = Depends(current_user)) -> dict:
        return refresh(curation_id, user["id"])
