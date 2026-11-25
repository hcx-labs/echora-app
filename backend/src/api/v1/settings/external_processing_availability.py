"""``GET /settings/external-processing/availability``."""

from __future__ import annotations

import uuid

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from worker.remote.settings import compute_for


class Availability(ResponseModel):
    """Whether the user may sync on Modal, and the preselected location."""

    available: bool
    default: str


class ModalAvailability(Endpoint):
    """Whether this user may process a sync on Modal, and the preselected choice."""

    method = "GET"
    path = "/external-processing/availability"
    response = Availability

    def handle(self, user: SessionUser = Depends(current_user)) -> dict[str, object]:
        try:
            default = compute_for(uuid.UUID(str(user["id"])), None)
            allowed = compute_for(uuid.UUID(str(user["id"])), "modal") == "modal"
        except PermissionError:
            allowed = False
            default = "local"
        return {"available": allowed, "default": default}
