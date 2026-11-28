"""``PUT /settings/models/karaoke/realign``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.requests import PutRequest
from backend.api.responses import ResponseModel
from backend.api.v1.settings._deps import admin_required
from core.db.connection import connect
from core.settings import store as settings


class RealignRequest(PutRequest):
    """The realign switch."""

    enabled: bool
    """Whether a full sync realigns karaoke of an older pipeline or model revision."""


class RealignState(ResponseModel):
    """The stored realign switch."""

    enabled: bool


class KaraokeRealignSwitch(Endpoint):
    """Whether full syncs bring every song's karaoke to the current revision.

    Off, a full sync still aligns songs without karaoke (and lyrics that changed); songs asked
    to be resynced (Processing information) are realigned by any sync.
    """

    method = "PUT"
    path = "/models/karaoke/realign"
    request = RealignRequest
    response = RealignState

    def handle(
        self, request: RealignRequest, user: SessionUser = Depends(admin_required)
    ) -> dict[str, object]:
        with connect() as connection:
            settings.set_karaoke_realign_outdated(connection, request.enabled)
        return {"enabled": request.enabled}
