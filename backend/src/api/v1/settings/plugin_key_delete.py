"""``DELETE /settings/integrations/navidrome/keys/{key_id}``."""

from __future__ import annotations

import uuid

from fastapi import Depends, Response

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from core.db.connection import connect
from core.errors import NotFoundError
from core.plugin import store


class RevokePluginKey(Endpoint):
    """Revoke one of the user's API keys (revoking twice keeps the first time)."""

    method = "DELETE"
    path = "/integrations/navidrome/keys/{key_id}"
    status_code = 204

    def handle(self, key_id: uuid.UUID, user: SessionUser = Depends(current_user)) -> Response:
        with connect() as db:
            if not store.revoke_key(db, key_id, user["id"]):
                raise NotFoundError("Integration key not found")
        return Response(status_code=204)
