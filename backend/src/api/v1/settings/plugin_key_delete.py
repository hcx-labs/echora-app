"""``DELETE /settings/integrations/navidrome/keys/{key_id}``."""

from __future__ import annotations

import uuid

from fastapi import Depends, Response

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.settings.plugin_integration import connect
from core.errors import NotFoundError


class RevokePluginKey(Endpoint):
    """Revoke one of the user's API keys (revoking twice keeps the first time)."""

    method = "DELETE"
    path = "/integrations/navidrome/keys/{key_id}"
    status_code = 204

    def handle(self, key_id: uuid.UUID, user: SessionUser = Depends(current_user)) -> Response:
        with connect() as db:
            row = db.execute(
                """UPDATE navidrome_api_keys SET revoked_at=coalesce(revoked_at,now())
                WHERE id=%s AND user_id=%s RETURNING id""",
                (key_id, user["id"]),
            ).fetchone()
            if row is None:
                raise NotFoundError("Integration key not found")
        return Response(status_code=204)
