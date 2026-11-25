"""``DELETE /settings/lastfm``."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import Depends, Response

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from core.db.models import UserPreference
from core.db.session import session_scope


class DisconnectLastFm(Endpoint):
    """Forget the Last.fm account and key."""

    method = "DELETE"
    path = "/lastfm"
    status_code = 204

    def handle(self, user: SessionUser = Depends(current_user)) -> Response:
        with session_scope() as session:
            preference = session.get(UserPreference, user["id"])
            if preference is not None:
                preference.lastfm_username = None
                preference.lastfm_api_key_encrypted = None
                preference.updated_at = datetime.now(UTC)
        return Response(status_code=204)
