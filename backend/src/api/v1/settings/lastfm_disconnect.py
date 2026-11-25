"""``DELETE /settings/lastfm``."""

from __future__ import annotations

from fastapi import Depends, Response

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from core.db.session import session_scope
from core.users import store as users


class DisconnectLastFm(Endpoint):
    """Forget the Last.fm account and key."""

    method = "DELETE"
    path = "/lastfm"
    status_code = 204

    def handle(self, user: SessionUser = Depends(current_user)) -> Response:
        with session_scope() as session:
            preference = users.preference(session, user["id"])
            if preference is not None:
                preference.lastfm_username = None
                preference.lastfm_api_key_encrypted = None
                users.touch(preference)
        return Response(status_code=204)
