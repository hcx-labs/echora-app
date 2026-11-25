"""``GET /auth/session``."""

from __future__ import annotations

from fastapi import Cookie, Response

from backend.api.endpoint import Endpoint
from backend.api.v1.auth._models import SessionState
from backend.sessions import store


class ReadSession(Endpoint):
    """The session's deadlines; reading never extends them."""

    method = "GET"
    path = "/session"
    response = SessionState

    def handle(
        self, response: Response, echora_session: str | None = Cookie(default=None)
    ) -> dict[str, object]:
        response.headers["Cache-Control"] = "no-store"
        return store.session_status(echora_session)
