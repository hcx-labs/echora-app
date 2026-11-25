"""``POST /auth/logout``."""

from __future__ import annotations

from fastapi import Cookie, Response

from backend.api.endpoint import Endpoint
from backend.sessions.store import delete_session


class Logout(Endpoint):
    """End the session and clear its cookie."""

    method = "POST"
    path = "/logout"
    status_code = 204

    def handle(self, echora_session: str | None = Cookie(default=None)) -> Response:
        if echora_session:
            delete_session(echora_session)
        # A new response: FastAPI's injected one has no status, so returning it sends nothing.
        response = Response(status_code=204)
        response.delete_cookie("echora_session", path="/")
        return response
