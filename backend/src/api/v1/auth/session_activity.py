"""``POST /auth/session/activity``."""

from __future__ import annotations

from urllib.parse import urlsplit

from fastapi import Cookie, Header, HTTPException, Response

from backend.api.endpoint import Endpoint
from backend.api.v1.auth._models import SessionState
from backend.config import ApiConfig, OidcConfig, SessionConfig
from backend.sessions import store


def trusted_origins() -> set[str]:
    """Origins allowed to report activity: configured ones and the OIDC redirect origins.

    Never inferred from Host or forwarded headers.
    """
    settings, oidc = SessionConfig.load(), OidcConfig.load()
    allowed = {
        value.strip()
        for value in (settings.trusted_origins + "," + ApiConfig.load().cors_origins).split(",")
        if value.strip()
    }
    for redirect in (oidc.redirect_uri, oidc.post_login_redirect):
        parsed = urlsplit(redirect)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            allowed.add(f"{parsed.scheme}://{parsed.netloc}")
    return allowed


def _forbidden(detail: str) -> HTTPException:
    return HTTPException(status_code=403, detail=detail, headers={"Cache-Control": "no-store"})


class RecordActivity(Endpoint):
    """Record explicit user activity, renewing the session when a renewal is due.

    Requires ``X-Echora-Activity: 1`` and a trusted ``Origin``, enforced here as well as by the
    web app's proxy, so a cross-site request cannot keep a session alive.
    """

    method = "POST"
    path = "/session/activity"
    response = SessionState

    def handle(
        self,
        response: Response,
        echora_session: str | None = Cookie(default=None),
        x_echora_activity: str | None = Header(default=None),
        origin: str | None = Header(default=None),
    ) -> dict[str, object]:
        if x_echora_activity != "1":
            raise _forbidden("Activity header required")
        if not origin or origin == "null" or origin not in trusted_origins():
            raise _forbidden("Untrusted activity origin")
        result = store.session_status(echora_session, activity=True)
        # Reissue the authoritative deadline even when another tab already renewed it.
        store.set_session_cookie(response, echora_session, result["expires_at"])
        response.headers["Cache-Control"] = "no-store"
        return result
