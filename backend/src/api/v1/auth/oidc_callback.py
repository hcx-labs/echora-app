"""``GET /auth/oidc/callback``."""

from __future__ import annotations

import logging

from authlib.integrations.starlette_client import OAuthError
from fastapi import HTTPException, Request, Response
from starlette.responses import RedirectResponse

from backend.api.endpoint import Endpoint
from backend.config import OidcConfig
from backend.sessions.oidc import oidc_client, sign_in
from backend.sessions.store import set_session_cookie

logger = logging.getLogger(__name__)


class OidcCallback(Endpoint):
    """Finish sign-in: open a session and redirect to the app.

    The provider's registered redirect URI (and ``ECHORA_OIDC_REDIRECT_URI``) must point here.
    """

    method = "GET"
    path = "/oidc/callback"
    name = "oidc_callback"

    async def handle(self, request: Request) -> Response:
        client = oidc_client()
        if client is None:
            raise HTTPException(status_code=503, detail="OIDC is not configured")
        try:
            token_data = await client.authorize_access_token(request)
            claims = token_data.get("userinfo") or await client.userinfo(token=token_data)
        except OAuthError as error:
            logger.warning("OIDC authentication failed: %s", error.error)
            raise HTTPException(status_code=401, detail="OIDC authentication failed") from error
        token, expires_at = sign_in(claims)
        response = RedirectResponse(OidcConfig.load().post_login_redirect, status_code=303)
        set_session_cookie(response, token, expires_at)
        return response
