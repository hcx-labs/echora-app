"""``GET /auth/oidc/start``."""

from __future__ import annotations

from fastapi import HTTPException, Request, Response

from backend.api.endpoint import Endpoint
from backend.config import OidcConfig
from backend.sessions.oidc import oidc_client


class OidcStart(Endpoint):
    """Redirect to the provider's sign-in page."""

    method = "GET"
    path = "/oidc/start"

    async def handle(self, request: Request) -> Response:
        client = oidc_client()
        if client is None:
            raise HTTPException(status_code=503, detail="OIDC is not configured")
        redirect_uri = OidcConfig.load().redirect_uri or str(request.url_for("oidc_callback"))
        return await client.authorize_redirect(request, redirect_uri)
