"""``GET /auth/oidc/status``."""

from __future__ import annotations

from backend.api.endpoint import Endpoint
from backend.api.v1.auth._models import OidcState
from backend.sessions.oidc import oidc_client


class OidcStatus(Endpoint):
    """Whether OIDC sign-in is configured."""

    method = "GET"
    path = "/oidc/status"
    response = OidcState

    def handle(self) -> dict[str, object]:
        return {"configured": oidc_client() is not None, "provider": "OIDC"}
