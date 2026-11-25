"""``GET /settings/external-ai/endpoint``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.v1.settings._deps import admin_required
from backend.api.v1.settings._external_ai import EndpointView, current, endpoint_view


class ExternalAIEndpoint(Endpoint):
    """The External AI endpoint settings (administrators only)."""

    method = "GET"
    path = "/external-ai/endpoint"
    response = EndpointView

    def handle(self, user: SessionUser = Depends(admin_required)) -> dict:
        return endpoint_view(current())
