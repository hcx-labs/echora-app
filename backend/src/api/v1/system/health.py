"""``GET /system/health``."""

from __future__ import annotations

from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel


class HealthStatus(ResponseModel):
    """The API is up."""

    status: str
    service: str


class Health(Endpoint):
    """Report that the API is up."""

    method = "GET"
    path = "/health"
    response = HealthStatus

    def handle(self) -> dict[str, str]:
        return {"status": "ok", "service": "echora-backend"}
