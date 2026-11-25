"""``GET /sync/connections/{connection_id}/catalog``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.sync._models import Catalog
from backend.sync.connections import owned_connection
from core.errors import IntegrationError
from core.integrations.navidrome import NavidromeClient


class ConnectionCatalog(Endpoint):
    """The connection's albums and artists."""

    method = "GET"
    path = "/connections/{connection_id}/catalog"
    response = Catalog

    def handle(self, connection_id: str, user: SessionUser = Depends(current_user)) -> dict:
        credentials = owned_connection(connection_id, user["id"])
        try:
            with NavidromeClient(*credentials) as client:
                return client.catalog()
        except Exception as error:
            raise IntegrationError("Could not read the Navidrome catalog") from error
