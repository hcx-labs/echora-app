"""``PUT /settings/navidrome``."""

from __future__ import annotations

import logging
import uuid

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.navidrome import Credentials
from backend.api.responses import ResponseModel
from core.db.session import session_scope
from core.errors import InvalidQueryError
from core.integrations.navidrome import NavidromeClient
from core.library.connections import save_connection
from core.library.membership import attach_user_library
from core.users import store as users

logger = logging.getLogger(__name__)


class NavidromeState(ResponseModel):
    """The saved, verified connection."""

    id: uuid.UUID
    url: str
    username: str
    server: str | None
    """The server version Navidrome reported."""


class UpdateNavidrome(Endpoint):
    """Verify a Navidrome account, save it and select it as the user's connection."""

    method = "PUT"
    path = "/navidrome"
    request = Credentials
    response = NavidromeState

    def handle(
        self, request: Credentials, user: SessionUser = Depends(current_user)
    ) -> dict[str, object]:
        credentials = request.as_tuple()
        try:
            with NavidromeClient(*credentials) as client:
                version = client.ping()
        except Exception as error:
            logger.warning("Navidrome connection test failed for %s: %s", request.url, error)
            raise InvalidQueryError("Could not connect to Navidrome") from error
        connection_id = uuid.UUID(save_connection(credentials, user["id"]))
        with session_scope() as session:
            preference = users.require_preference(session, user["id"])
            preference.navidrome_connection_id = connection_id
            users.touch(preference)
        attach_user_library(user["id"], credentials[0])
        return {
            "id": connection_id,
            "url": credentials[0],
            "username": credentials[1],
            "server": version,
        }
