"""``PUT /settings/lastfm``."""

from __future__ import annotations

import logging

import httpx
from fastapi import Depends
from pydantic import Field, SecretStr

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.requests import PutRequest
from backend.api.responses import ResponseModel
from core.db.session import session_scope
from core.errors import InvalidQueryError
from core.integrations.config import LastfmConfig
from core.library.connections import credential_cipher
from core.users import store as users

logger = logging.getLogger(__name__)


class LastFmUpdate(PutRequest):
    """A Last.fm user name and API key (surrounding whitespace is dropped)."""

    username: str = Field(min_length=1, max_length=120)
    api_key: SecretStr


class LastFmConnected(ResponseModel):
    """The verified account."""

    connected: bool
    username: str
    scrobbles: int


class UpdateLastFm(Endpoint):
    """Verify a Last.fm account with one API call, then store it (the key encrypted)."""

    method = "PUT"
    path = "/lastfm"
    request = LastFmUpdate
    response = LastFmConnected

    def handle(
        self, request: LastFmUpdate, user: SessionUser = Depends(current_user)
    ) -> dict[str, object]:
        api_key = request.api_key.get_secret_value().strip()
        username = request.username.strip()
        try:
            response = httpx.get(
                "https://ws.audioscrobbler.com/2.0/",
                params={
                    "method": "user.getrecenttracks",
                    "user": username,
                    "api_key": api_key,
                    "format": "json",
                    "limit": 1,
                },
                timeout=LastfmConfig.load().validation_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                raise ValueError(str(payload.get("message") or "Last.fm rejected these settings"))
            total = int(payload.get("recenttracks", {}).get("@attr", {}).get("total") or 0)
        except Exception as error:
            logger.warning("Last.fm verification failed for %s: %s", request.username, error)
            raise InvalidQueryError("Could not verify the Last.fm account") from error
        with session_scope() as session:
            preference = users.require_preference(session, user["id"])
            preference.lastfm_username = username
            preference.lastfm_api_key_encrypted = credential_cipher().encrypt(api_key.encode())
            users.touch(preference)
        return {"connected": True, "username": username, "scrobbles": total}
