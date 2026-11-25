"""``GET /settings``."""

from __future__ import annotations

from fastapi import Depends
from sqlalchemy import text

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from core.db.models import NavidromeConnection, User, UserPreference
from core.db.session import session_scope
from core.errors import NotFoundError


class Profile(ResponseModel):
    """The account."""

    username: str
    email: str | None
    display_name: str | None
    is_admin: bool


class ModelSwitches(ResponseModel):
    """Instance-wide model switches, for syncs here and on Modal."""

    karaoke_processing_enabled: bool
    hum_processing_enabled: bool
    transcription_processing_enabled: bool
    karaoke_modal_enabled: bool
    hum_modal_enabled: bool
    transcription_modal_enabled: bool
    huggingface_account: str | None
    """The account of the stored Hugging Face token (the token itself is never returned)."""
    has_huggingface_token: bool


class ConnectionSummary(ResponseModel):
    """The selected Navidrome connection."""

    id: str
    url: str
    username: str


class LastFmState(ResponseModel):
    """Whether Last.fm is connected."""

    connected: bool
    username: str | None


class UserSettings(ResponseModel):
    """Everything the settings page shows."""

    profile: Profile
    models: ModelSwitches
    timezone: str
    navidrome: ConnectionSummary | None
    lastfm: LastFmState


class GetSettings(Endpoint):
    """The user's settings and the instance's model switches."""

    method = "GET"
    path = ""
    response = UserSettings

    def handle(self, user: SessionUser = Depends(current_user)) -> dict[str, object]:
        with session_scope() as session:
            stored_user = session.get(User, user["id"])
            preference = session.get(UserPreference, user["id"])
            connection = (
                session.get(NavidromeConnection, preference.navidrome_connection_id)
                if preference and preference.navidrome_connection_id
                else None
            )
            if stored_user is None or preference is None:
                raise NotFoundError("User settings are unavailable")
            switches = session.execute(
                text(
                    """SELECT karaoke_processing_enabled, hum_processing_enabled,
                          transcription_processing_enabled, karaoke_modal_enabled,
                          hum_modal_enabled, transcription_modal_enabled,
                          hf_token_account, hf_token_encrypted IS NOT NULL
                   FROM analysis_settings WHERE singleton=true"""
                )
            ).one_or_none()
            return {
                "profile": {
                    "username": stored_user.username,
                    "email": stored_user.email,
                    "display_name": stored_user.display_name,
                    "is_admin": stored_user.is_admin,
                },
                "models": _models(switches),
                "timezone": preference.timezone,
                "navidrome": None
                if connection is None
                else {
                    "id": str(connection.id),
                    "url": connection.url,
                    "username": connection.username,
                },
                "lastfm": {
                    "connected": bool(
                        preference.lastfm_username and preference.lastfm_api_key_encrypted
                    ),
                    "username": preference.lastfm_username,
                },
            }


def _models(row) -> dict[str, object]:
    """Karaoke, hum and the Modal karaoke/hum switches default to on without a settings row."""
    return {
        "karaoke_processing_enabled": True if row is None else bool(row[0]),
        "hum_processing_enabled": True if row is None else bool(row[1]),
        "transcription_processing_enabled": bool(row and row[2]),
        "karaoke_modal_enabled": True if row is None else bool(row[3]),
        "hum_modal_enabled": True if row is None else bool(row[4]),
        "transcription_modal_enabled": bool(row and row[5]),
        "huggingface_account": row[6] if row else None,
        "has_huggingface_token": bool(row and row[7]),
    }
