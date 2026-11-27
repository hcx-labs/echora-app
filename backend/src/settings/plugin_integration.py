"""The Navidrome plugin integration: each user's ranking profile and API keys.

Keys are ``echora_nd_<random>`` secrets shown once; only their SHA-256 is stored. ``authenticate``
is the plugin endpoints' dependency: a live key of an enabled integration, of an unblocked user,
for the integration's own connection.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from fastapi import Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from core.db.connection import connect
from core.plugin import store


class RankingProfile(BaseModel):
    """How the plugin ranks and what it serves, as stored in ``navidrome_integrations.profile``."""

    model_config = ConfigDict(extra="forbid")
    musical_weight: int = Field(default=80, ge=0, le=100)
    musical_semantic_weight: int = Field(default=70, ge=0, le=100)
    missing_lyrics: Literal["audio", "exclude"] = "audio"
    max_per_artist: int = Field(default=2, ge=1, le=20)
    serve_lyrics: bool = True
    include_translations: bool = True
    lyrics_format: Literal["ttml", "lrc"] = "ttml"
    # Karaoke smoothing inside words (core.lyrics.smoothing), applied to TTML syllable timing.
    karaoke_close_gaps: bool = False
    karaoke_even_pacing: bool = False
    karaoke_min_syllable: bool = False
    key_expiry_days: int = Field(default=90, ge=1, le=3650)


def digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def profile_from(row: dict[str, Any] | None) -> RankingProfile:
    """The stored profile, or the defaults without an integration row."""
    return RankingProfile.model_validate(row["profile"] if row else {})


def issue_key(db, user_id, connection_id, label: str, days: int) -> dict[str, object]:
    """Create a key; the returned ``secret`` is the only copy."""
    now = datetime.now(UTC)
    secret = "echora_nd_" + secrets.token_urlsafe(32)
    identifier = uuid.uuid4()
    expiry = now + timedelta(days=days)
    store.add_key(
        db,
        identifier,
        user_id,
        connection_id,
        label.strip(),
        secret[:18],
        digest(secret),
        now,
        expiry,
    )
    return {"id": str(identifier), "label": label.strip(), "secret": secret, "expires_at": expiry}


def authenticate(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """The key's user, connection and profile (401 for a missing, revoked or expired key)."""
    unauthorized = HTTPException(
        401, "Invalid or expired integration key", headers={"WWW-Authenticate": "Bearer"}
    )
    if not authorization or not authorization.startswith("Bearer "):
        raise unauthorized
    with connect() as db:
        row = store.key_holder(db, digest(authorization[7:]))
        if row is None:
            raise unauthorized
        store.record_key_use(db, row["key_id"])
    row["profile"] = profile_from(row)
    return row
