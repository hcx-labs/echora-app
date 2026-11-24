"""Stored Navidrome connections and the cipher for every stored credential."""

from __future__ import annotations

from __future__ import annotations
from core.library.config import CredentialsConfig
from datetime import datetime, timezone
import uuid
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import select
from core.db.session import session_scope
from core.db.models import NavidromeConnection


def credential_cipher() -> Fernet:
    key = CredentialsConfig.load().credential_encryption_key.get_secret_value()
    if not key:
        raise RuntimeError("CREDENTIAL_ENCRYPTION_KEY is required")
    return Fernet(key.encode())


def load_connection(
    connection_id: str, user_id: uuid.UUID | None = None
) -> tuple[str, str, str] | None:
    try:
        identifier = uuid.UUID(connection_id)
    except ValueError:
        return None
    with session_scope() as session:
        filters = [NavidromeConnection.id == identifier]
        if user_id is not None:
            filters.append(NavidromeConnection.owner_user_id == user_id)
        stored = session.scalar(select(NavidromeConnection).where(*filters))
        if stored is None:
            return None
        stored.last_used_at = datetime.now(timezone.utc)
        url, username, encrypted_password = (
            stored.url,
            stored.username,
            bytes(stored.encrypted_password),
        )
    try:
        password = credential_cipher().decrypt(encrypted_password).decode()
    except InvalidToken as error:
        raise RuntimeError("Stored Navidrome credentials cannot be decrypted") from error
    return url, username, password
