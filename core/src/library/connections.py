"""Stored Navidrome connections and the cipher for every stored credential."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import select

from core.db.models import NavidromeConnection
from core.db.session import session_scope
from core.library.config import CredentialsConfig


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
        stored.last_used_at = datetime.now(UTC)
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


def save_connection(credentials: tuple[str, str, str], user_id: uuid.UUID) -> str:
    """Store (or refresh the password of) the user's connection; returns its id."""
    url, username, password = credentials
    encrypted = credential_cipher().encrypt(password.encode())
    with session_scope() as session:
        stored = session.scalar(
            select(NavidromeConnection).where(
                NavidromeConnection.owner_user_id == user_id,
                NavidromeConnection.url == url,
                NavidromeConnection.username == username,
            )
        )
        if stored is None:
            stored = NavidromeConnection(
                url=url, username=username, encrypted_password=encrypted, owner_user_id=user_id
            )
            session.add(stored)
            session.flush()
        else:
            stored.encrypted_password = encrypted
            stored.last_used_at = datetime.now(UTC)
        return str(stored.id)
