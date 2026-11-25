"""Configuration of stored credentials."""

from __future__ import annotations

from pydantic import SecretStr

from core.config import EchoraConfig, secret


class CredentialsConfig(EchoraConfig, env_prefix="ECHORA_"):
    """The key that encrypts stored credentials (Navidrome passwords, provider keys)."""

    credential_encryption_key: SecretStr = secret(
        legacy="CREDENTIAL_ENCRYPTION_KEY", description="Fernet key."
    )
