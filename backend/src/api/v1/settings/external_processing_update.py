"""``PUT /settings/external-processing``."""

from __future__ import annotations

from datetime import datetime

from fastapi import Depends, HTTPException
from pydantic import Field, SecretStr, field_validator

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.requests import PutRequest
from backend.api.v1.settings._deps import admin_required
from backend.api.v1.settings._external_processing import ExternalProcessingView
from worker.remote.config import RemoteConfig
from worker.remote.constants import GPU_TYPES
from worker.remote.session import ModalConfig, workspace
from worker.remote.settings import (
    _cipher,
    _connect,
    _has_hf_token,
    _public,
    _record,
    _setup_job,
    load,
)


class ExternalProcessingUpdate(PutRequest):
    """The Modal settings. ``token_secret``: omitted or null keeps it, "" clears it."""

    enabled: bool = False
    token_id: str = Field(default="", max_length=200)
    token_secret: SecretStr | None = Field(default=None, max_length=1000, exclude=True)
    gpu: str = "L40S"
    default_compute: str = Field(default="local", pattern="^(local|modal)$")
    allow_users: bool = False

    @field_validator("token_id")
    @classmethod
    def token(cls, value: str) -> str:
        value = value.strip()
        if value and not value.startswith("ak-"):
            raise ValueError("Modal token IDs start with ak-")
        return value

    @field_validator("gpu")
    @classmethod
    def gpu_type(cls, value: str) -> str:
        if value not in GPU_TYPES:
            raise ValueError("Unsupported GPU type")
        return value


class ExternalProcessingSettingsUpdate(Endpoint):
    """Save the Modal settings, checking a new token against Modal (administrators only).

    New credentials or another GPU type mark the deployment unprepared.
    """

    method = "PUT"
    path = "/external-processing"
    request = ExternalProcessingUpdate
    response = ExternalProcessingView

    def handle(
        self, body: ExternalProcessingUpdate, user: SessionUser = Depends(admin_required)
    ) -> dict:
        with _connect() as db:
            current = load(db)
            secret = (
                body.token_secret.get_secret_value().strip()
                if body.token_secret is not None
                else None
            )
            encrypted = current["token_secret_encrypted"]
            if secret is not None:
                encrypted = _cipher().encrypt(secret.encode()) if secret else None
            credentials_changed = secret is not None or body.token_id != current["token_id"]
            workspace_name = current["workspace"]
            if body.enabled and not (body.token_id and encrypted):
                raise HTTPException(422, "Enter a Modal token ID and secret before enabling Modal")
            if credentials_changed:
                workspace_name = None
                if body.token_id and encrypted:
                    candidate = ModalConfig(
                        body.token_id,
                        _cipher().decrypt(bytes(encrypted)).decode(),
                        body.gpu,
                        RemoteConfig.load().image,
                    )
                    try:
                        workspace_name = workspace(candidate)
                    except Exception:  # noqa: BLE001 - Modal's errors may carry the token
                        raise HTTPException(422, "Modal did not accept this token") from None
            # A new workspace or GPU type needs a new deployment before use.
            stale = credentials_changed or body.gpu != current["gpu"]
            values = {
                "enabled": body.enabled,
                "token_id": body.token_id,
                "token_secret_encrypted": encrypted,
                "gpu": body.gpu,
                "default_compute": body.default_compute,
                "allow_users": body.allow_users,
                "workspace": workspace_name,
                "updated_at": datetime.now().astimezone(),
            }
            if stale:
                values.update(
                    status="unprepared",
                    status_detail=None,
                    deployed_code=None,
                    deployed_models=None,
                    checked_at=None,
                )
            _record(db, **values)
            db.commit()
            return _public(load(db), _setup_job(db), _has_hf_token(db))
