"""External processing endpoints: Modal settings (admin), preparation, and availability."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from worker.remote.config import RemoteConfig
from worker.remote.constants import GPU_TYPES
from worker.remote.session import ModalConfig, workspace
from worker.remote.settings import (
    SETUP_JOB,
    _cipher,
    _connect,
    _has_hf_token,
    _public,
    _record,
    _setup_job,
    compute_for,
    config,
    load,
)


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    token_id: str = Field(default="", max_length=200)
    # Omitted or null keeps the stored secret; an empty string clears it.
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


class RedactedRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def redacted(request):
            try:
                return await handler(request)
            except RequestValidationError:
                # Model-level validation errors can echo the body, including the secret.
                raise HTTPException(422, "Invalid External processing settings") from None

        return redacted


def router(require_user):
    api = APIRouter(
        prefix="/settings/external-processing", tags=["settings"], route_class=RedactedRoute
    )

    def admin(user=Depends(require_user)):
        if not user.get("is_admin"):
            raise HTTPException(403, "Administrator access required")
        return user

    @api.get("")
    def read_settings(user=Depends(admin)):
        with _connect() as db:
            return _public(load(db), _setup_job(db), _has_hf_token(db))

    @api.put("")
    def write_settings(body: SettingsUpdate, user=Depends(admin)):
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
                    except Exception:
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

    @api.post("/prepare", status_code=202)
    def start_setup(user=Depends(admin)):
        import core.jobs.queue as jobs

        if config() is None:
            raise HTTPException(409, "Enable Modal and save a token first")
        return jobs.enqueue(SETUP_JOB, "analysis", user["id"], payload={}, dedupe_key=SETUP_JOB)

    @api.get("/availability")
    def availability(user=Depends(require_user)):
        """Whether this user may process a sync on Modal, and the preselected choice."""
        try:
            default = compute_for(uuid.UUID(str(user["id"])), None)
            allowed = compute_for(uuid.UUID(str(user["id"])), "modal") == "modal"
        except PermissionError:
            allowed = False
            default = "local"
        return {"available": allowed, "default": default}

    return api
