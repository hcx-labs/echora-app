"""``GET /settings/external-processing``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.v1.settings._deps import admin_required
from backend.api.v1.settings._external_processing import ExternalProcessingView
from core.db.connection import connect
from worker.remote.settings import _has_hf_token, _public, _setup_job, load


class ExternalProcessingRead(Endpoint):
    """The Modal settings and deployment state (administrators only)."""

    method = "GET"
    path = "/external-processing"
    response = ExternalProcessingView

    def handle(self, user: SessionUser = Depends(admin_required)) -> dict:
        with connect() as db:
            return _public(load(db), _setup_job(db), _has_hf_token(db))
