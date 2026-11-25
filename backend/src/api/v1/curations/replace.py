"""``PUT /curations/{curation_id}``."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from fastapi import Depends
from sqlalchemy import select

import worker.curation.jobs as curation_jobs
from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.curations._models import CurationReplaceRequest, CurationSaved
from backend.api.v1.curations._shared import recipe_columns, refresh, saved, unique_name
from core.db.models import Curation
from core.db.session import session_scope
from core.errors import NotFoundError


class ReplaceCuration(Endpoint):
    """Replace a saved recipe and generate a new playlist revision."""

    method = "PUT"
    path = "/{curation_id}"
    request = CurationReplaceRequest
    response = CurationSaved
    status_code = 202

    def handle(
        self,
        curation_id: uuid.UUID,
        request: CurationReplaceRequest,
        user: SessionUser = Depends(current_user),
    ) -> dict:
        with (
            curation_jobs.locked(curation_id) as lock_connection,
            unique_name(),
            session_scope() as session,
        ):
            curation = session.scalar(
                select(Curation).where(Curation.id == curation_id, Curation.user_id == user["id"])
            )
            if curation is None:
                raise NotFoundError("Curation not found")
            curation_jobs.assert_mutable(lock_connection, curation_id)
            for column, value in recipe_columns(request).items():
                setattr(curation, column, value)
            curation.next_refresh_at = (
                datetime.now(UTC) + timedelta(hours=curation.refresh_interval_hours)
                if request.refresh_enabled
                else None
            )
            curation.updated_at = datetime.now(UTC)
        return saved(request, curation_id, refresh(curation_id, user["id"]))
