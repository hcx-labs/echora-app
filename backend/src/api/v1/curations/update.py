"""``PATCH /curations/{curation_id}``."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from fastapi import Depends
from sqlalchemy import select

import worker.curation.jobs as curation_jobs
from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.curations._models import CurationChanged, CurationUpdateRequest
from core.db.models import Curation
from core.db.session import session_scope
from core.errors import NotFoundError


class UpdateCuration(Endpoint):
    """Change a curation's schedule, refresh mode, size or familiarity."""

    method = "PATCH"
    path = "/{curation_id}"
    request = CurationUpdateRequest
    response = CurationChanged

    def handle(
        self,
        curation_id: uuid.UUID,
        request: CurationUpdateRequest,
        user: SessionUser = Depends(current_user),
    ) -> dict:
        values = request.model_dump(exclude_none=True)
        with curation_jobs.locked(curation_id) as lock_connection:
            with session_scope() as session:
                curation = session.scalar(
                    select(Curation).where(
                        Curation.id == curation_id, Curation.user_id == user["id"]
                    )
                )
                if curation is None:
                    raise NotFoundError("Curation not found")
                curation_jobs.assert_mutable(lock_connection, curation_id)
                for key, value in values.items():
                    setattr(curation, key, value)
                if "refresh_enabled" in values:
                    curation.next_refresh_at = (
                        datetime.now(UTC) + timedelta(hours=curation.refresh_interval_hours)
                        if values["refresh_enabled"]
                        else None
                    )
                curation.updated_at = datetime.now(UTC)
            return {"id": str(curation_id), **values}
