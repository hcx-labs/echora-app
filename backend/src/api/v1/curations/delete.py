"""``DELETE /curations/{curation_id}``."""

from __future__ import annotations

import uuid

from fastapi import Depends, Response
from sqlalchemy import select

import worker.curation.jobs as curation_jobs
from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from core.db.models import Curation
from core.db.session import session_scope
from core.errors import ConflictError, NotFoundError
from core.integrations.navidrome import NavidromeClient
from core.library.connections import load_connection


class DeleteCuration(Endpoint):
    """Delete a curation and, unless ``delete_navidrome=false``, its Navidrome playlist."""

    method = "DELETE"
    path = "/{curation_id}"
    status_code = 204

    def handle(
        self,
        curation_id: uuid.UUID,
        delete_navidrome: bool = True,
        user: SessionUser = Depends(current_user),
    ) -> Response:
        with curation_jobs.locked(curation_id) as lock_connection:
            with session_scope() as session:
                curation = session.scalar(
                    select(Curation).where(
                        Curation.id == curation_id, Curation.user_id == user["id"]
                    )
                )
                if curation is None:
                    raise NotFoundError("Curation not found")
                curation_jobs.assert_mutable(
                    lock_connection, curation_id, deleting=True, delete_remote=delete_navidrome
                )
                connection_id, playlist_id = (
                    curation.navidrome_connection_id,
                    curation.navidrome_playlist_id,
                )

            # Delete remotely first so a Navidrome failure does not leave an orphaned
            # playlist after Echora has already forgotten its ID.
            if delete_navidrome and playlist_id:
                credentials = load_connection(str(connection_id))
                if credentials is None:
                    raise ConflictError(
                        "Navidrome connection is unavailable; keep the Navidrome playlist "
                        "or reconnect the server"
                    )
                with NavidromeClient(*credentials) as client:
                    client.delete_playlist(str(playlist_id))

            with session_scope() as session:
                curation = session.scalar(
                    select(Curation).where(
                        Curation.id == curation_id, Curation.user_id == user["id"]
                    )
                )
                if curation is not None:
                    session.delete(curation)
            return Response(status_code=204)
