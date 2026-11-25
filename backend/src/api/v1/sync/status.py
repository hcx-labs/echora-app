"""``GET /sync/connections/{connection_id}/status``."""

from __future__ import annotations

import uuid

import psycopg
from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.sync._models import SyncStatus
from backend.sync.connections import owned_connection
from core.db.config import DatabaseConfig
from core.errors import IntegrationError
from core.integrations.navidrome import NavidromeClient
from core.library.membership import relink_renumbered_sources
from worker.recordings.search import index_status


class ConnectionSyncStatus(Endpoint):
    """Scan the connection's library: what is analyzed, what is missing, fingerprint coverage."""

    method = "GET"
    path = "/connections/{connection_id}/status"
    response = SyncStatus

    def handle(self, connection_id: str, user: SessionUser = Depends(current_user)) -> dict:
        credentials = owned_connection(connection_id, user["id"])
        try:
            with NavidromeClient(*credentials) as client:
                tracks = client.all_tracks()
        except Exception as error:
            raise IntegrationError("Could not scan the Navidrome library") from error
        namespace = uuid.uuid5(uuid.NAMESPACE_URL, credentials[0].rstrip("/"))
        with (
            psycopg.connect(DatabaseConfig.load().database_url) as connection,
            connection.cursor() as cursor,
        ):
            # Heal a renumbered library here too, so it never shows as unanalyzed.
            relink_renumbered_sources(connection, credentials[0], tracks)
            cursor.execute(
                """
                SELECT ts.external_id FROM track_sources ts
                JOIN libraries l ON l.id=ts.library_id
                WHERE l.namespace=%s AND ts.source_type='subsonic'
                """,
                (namespace,),
            )
            processed = {str(row[0]) for row in cursor.fetchall()}
            recording_index = index_status(
                connection, namespace, [track.id for track in tracks], user["id"]
            )
        missing = [track for track in tracks if track.id not in processed]
        return {
            "server": credentials[0],
            "total": len(tracks),
            "processed": len(tracks) - len(missing),
            "missing": len(missing),
            "recording_index": recording_index,
            "tracks": [
                {
                    "id": track.id,
                    "title": track.title,
                    "artist": track.artist,
                    "album": track.album,
                    "duration": track.duration,
                    "cover_art": track.raw.get("coverArt"),
                }
                for track in missing[:50]
            ],
        }
