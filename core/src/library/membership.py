"""Which libraries and tracks a user can see, kept in step with the music server."""

from __future__ import annotations

from __future__ import annotations
import uuid
import psycopg
from psycopg.rows import dict_row
from core.revisions import SUPPORTED_PROFILE_MODELS
from core.settings import get_settings


def attach_user_library(user_id: uuid.UUID, source_url: str) -> None:
    namespace = uuid.uuid5(uuid.NAMESPACE_URL, source_url.rstrip("/"))
    with psycopg.connect(get_settings().database_url) as connection, connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO user_libraries (user_id, library_id)
               SELECT %s, id FROM libraries WHERE namespace=%s
               ON CONFLICT DO NOTHING""",
            (user_id, namespace),
        )


def reconcile_user_tracks(
    user_id: uuid.UUID, source_url: str, external_ids: list[str]
) -> dict[str, int]:
    from core.library.visibility import update_memberships

    namespace = uuid.uuid5(uuid.NAMESPACE_URL, source_url.rstrip("/"))
    with (
        psycopg.connect(get_settings().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        # A first full catalog can precede all source/identity discovery.
        cursor.execute(
            """INSERT INTO libraries (name, root_path, namespace)
               VALUES ('Navidrome', %s, %s)
               ON CONFLICT (namespace) DO UPDATE SET root_path=EXCLUDED.root_path
               RETURNING id""",
            (source_url.rstrip("/"), namespace),
        )
        library_id = cursor.fetchone()["id"]
        return update_memberships(connection, library_id, user_id, external_ids, full=True)


def user_audio_track_ids(user_id: uuid.UUID) -> list[uuid.UUID]:
    with psycopg.connect(get_settings().database_url) as connection, connection.cursor() as cursor:
        cursor.execute(
            """SELECT DISTINCT utl.track_id
               FROM user_track_links utl
               JOIN current_embeddings e ON e.track_id=utl.track_id
               JOIN analysis_runs ar ON ar.id=e.run_id
               WHERE utl.user_id=%s
                 AND e.embedding_type='audio-track' AND e.window_index IS NULL
                 AND ar.model_name=ANY(%s)
               ORDER BY utl.track_id""",
            (user_id, list(SUPPORTED_PROFILE_MODELS)),
        )
        return [row[0] for row in cursor.fetchall()]


def relink_renumbered_sources(connection, url: str, tracks) -> int:
    from core.library.relink import relink_sources

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT id FROM libraries WHERE namespace=%s",
            (uuid.uuid5(uuid.NAMESPACE_URL, url.rstrip("/")),),
        )
        library = cursor.fetchone()
    if library is None:
        return 0
    relinked = relink_sources(connection, library[0], {track.id: track.raw for track in tracks})
    connection.commit()
    return relinked

