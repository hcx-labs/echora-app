"""``GET /curations``."""

from __future__ import annotations

import psycopg
from fastapi import Depends
from psycopg.rows import dict_row

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.curations._models import CurationList
from core.db.config import DatabaseConfig


class ListCurations(Endpoint):
    """The user's curations with their latest revision's songs."""

    method = "GET"
    path = ""
    response = CurationList

    def handle(self, user: SessionUser = Depends(current_user)) -> dict:
        with (
            psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(
                """SELECT c.*, latest.recipe,
                       coalesce((SELECT jsonb_agg(jsonb_build_object('id', pt.id, 'title', pt.title, 'artist', pt.artist, 'album', pt.album)) FROM tracks pt WHERE pt.id=ANY(c.positive_track_ids)), '[]') AS positive_tracks,
                       coalesce((SELECT jsonb_agg(jsonb_build_object('id', nt.id, 'title', nt.title, 'artist', nt.artist, 'album', nt.album)) FROM tracks nt WHERE nt.id=ANY(c.negative_track_ids)), '[]') AS negative_tracks,
                       coalesce(jsonb_agg(jsonb_build_object(
                         'id', t.id, 'title', t.title, 'artist', t.artist, 'album', t.album,
                         'duration_seconds', t.duration_seconds, 'source_id', crt.source_id,
                         'position', crt.position, 'score', crt.score,
                         'percentile', nullif(crt.evidence->>'percentile','')::real,
                         'retained', coalesce((crt.evidence->>'retained')::boolean, false),
                         'evidence', crt.evidence,
                         'cover_art', (
                           SELECT max(ts.source_data->>'coverArt')
                           FROM user_track_links utl
                           JOIN track_sources ts ON ts.library_id=utl.library_id
                                                 AND ts.track_id=utl.track_id
                                                 AND ts.external_id=crt.source_id
                           WHERE utl.user_id=c.user_id AND utl.track_id=crt.track_id
                         )
                       ) ORDER BY crt.position) FILTER (WHERE crt.track_id IS NOT NULL), '[]') AS tracks
                   FROM curations c
                   LEFT JOIN LATERAL (
                     SELECT id, recipe FROM curation_revisions
                     WHERE curation_id=c.id ORDER BY revision_number DESC LIMIT 1
                   ) latest ON true
                   LEFT JOIN curation_revision_tracks crt ON crt.revision_id=latest.id
                   LEFT JOIN tracks t ON t.id=crt.track_id
                   WHERE c.user_id=%s GROUP BY c.id, latest.recipe ORDER BY c.updated_at DESC""",
                (user["id"],),
            )
            return {"curations": cursor.fetchall()}
