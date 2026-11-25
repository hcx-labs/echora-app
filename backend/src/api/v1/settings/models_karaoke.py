"""``PUT /settings/models/karaoke``."""

from __future__ import annotations

import psycopg
from fastapi import Depends
from psycopg.rows import dict_row

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.v1.settings._deps import admin_required
from backend.api.v1.settings._models import ModelSwitch, SwitchWithBacklog
from core.db.config import DatabaseConfig
from worker.kara.pipeline import KARAOKE_PIPELINE_REVISION


class KaraokeSwitch(Endpoint):
    """Turn karaoke alignment on or off for syncs here or on Modal."""

    method = "PUT"
    path = "/models/karaoke"
    request = ModelSwitch
    response = SwitchWithBacklog

    def handle(
        self, request: ModelSwitch, user: SessionUser = Depends(admin_required)
    ) -> dict[str, object]:
        column = (
            "karaoke_modal_enabled" if request.location == "modal" else "karaoke_processing_enabled"
        )
        with (
            psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(
                f"""INSERT INTO analysis_settings
                     (singleton, {column}, karaoke_bound_to_synced_lines, updated_at)
                   VALUES (true,%s,false,now()) ON CONFLICT (singleton) DO UPDATE
                   SET {column}=EXCLUDED.{column},
                       karaoke_bound_to_synced_lines=false, updated_at=now()""",
                (request.enabled,),
            )
            pending = 0
            if request.enabled:
                cursor.execute(
                    """SELECT count(*) AS pending FROM lyrics l
                       WHERE l.text IS NOT NULL
                         AND coalesce((l.provenance->>'synced')::boolean,false)
                         AND NOT EXISTS (SELECT 1 FROM karaoke_lyrics_variants kv
                                         WHERE kv.track_id=l.track_id AND kv.bounded=false
                                           AND kv.provenance->>'pipeline_revision'=%s)""",
                    (KARAOKE_PIPELINE_REVISION,),
                )
                pending = int(cursor.fetchone()["pending"])
        return {"enabled": request.enabled, "location": request.location, "pending": pending}
